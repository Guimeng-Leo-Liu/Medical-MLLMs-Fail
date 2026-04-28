import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from measure_attention import (
    ANSWER_INSTRUCTION,
    GENERAL_QUESTION,
    HUATUO_NUM_IMG_TOKENS,
    HUATUO_PATCHES,
    HUATUO_SIZE,
    compute_metrics,
    convert_bbox_after_resize,
    get_dataset_paths,
    get_num_layers,
    get_required_attentions,
    get_tokens_covering_bbox,
    infer_model_type,
    load_backend,
    load_questions,
    parse_args,
    process_input_image_and_qs_qwen,
    uses_xyxy_bbox,
)


def get_attention_qwen(inputs, general_inputs, pos, pos_end, layers, backend):
    output = backend["model"](
        **inputs, output_attentions=True, output_hidden_states=False, return_dict=True, use_cache=False
    )
    general_output = backend["model"](
        **general_inputs, output_attentions=True, output_hidden_states=False, return_dict=True, use_cache=False
    )
    attentions = get_required_attentions(output, "Qwen")
    general_attentions = get_required_attentions(general_output, "Qwen general prompt")

    att = np.array(
        [
            attentions[layer][0, :, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()
            for layer in layers
        ]
    )
    general_att = np.array(
        [
            general_attentions[layer][0, :, -1, pos:pos_end].to(torch.float32).detach().cpu().numpy()
            for layer in layers
        ]
    )
    return att, general_att


def get_huatuo_attention_for_prompt(prompt, image_path, layers, backend):
    model_output, input_ids, _ = backend["bot"].inference(prompt, image_path, return_att_map=True)
    attentions = get_required_attentions(model_output, "HuatuoGPT-Vision")

    input_ids = input_ids[0].cpu()
    image_token_positions = torch.where(input_ids == -200)[0]
    if image_token_positions.numel() == 0:
        raise ValueError("Cannot find Huatuo image placeholder token (-200) in input_ids.")
    image_start = int(image_token_positions[0].item())

    return np.array(
        [
            attentions[layer][0, :, -1, image_start : image_start + HUATUO_NUM_IMG_TOKENS]
            .to(torch.float32)
            .detach()
            .cpu()
            .numpy()
            for layer in layers
        ]
    )


def generate_attention_maps_huatuo(question, image_path, layers, backend):
    prompt = f"{question} {ANSWER_INSTRUCTION}"
    general_prompt = f"{GENERAL_QUESTION} {ANSWER_INSTRUCTION}"

    att_maps = get_huatuo_attention_for_prompt(prompt, image_path, layers, backend)
    general_att_maps = get_huatuo_attention_for_prompt(general_prompt, image_path, layers, backend)

    return att_maps, general_att_maps, (HUATUO_PATCHES, HUATUO_PATCHES), HUATUO_SIZE


def get_sample_attention(sample, data_path, dataset, backend, layers):
    image_name = sample["image"]
    image_path = os.path.join(data_path, image_name)
    original_size = Image.open(image_path).size
    question = sample["question"]
    original_bboxes = sample.get("bbox", sample.get("original_bboxes"))
    if original_bboxes is None:
        raise KeyError(f"Sample has no bbox or original_bboxes field: {sample}")

    if backend["type"] == "huatuogpt_vision":
        att_maps, general_att_maps, token_shape, new_size = generate_attention_maps_huatuo(
            question, image_path, layers=layers, backend=backend
        )
    else:
        inputs, general_inputs, token_shape, new_size, pos, pos_end = process_input_image_and_qs_qwen(
            image_path, question, backend
        )
        att_maps, general_att_maps = get_attention_qwen(
            inputs, general_inputs, pos, pos_end, layers=layers, backend=backend
        )

    xyxy_format = uses_xyxy_bbox(dataset, image_name)
    gt_tokens = set()
    for bbox in original_bboxes:
        resized_bbox = convert_bbox_after_resize(bbox, original_size, new_size, xyxy_format)
        gt_tokens.update(
            get_tokens_covering_bbox(
                resized_bbox,
                num_tokens=token_shape,
                img_size=new_size,
                xyxy_format=xyxy_format,
            )
        )

    return att_maps, general_att_maps, list(gt_tokens), token_shape


def save_head_metrics(output_root, model_name, dataset, q_type, attention_ratios, attention_kls, attention_js):
    save_path = os.path.join(output_root, model_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(attention_ratios, f"{save_path}/{dataset}_{q_type}_attention_head_ratios_normalized.pt")
    torch.save(attention_kls, f"{save_path}/{dataset}_{q_type}_attention_head_kl_normalized.pt")
    torch.save(attention_js, f"{save_path}/{dataset}_{q_type}_attention_head_js_normalized.pt")


def main():
    args = parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    model_name = os.path.basename(os.path.normpath(args.model_path))
    model_type = infer_model_type(args.model_path, args.model_type)
    backend = load_backend(
        args.model_path,
        args.model_base_path,
        model_type=model_type,
        device=args.device,
        huatuo_repo=args.huatuo_repo,
        qwen_vl_utils_path=args.qwen_vl_utils_path,
    )

    attention_ratios_normalized = []
    attention_kl_normalized = []
    attention_js_normalized = []

    input_path, data_path = get_dataset_paths(args)
    questions = load_questions(input_path)
    layers = range(get_num_layers(backend))

    with torch.no_grad():
        for sample in tqdm(questions):
            att_maps, general_att_maps, gt_tokens, token_shape = get_sample_attention(
                sample, data_path, args.dataset, backend, layers
            )
            att_ratio, att_kl, att_js = compute_metrics(att_maps, general_att_maps, gt_tokens, token_shape)

            attention_ratios_normalized.append(att_ratio)
            attention_kl_normalized.append(att_kl)
            attention_js_normalized.append(att_js)

    save_head_metrics(
        args.output_root,
        model_name,
        args.dataset,
        args.q_type,
        attention_ratios_normalized,
        attention_kl_normalized,
        attention_js_normalized,
    )


if __name__ == "__main__":
    main()
