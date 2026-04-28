import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."
GENERAL_QUESTION = "Write a general description of the image."

HUATUO_NUM_IMG_TOKENS = 576
HUATUO_PATCHES = 24
HUATUO_SIZE = (336, 336)


def prepend_sys_path(path):
    if path and os.path.isdir(path):
        abs_path = os.path.abspath(path)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)


def infer_model_type(model_path, explicit_model_type="auto"):
    if explicit_model_type and explicit_model_type != "auto":
        return explicit_model_type
    lower_path = model_path.lower()
    if "huatuogpt" in lower_path:
        return "huatuogpt_vision"
    if "qwen3" in lower_path:
        return "qwen3vl"
    if "qwen2" in lower_path or "lingshu" in lower_path:
        return "qwen2_5vl"
    raise ValueError(f"Cannot infer model type from model_path: {model_path}. Please set --model_type.")


def load_backend(model_path, model_base_path, model_type, device, huatuo_repo, qwen_vl_utils_path):
    if model_type in {"qwen3vl", "qwen2_5vl"}:
        from transformers import AutoProcessor

        prepend_sys_path(qwen_vl_utils_path)
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Cannot import qwen_vl_utils. Install qwen-vl-utils or set "
                "--qwen_vl_utils_path to the local Qwen3-VL/qwen-vl-utils/src directory."
            ) from exc

        if model_type == "qwen3vl":
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map="auto",
            )
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json")) and not os.path.exists(
                os.path.join(model_path, "config.json")
            )
            if is_adapter:
                if not model_base_path:
                    raise ValueError("--model_base_path is required when loading a LoRA adapter.")
                print(f"Loading base model from: {model_base_path}")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_base_path,
                    dtype="auto",
                    attn_implementation="eager",
                    device_map="auto",
                )
                from peft import PeftModel

                print(f"Loading adapter from: {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
            else:
                print(f"Loading full model from: {model_path}")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    dtype="auto",
                    attn_implementation="eager",
                    device_map="auto",
                )

        try:
            processor = AutoProcessor.from_pretrained(model_path)
        except Exception:
            if not model_base_path:
                raise
            processor = AutoProcessor.from_pretrained(model_base_path)

        processor.image_processor.max_pixels = 50176
        processor.image_processor.min_pixels = 784
        image_patch_size = getattr(processor.image_processor, "patch_size", 14)
        merge_size = getattr(processor.image_processor, "merge_size", 2)

        return {
            "type": model_type,
            "model": model,
            "processor": processor,
            "process_vision_info": process_vision_info,
            "image_patch_size": image_patch_size,
            "merge_size": merge_size,
        }

    if model_type == "huatuogpt_vision":
        prepend_sys_path(huatuo_repo)
        cli_path = os.path.join(huatuo_repo, "cli_new.py")
        if not os.path.exists(cli_path):
            raise FileNotFoundError(
                f"Cannot find cli_new.py at {cli_path}. Move cli_new.py into the HuatuoGPT-Vision folder "
                "or pass the correct --huatuo_repo path."
            )
        from cli_new import HuatuoChatbot  # type: ignore[import-not-found]

        bot = HuatuoChatbot(model_path, device=device)
        return {
            "type": model_type,
            "bot": bot,
        }

    raise ValueError(f"Unsupported model_type: {model_type}")


def make_gt_mask(gt_tokens, num_cols, num_visual_tokens):
    if not gt_tokens:
        raise ValueError("No visual tokens overlap the ground-truth bounding boxes.")

    gt_mask = np.zeros(num_visual_tokens)
    for x_pos, y_pos in gt_tokens:
        gt_mask[y_pos * num_cols + x_pos] = 1
    return gt_mask / gt_mask.sum()


def attention_ratio_vectorized(visual_attentions, gt_tokens, num_cols, num_tokens):
    if not gt_tokens:
        raise ValueError("No visual tokens overlap the ground-truth bounding boxes.")

    token_indices = np.array([token[1] * num_cols + token[0] for token in gt_tokens])
    relevant_attention = np.sum(visual_attentions[..., token_indices], axis=-1)
    average_attention = np.sum(visual_attentions, axis=-1) / num_tokens * len(token_indices)
    return relevant_attention / (average_attention + 1e-8)


def js_divergence_vectorized(att_map, gt_tokens, num_cols, epsilon=1e-8):
    att_map = att_map / (att_map.sum(axis=-1, keepdims=True) + epsilon)
    gt_mask = make_gt_mask(gt_tokens, num_cols, att_map.shape[-1])

    m = 0.5 * (att_map + gt_mask)
    kl_att_m = np.sum(att_map * np.log((att_map + epsilon) / (m + epsilon)), axis=-1)
    kl_gt_m = np.sum(gt_mask * np.log((gt_mask + epsilon) / (m + epsilon)), axis=-1)
    js_div = 0.5 * (kl_att_m + kl_gt_m)
    return js_div


def kl_divergence_vectorized(att_map, gt_tokens, num_cols, epsilon=1e-8):
    att_map = att_map / (att_map.sum(axis=-1, keepdims=True) + epsilon)
    gt_mask = make_gt_mask(gt_tokens, num_cols, att_map.shape[-1])

    epsilon = 1e-12
    att_map = np.clip(att_map, epsilon, 1)
    gt_mask = np.clip(gt_mask, epsilon, 1)
    return np.sum(gt_mask * np.log(gt_mask / att_map), axis=-1)


def uses_xyxy_bbox(dataset, image_name):
    return dataset == "VGMED" and not image_name.startswith("Slake")


def convert_bbox_after_resize(bbox, original_size, resized_size, xyxy_format):
    w_orig, h_orig = original_size
    w_new, h_new = resized_size
    scale_x = w_new / w_orig
    scale_y = h_new / h_orig

    if xyxy_format:
        x_min, y_min, x_max, y_max = bbox
        return x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y

    x_min, y_min, width, height = bbox
    return x_min * scale_x, y_min * scale_y, width * scale_x, height * scale_y


def get_tokens_covering_bbox(bbox, num_tokens, img_size, xyxy_format):
    img_width, img_height = img_size
    num_rows, num_cols = num_tokens
    token_width = img_width / num_cols
    token_height = img_height / num_rows

    if xyxy_format:
        x_min, y_min, x_max, y_max = bbox
    else:
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height

    x_min_token = int(max(np.floor(x_min / token_width), 0))
    y_min_token = int(max(np.floor(y_min / token_height), 0))
    x_max_token = int(np.ceil(x_max / token_width))
    y_max_token = int(np.ceil(y_max / token_height))
    x_max_token = int(min(x_max_token, num_cols))
    y_max_token = int(min(y_max_token, num_rows))
    return [(x, y) for x in range(x_min_token, x_max_token) for y in range(y_min_token, y_max_token)]


def make_qwen_message(image_path, text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
    ]


def process_input_image_and_qs_qwen(image_path, qs, backend):
    messages_query = make_qwen_message(image_path, f"{qs} {ANSWER_INSTRUCTION}")
    image_inputs, _ = backend["process_vision_info"](
        messages_query,
        image_patch_size=backend["image_patch_size"],
    )

    text_query = backend["processor"].apply_chat_template(
        messages_query, tokenize=False, add_generation_prompt=True
    )
    inputs = backend["processor"](
        text=[text_query],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(backend["model"].device)

    messages_general = make_qwen_message(image_path, f"{GENERAL_QUESTION} {ANSWER_INSTRUCTION}")
    text_general = backend["processor"].apply_chat_template(
        messages_general, tokenize=False, add_generation_prompt=True
    )
    general_inputs = backend["processor"](
        text=[text_general],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(backend["model"].device)

    image_inputs_aux = backend["processor"].image_processor(images=image_inputs)
    output_shape = image_inputs_aux["image_grid_thw"].numpy().squeeze(0)[1:] / backend["merge_size"]
    output_shape = output_shape.astype(int)

    vision_start_token_id = backend["processor"].tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_token_id = backend["processor"].tokenizer.convert_tokens_to_ids("<|vision_end|>")
    pos = inputs["input_ids"].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs["input_ids"].tolist()[0].index(vision_end_token_id)
    return inputs, general_inputs, output_shape, image_inputs[0].size, pos, pos_end


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
            attentions[layer][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).detach().cpu().numpy()
            for layer in layers
        ]
    )
    general_att = np.array(
        [
            general_attentions[layer][0, :, -1, pos:pos_end]
            .mean(dim=0)
            .to(torch.float32)
            .detach()
            .cpu()
            .numpy()
            for layer in layers
        ]
    )
    return att, general_att


def get_required_attentions(model_output, model_name):
    attentions = getattr(model_output, "attentions", None)
    if attentions is None and isinstance(model_output, dict):
        attentions = model_output.get("attentions")
    if attentions is None:
        raise RuntimeError(
            f"{model_name} did not return attentions. Make sure the model is using eager attention "
            "instead of sdpa or flash attention before calling with output_attentions=True."
        )
    return attentions


def generate_attention_maps_huatuo(question, image_path, layers, backend):
    prompt = f"{question} {ANSWER_INSTRUCTION}"
    general_prompt = f"{GENERAL_QUESTION} {ANSWER_INSTRUCTION}"

    model_output, input_ids, _ = backend["bot"].inference(prompt, image_path, return_att_map=True)
    attentions = get_required_attentions(model_output, "HuatuoGPT-Vision")
    input_ids = input_ids[0].cpu()
    index = torch.where(input_ids == -200)[0]
    if index.numel() == 0:
        raise ValueError("Cannot find Huatuo image placeholder token (-200) in input_ids.")
    index = int(index[0].item())
    att_maps = np.array(
        [
            attentions[layer][0, :, -1, index : index + HUATUO_NUM_IMG_TOKENS]
            .mean(dim=0)
            .to(torch.float32)
            .detach()
            .cpu()
            .numpy()
            for layer in layers
        ]
    )

    model_output, input_ids, _ = backend["bot"].inference(general_prompt, image_path, return_att_map=True)
    attentions = get_required_attentions(model_output, "HuatuoGPT-Vision")
    input_ids = input_ids[0].cpu()
    index = torch.where(input_ids == -200)[0]
    if index.numel() == 0:
        raise ValueError("Cannot find Huatuo image placeholder token (-200) in input_ids.")
    index = int(index[0].item())
    general_att_maps = np.array(
        [
            attentions[layer][0, :, -1, index : index + HUATUO_NUM_IMG_TOKENS]
            .mean(dim=0)
            .to(torch.float32)
            .detach()
            .cpu()
            .numpy()
            for layer in layers
        ]
    )

    return att_maps, general_att_maps, (HUATUO_PATCHES, HUATUO_PATCHES), HUATUO_SIZE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["VGMED", "COCO"], required=True)
    parser.add_argument("--q_type", type=str, choices=["localization", "attribute"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base_path", type=str, required=False)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["qwen2_5vl", "qwen3vl", "huatuogpt_vision"],
        required=True
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--huatuo_repo",
        type=str,
        default="./HuatuoGPT-Vision",
        help="Local HuatuoGPT-Vision repo path. Defaults to ./HuatuoGPT-Vision.",
    )
    parser.add_argument("--qwen_vl_utils_path", type=str)
    parser.add_argument("--vgmed_root", type=str)
    parser.add_argument("--coco_root", type=str)
    parser.add_argument("--coco_annotation_root", type=str)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "VGMED" and not args.vgmed_root:
        parser.error("--vgmed_root is required when --dataset VGMED.")
    if args.dataset == "COCO":
        if not args.coco_root:
            parser.error("--coco_root is required when --dataset COCO.")
        if not args.coco_annotation_root:
            parser.error("--coco_annotation_root is required when --dataset COCO.")

    return args


def get_dataset_paths(args):
    if args.dataset == "VGMED":
        return os.path.join(args.vgmed_root, f"VGMED_{args.q_type}.jsonl"), args.vgmed_root
    return os.path.join(args.coco_annotation_root, f"COCO_{args.q_type}.jsonl"), args.coco_root


def load_questions(input_path):
    with open(input_path, "r") as infile:
        return [json.loads(line) for line in infile]


def get_num_layers(backend):
    if backend["type"] == "huatuogpt_vision":
        return backend["bot"].model.config.num_hidden_layers
    return backend["model"].config.num_hidden_layers


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


def compute_metrics(att_maps, general_att_maps, gt_tokens, token_shape):
    num_tokens = token_shape[0] * token_shape[1]
    num_cols = token_shape[1]
    normalized_att_maps = att_maps / general_att_maps
    return (
        attention_ratio_vectorized(normalized_att_maps, gt_tokens, num_cols=num_cols, num_tokens=num_tokens),
        kl_divergence_vectorized(normalized_att_maps, gt_tokens, num_cols=num_cols),
        js_divergence_vectorized(normalized_att_maps, gt_tokens, num_cols=num_cols),
    )


def save_metrics(output_root, model_name, dataset, q_type, attention_ratios, attention_kls, attention_js):
    save_path = os.path.join(output_root, model_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(attention_ratios, f"{save_path}/{dataset}_{q_type}_attention_ratios_normalized.pt")
    torch.save(attention_kls, f"{save_path}/{dataset}_{q_type}_attention_kl_normalized.pt")
    torch.save(attention_js, f"{save_path}/{dataset}_{q_type}_attention_js_normalized.pt")


def main():
    args = parse_args()

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

    save_metrics(
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
