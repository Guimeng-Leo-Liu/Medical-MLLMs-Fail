import copy
import functools
import io
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

NUM_IMG_TOKENS = 576
PATCHES = 24
SIZE = (336, 336)


def load_image(image_file):
    if isinstance(image_file, Image.Image):
        return image_file.convert("RGB")
    if isinstance(image_file, bytes):
        return Image.open(io.BytesIO(image_file)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def load_images(image_files):
    return [load_image(image_file) for image_file in image_files]


def get_required_attentions(model_output, model_name):
    attentions = getattr(model_output, "attentions", None)
    if attentions is None and isinstance(model_output, dict):
        attentions = model_output.get("attentions")
    if attentions is None:
        raise RuntimeError(
            f"{model_name} did not return attentions. Make sure the model uses eager attention "
            "instead of sdpa or flash attention before calling with output_attentions=True."
        )
    return attentions


def get_model_dtype(model):
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(model.parameters()).dtype


def get_llm_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def is_qwen_style_model(model):
    return hasattr(model, "model") and hasattr(model.model, "language_model")


def get_visual_range_huatuo(input_ids):
    input_ids = input_ids[0].cpu() if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2 else input_ids
    if isinstance(input_ids, np.ndarray):
        matches = np.where(input_ids == -200)[0]
        if len(matches) == 0:
            raise ValueError("Cannot find Huatuo image placeholder token (-200) in input_ids.")
        start = int(matches[0])
    else:
        matches = torch.where(input_ids == -200)[0]
        if matches.numel() == 0:
            raise ValueError("Cannot find Huatuo image placeholder token (-200) in input_ids.")
        start = int(matches[0].item())
    return start, start + NUM_IMG_TOKENS


def flatten_input_ids(input_ids):
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().cpu().numpy()
    input_ids = np.asarray(input_ids)
    if input_ids.ndim == 2:
        input_ids = input_ids[0]
    return input_ids


def get_visual_range_qwen(processor, input_ids):
    input_ids = flatten_input_ids(input_ids)
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    start_matches = np.where(input_ids == vision_start_token_id)[0]
    end_matches = np.where(input_ids == vision_end_token_id)[0]
    if len(start_matches) == 0 or len(end_matches) == 0:
        raise ValueError("Cannot find Qwen vision_start/vision_end tokens in input_ids.")
    pos = int(start_matches[0]) + 1
    pos_end = int(end_matches[0])
    return pos, pos_end


def import_process_vision_info():
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        local_qwen_utils = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Qwen3-VL",
            "qwen-vl-utils",
            "src",
        )
        if os.path.isdir(local_qwen_utils) and local_qwen_utils not in sys.path:
            sys.path.insert(0, local_qwen_utils)
            try:
                from qwen_vl_utils import process_vision_info
                return process_vision_info
            except ImportError:
                pass
        raise ImportError(
            "Cannot import qwen_vl_utils. Install qwen-vl-utils or add "
            "Qwen3-VL/qwen-vl-utils/src to PYTHONPATH before using backend='qwen'."
        ) from exc
    return process_vision_info


def set_block_attn_hooks_llava(model, from_to_index_per_layer, opposite=False, block_desc=None):
    """
    Block selected attention edges for LLaVA/Huatuo-style and Qwen-style models.
    """
    llm_layers = get_llm_layers(model)
    qwen_style = is_qwen_style_model(model)

    def wrap_attn_forward(forward_fn, model_, from_to_index_, opposite_, block_desc_):
        @functools.wraps(forward_fn)
        def wrapper_fn(*args, **kwargs):

            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for (k, v) in kwargs.items():
                new_kwargs[k] = v

            if qwen_style:
                num_tokens = kwargs["position_ids"].shape[-1]
            else:
                num_tokens = kwargs["position_ids"][0][-1].item() + 1
            q_length = kwargs["hidden_states"][0].size(0)

            if q_length==1:
                if block_desc_ and block_desc_.split("->")[-1]=="Last":
                    from_to_index=[(0, t) for _, t in from_to_index_]
                else:
                    from_to_index = []

            else:
                from_to_index=from_to_index_

            if opposite_:
                if q_length == 1:
                    attn_mask = torch.zeros((q_length, num_tokens), dtype=torch.uint8)
                else:
                    attn_mask = torch.tril(torch.zeros((q_length, num_tokens), dtype=torch.uint8))

                if from_to_index !=[]:
                    rows, cols = zip(*from_to_index)
                    attn_mask[rows, cols] = 1
            else:
                if q_length == 1:
                    attn_mask = torch.ones((q_length, num_tokens), dtype=torch.uint8)
                else:
                    attn_mask = torch.tril(torch.ones((q_length, num_tokens), dtype=torch.uint8)) # set the upper triangular part of a matrix (the part above the main diagonal) to zero.

                if from_to_index !=[]:
                    rows, cols = zip(*from_to_index)
                    # print(f"rows: {rows}, cols: {cols}")
                    attn_mask[rows, cols] = 0


            attn_mask = attn_mask.repeat(1, 1, 1, 1)

            model_dtype = get_model_dtype(model_)
            attn_mask = attn_mask.to(dtype=model_dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(model_dtype).min
            attn_mask = attn_mask.to(model_.device)
            new_kwargs["attention_mask"] = attn_mask
            return forward_fn(*new_args, **new_kwargs)

        return wrapper_fn

    hooks = []
    for i in from_to_index_per_layer.keys():
        hook = llm_layers[i].self_attn.forward
        llm_layers[i].self_attn.forward = wrap_attn_forward(
            llm_layers[i].self_attn.forward,
            model,
            from_to_index_per_layer[i],
            opposite,
            block_desc,
        )
        hooks.append((i, hook))

    return hooks



def remove_wrapper_llava(model, hooks):
    llm_layers = get_llm_layers(model)
    for layer, original_forward in hooks:
        llm_layers[layer].self_attn.forward = original_forward

def convert_bounding_box_to_resized(bboxes, original_size, new_size):
    # Extract original image size and resized image size
    W_orig, H_orig = original_size
    W_new, H_new = new_size

    # Compute scaling factors
    scale_x = W_new / W_orig
    scale_y = H_new / H_orig

    # Handle both single bbox and list of bboxes
    if not isinstance(bboxes[0], list) and not isinstance(bboxes[0], tuple):
        bboxes = [bboxes]
    
    resized_bboxes = []
    for bbox in bboxes:
        # Extract the bounding box in the original image
        x_min, y_min, width, height = bbox

        # Convert to new image coordinates
        x_min_new = x_min * scale_x
        y_min_new = y_min * scale_y
        width_new = width * scale_x
        height_new = height * scale_y

        resized_bboxes.append([x_min_new, y_min_new, width_new, height_new])
    
    return resized_bboxes


def get_tokens_covering_bbox(bboxes, num_tokens=24):

    img_width, img_height = SIZE
    token_width = img_width / num_tokens
    token_height = img_height / num_tokens
    
    # Convert single bbox to list for uniform processing
    if not isinstance(bboxes, list):
        bboxes = [bboxes]
    
    selected_tokens = set()  # Use set to avoid duplicate tokens
    
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        
        x_min_token = int(np.floor(x_min / token_width))
        y_min_token = int(np.floor(y_min / token_height))
        x_max_token = int(np.ceil((x_min + width) / token_width))
        y_max_token = int(np.ceil((y_min + height) / token_height))

        x_max_token = min(x_max_token, num_tokens - 1)
        y_max_token = min(y_max_token, num_tokens - 1)
        
        ### Note it is (y, x) here
        bbox_tokens = [(y, x) for x in range(x_min_token, x_max_token) 
                               for y in range(y_min_token, y_max_token)]
        selected_tokens.update(bbox_tokens)

    return list(selected_tokens)

def draw_tokens_on_image(img, token_indices, num_tokens=24, color=(0, 255, 0), thickness=1):
    # Load the image
    h, w = SIZE  # Get image dimensions

    # Compute grid cell size
    token_width = w / num_tokens
    token_height = h / num_tokens
    
    # Create a copy of the image if it's a tensor
    if isinstance(img, torch.Tensor):
        img_np = img.numpy().copy()
    else:
        img_np = np.array(img).copy()
    
    # Make sure the image is in the right format for OpenCV
    if len(img_np.shape) == 2:  # If it's a grayscale image
        img_np = cv2.cvtColor(img_np.astype(np.float32), cv2.COLOR_GRAY2BGR)
    
    # Convert to uint8 if it's float
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        img_np = (img_np * 255).astype(np.uint8)

    # Already switch x and y here
    for y_token, x_token in token_indices:
        # Compute the top-left corner pixel coordinates of the token
        x_pixel = int(x_token * token_width)
        y_pixel = int(y_token * token_height)

        # Draw the rectangle (top-left, bottom-right)
        cv2.rectangle(img_np, (x_pixel, y_pixel),
                      (int((x_token + 1) * token_width), int((y_token + 1) * token_height)),
                      color, thickness)

    return img_np

def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap

def visualize_attention_map(att_map, image_path, gt_token=None, threshold=None, mask=None, v_minmax=None):
    import matplotlib.pyplot as plt

    att_map_scaled = torch.tensor(att_map)
    att_map_scaled = normalize(att_map_scaled)

    # To only show the bbox area
    if mask and (gt_token is not None):
        mask = np.zeros_like(att_map_scaled)
        for x, y in gt_token:
            mask[x, y] = 1

        att_map_scaled = att_map_scaled * mask

    attn_over_image = torch.nn.functional.interpolate(
        att_map_scaled.unsqueeze(0).unsqueeze(0), 
        size=(336,336), 
        mode='nearest', 
    ).squeeze()

    np_img = np.array(Image.open(image_path).resize((336, 336)))
    img_with_attn, _ = show_mask_on_image(np_img, attn_over_image.numpy())

    if gt_token is not None:
        np_img = draw_tokens_on_image(np_img, gt_token)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(np_img)
    axes[0].axis("off")

    if v_minmax is not None:
        axes[1].imshow(att_map_scaled, vmin=v_minmax[0], vmax=v_minmax[1])
    else:
        axes[1].imshow(att_map_scaled)
        v_minmax = (att_map_scaled.min(), att_map_scaled.max())
    axes[1].axis("off")

    axes[2].imshow(img_with_attn)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    if v_minmax is not None:
        return v_minmax
    


def qwen3_vl_inference(
    model,
    processor,
    question,
    image_path,
    return_att_map=False,
    image_patch_size=None,
    max_new_tokens=64,
):
    if not question:
        question = "Write a general description of the image. Answer the question using a single word or phrase."
    if isinstance(image_path, (str, bytes, Image.Image)):
        image_path = [image_path]

    images = load_images(image_path)
    image_input = images[0] if len(images) == 1 else images
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_input},
                {"type": "text", "text": question},
            ],
        }
    ]

    if return_att_map:
        process_vision_info = import_process_vision_info()
        if image_patch_size is None:
            image_patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 16)
        try:
            image_inputs, _ = process_vision_info(messages, image_patch_size=image_patch_size)
        except TypeError:
            image_inputs, _ = process_vision_info(messages)

        text_query = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text_query],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
        return output, inputs["input_ids"], None

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return None, None, output_text[0]


def model_inference(bot, image_path, question=None):
    if not question:
        question = 'Write a general description of the image. Answer the question using a single word or phrase.'
    model_output, input_ids, response_qs = bot.inference(question, image_path, return_att_map=True)
    return model_output, input_ids, response_qs



def extract_huatuo_attention_maps(model_output, input_ids, layers, Q=False, top_sensitive_heads=None):
    attentions = get_required_attentions(model_output, "HuatuoGPT-Vision")
    index, _ = get_visual_range_huatuo(input_ids)
    
    att_maps = {}

    if top_sensitive_heads is not None:
        for layer_head in top_sensitive_heads:
            layer, head = layer_head[0], layer_head[1]
            if Q:
                att_maps[(layer, head)] = attentions[layer][0, head, index+NUM_IMG_TOKENS:, index:index+NUM_IMG_TOKENS].mean(dim=(0)).to(torch.float32).detach().cpu().numpy().reshape(PATCHES, PATCHES)
            else:
                att_maps[(layer, head)] = attentions[layer][0, head, -1, index:index+NUM_IMG_TOKENS].to(torch.float32).detach().cpu().numpy().reshape(PATCHES, PATCHES)
    else:
        for layer in layers:
            if Q:
                att_maps[layer] = attentions[layer][0, :, index+NUM_IMG_TOKENS:, index:index+NUM_IMG_TOKENS].mean(dim=(0,1)).to(torch.float32).detach().cpu().numpy().reshape(PATCHES, PATCHES)

            else:
                att_maps[layer] = attentions[layer][0, :, -1, index:index+NUM_IMG_TOKENS].mean(dim=0).to(torch.float32).detach().cpu().numpy().reshape(PATCHES, PATCHES)
        
    return att_maps


def attention_triage_huatuo(bot, image, prompt, norm=True, Q=False, top_sensitive_heads=None):
    model_output, input_ids, _ = model_inference(bot, image, prompt)
    num_layers = len(get_required_attentions(model_output, "HuatuoGPT-Vision"))
    att_maps = extract_huatuo_attention_maps(model_output, input_ids, layers=[l for l in range(num_layers)], Q=Q, top_sensitive_heads=top_sensitive_heads)

    if norm:
        model_general_output, input_general_ids, _ = model_inference(bot, image)
        num_general_layers = len(get_required_attentions(model_general_output, "HuatuoGPT-Vision"))
        general_att_maps = extract_huatuo_attention_maps(model_general_output, input_general_ids, layers=[l for l in range(num_general_layers)], Q=Q, top_sensitive_heads=None)

    att_map_list = {}
    for layer in att_maps.keys():
        att_map = att_maps[layer]
        if norm:
            try:
                general_att_map = general_att_maps[layer]
            except:
                general_att_map = general_att_maps[layer[0]]
            att_map = att_map / (general_att_map + 1e-10)
        att_map_list[layer] = {
                    'att_map': att_map.tolist()
                }
    
    return att_map_list, input_ids


def extract_qwen_attention_maps(model_output, processor, input_ids, layers, Q=False, top_sensitive_heads=None):
    attentions = get_required_attentions(model_output, "Qwen")
    pos, pos_end = get_visual_range_qwen(processor, input_ids)
    att_maps = {}

    if top_sensitive_heads is not None:
        for layer_head in top_sensitive_heads:
            layer, head = layer_head[0], layer_head[1]
            if Q:
                att_maps[(layer, head)] = (
                    attentions[layer][0, head, pos_end:, pos:pos_end]
                    .mean(dim=0)
                    .to(torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                att_maps[(layer, head)] = (
                    attentions[layer][0, head, -1, pos:pos_end]
                    .to(torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                )
    else:
        for layer in layers:
            if Q:
                att_maps[layer] = (
                    attentions[layer][0, :, pos_end:, pos:pos_end]
                    .mean(dim=(0, 1))
                    .to(torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                att_maps[layer] = (
                    attentions[layer][0, :, -1, pos:pos_end]
                    .mean(dim=0)
                    .to(torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                )
    return att_maps


def attention_triage_qwen(
    model,
    processor,
    image,
    prompt,
    norm=True,
    Q=False,
    top_sensitive_heads=None,
    image_patch_size=None,
):
    model_output, input_ids, _ = qwen3_vl_inference(
        model,
        processor,
        prompt,
        image,
        return_att_map=True,
        image_patch_size=image_patch_size,
    )
    num_layers = len(get_required_attentions(model_output, "Qwen"))
    att_maps = extract_qwen_attention_maps(
        model_output,
        processor,
        input_ids,
        layers=[l for l in range(num_layers)],
        Q=Q,
        top_sensitive_heads=top_sensitive_heads,
    )

    if norm:
        model_general_output, input_general_ids, _ = qwen3_vl_inference(
            model,
            processor,
            None,
            image,
            return_att_map=True,
            image_patch_size=image_patch_size,
        )
        num_general_layers = len(get_required_attentions(model_general_output, "Qwen general prompt"))
        general_att_maps = extract_qwen_attention_maps(
            model_general_output,
            processor,
            input_general_ids,
            layers=[l for l in range(num_general_layers)],
            Q=Q,
            top_sensitive_heads=None,
        )

    att_map_list = {}
    for layer in att_maps.keys():
        att_map = att_maps[layer]
        if norm:
            try:
                general_att_map = general_att_maps[layer]
            except KeyError:
                general_att_map = general_att_maps[layer[0]]
            att_map = att_map / (general_att_map + 1e-10)
        att_map_list[layer] = {"att_map": att_map.tolist()}

    return att_map_list, input_ids


def build_attention_knockout_map(att_map_list, att_map_layer, top_sensitive_heads=None):
    if top_sensitive_heads is not None:
        att_map = []
        for layer_head in top_sensitive_heads:
            layer, head = layer_head[0], layer_head[1]
            att_map.append(att_map_list[(layer, head)]["att_map"])
        return np.array(att_map).mean(axis=0)
    if att_map_layer is None:
        raise ValueError("att_map_layer is required when top_sensitive_heads is not provided.")
    return np.array(att_map_list[att_map_layer]["att_map"])


def select_low_attention_indices(att_map, percentile, magnitude_threshold):
    flattened_att_map = att_map.flatten()
    if percentile is not None:
        threshold = np.percentile(flattened_att_map, percentile)
        return np.where(flattened_att_map < threshold)[0]
    if magnitude_threshold is not None:
        threshold = magnitude_threshold
        flattened_att_map = normalize(flattened_att_map.copy())
        return np.where(flattened_att_map < threshold)[0]
    raise ValueError("Either percentile or magnitude_threshold must be provided.")


# 2nd pass
def attention_knockout_huatuo(bot, image, prompt, att_map_list, input_ids, magnitude_threshold, att_map_layer, percentile, masked_layer, top_sensitive_heads=None):
    
    image_start_token_index = np.where(input_ids==-200)[0][0]
    last_token_index = [i for i in range(image_start_token_index + NUM_IMG_TOKENS, input_ids.shape[-1] + NUM_IMG_TOKENS - 1)]
    if not isinstance(last_token_index, list):
        last_token_index = [last_token_index]

    block_config = {}

    att_map = build_attention_knockout_map(att_map_list, att_map_layer, top_sensitive_heads=top_sensitive_heads)
    low_attention_indices = select_low_attention_indices(att_map, percentile, magnitude_threshold)
        

    image_range = [image_start_token_index + patch_index for patch_index in low_attention_indices]

    block_ids = [image_range, last_token_index]
    temp = [(stok1, stok0) for stok0 in block_ids[0] for stok1 in block_ids[1]]

    for layer in masked_layer:
        block_config[layer] = copy.deepcopy(temp)

    block_attn_hooks = set_block_attn_hooks_llava(bot.model, block_config, block_desc="Image->Question")
    try:
        with torch.inference_mode():
            _, input_ids, response_qs = bot.inference(prompt, image)
    finally:
        remove_wrapper_llava(bot.model, block_attn_hooks)

    return response_qs


def attention_knockout_qwen(
    model,
    processor,
    image,
    prompt,
    att_map_list,
    input_ids,
    magnitude_threshold,
    att_map_layer,
    percentile,
    masked_layer,
    top_sensitive_heads=None,
    image_patch_size=None,
):
    input_ids = flatten_input_ids(input_ids)
    pos, pos_end = get_visual_range_qwen(processor, input_ids)
    last_token_index = [i for i in range(pos_end, input_ids.shape[-1])]
    if not isinstance(last_token_index, list):
        last_token_index = [last_token_index]

    att_map = build_attention_knockout_map(att_map_list, att_map_layer, top_sensitive_heads=top_sensitive_heads)
    low_attention_indices = select_low_attention_indices(att_map, percentile, magnitude_threshold)

    image_range = [pos + patch_index for patch_index in low_attention_indices]
    block_ids = [image_range, last_token_index]
    temp = [(stok1, stok0) for stok0 in block_ids[0] for stok1 in block_ids[1]]

    block_config = {}
    for layer in masked_layer:
        block_config[layer] = copy.deepcopy(temp)

    block_attn_hooks = set_block_attn_hooks_llava(model, block_config, block_desc="Image->Question")
    try:
        with torch.inference_mode():
            _, _, response_qs = qwen3_vl_inference(
                model,
                processor,
                prompt,
                image,
                return_att_map=False,
                image_patch_size=image_patch_size,
            )
    finally:
        remove_wrapper_llava(model, block_attn_hooks)

    return response_qs


def VGRefine(
    bot=None,
    image=None,
    text=None,
    masked_layer=None,
    top_sensitive_heads=None,
    percentile=80,
    magnitude_threshold=None,
    att_map_layer=None,
    Q=False,
    norm=True,
    backend="huatuo",
    model=None,
    processor=None,
    image_patch_size=None,
):

    if top_sensitive_heads is None or len(top_sensitive_heads) == 0:
        raise ValueError("VGRefine requires a non-empty top_sensitive_heads list.")
    if image is None:
        raise ValueError("VGRefine requires an image path or PIL image.")
    if text is None:
        raise ValueError("VGRefine requires a text question.")
    if masked_layer is None:
        raise ValueError("VGRefine requires masked_layer.")

    prompt = f"{text} Answer the question using a single word or phrase."
    backend = backend.lower().replace("-", "_")
    if backend == "auto":
        backend = "qwen" if model is not None and processor is not None else "huatuo"

    if backend in {"huatuo", "huatuogpt", "huatuogpt_vision", "llava"}:
        
        if bot is None:
            raise ValueError("backend='huatuo' requires bot.")

        att_map_list, input_ids = attention_triage_huatuo(
            bot=bot,
            image=image,
            prompt=prompt,
            Q=Q,
            norm=norm,
            top_sensitive_heads=top_sensitive_heads,
        )

        response_qs = attention_knockout_huatuo(
            bot=bot,
            image=image,
            prompt=prompt,
            att_map_list=att_map_list,
            input_ids=input_ids[0].cpu().numpy(),
            att_map_layer=att_map_layer,
            percentile=percentile,
            magnitude_threshold=magnitude_threshold,
            masked_layer=masked_layer,
            top_sensitive_heads=top_sensitive_heads,
        )
    elif backend in {"qwen", "qwen3vl", "qwen3_vl", "qwen2_5vl", "qwen2.5vl"}:

        if model is None or processor is None:
            raise ValueError("backend='qwen' requires model and processor.")

        att_map_list, input_ids = attention_triage_qwen(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            Q=Q,
            norm=norm,
            top_sensitive_heads=top_sensitive_heads,
            image_patch_size=image_patch_size,
        )

        response_qs = attention_knockout_qwen(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            att_map_list=att_map_list,
            input_ids=input_ids[0].cpu().numpy(),
            att_map_layer=att_map_layer,
            percentile=percentile,
            magnitude_threshold=magnitude_threshold,
            masked_layer=masked_layer,
            top_sensitive_heads=top_sensitive_heads,
            image_patch_size=image_patch_size,
        )
    else:
        raise ValueError(f"Unsupported VGRefine backend: {backend}")

    return {
        "response": response_qs,
        "attention_maps": att_map_list,
        "input_ids": input_ids,
        "prompt": prompt,
    }
