# [ICLR'26] How Do Medical MLLMs Fail? A Study on Visual Grounding in Medical Images

This repository measures how medical multimodal large language models attend to image regions that are relevant to visual-question-answering samples.

**Links:** [Project Page](https://guimeng-leo-liu.github.io/Medical-MLLMs-Fail/) | [Paper](https://arxiv.org/pdf/2603.14323v1) | [Dataset](https://huggingface.co/datasets/guimeng-liu/VGMED)

- `measure_attention.py`: computes layer-level metrics.
- `measure_attention_head.py`: computes head-level metrics.

## Clone This Repository

Start by cloning this repository and entering the project directory:

```bash
git clone https://github.com/Guimeng-Leo-Liu/Medical-MLLMs-Fail.git
cd Medical-MLLMs-Fail
```

## Clone Support Repositories

Clone the model support repositories inside `Medical-MLLMs-Fail/`:

```bash
git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git
git clone https://github.com/QwenLM/Qwen3-VL.git
```

The scripts use:

- `./HuatuoGPT-Vision` as the default HuatuoGPT-Vision support repo path.
- `--qwen_vl_utils_path ./Qwen3-VL/qwen-vl-utils/src` for local Qwen visual preprocessing.

Only pass `--huatuo_repo /custom/path/HuatuoGPT-Vision` if you cloned HuatuoGPT-Vision outside this project folder.

**Important:** for HuatuoGPT-Vision attention extraction, use the modified `cli_new.py` from this repository:

```bash
cp ./cli_new.py ./HuatuoGPT-Vision/cli_new.py
```

## Environment Setup

Create and activate a Python environment:

```bash
conda create -n mllm python=3.10
conda activate mllm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The requirements file is a curated version of the local `mllm` environment. If your CUDA version is different, install the matching PyTorch build first, then install the remaining packages from `requirements.txt`.

# Quantifying MLLMs’ Visual Grounding

## Dataset Setup

Download VGMED from Hugging Face:

https://huggingface.co/datasets/guimeng-liu/VGMED

The evaluation JSONL files are:

- `VGMED_attribute.jsonl`
- `VGMED_localization.jsonl`
- `COCO_attribute.jsonl`
- `COCO_localization.jsonl`

The scripts also need the corresponding image folders:

- VGMED images referenced by the VGMED JSONL files.
- COCO `val2014` images referenced by the COCO JSONL files. Download COCO images from https://cocodataset.org/#download.

## Path Configuration

The scripts do not assume dataset or output paths. Pass them from the command line.

For VGMED runs, provide:

- `--vgmed_root`: folder containing VGMED images and `VGMED_<q_type>.jsonl`
- `--output_root`

For COCO runs, provide:

- `--coco_root`: folder containing COCO `val2014` images
- `--coco_annotation_root`: folder containing `COCO_<q_type>.jsonl`
- `--output_root`

Use these path flags in the commands below.

## Measure Visual Grounding

To measure visual grounding on general-scene or medical samples, use the following script:

```bash
python ./measure_attention.py \
  --dataset VGMED \
  --q_type localization \
  --model_path FreedomIntelligence/HuatuoGPT-Vision-7B \
  --model_type huatuogpt_vision \
  --vgmed_root /path/to/VGMED \
  --output_root /path/to/outputs
```

The main arguments are:

- `--dataset`: `VGMED` or `COCO`
- `--q_type`: `localization` or `attribute`
- `--model_path`: Hugging Face model name or local checkpoint path
- `--model_base_path`: base model path, required when loading a LoRA adapter
- `--model_type`: `qwen2_5vl`, `qwen3vl`, or `huatuogpt_vision`
- `--output_root`: folder for saved attention metrics

We can identify the most visually sensitive layer, and to be used as `MASKED_LAYER` in VGRefine.


# Visual Grounding Refinement

## Head-Level Visual Grounding

Before VGRefine, we need to identify the top $k$ most visually sensitive head `TOP_SENSITIVE_HEADS`. 

This script keeps every attention head and saves metrics with shape `[num_layers, num_heads]` for each sample:

```bash
python ./measure_attention_head.py \
  --dataset VGMED \
  --q_type localization \
  --model_path FreedomIntelligence/HuatuoGPT-Vision-7B \
  --model_type huatuogpt_vision \
  --vgmed_root /path/to/VGMED \
  --output_root /path/to/outputs
```

## VGRefine Example

Here's an example of our proposed VGRefine, an inference time method that improves
visual grounding by refining internal attention distributions. 

The example is built on HuatuoGPT-Vision. Replace the model and corresponding `MASKED_LAYER` and `TOP_SENSITIVE_HEADS` for your own setting.


```python
import os
import sys

sys.path.insert(0, "./HuatuoGPT-Vision")

from cli_new import HuatuoChatbot
from eval_utils import VGRefine

bot = HuatuoChatbot(
    "FreedomIntelligence/HuatuoGPT-Vision-7B",
    device="cuda:0",
)

MASKED_LAYER = [16]

TOP_SENSITIVE_HEADS=[
  [15, 4], [16, 5], [15, 1], [10, 6], [4, 0], 
  [15, 24], [19, 6], [8, 3], [19, 18], [18, 10], 
  [4, 17], [16, 2], [16, 17], [16, 1], [14, 23], 
  [6, 18], [18, 19], [6, 11], [0, 2], [17, 26]
  ]

result = VGRefine(
    bot=bot,
    backend="huatuo",
    image="/path/to/image.jpg",
    text="What abnormality is visible in the image?",
    masked_layer=MASKED_LAYER,
    top_sensitive_heads=TOP_SENSITIVE_HEADS,
)

print(result["response"])
```

## Citation

```bibtex
@inproceedings{liu2026how,
  title     =   {How Do Medical MLLMs Fail?  A Study on Visual Grounding in Medical Images},
  author    =   {Guimeng Liu 
              and Tianze Yu 
              and Somayeh Ebrahimkhani 
              and Lin Zhi Zheng Shawn 
              and Kok Pin Ng 
              and Ngai-Man Cheung},
  booktitle =  {The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=dXshexyFKx}
}
```
