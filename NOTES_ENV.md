# Model environments

**Current v2 deployment:** use [JUPYTER.md](JUPYTER.md). The existing server Qwen
environment and 32B offline cache have now been exercised by the user. Keep them.
Add a separate Paddle 3.2/cu126 + PaddleX OCR-core 3.7 environment, plus the small
recovery/render dependencies in Qwen. CPU contract and PDF rendering checks have
passed locally; the new Paddle model path has not been run on H200. The original
DeepSeek setup notes below are retained for historical reproduction only.

Local work has not installed packages, downloaded model weights, or used a GPU.
These are proposed Linux environments for the A100/H200 host, not a tested GPU setup.
Keep the existing deblur environment unchanged. Use separate OCR and Qwen virtual
environments: DeepSeek's official requirements pin Transformers 4.46.3, while this
Qwen implementation targets 4.57.1. This separation is a dependency choice; DeepSeek's
single-image limitation comes from its public `infer` wrapper.

Install each requirements file only inside its own environment after checking the
host driver. Both files select the official PyTorch 2.6 CUDA 12.4 wheels. DeepSeek's
README demonstrates CUDA 11.8; our CUDA 12.4 selection is for the intended Hopper
host and has not yet been verified on that GPU. For OCR only, install FlashAttention
2.7.3 **after** Torch, using the commented command in requirements-ocr.txt. A source
build needs compatible CUDA development tools; prefer a matching CUDA 12.4 toolkit,
not just the driver or a runtime-only container. Qwen uses PyTorch SDPA and the
official processor, so it needs neither flash-attn nor qwen-vl-utils. Do not install
either file into the current local environment merely to inspect the scripts.
Record the resulting complete package freeze, Python/driver/toolkit versions, and
this repository revision with every target-host run.

FlashAttention **2** version 2.7.3 advertises Ampere/Ada/Hopper and CUDA >=11.7;
its build script emits sm90 with nvcc >=11.8. The CUDA >=12.3 requirement in that
README's separate Hopper section refers to FlashAttention **3**, which this project
does not use. CUDA 12.4 is a coherent proposed build/runtime selection, not a claim
that every FlashAttention-2 Hopper installation inherently requires CUDA 12.

## Public model metadata checked on 2026-09-22

| Model | Revision | Safetensors only |
|---|---|---:|
| deepseek-ai/DeepSeek-OCR-2 | `aaa02f3811945a91062062994c5c4a3f4c0af2b0` | 6,778,573,880 bytes |
| Qwen/Qwen3-VL-8B-Instruct | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 17,534,339,512 bytes |
| Qwen/Qwen3-VL-32B-Instruct | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | 66,714,912,872 bytes |

The public Hugging Face metadata API returned `gated=false` and `private=false` for
all three models. No account token was needed for that metadata request. These are
weight-file sizes, not peak GPU memory or a sufficient total-disk reservation;
downloads/caches, runtime packages, checkpoints and outputs require extra storage.
No weights were fetched to obtain these numbers.

## OCR behavior

`ocr.py` calls the official HF `infer` separately for Blur and Out with
`eval_mode=True`, which returns decoded text. It does not parse stdout. Each pair
is appended to JSONL and flushed; successful matching records are skipped on rerun,
while failed records are retried. Changed input file hashes or inference settings
invalidate the cached result. A local `--model-path` also contributes a file
size/mtime inventory to its fingerprint; its recorded upstream revision remains
the revision declared in config, not proof that arbitrary local weights match it.
Prefer an untouched snapshot of that exact revision. `--offline` disables remote
model/tokenizer loading; it requires a complete pre-populated local snapshot/cache.

The inspected DeepSeek wrapper fixes generation to 8192 new tokens internally and
does not accept a `max_new_tokens` argument. `eval_mode=True` also has its own
repetition setting; the pinned code determines those settings. Input dimensions
at most 768 on both sides do not automatically generate local crops, even with
`crop_mode=True`. Prepare paragraph crops explicitly when needed.

Sources: [PyTorch wheels](https://pytorch.org/get-started/previous-versions/#v260),
[FlashAttention 2.7.3](https://github.com/Dao-AILab/flash-attention/tree/v2.7.3),
[DeepSeek requirements](https://github.com/deepseek-ai/DeepSeek-OCR-2/blob/main/requirements.txt),
[DeepSeek model implementation](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/aaa02f3811945a91062062994c5c4a3f4c0af2b0/modeling_deepseekocr2.py),
[Qwen training](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune).
