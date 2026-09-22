# deblurstage2

**在 JupyterLab 上开始：** 阅读 [JUPYTER.md](JUPYTER.md)，把本项目克隆到 `qaoa` 的同级目录，对现有 4/14 号的 157 张完整 Out 进行正式 OCR 与转写。

## 模型权重怎样使用

这套程序在服务器上运行模型。先从 Hugging Face 下载固定版本的权重、配置和 tokenizer，保存在服务器共享缓存；之后 GPU 作业直接读取缓存。**它不调用 DeepSeek / Qwen 的收费推理 API，也不需要它们的 API key。** 已准备好完整缓存时，三个模型入口均支持 `--offline`；下载和 GPU 推理可以分开进行。

2026-09-22 核对的官方 Hugging Face 元数据如下。三个仓库均为公开、非 gated，当前不需要先申请 Hugging Face 模型访问审批；登录可用于账户限额等需要。许可证均标注 Apache-2.0，使用与分发仍应遵守各仓库许可。ICRN 的账号、联网、存储和作业权限由本校规则决定，目前没有核实。

|模型|用途|仅 safetensors 权重大小，十进制|
|---|---|---:|
|[DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)|分别读取 Blur 和 Out|约 6.8 GB|
|[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)|快速基线 / 较低显存备选|约 17.5 GB|
|[Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)|默认模型，H200 主方案；独立 LoRA 底座|约 66.7 GB|

上述数字不是显存峰值，也不是完整磁盘需求。环境、缓存、检查点和输出都需要额外空间。固定的 commit SHA 在 `config.json` 和 [NOTES_ENV.md](NOTES_ENV.md) 中；本地开发未下载权重、运行 GPU 推理或开始训练。

## 做什么

```text
冻结的 deblur：Blur → Out
                      ↓
Blur、Out 分别进入冻结 DeepSeek-OCR2 → 两份文字候选
        两张图 + 两份候选 → 冻结 Qwen3-VL → 恢复文字
```

输出是文字；这里没有重训 deblur、修改其 Out 像素或微调 DeepSeek-OCR2。当前直接使用现有结果进行正式 OCR 与转写，不要求其他训练文档的 Out。

主要入口为 `prepare.py`、`ocr.py`、`restore.py`、`train.py`，共享逻辑和参数放在 `common.py`、`config.json`。训练直接复用 Hugging Face Trainer 与 PEFT，不复制整套上游训练仓库。输入数据、识别结果和权重不提交到 Git。

## 服务器准备

以下是用户后续在服务器执行的 Linux 命令，尚未在 ICRN 实机验证。目标为 **单 H200 141 GB + Qwen 32B**，优先文字效果；A100 80 GB 可作替代，但 32B 的显存余量更紧。先核对实际分配的 GPU、驱动、CUDA 编译工具、磁盘配额和网络方式。没有预设 ICRN 主机名、module、分区或调度命令。

下面从 GitHub 克隆代码。数据通过实验室允许的文件传输方式另行搬运；全量数据包上传与运行步骤见 [JUPYTER.md](JUPYTER.md)。

```bash
# 替换成有足够空间、GPU 节点也能访问的真实路径。
export WORK_ROOT=/replace/with/server/workspace
export HF_HOME="$WORK_ROOT/hf-cache"
git clone https://github.com/Yijing-Zuo/deblurstage2.git "$WORK_ROOT/deblurstage2"
cd "$WORK_ROOT/deblurstage2"
```

保留原 deblur 环境；新建两个独立环境。DeepSeek 官方接口使用 Transformers 4.46.3，本实现的 Qwen 入口使用 4.57.1，不能把两份 requirements 合并安装。以下选 Python 3.11；环境文件亦允许 3.10。

```bash
conda create -n deblur-ocr python=3.11 -y
conda activate deblur-ocr
python -m pip install -r requirements-ocr.txt
python -m pip install flash-attn==2.7.3 --no-build-isolation
python -m pip check

conda create -n deblur-qwen python=3.11 -y
conda activate deblur-qwen
python -m pip install -r requirements-qwen.txt
python -m pip check
```

两个环境均固定 Torch 2.6.0 + CUDA 12.4 wheel，但 Transformers 版本不同。DeepSeek 需要 FlashAttention，其源码安装可能需要匹配的 CUDA 12.4 toolkit / 编译器；`nvidia-smi` 显示的驱动能力不等于已安装 toolkit。Qwen 入口使用 PyTorch SDPA，不需要额外安装 FlashAttention。若服务器环境不兼容，先调整并记录版本，不在原 deblur 环境反复覆盖依赖。

安装与下载放在学校允许的联网节点进行；GPU 推理和训练须在分配到的 GPU 作业中进行，不在登录节点加载模型。进入每个 GPU 作业时重新设置相同的 `HF_HOME`，激活相应环境并切换到仓库目录。

### 下载一次，以后离线读取

下面只下载文件，不启动推理。主方案准备 OCR2 和 32B；8B 是可选基线。

```bash
conda activate deblur-ocr
hf download deepseek-ai/DeepSeek-OCR-2 \
  --revision aaa02f3811945a91062062994c5c4a3f4c0af2b0
hf download Qwen/Qwen3-VL-32B-Instruct \
  --revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb

# 可选：8B 必须使用自己的权重和 adapter。
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

也可以把完整 snapshot 放到自选目录，再向对应脚本传 `--model-path /absolute/path/to/snapshot --offline`。不要把仅有 safetensors 的目录当成完整模型，也不要将不同 revision 的配置、代码和权重混放。DeepSeek 使用仓库自带模型代码，加载时固定 revision；使用本地目录时应保留对应 snapshot 原貌。

## 阶段 I：现有 Out → 文字

### 1. 对齐输入

当前 4/14 有 157 张拼接 Out。将下面原工作区的目录结构复制到服务器数据目录：

```text
outputs/results_Out_merged_512x768_20260915/*.png
outputs/results_original_Blur_GT_fullpages/verified_4_14.json
outputs/results_original_Blur_GT_fullpages/Blur/*.webp
```

GT 不参与这一步推理。`verified_4_14.json` 提供实际 ROI；不能凭文件名把 Out 和整页 Blur 直接配对，也不能认为 512×768 一定是完整段落。以下直接导入全量 157 张。

```bash
conda activate deblur-ocr
export DEBLUR_DATA=/replace/with/copied/deblur/data-root
python prepare.py \
  --pairs "$DEBLUR_DATA/outputs/results_original_Blur_GT_fullpages/verified_4_14.json" \
  --out-dir "$DEBLUR_DATA/outputs/results_Out_merged_512x768_20260915" \
  --output data/samples.jsonl
```

这个入口将裁剪后的 Blur 与对应 Out 一起写入 `data/images/`，生成相对路径索引。也可以先在本地导入，再将整个 `data/` 传到服务器，跳过服务器导入步骤。

自备样本用 `--manifest` 导入。每行需要 `id`、`document_id`、`deblur_run`、`split`、`blur`、`out`；建议同时提供可区分文档版本的 `document_uid`。图片相对路径以输入 JSONL 所在目录为基准。已对齐图片直接配对；需要裁剪时提供 `blur_box` / `out_box`，坐标为左上角原点的 `[x0,y0,x1,y1]`、右下边界不包含在内。

```bash
python prepare.py --manifest data/input_pairs.jsonl --output data/samples.jsonl
```

`examples/samples.jsonl` 只展示字段，不附带实际图像。不要向 OCR 输入含 Clear 的三列比较 PDF、图像拼表或原 GT；这些包含答案。0 号缺块数据只能按实际可见区域配对，不能白填后当完整段落。旧 2 号属于另一批 deblur run，不用于这套当前 run 的训练或替代测试输入。

### 2. 缓存两份 OCR 候选

在分配到的 GPU 上：

```bash
conda activate deblur-ocr
python ocr.py --samples data/samples.jsonl \
  --output runs/candidates.jsonl --offline
```

DeepSeek 的公开 `infer` 接口一次处理一张图，脚本分别识别 Blur 和 Out；不是把两图拼成一张。匹配输入哈希和设置的成功记录会复用，失败记录会重试。候选存为 JSONL，后续训练和推理使用同一份格式。

### 3. Qwen 合并图像证据与候选

先结束 OCR 进程、释放其显存，再切换环境：

```bash
conda activate deblur-qwen
python restore.py --samples data/samples.jsonl \
  --candidates runs/candidates.jsonl --model-size 32b \
  --output runs/restored_32b.jsonl --offline

# 可选基线：复用 OCR 缓存，另外加载 8B。
python restore.py --samples data/samples.jsonl \
  --candidates runs/candidates.jsonl --model-size 8b \
  --output runs/restored_8b.jsonl --offline
```

Qwen 同时接收两张图和两份 OCR 候选。默认不使用 adapter；省略 `--model-size` 时使用配置中的 **32B**。直接处理全量输入，检查输出的 `status` 与 `text`：`error` 或 `truncated` 不能算作完整成功。改动图片、模型或相关配置会使原结果不能直接复用。显存峰值取决于双图 token、序列长度和输出预算，当前尚未在 H200 实测。

## 当前不执行 LoRA 微调

其他训练文档的 Out 不存在，也不会提供。当前正式方案直接使用冻结的 DeepSeek-OCR2 和预训练 Qwen3-VL，完成现有 Out 的 OCR 与文字转写，不需要训练数据。

`train.py` 与训练导出功能保留为可选工具，但其原始设计依赖真实训练 Out，**不适用于当前可用材料，也不是运行系统的前置步骤**。不再要求收集其他文档的 Out。若以后仍需微调，应基于实际可用的 Blur、清晰文档或已核对文字重新设计训练数据，再实施。

当前本地检查仅使用 CPU：`python -m unittest discover -s tests -v` 的 19 项检查、四入口的 `--help` / 语法检查，以及现有 4/14 号全部 157 对的 Blur 裁剪与 Out 导入已通过。模型加载、实际 HF processor / adapter 集成、GPU 显存和文字恢复效果仍待服务器验证。

`.gitignore` 排除 `data/`、`runs/`、`models/`、缓存、检查点和凭据。推送前仍需检查 `git status`；本地开发和文档准备不会自动提交或推送。
