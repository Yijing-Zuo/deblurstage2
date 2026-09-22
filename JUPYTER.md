# JupyterLab：在 qaoa 旁运行 deblurstage2

在 JupyterLab 的 **Terminal** 中执行下面的 Bash 命令，直接处理现有 4/14 号的全部 157 张 Out。无需先建 Notebook，也不要在已有 qaoa 环境中安装依赖。截图显示已分配 H200 NVL、约 143771 MiB 显存；实际可用显存仍以运行时为准。目前没有与服务器连接，下面是供你在服务器执行的命令。

## 1. 确认目录，再下载代码

截图的终端位于 `~`，但左侧文件浏览器的 `/` 不代表 Linux 系统根目录。先确认终端当前目录确实包含 qaoa：

```bash
pwd
ls -d qaoa
```

若第二条显示 `qaoa`，在这个目录执行：

```bash
git clone https://github.com/Yijing-Zuo/deblurstage2.git
cd deblurstage2
export HF_HOME="$PWD/hf-cache"
```

结果为同级的 `qaoa/` 和 `deblurstage2/`。若找不到 qaoa，先进入它的真实父目录再 clone，不要直接 `cd /`。如果已经 clone，进入已有 `deblurstage2` 后使用 `git pull --ff-only`，不重复克隆。

GitHub 只含代码、配置与文档，**不包含真实图片、权重或训练输出**。

## 2. 检查环境与空间

```bash
command -v conda
command -v python
command -v nvcc
python --version
nvcc --version
nvidia-smi
df -h .
```

`nvcc` 不存在时，该条命令报错不影响此前检查。OCR2 + 32B 的权重文件合计约 74 GB，完整缓存、两个环境、训练检查点还需额外空间；若学校有磁盘配额，也要单独核对配额。8B 是另加约 17.5 GB 的可选下载。

截图中 `nvidia-smi` 的 CUDA 12.8 表示驱动支持的 CUDA 版本，**不证明已安装 CUDA 编译工具**。本项目使用 Torch 的 CUDA 12.4 wheel，无需因为截图显示 12.8 就修改 GPU 驱动。OCR 的 FlashAttention 源码安装还需要兼容的编译器与 CUDA toolkit，建议匹配 CUDA 12.4；Qwen 使用 SDPA，不需要安装 FlashAttention。

如果没有 conda，先使用学校提供的环境管理方式；不要猜测 conda 安装路径。如果 conda 存在但 `conda activate` 提示 shell 未初始化，可在当前终端执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
```

## 3. 建两个独立环境

```bash
conda create -n deblur-ocr python=3.11 -y
conda activate deblur-ocr
python -m pip install -r requirements-ocr.txt
```

**先确认 `nvcc --version` 及 CUDA toolkit 兼容，再执行下一条。** 如果缺少 nvcc 或版本不匹配，先通过学校提供的 toolkit/module 或管理员解决，不要强行安装或反复覆盖现有环境。

```bash
python -m pip install flash-attn==2.7.3 --no-build-isolation
python -m pip check
python -c "import torch, flash_attn; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"

conda create -n deblur-qwen python=3.11 -y
conda activate deblur-qwen
python -m pip install -r requirements-qwen.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

不要合并两份 requirements：DeepSeek 与 Qwen 使用不同的 Transformers 版本。这些安装命令会下载依赖，但不会启动模型推理或训练。

## 4. 下载固定版本的模型

主方案只需 OCR2 和 32B。当前两个仓库均公开、无需申请模型访问权限；不需要 DeepSeek/Qwen 的付费 API key。下载需要服务器有网络，文件会缓存到 `hf-cache/`，首次下载后可重复使用。

```bash
conda activate deblur-ocr
export HF_HOME="$PWD/hf-cache"
hf download deepseek-ai/DeepSeek-OCR-2 \
  --revision aaa02f3811945a91062062994c5c4a3f4c0af2b0
hf download Qwen/Qwen3-VL-32B-Instruct \
  --revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb
```

这些版本与 `config.json` 一致。若 GPU 节点不能联网，可以在允许联网的节点下载到两边都能访问的磁盘；之后每个终端都设置同一个 `HF_HOME`。`hf-cache/` 已被 Git 忽略。

## 5. 上传完整数据包

将另行提供的 **`deblurstage2-docsity.zip`** 通过 JupyterLab 左侧上传按钮传到 `deblurstage2/`。它包含 4/14 号全部 157 对已对齐的 Blur/Out 和相对路径索引，不包含 Clear/GT。0 号存在缺块，不拼补成完整图混入这批数据。解压：

```bash
mkdir -p data
unzip deblurstage2-docsity.zip -d data
ls data/docsity/samples.jsonl data/docsity/images/
```

应得到：

```text
deblurstage2/
  data/docsity/samples.jsonl
  data/docsity/images/Blur_4_004.png
  data/docsity/images/Out_4_004.png
  data/docsity/images/...
```

这份完整数据已完成坐标对齐，无需再运行 prepare。将来加入新数据时，按 [README 的输入准备步骤](README.md#阶段-i现有-out--文字) 导入；不要把含答案的三列比较 PDF 输入模型。

## 6. 顺序运行 OCR 和 32B

下面直接对全部 157 对图片进行正式 OCR 和文字转写，不设置样本数限制。两个程序顺序运行，OCR 进程结束后再启动 Qwen，以释放显存。这一步运行模型推理，不更新权重。

```bash
conda activate deblur-ocr
python ocr.py --samples data/docsity/samples.jsonl \
  --output runs/docsity_candidates.jsonl --offline

conda activate deblur-qwen
python restore.py --samples data/docsity/samples.jsonl \
  --candidates runs/docsity_candidates.jsonl --model-size 32b \
  --output runs/docsity_32b.jsonl --offline
```

在 JupyterLab 文件浏览器打开 `runs/docsity_candidates.jsonl` 查看 OCR 候选，打开 `runs/docsity_32b.jsonl` 查看最终 `text`。检查记录的 `status`：`error` 或 `truncated` 需要处理，不能当作完整结果。

### 提高长度预算后重新运行

当前配置将单次输出上限从 2048 提高到 **32768 tokens**，输入加预留输出上限从 8192 提高到 **131072 tokens**。模型遇到结束标记会提前停止，并非每页都生成 32768 tokens。上限留作不结束或重复生成时的兜底；当前没有专门的死循环检测，触顶仍标记 `truncated`，不静默当作完整结果。

如果旧 Qwen 进程仍在运行，在它所在的 Terminal 按一次 **Ctrl+C**，等 shell 提示符回来。然后执行下面的命令更新配置并正式重跑 Qwen；不用重新运行 OCR，也不用重新下载权重。

```bash
git pull --ff-only
conda activate deblur-qwen
export HF_HOME="$PWD/hf-cache"
python restore.py --samples data/docsity/samples.jsonl \
  --candidates runs/docsity_candidates.jsonl --model-size 32b \
  --output runs/docsity_32b_long.jsonl --offline
```

新结果保存到 `runs/docsity_32b_long.jsonl`，全部处理结束后生成对应 `.md`。配置变化会使旧 Qwen 缓存失效，因此整批图片对都会按新预算重跑；旧结果文件保留。更长的实际输入或输出会增加耗时和显存，仍需检查最终状态及文字内容。

如需 8B 对照，先下载它，再复用同一份 OCR 候选：

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
python restore.py --samples data/docsity/samples.jsonl \
  --candidates runs/docsity_candidates.jsonl --model-size 8b \
  --output runs/docsity_8b.jsonl --offline
```

新开终端后，重新进入项目并恢复缓存变量与对应环境，例如：

```bash
cd /实际的qaoa父目录/deblurstage2
export HF_HOME="$PWD/hf-cache"
conda activate deblur-qwen
```

Jupyter 会话/GPU 分配到期可能终止进程，浏览器窗口本身不保证任务持续运行。较长训练应遵循学校提供的持久作业方式。

## 7. 当前使用预训练模型，不做微调

其他训练文档的 Out 不存在，也不会提供。当前正式任务直接使用预训练 OCR2 与 Qwen，完成这 157 对图片的 OCR 与转写，无需准备训练数据或执行 `train.py`。原有依赖真实训练 Out 的 LoRA 方案不纳入当前流程；若以后需要微调，必须按实际可用材料重新设计方案。
