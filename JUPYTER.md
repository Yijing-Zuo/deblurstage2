# JupyterLab：更新到字符证据恢复架构

在现有 JupyterLab **Terminal** 执行 Bash 命令。继续使用 `~/deblurstage2`、现有 157 对图片、`deblur-qwen` 和 `hf-cache`。不重建 Qwen、不重下同一份 32B、不需要其他训练 Out。以下直接运行完整实验，不包含模型 smoke test。

先停止仍在运行的旧 `restore.py` / `restore_local.py`：在其终端按 Ctrl+C，等 shell 提示符返回。

## 1. 更新代码

```bash
cd ~/deblurstage2 &&
git -c safe.directory="$PWD" pull --ff-only &&
git -c safe.directory="$PWD" log -1 --oneline
```

`safe.directory` 只信任当前命令使用的项目，处理此前 ownership 报错。更新报错时不要继续用旧代码运行。已有本地修改时不要 `reset --hard` 覆盖；Git 会指出冲突文件。

每个新 Terminal 设置同一个模型缓存路径：

```bash
cd ~/deblurstage2
export HF_HOME="$PWD/hf-cache"
```

若 `conda activate` 未初始化，先执行 `source "$(conda info --base)/etc/profile.d/conda.sh"`。

## 2. 新建专用 OCR 环境并下载小模型

`deblur-paddle` 只创建一次；已创建时从激活开始。

```bash
conda create -n deblur-paddle -c conda-forge python=3.11 libgl=1.7 -y &&
conda activate deblur-paddle &&
python -m pip install paddlepaddle-gpu==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ &&
python -m pip install -r requirements-paddle.txt &&
python -m pip check &&
python ocr_lines.py --download-only
```

通过 `paddlex[ocr-core]==3.7.0` 直接调用官方检测/识别器，不安装 PaddleOCR-VL、TensorRT 或新 Transformers。三套模型按固定 SHA 下载至 HF 缓存；`--download-only` 不加载 GPU。

截图驱动 570.211.01 满足官方 cu126 wheel 的驱动下限。无需编译 FlashAttention。已核查官方接口与版本声明，但未在本地进行 H200 真实模型加载。

若已经按旧命令装好环境，运行时出现 `ImportError: libGL.so.1`，补装一次即可，不必重建环境或重新下载权重：

```bash
conda activate deblur-paddle &&
conda install -c conda-forge --freeze-installed libgl=1.7 -y
```

`libgl` 在当前 Conda 环境提供 OpenCV 所需的原生共享库；`pip check` 通过只说明 Python 包依赖声明相容，不证明原生动态库齐全。PaddleX 依赖普通版 `opencv-contrib-python`，保留当前包，不同时叠装 headless 版。下面 OCR 命令临时加入当前环境的库目录，仅影响该进程，切换 Qwen 后不会残留 Paddle 库路径。[libgl 官方配方](https://github.com/conda-forge/libglvnd-feedstock/blob/main/recipe/meta.yaml)、[PaddleX 依赖声明](https://github.com/PaddlePaddle/PaddleX/blob/v3.7.0/setup.py)。

## 3. 给现有 Qwen 环境补小依赖

```bash
conda activate deblur-qwen &&
python -m pip install -c requirements-qwen.txt \
  -r requirements-recovery.txt -r requirements-render.txt &&
python -m pip check
```

约束文件保留原 Torch、Transformers、NumPy 等版本；新增 SymSpell 词典和 PyMuPDF 排版。不要升级 Transformers 或重装旧 DeepSeek 环境。以下三步可离线运行。

## 4. 正式运行全部 157 对图的 OCR

```bash
conda activate deblur-paddle &&
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
python ocr_lines.py \
  --samples data/docsity/samples.jsonl \
  --output runs/v2/evidence.jsonl \
  --offline
```

分别检测 Blur/Out，建立共用行裁剪，每行保存两个识别器 × 两种图像的字符分布。字符范围为英文大小写、数字、标点与空格。裁剪和 NPZ 在 `runs/v2/evidence.assets/`，不要只移动 JSONL 丢掉 assets。

成功结束后进入下一步。`error` 会打印原因；修复后重复命令，完整未变的页面跳过。

若旧版在保存时出现 `TypeError: Object of type int64 is not JSON serializable`，更新代码后重复上面的 OCR 命令即可，无需重装依赖。此次保存修复保留现有 OCR 缓存标识：已有完整页面校验通过后显示 `cached`，出错页重算。保留原 `evidence.jsonl` 与 `evidence.assets/`；不要把未完成的 `.tmp` 文件当成正式结果。

## 5. 正式运行 Qwen 32B 局部恢复

```bash
conda activate deblur-qwen &&
python recover.py \
  --samples data/docsity/samples.jsonl \
  --evidence runs/v2/evidence.jsonl \
  --model-size 32b \
  --output runs/v2/recovery.jsonl \
  --offline
```

复用现有权重，不做训练。Qwen 处理真实行中的局部分歧，可提出短词，再由字符概率评分。32768-token 输出预算继续保留，并检测重复循环；有限预算仍存在，触顶不会静默当完整结果。

进度形如 `[123/4800] 4_004:l0012: ok`；行数由实际检测确定，此处只是格式示例。`ok` 表示结构有效，不表示文字正确；`review` 保留结果并记录异常；`error` 使程序结束时返回非零退出码。重复命令重试失败行。

查看 `recovery.jsonl` 的 `text`，详细候选在 `recovery.lines.jsonl`。页面文件随运行更新，含 `pending` 时尚未完成。

## 6. 导出白底页面和四列 PDF

为了得到 **Clear / Output / Blur / Recovered** 四列，将另行提供的 `deblurstage2-clear-references.zip` 上传到项目目录，首次解压：

```bash
unzip -n deblurstage2-clear-references.zip -d data
```

包内是当前 157 个匹配样本的 512×768 Clear 裁剪与独立清单，仅供 renderer 使用。然后运行：

```bash
conda activate deblur-qwen &&
python render.py \
  --samples data/docsity/samples.jsonl \
  --recovery runs/v2/recovery.jsonl \
  --clear-references data/docsity_references/clear.jsonl \
  --output runs/v2/render
```

只用 CPU，生成：

- `runs/v2/render/recovered/*.png`：512×768 白底文字页。
- `runs/v2/render/recovered.pdf`：可复制文字的 PDF。
- `runs/v2/render/comparison.pdf`：四列对照。
- `runs/v2/render/report.json`：排版与覆盖诊断。

Clear 不传给 OCR/Qwen。不上传参考包也能先看白底结果：删除 `--clear-references ...` 行即可，此时对照为 Output / Blur / Recovered 三列。

## 切换 8B、续跑和调参

8B 尚未缓存时联网下载一次：

```bash
conda activate deblur-qwen
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

复用 OCR，另存 8B 结果：

```bash
python recover.py \
  --samples data/docsity/samples.jsonl \
  --evidence runs/v2/evidence.jsonl \
  --model-size 8b \
  --output runs/v2/recovery_8b.jsonl \
  --offline
```

中断后激活对应环境、设置 `HF_HOME`，重复原命令。OCR 以完整页面、Qwen 以完整行为单位恢复；中断中的页/行重算，不恢复 GPU KV cache。末尾不完整日志被忽略，中间记录损坏明确报错。

Qwen 的 `ok/review` 默认复用，`--retry-review` 可重算 review 行。修改 `RECOVERY_PROMPT.md`、词典或相关参数自动使恢复缓存失效。只改字体/尺寸时仅重跑 render。不要并发写同一路径。

浏览器断开与服务器进程结束不同；GPU 作业/会话过期仍可能终止进程，缓存不保证进程常驻。

当前不要执行旧 `train.py`。这次运行使用预训练模型；小 OCR 微调应根据新输出与已有训练 Blur/可靠转写准备，0/4/14 留出页不用于训练。
