# JupyterLab：Out 专用 PaddleOCR 微调

在 ICRN 的 JupyterLab **Terminal** 按顺序执行。项目仍是 `~/deblurstage2`，复用现有图片、`deblur-paddle` 和 `hf-cache`。这套命令直接准备正式数据、训练和推理，没有模型 smoke test。先让旧 OCR/Qwen 进程停止，避免占用同一 GPU。

## 1. 更新代码和现有推理环境

```bash
cd ~/deblurstage2 &&
git -c safe.directory="$PWD" pull --ff-only &&
git -c safe.directory="$PWD" log -1 --oneline
```

更新失败就先处理 Git 提示，不要继续运行旧代码；不需要 `reset --hard`。随后：

```bash
conda activate deblur-paddle &&
python -m pip install -r requirements-render.txt &&
python -m pip check
export HF_HOME="$PWD/hf-cache"
```

保留现有 Paddle GPU、PaddleX 和 OpenCV。若新终端无法激活 Conda，先执行 `source "$(conda info --base)/etc/profile.d/conda.sh"`。

## 2. 上传标签包，生成正式训练数据

将提供的 **`deblurstage2-v3-labels.zip`** 上传到项目目录。包中只有 doc4 标签和审计记录；不必重传 Out 或原始 PDF。

```bash
unzip -n deblurstage2-v3-labels.zip -d data &&
python prepare_ocr.py \
  --samples data/docsity/samples.jsonl \
  --labels data/doc4_out_labels.jsonl \
  --output data/ocr_out
```

当前材料应打印 **4,807 train / 1,220 validation**。程序会核对标签和实际 Out 哈希，不匹配时停止。输出 `data/ocr_out/stage2_samples.jsonl` 供后续推理和排版使用。4 号前 59 个可用页面训练、最后 15 个页面验证；14 号 83 个区域用于跨文档诊断。

## 3. 一次性建立训练环境并下载官方训练权重

训练依赖 Albumentations 所需的 headless OpenCV，与原 PaddleX 环境的 contrib OpenCV 会共用 `cv2` 文件。为避免覆盖现有可用环境，单独创建一个 **`deblur-paddle-train`**；模型、图片和训练结果仍放在同一项目目录共享。不安装 Qwen、Torch 或新 CUDA toolkit。

```bash
conda create -n deblur-paddle-train -c conda-forge python=3.11 -y &&
conda activate deblur-paddle-train &&
python -m pip install paddlepaddle-gpu==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ &&
python -m pip install -r requirements-train.txt &&
python -m pip check
```

以下首次执行。已有 HF 静态模型用于推理，不能直接代替训练检查点：

```bash
mkdir -p vendor models &&
git clone --branch v3.7.0 --depth 1 \
  https://github.com/PaddlePaddle/PaddleOCR.git vendor/PaddleOCR &&
wget -c -O models/PP-OCRv6_medium_rec_pretrained.pdparams \
  https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams
```

若 `vendor/PaddleOCR` 已存在，跳过 clone；若下载中断，只重复 wget 行。包装器验证上游提交为 `b03f46425e8ff4442b268ce449e3eef758146cd4`，不使用会漂移的主分支。

共享挂载可能触发 Git 的 `dubious ownership`。新版包装器只对指定的 PaddleOCR 绝对路径在检查命令中临时设置 `safe.directory`，不修改全局配置或文件所有权。若旧版在 `upstream_directory` 报此错误，先更新项目代码，再重复训练命令即可；此时训练尚未开始，不需要 `--resume`，也不用重新下载权重。

## 4. 正式微调并导出

```bash
conda activate deblur-paddle-train &&
python train_ocr.py train \
  --paddleocr vendor/PaddleOCR \
  --data data/ocr_out \
  --output runs/v3/train \
  --pretrained models/PP-OCRv6_medium_rec_pretrained.pdparams
```

默认训练 20 epochs，batch 16、学习率 `1e-5`；用 doc4 验证集选择最佳权重。具体耗时以 H200 实际日志为准，本地未进行 GPU 训练。完成后：

```bash
python train_ocr.py export \
  --paddleocr vendor/PaddleOCR \
  --run runs/v3/train \
  --output models/ocr_out
```

导出 `inference.json`、`inference.pdiparams`、`inference.yml` 和训练来源记录。导出目录必须为空；更换训练时使用新的目录，避免把不同实验混在一起。

## 5. 运行相同输入下的原模型与微调模型

切回现有 PaddleX 推理环境，补齐或复用固定版静态模型缓存：

```bash
conda activate deblur-paddle &&
export HF_HOME="$PWD/hf-cache" &&
python ocr_lines.py --download-only
```

下载命令只处理小 OCR 模型文件，不加载 GPU。以下两次均为完整的 14 号跨文档实验。原模型也使用新的 Out 行几何，用它比较才不会把几何变化误认为微调收益。

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
python ocr_lines.py \
  --samples data/ocr_out/stage2_samples.jsonl \
  --documents 14 \
  --output runs/v3/baseline_14.jsonl \
  --offline
```

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
python ocr_lines.py \
  --samples data/ocr_out/stage2_samples.jsonl \
  --documents 14 \
  --model-dir models/ocr_out \
  --detections runs/v3/baseline_14.jsonl \
  --output runs/v3/finetuned_14.jsonl \
  --offline
```

第二条复用第一条的 Out 检测框，只重新运行微调识别器。均不运行 Qwen，不读取 Blur/Clear。已有 `runs/v2/evidence.jsonl` 也能传给第一条的 `--detections`，前提是覆盖全部所选页面且检测器参数和 Out 哈希一致。新的识别概率必须重新计算。

如需看全部 157 个区域，在另一条推理命令中去掉 `--documents 14`，去掉仅含 14 号的 `--detections`，另存 `runs/v3/finetuned_all.jsonl`。4 号会明确标记 train/validation，不能当成跨文档成果。

## 6. 生成白底页面和对照 PDF

已有 `data/docsity_references/clear.jsonl` 时直接运行。没有时，将此前的 `deblurstage2-clear-references.zip` 上传到项目目录并执行 `unzip -n deblurstage2-clear-references.zip -d data`；也可以先省略 `--clear-references`，得到三列对照。

```bash
python render.py \
  --samples data/ocr_out/stage2_samples.jsonl \
  --documents 14 \
  --predictions runs/v3/finetuned_14.jsonl \
  --clear-references data/docsity_references/clear.jsonl \
  --output runs/v3/render_finetuned_14 \
  --allow-incomplete
```

`--allow-incomplete` 保留漏识别位置并标记为诊断输出，方便查看真实效果；不代表每一行都恢复成功。再运行同一命令，把 predictions 改为 `runs/v3/baseline_14.jsonl`、output 改为 `runs/v3/render_baseline_14`，即可对比原模型。

生成 `recovered/*.png`、`recovered.pdf`、`comparison.pdf` 和 `report.json`，均在对应 render 目录。四列顺序为 Clear / Out / Blur / Recovered。Clear 只参与展示；没有输入识别器。

## 中断后继续

新终端先执行 `cd ~/deblurstage2`，按下面的步骤激活对应环境。

- 训练：`conda activate deblur-paddle-train`，重复第 4 步 train 命令，末尾加 `--resume latest`。要求 `.pdparams`、`.pdopt`、`.states` 三份文件齐全；恢复最近保存的 epoch/优化器状态，未保存的进度重跑。
- 推理：`conda activate deblur-paddle`、`export HF_HOME="$PWD/hf-cache"`，重复原推理命令；已完整保存且哈希匹配的页面显示 `cached`，中断中的页面重算。
- 排版：在 `deblur-paddle` 重复 render 命令，无须重跑模型。

同一训练目录续训时，数据、配置、包装代码和初始权重必须与原来一致。浏览器断线不一定终止进程，但服务器作业结束仍会终止运行；这些命令没有额外建立后台守护进程。
