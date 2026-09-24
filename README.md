# deblurstage2

读取 deblur 的 **Out**，针对它的字形失真微调 PaddleOCR，输出英文文字、512×768 白底页面和可复制文字 PDF。当前默认是 v3，运行命令见 [JUPYTER.md](JUPYTER.md)。

```text
训练：doc4 Out + 对齐的 PDF 原文 → 短词组裁剪 → 官方 PaddleOCR 识别器微调
推理：Out → 行检测 / 几何整理 → 微调识别器 → 英文字符解码 → PNG / PDF
```

主流程不加载 Qwen、DeepSeek 或 Torch。复用现有 `deblur-paddle`、检测模型、HF 缓存和 Out 图片；训练调用官方 PaddleOCR v3.7.0 脚本及独立的 `.pdparams` 预训练权重。训练环境单独建立，避免训练依赖与 PaddleX 的 OpenCV 包互相覆盖。

## 本次数据划分

| 用途 | 数据 | 已准备的样本 |
| --- | --- | --- |
| stage2 训练 | 4 号前 59 个可用页面 | 4,807 个短词组 |
| stage2 验证、选择权重 | 4 号最后 15 个可用页面 | 1,220 个短词组 |
| 跨文档诊断 | 14 号的 83 个 Out 区域 | 不进入训练与选权重 |

原 `split=test` 表示 stage1 deblur 的划分，保持不变；新的 `stage2_split` 单独标记用途。4 号已参与 stage2 训练，不能再称为 stage2 独立测试。14 号是当前跨文档检查；历史中已查看过它的结果，不宣称为从未接触过的盲测集。

6,027 个标签来自已有匹配 PDF 的文字层，经过配准映射到 Out 坐标，未使用 OCR/Qwen 输出作为监督。检查 PDF、Clear、Out 哈希，排除裁到一半的词和水印。标签是 `geometry_checked`，不冒充逐词人工校对。8 个带重音的词被明确排除；标点、连字的归一化保留审计记录。

官方识别器训练时只有 40 个 CTC 时间步，因此按完整单词边界组成短语，检查字符数及重复字符的额外时间步需求。**不截断标签。** 推理仍在完整 Out 页面上检测行，并使用动态行宽。训练裁剪允许用 doc4 Clear/PDF 定位；推理只读取 Out。Clear 和 Blur 仅可另传给 renderer 制作对照。

## 入口

| 文件 | 职责 |
| --- | --- |
| `export_ocr_labels.py` | 从本地已验证 PDF/Clear 几何导出 doc4 标签；服务器使用提供的标签包即可 |
| `prepare_ocr.py` | 校验标签与 Out，按原页面划分训练/验证，生成裁剪和 stage2 清单 |
| `train_ocr.py` | 薄封装：调用固定版本官方训练、完整检查点续训、静态模型导出 |
| `ocr_lines.py` | Out 行检测、整理、字符概率与文本输出；`--model-dir` 切换微调权重 |
| `render.py` | 白底 PNG、可复制 PDF、Clear / Out / Blur / Recovered 四列对照 |

`configs/ocr_out.yml` 默认 20 epochs、batch 16、学习率 `1e-5`；只训练识别器，检测器保持原权重。保留官方字典和模型结构，以复用预训练权重。推理输出限制为 A–Z/a–z、0–9、ASCII 标点和空格；不做全局 `1/l`、`0/O` 替换，不强行把乱码改成词典中的词。

## 输出与复用

`ocr_lines.py` 输出 JSONL，同时在相邻 `.assets/` 目录保留行裁剪和原始字符概率。renderer 直接读取该 JSONL；`ok` 只表示步骤完成，不代表文字正确。空行、可疑几何、覆盖缺口标记为 `review`；真正失败为 `error`。

- 同一推理命令重跑，复用完整且哈希匹配的页面；失败或变化页面重算。
- `--detections runs/v2/evidence.jsonl` 可复用相同 Out、检测器和参数的旧检测框。新裁剪与新权重必须重新识别，旧 NPZ 不会冒充微调结果。
- 训练中断使用 `--resume latest`，同时要求模型、优化器和训练状态文件；恢复到最近保存的检查点。
- 只改排版，只重跑 `render.py`。不要同时向同一输出文件写入。

renderer 生成 `recovered/*.png`、`recovered.pdf`、`comparison.pdf` 和 `report.json`。默认保留源坐标；文字放不下会记录扩展或追加页，不静默裁字。`--allow-incomplete` 明确允许把缺失结果绘成诊断图。

旧 Qwen/DeepSeek 代码保留用于历史复现，退出默认入口。v2 需要显式 `ocr_lines.py --pipeline v2`，历史命令见 [JUPYTER_V2.md](JUPYTER_V2.md)。不要对当前数据执行旧 `train.py`。

本地校验使用 `python -m unittest discover -s tests -v`，覆盖标签映射、划分、CTC、训练/导出调用、缓存和 PDF。CPU 校验不等于 H200 真实训练，也不证明微调效果已提高。真实图片、标签、权重和实验结果不提交到 Git。

上游依据：[固定版识别配置](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml)、[官方微调文档](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/docs/version2.x/ppocr/model_train/finetune.en.md)。