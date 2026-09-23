# deblurstage2

从配准的 Blur/Out 恢复英文文字，输出 512×768 白底页面、可复制文字 PDF，以及同一样本的 Clear / Output / Blur / Recovered 四列对照。

**当前入口：`ocr_lines.py` → `recover.py` → `render.py`。云端完整命令见 [JUPYTER.md](JUPYTER.md)。** 使用已有 H200、`deblur-qwen`、Qwen 32B 缓存和 157 对图片；新增独立 Paddle 环境。程序在服务器本地运行公开权重，不调用收费推理 API。

```text
Blur + Out
  → 双图行检测、共用裁剪、分栏阅读顺序
  → PP-OCRv6 medium + v5 英文识别器：保存字符概率
  → 英文字符约束、CTC 候选、字符组混淆与词典召回
  → Qwen 局部选择 / 短词提议 → 缓存概率再评分
  → 按行装配 → 白底 PNG / 可复制 PDF / 对照 PDF
```

Qwen 可切换 8B/32B；默认复用已下载的 32B。它不再自由续写整页。允许保留 gibberish，允许提出原 OCR 中没有的新词；每次改动记录原读法、候选与各视觉来源的支持。正文限制为 A–Z/a–z、0–9、ASCII 标点与空格，行之间由程序换行。数字不作全局 `1→l`、`0→O` 替换。

## 输入与输出

现有 `data/docsity/samples.jsonl` 直接使用，不重新上传、不重新跑 base deblur。每行包含：

```json
{"id":"4_004","document_id":"4","deblur_run":"docsity_20260915","split":"test","blur":"images/Blur_4_004.png","out":"images/Out_4_004.png"}
```

图片路径相对 manifest 所在目录，两图必须配准、尺寸一致。512×768 是已有区域尺寸，不保证等于原 PDF 全页。当前 4/14 共 157 对；0 号缺块未拼补混入。其他训练文档的 Out 不存在，运行不依赖它们。

| 文件 | 内容 |
| --- | --- |
| `runs/v2/evidence.jsonl` | 行位置、阅读顺序、模型身份和缓存索引 |
| `runs/v2/evidence.assets/` | 同坐标行裁剪，以及保留原概率的压缩 NPZ |
| `runs/v2/recovery.lines.jsonl` | 每行候选、CTC 分数、Qwen 回复、选择及中断恢复日志 |
| `runs/v2/recovery.jsonl` | 最终按页组织的文字和坐标 |
| `runs/v2/render/recovered/*.png` | 512×768 白底文字页 |
| `runs/v2/render/recovered.pdf` | 可选择和复制文字的 PDF |
| `runs/v2/render/comparison.pdf` | 有 Clear 参考时四列；没有时为明确标注的三列 |
| `runs/v2/render/report.json` | 排版、溢出、覆盖诊断和各页状态 |

Clear 只通过 renderer 独立的 `--clear-references` 清单进入对照图，不进入 OCR、Qwen 或词候选。格式为 `{"id":"4_004","clear":"images/Clear_4_004.png"}`，路径相对该清单；必须和当前样本 ID、尺寸对应。不要输入旧 2 号的三列比较 PDF 代替当前 4/14 数据。

`ok` 表示步骤完成且结构合法，**不表示识别正确**。`review` 保留有效文字，记录被拒绝的提议、可疑行几何或覆盖缺口。`error` 保留错误原因并在下次运行时重试。默认不把缺行/错误结果排版成完整成品；需要诊断图时才使用 `render.py --allow-incomplete`。

字号根据行框自适应；文字确实放不下时不裁尾，而是标记位置并将全文放在额外白底页，详情写入 report。

## 模型与环境

- 检测：`PP-OCRv6_medium_det`。
- 字符识别：`PP-OCRv6_medium_rec` 与 `en_PP-OCRv5_mobile_rec`。
- 局部语言判断：原 `Qwen/Qwen3-VL-32B-Instruct`，可选 8B。
- OCR 用 Paddle GPU 3.2.0 cu126 + `paddlex[ocr-core]==3.7.0` 静态接口；省去 PaddleOCR 包装层和可选 VLM 后端。
- Qwen 沿用 Torch 2.6.0 cu124 / Transformers 4.57.1；只增加词典和 CPU 排版依赖。

默认模型在 `config.json` 锁定 immutable revision。OCR 的 `--download-only` 只下载三套静态模型必要文件；`--offline` 只读缓存。OCR 和 Qwen 顺序执行，通过文件交接，两个框架不混装到同一环境。

## 代码与可调设置

| 文件 | 职责 |
| --- | --- |
| `ocr_lines.py` | 检测、行几何、共同裁剪与阶段缓存 |
| `paddle_ctc.py` | 锁版本的 PaddleX 概率提取适配 |
| `ctc.py` | 受限解码、完整 CTC 路径求和 |
| `candidates.py` | 字符组编辑、词表召回、拆合词和区间装配 |
| `recover.py` | 已有 Qwen 的局部候选选择与短提议 |
| `render.py` | 字体排版、PNG/PDF 和同样本对照 |

`config.json` 的 `v2` 段控制检测、字符范围、候选搜索、Qwen 图像与输出预算。`RECOVERY_PROMPT.md` 可直接编辑；`--lexicon` 可换成自备 `word count` 英文词典。允许保留词表外名称，不会把最高频近邻直接当答案。

字符概率是识别器的视觉支持，不是恢复正确率；CTC 时间步也不是精确字母框。保持动态行宽，不强制压成 320 像素。几何使用共享轴对齐裁剪，保留原检测多边形，未实现任意旋转/复杂弯曲文本的自动校正。

## 缓存与复核

同一命令重新执行即可：OCR 复用完整且哈希匹配的页面，Qwen 复用完整的行，失败/变化部分重算。丢失或被修改的图像/NPZ 会拒绝恢复，先重跑 OCR 重建该页。不要让两个作业同时写同一输出路径。

改 Qwen 提示或词典不重跑 OCR；只改排版不重跑模型。改变字符范围需要重新生成 OCR 证据。旧 DeepSeek 字符串和旧 Qwen JSONL 无法转为字符概率，不作为 v2 缓存。

本地 CPU 检查：

```bash
python -m unittest discover -s tests -v
```

覆盖 CTC 与枚举路径对照、概率/字典、数字和字符限制、候选边界、分栏几何、文件完整性、中断日志、模拟模型的跨阶段合同，以及 PDF 文字和 PNG 一致性。未执行模型 smoke test；本地复核不等于在 H200 实测新模型，也不证明恢复效果已改善。

`ocr.py`、`restore.py`、`restore_local.py` 保留作历史复现，已退出默认流程。`train.py` 的旧 LoRA 数据设计需要训练 Out，不适用于当前材料。未来 OCR 微调应使用已有训练文档的 Blur/可靠行转写和上游训练循环，保持 0/4/14 留出；当前三步运行都是推理。

设计与限制见 [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md)。Git 不包含真实文档图片、模型、运行结果或凭据。
