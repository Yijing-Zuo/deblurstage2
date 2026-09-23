# 局部读图：复用现有 Qwen，正式处理全部 157 对

本入口直接读取配对的 Blur/Out 局部图，不读取 DeepSeek 文字候选。使用已有 `deblur-qwen` 环境、模型缓存和样本清单；无需重跑 OCR、安装新依赖或训练。旧 `restore.py` 仍保留用于比较。

## 在 JupyterLab Terminal 运行

旧推理已停止时，执行：

```bash
cd ~/deblurstage2 &&
git -c safe.directory="$PWD" pull --ff-only &&
git -c safe.directory="$PWD" log -1 --oneline &&
conda activate deblur-qwen &&
export HF_HOME="$PWD/hf-cache" &&
python restore_local.py \
  --samples data/docsity/samples.jsonl \
  --model-size 32b \
  --output runs/docsity_32b_local.jsonl \
  --offline
```

这会正式处理清单中的全部 157 对，没有单样本测试步骤。`&&` 保证更新失败时不会继续启动旧代码；`safe.directory` 仅对本次 Git 命令指定当前项目。命令不需要 `--candidates`。

已缓存 8B 权重时，把 `--model-size 32b` 改为 `8b`，并将输出改为 `runs/docsity_8b_local.jsonl` 即可。两种规模使用各自的权重，不能把 32B 文件当作 8B 模型。

## 本轮怎样读图

- 默认每个核心区域高约 192 个原图像素，在附近行间空白处调整边界；当前图像约相当于 6–7 行。上下各保留约 32 像素上下文。
- 核心区域连续覆盖整图，互不重叠；上下文可以重叠。两张图裁剪坐标一致，新增白色侧边中的红括号标出本次应转写的核心区域。
- 局部图默认放大约 2 倍；本入口每张图像像素预算为 786432。裁剪和插值改善模型对已有字形的查看，不增加原图中不存在的笔画。
- Qwen只返回当前核心区的 `{"text": "..."}`，括号外用于理解上下文。程序按核心坐标顺序拼接，不使用模糊字符串去重，也不让模型重新改写全文。

模型仍可能误读 Out 中已经错误的字形。局部块大小是工程起点，不是已验证的最优值；没有承诺这次输出一定恢复正确。沿用 `config.json` 的大输出预算，循环、截断或格式异常另外记录。

## 看结果与继续运行

同一输出目录会得到：

| 文件 | 内容 |
| --- | --- |
| `docsity_32b_local.jsonl` | 按原样本汇总的正文与运行状态 |
| `docsity_32b_local.md` | 便于直接阅读的正文 |
| `docsity_32b_local.blocks.jsonl` | 每块坐标、原始模型回答与运行信息 |

`ok` 表示完成生成与格式处理，不是语义正确认证。`review` 表示含疑读、解释性输出、异常语言或空白等需要检查的内容；`format_error`、`truncated`、`loop`、`error` 分别表示格式不合要求、达到长度上限、检测到循环或执行失败。原始回答保留在块记录中；JSON能够解析也不代表英文是真实原文。

每块完成后写入并刷新结果。进程确实停止后，用**相同命令与输出路径**继续，会复用已完整写入、指纹一致且已完成解析的 `ok` 和 `review` 块，避免每次都重新生成永久不可读的 `[unclear]`；被打断的当前块需要重跑。若强杀或存储故障留下不完整 JSON 行，需要先处理损坏记录，程序不会静默删除它。修改图片、提示词或推理参数后，相应旧块不能直接复用。若想以相同配置从头重新生成，请换一个新的输出文件名。连接断开但进程仍在运行时，不要同时启动第二份写入同一输出路径的作业。

本地已检查全部 157 对图像的 628 个局部块：核心覆盖完整、配对尺寸一致，放大后的最大单图面积为 604608 像素，低于本入口预算。图像布局、解析、拼接和缓存通过 CPU 检查；没有执行模型 smoke test。本实现尚未在服务器 GPU 上验证恢复质量；检查具体可读词句、漏行、数字和专名，比仅看 `ok` 更有用。

像素预算沿用 [Qwen 官方 processor 的控制方式](https://github.com/QwenLM/Qwen3-VL#pixel-control-via-official-processor)。实际编码的 `image_grid_thw` 与输入尺寸会写入每块记录，方便核对。`--core-height`、`--context`、`--scale` 可调整裁剪，首次正式运行使用上面的默认命令即可。
