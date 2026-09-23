# 英文文档恢复提示词的依据

2026-09-22。当前提示词是针对本项目输入设计的候选方案，尚未在服务器上比较新旧恢复效果，不能称为已验证的最优提示词。代码保持一次 Qwen 调用；8B/32B 共用 [PROMPT.md](PROMPT.md)。

## 为什么修改

4_010 的 Out 图本身就有清晰但错误的字形，旧 Qwen 输出与这些伪字相近。旧提示强调忠实转写可见文字，没有明确原文为英文，容易把 deblur 的错误当成要保留的内容。这些结果只有约 600–800 个生成 tokens，增加输出预算不能解决识别问题。`status: ok` 是运行状态，不是质量评价。

新提示明确恢复原始英文，主动结合英文词汇、语法和区域上下文纠正错字、漏字、黏连与断词；不限制为一两个字符的修改。与此同时，词长、字形、位置和可辨的相邻文字仍是约束，不能根据主题补写段落。保留专名、公式、数字和已有可读文字，裁剪外内容不补全，无法恢复的局部用 `[unclear]`。这些是本项目的设计选择，不是论文已在 Docsity 上验证的结论。

## 本轮定向参考

| 来源与阅读位置 | 直接相关的依据 | 本项目采用与边界 |
| --- | --- | --- |
| [Qwen3-VL 官方 README：Transformers / Multi image inference](https://github.com/QwenLM/Qwen3-VL#using--transformers-to-chat) | 官方示例将多张图和任务文本放在同一 user 消息中，经 chat template 输入模型。 | 保持现有双图加提示词接口；不必为提示词修改增加调用层。官方示例不证明我们的恢复效果。 |
| [olmOCR 的实际 prompts.py](https://github.com/allenai/olmocr/blob/main/olmocr/prompts/prompts.py) | 包含图像加原始文本的提示、明确输出格式和跨页句子保留要求，也包含不使用文本锚点的模板。 | 明确候选身份、阅读顺序与裁剪边界。没有照搬其去页眉等规则，也不据此认定提供 OCR 在所有模型上都更好。 |
| [HIPE-OCRepair 2026，§5–6](https://arxiv.org/html/2607.08143v1) | 系统描述中有语言元信息，以及针对拆词、黏词、错字符、漏字符的提示设计；部分强系统还经过专门训练。 | 明确 English 和错误类型。语言提示的先例不等于单独增加 English 的消融证据；不把训练系统的收益归给提示词。 |
| [Post-Correction of Historical Text Transcripts with Large Language Models，Table 2 / §5](https://aclanthology.org/2024.latechclfl-1.14.pdf) | 比较多种纠错提示及少样本设置，没有一种设置在所有模型和数据上稳定占优；随机少样本可能降低效果。 | 具体说明识别错误，避免笼统要求改语法或润色；本轮不加入未经验证的示例，尤其不加入当前文档答案。 |
| [OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches，§6–7](https://arxiv.org/html/2502.01205v1) | 英文与芬兰语表现不同；片段长度及上下文会影响纠错，复杂上下文提示对不同模型效果不一。 | 保留当前区域内的多句上下文，不拆成孤立词；不据此承诺全文输入或更长提示一定更好。 |
| [Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents，§4.2 / §6.2](https://arxiv.org/html/2504.00414v1#S4.SS2) | 图像加外部 OCR 候选的后纠错有效；对同一模型自己的转写再次纠正未得到显著改善。 | 保留图像核对和已有外部 OCR，不增加反复自我润色循环。历史德语目录的实验不能直接代表严重 deblur 伪字。 |
| [LLM-Aided OCR：process_chunk 实际源码](https://github.com/Dicklesworthstone/llm_aided_ocr/blob/main/llm_aided_ocr.py) | 提示列出字形混淆、断词、保留结构、不添加内容和仅输出正文等要求。 | 借鉴具体错误类型与输出约束；项目实现是参考案例，不是最优提示词的实验结论。 |

## 使用方式与限制

`common.make_prompt()` 读取本仓库的 `PROMPT.md`，再追加两个 OCR 候选 JSON。推理和可选训练共用这一入口；完整提示词参与推理缓存指纹。更改文件后重启 Qwen，新模板不会复用旧模板的结果，OCR 候选与权重无需更新。输出保留 `prompt_template_hash` 便于确认实际版本。

当前是预训练 Qwen3-VL-Instruct 的一次推理，没有微调，也没有增加第二次“自检”调用。提示中的返回前核对只是任务要求，不是已经证实有效的独立验证器。`[unclear]` 也不能当作可靠的错误检测或置信度。英文先验可以缩小候选范围，不能保证补回原图已丢失的信息；应从正式输出中确认是否恢复了更多有依据的词句，而不只看句子是否通顺。
