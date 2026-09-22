"""Run frozen DeepSeek-OCR2 separately on Blur/Out; append resumable JSONL."""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

from common import fingerprint, load_config, load_samples, read_jsonl, source_hashes


def completed_keys(rows):
    # Downstream readers use the last attempt for each ID, including failures.
    latest = {row["id"]: row for row in rows}
    return {(row["id"], row.get("fingerprint")) for row in latest.values()
            if row.get("status") == "ok"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--samples", default="data/samples.jsonl")
    parser.add_argument("--output", default="runs/candidates.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-path", help="Existing local model/snapshot directory")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    cfg = load_config(args.config)["ocr"]
    revision = cfg.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("ocr.revision must be a complete upstream commit SHA")
    source = cfg["id"]
    local_stamp = None
    if args.model_path:
        source = str(Path(args.model_path).resolve())
        if not Path(source).is_dir():
            parser.error("--model-path must be an existing model directory")
        local_stamp = [(str(p.relative_to(source)), p.stat().st_size,
                        p.stat().st_mtime_ns)
                       for p in sorted(Path(source).rglob("*")) if p.is_file()]

    settings = {
        "prompt": "<image>\nFree OCR. ",
        "base_size": cfg.get("base_size", 1024),
        "image_size": cfg.get("image_size", 768),
        "crop_mode": cfg.get("crop_mode", True),
        "eval_mode": True,
        "save_results": False,
    }
    output = Path(args.output)
    done = completed_keys(read_jsonl(output) if output.exists() else [])
    pending = []
    for row in load_samples(args.samples)[:args.limit]:
        input_hashes = source_hashes(row)
        key = fingerprint({"inputs": input_hashes, "config": cfg,
                           "source": source, "local_stamp": local_stamp,
                           "settings": settings, "adapter_version": 1})
        if (row["id"], key) not in done:
            pending.append((row, key, input_hashes))
    if not pending:
        print("All requested samples already have matching successful OCR results.")
        return 0

    # Heavy imports are deferred so --help and a completed resume need no GPU stack.
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-OCR2 requires the configured CUDA environment")
    load_args = {"trust_remote_code": True, "local_files_only": args.offline}
    if not args.model_path:
        load_args.update(revision=revision, code_revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(source, **load_args)
    model = AutoModel.from_pretrained(
        source, _attn_implementation="flash_attention_2",
        use_safetensors=True, **load_args,
    ).eval().cuda().to(torch.bfloat16)
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with output.open("a", encoding="utf-8") as sink:
        for row, key, input_hashes in pending:
            result = {"id": row["id"], "blur_text": "", "out_text": "",
                      "status": "ok", "fingerprint": key,
                      "backend": "deepseek-ocr2-hf", "revision": revision,
                      "model": cfg["id"], "model_source": source,
                      "source_hashes": input_hashes, "settings": settings}
            for view in ("blur", "out"):
                try:
                    # infer() creates files even in eval_mode; isolate each call.
                    with tempfile.TemporaryDirectory(prefix="deblurstage2-ocr-") as tmp:
                        value = model.infer(tokenizer, image_file=row[view],
                                            output_path=tmp, **settings)
                    if not isinstance(value, str):
                        raise TypeError("Expected infer(eval_mode=True) to return text")
                    result[f"{view}_text"] = value
                except Exception as exc:
                    result["status"] = "error"
                    result.setdefault("errors", {})[view] = f"{type(exc).__name__}: {exc}"
                    torch.cuda.empty_cache()
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            failures += result["status"] != "ok"
            print(f'{row["id"]}: {result["status"]}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
