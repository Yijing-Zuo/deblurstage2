"""Transcribe paired Blur/Out regions with a frozen Qwen3-VL and optional LoRA."""
import argparse
import json
from pathlib import Path

from common import (fingerprint, hash_file, load_config, load_samples, make_prompt,
                    model_settings, read_jsonl, source_hashes)


def base_identity(config, settings):
    return {"id": config["models"][settings["size"]]["id"],
            "revision": settings["revision"], "size": settings["size"]}


def image_messages(paths, prompt):
    from PIL import Image
    content = []
    for path in paths:
        with Image.open(path) as image:
            content.append({"type": "image", "image": image.convert("RGB")})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_processor(settings, config, offline):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        settings.get("path", settings["id"]), revision=settings["revision"], local_files_only=offline)
    ip = processor.image_processor
    ip.size["longest_edge"] = config["image_max_pixels"]
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = config["image_max_pixels"]
    return processor


def encode(processor, messages, generation=False):
    batch = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation,
        return_dict=True, return_tensors="pt", truncation=False)
    if "image_grid_thw" not in batch or len(batch["image_grid_thw"]) != 2:
        raise ValueError("Expected exactly two encoded images (Blur, Out)")
    return batch


def adapter_signature(path, config, settings, offline=False):
    from peft import PeftConfig
    directory = Path(path).resolve()
    meta = json.loads((directory / "training_metadata.json").read_text(encoding="utf-8"))
    expected = base_identity(config, settings)
    if meta["base_model"] != expected:
        raise ValueError(f"Adapter base mismatch: {meta['base_model']} != {expected}")
    if meta["prompt_template_hash"] != fingerprint(make_prompt("", "")):
        raise ValueError("Adapter was trained with a different prompt template")
    peft_config = PeftConfig.from_pretrained(str(directory), local_files_only=offline)
    if peft_config.base_model_name_or_path not in {
            expected["id"], settings.get("path", settings["id"]), meta["loaded_from"]}:
        raise ValueError("PEFT base_model_name_or_path disagrees with training metadata")
    files = sorted(directory.glob("adapter_model.*"))
    if not files:
        raise ValueError("Adapter has no adapter_model weights")
    files += [directory / "adapter_config.json", directory / "training_metadata.json"]
    return fingerprint({file.name: hash_file(file) for file in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model-size", choices=("8b", "32b"))
    parser.add_argument("--model-path")
    parser.add_argument("--adapter")
    parser.add_argument("--output", default="runs/restored.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = load_config(args.config)
    settings = model_settings(config, args.model_size, args.model_path)
    prompt_hash = fingerprint(make_prompt("", ""))
    print(f"Prompt: PROMPT.md ({prompt_hash[:12]}); output budget: "
          f"{config['max_new_tokens']} tokens", flush=True)
    source = settings.get("path", settings["id"])
    samples = load_samples(args.samples)[:args.limit]
    candidates = {row["id"]: row for row in read_jsonl(args.candidates)}
    adapter_hash = adapter_signature(args.adapter, config, settings, args.offline) if args.adapter else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    old = read_jsonl(output) if output.exists() else []
    latest = {row["id"]: row for row in old}
    completed = {row["fingerprint"]: row for row in latest.values()
                 if row.get("status") == "ok"}
    results, pending = [], []
    for sample in samples:
        candidate = candidates.get(sample["id"])
        hashes = source_hashes(sample)
        if not candidate or candidate.get("status") != "ok":
            raise ValueError(f"{sample['id']}: missing successful OCR candidate")
        if candidate.get("source_hashes") != hashes:
            raise ValueError(f"{sample['id']}: candidate image hashes do not match")
        prompt = make_prompt(candidate["blur_text"], candidate["out_text"])
        key = fingerprint({"id": sample["id"], "images": hashes, "candidate": candidate,
                           "config": config, "model": settings, "prompt": prompt,
                           "adapter": adapter_hash,
                           "implementation": hash_file(__file__)})
        if key in completed:
            results.append(completed[key])
        else:
            pending.append((sample, prompt, key, hashes))
    if pending:
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; no model will be loaded on CPU")
        processor = load_processor(settings, config, args.offline)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            source, revision=settings["revision"], local_files_only=args.offline,
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False,
                                              local_files_only=args.offline)
        model.eval()
        with output.open("a", encoding="utf-8") as stream:
            for sample, prompt, key, hashes in pending:
                row = {"id": sample["id"], "fingerprint": key, "source_hashes": hashes,
                       "model": settings, "adapter_hash": adapter_hash,
                       "prompt_template_hash": prompt_hash}
                batch = generated = tokens = None
                try:
                    batch = encode(processor, image_messages([sample["blur"], sample["out"]], prompt), True)
                    length = batch["input_ids"].shape[1]
                    if length + config["max_new_tokens"] > config["max_seq_length"]:
                        raise ValueError("Input plus output budget exceeds max_seq_length; crop or raise the limit")
                    with torch.inference_mode():
                        generated = model.generate(**batch.to(model.device), do_sample=False,
                                                   max_new_tokens=config["max_new_tokens"])
                    tokens = generated[0, length:]
                    eos = model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else (eos or [])
                    ended = bool(len(tokens) and int(tokens[-1]) in eos)
                    truncated = len(tokens) >= config["max_new_tokens"] and not ended
                    text = processor.decode(tokens, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False).strip()
                    if not text:
                        raise ValueError("Model returned an empty transcription")
                    row.update(text=text,
                               generated_tokens=len(tokens), status="truncated" if truncated else "ok")
                except Exception as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                    batch = generated = tokens = None
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                results.append(row)
                print(f"{row['id']}: {row['status']}", flush=True)
    by_id = {row["id"]: row for row in results}
    sections = [f"## {sample['id']}\n\n[{by_id[sample['id']]['status']}]\n\n"
                + by_id[sample["id"]].get("text", by_id[sample["id"]].get("error", ""))
                for sample in samples]
    output.with_suffix(".md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    if any(row["status"] != "ok" for row in results):
        raise SystemExit("Some regions failed or reached the output limit; see JSONL/Markdown")


if __name__ == "__main__":
    main()
