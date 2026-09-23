"""Read overlapping local Blur/Out crops with existing Qwen weights; no OCR candidates."""
import argparse
import json
from pathlib import Path

from PIL import Image

from common import fingerprint, hash_file, load_config, load_samples, model_settings, read_jsonl, source_hashes
from local_regions import crop_pair, parse_transcription, repeat_tail, review_flags, split_regions
from restore import encode, load_processor


def plan_jobs(samples, config, settings, prompt, core_height, context, scale):
    root = Path(__file__).parent
    implementation = {name: hash_file(root / name) for name in
                      ("restore_local.py", "local_regions.py", "restore.py", "common.py")}
    jobs = []
    for sample in samples:
        hashes = source_hashes(sample)
        with Image.open(sample["out"]) as out, Image.open(sample["blur"]) as blur:
            if out.size != blur.size:
                raise ValueError(f"{sample['id']}: Blur/Out sizes differ")
            regions = split_regions(out, core_height, context)
        for region in regions:
            key = fingerprint({"id": sample["id"], "sources": hashes, "region": region,
                               "scale": scale, "config": config, "model": settings,
                               "prompt": prompt, "implementation": implementation})
            jobs.append({"id": sample["id"], "block_id": f"{sample['id']}:{region['index']:03d}",
                         "fingerprint": key, "source_hashes": hashes, **region})
    return jobs


def save_pages(output, samples, jobs, results):
    """Spatial ownership: each core contributes once, even if its context overlaps."""
    pages, markdown = [], []
    for sample in samples:
        planned = [job for job in jobs if job["id"] == sample["id"]]
        blocks = [results[job["fingerprint"]] for job in planned if job["fingerprint"] in results]
        by_index = {block["index"]: block for block in blocks}
        text = "\n\n".join(by_index[job["index"]].get("text", "[unread block]")
                            if job["index"] in by_index else "[pending block]" for job in planned)
        status = "pending" if len(blocks) < len(planned) else (
            "ok" if all(block["status"] == "ok" for block in blocks) else "review")
        page = {"id": sample["id"], "mode": "local", "status": status, "text": text,
                "blocks_completed": len(blocks), "blocks_total": len(planned),
                "blocks": [{key: block[key] for key in ("block_id", "core_box", "status", "flags")
                            if key in block} for block in blocks]}
        pages.append(page)
        markdown.append(f"## {sample['id']} [{status}]\n\n{text}")
    for path, content in ((output, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pages)),
                          (output.with_suffix(".md"), "\n\n".join(markdown) + "\n")):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model-size", choices=("8b", "32b"))
    parser.add_argument("--model-path")
    parser.add_argument("--output", default="runs/restored_local.jsonl")
    parser.add_argument("--core-height", type=int, default=192)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.core_height < 32 or args.context < 0 or args.scale < 1:
        parser.error("core-height >=32, context >=0 and scale >=1 required")
    output = Path(args.output)
    if output.suffix != ".jsonl":
        parser.error("--output must end in .jsonl")
    # Separate visual budget, unchanged weights/dependencies and generation budget.
    config = {**load_config(args.config), "image_max_pixels": 786432}
    settings = model_settings(config, args.model_size, args.model_path)
    prompt = Path(__file__).with_name("LOCAL_PROMPT.md").read_text(encoding="utf-8-sig").strip()
    samples = load_samples(args.samples)
    jobs = plan_jobs(samples, config, settings, prompt, args.core_height, args.context, args.scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    blocks_path = output.with_suffix(".blocks.jsonl")
    history = read_jsonl(blocks_path) if blocks_path.exists() else []
    latest = {row["block_id"]: row for row in history}
    results = {row["fingerprint"]: row for row in latest.values() if row["status"] in {"ok", "review"}}
    pending = [job for job in jobs if job["fingerprint"] not in results]
    print(f"Local image-only mode: {len(samples)} regions, {len(jobs)} blocks; {len(pending)} pending.\n"
          f"LOCAL_PROMPT.md: {fingerprint(prompt)[:12]}; core={args.core_height}, context={args.context}, "
          f"scale={args.scale}; output budget={config['max_new_tokens']} tokens", flush=True)
    save_pages(output, samples, jobs, results)
    if pending:
        import torch
        from transformers import Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList

        class StopRepeat(StoppingCriteria):
            def __init__(self, prefix):
                self.prefix, self.loop = prefix, False

            def __call__(self, input_ids, scores, **kwargs):
                count = input_ids.shape[1] - self.prefix
                if count >= 64 and count % 32 == 0:
                    self.loop = repeat_tail(input_ids[0, max(self.prefix, input_ids.shape[1] - 512):].tolist())
                return self.loop

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; no model will be loaded on CPU")
        processor = load_processor(settings, config, args.offline)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            settings.get("path", settings["id"]), revision=settings["revision"],
            local_files_only=args.offline, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0}).eval()
        samples_by_id = {sample["id"]: sample for sample in samples}
        with blocks_path.open("a", encoding="utf-8") as stream:
            for number, job in enumerate(pending, 1):
                row = {**job, "model": settings, "prompt_template_hash": fingerprint(prompt)}
                batch = generated = tokens = None
                try:
                    images = crop_pair(samples_by_id[job["id"]], job, args.scale)
                    messages = [{"role": "user", "content": [
                        *({"type": "image", "image": image} for image in images),
                        {"type": "text", "text": prompt}]}]
                    batch = encode(processor, messages, True)
                    length = batch["input_ids"].shape[1]
                    row.update(input_sizes=[list(image.size) for image in images],
                               image_grid_thw=batch["image_grid_thw"].tolist(), input_tokens=length)
                    if length + config["max_new_tokens"] > config["max_seq_length"]:
                        raise ValueError("Input plus output budget exceeds max_seq_length")
                    stop = StopRepeat(length)
                    with torch.inference_mode():
                        generated = model.generate(**batch.to(model.device), do_sample=False,
                            max_new_tokens=config["max_new_tokens"], stopping_criteria=StoppingCriteriaList([stop]))
                    tokens = generated[0, length:]
                    raw = processor.decode(tokens, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False).strip()
                    row.update(raw=raw, generated_tokens=len(tokens))
                    eos = model.generation_config.eos_token_id
                    eos = [eos] if isinstance(eos, int) else (eos or [])
                    ended = bool(len(tokens) and int(tokens[-1]) in eos)
                    if stop.loop:
                        row.update(status="loop")
                    elif len(tokens) >= config["max_new_tokens"] and not ended:
                        row.update(status="truncated")
                    else:
                        try:
                            text = parse_transcription(raw)
                        except (ValueError, TypeError) as error:
                            row.update(status="format_error", error=str(error))
                        else:
                            flags = review_flags(text) if text else ["empty_transcription"]
                            row.update(text=text, flags=flags, status="review" if flags else "ok")
                except Exception as error:
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                    batch = generated = tokens = None
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                finally:
                    batch = generated = tokens = None
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                results[job["fingerprint"]] = row
                save_pages(output, samples, jobs, results)
                print(f"[{number}/{len(pending)}] {job['block_id']}: {row['status']}", flush=True)
    if any(results[job["fingerprint"]]["status"] not in {"ok", "review"} for job in jobs):
        raise SystemExit("Some blocks failed, looped or were truncated; inspect .blocks.jsonl and rerun to retry")
    print(f"Saved {output} and {output.with_suffix('.md')}. 'ok' is not an accuracy score.", flush=True)


if __name__ == "__main__":
    main()
