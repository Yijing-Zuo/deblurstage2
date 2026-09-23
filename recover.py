"""Recover English line spans from cached CTC evidence with the existing Qwen."""
import argparse
import json
import re
from pathlib import Path

from PIL import Image

from candidates import apply_choices, build_candidates, load_lexicon, score_candidate
from common import (append_record, atomic_write_jsonl, fingerprint, hash_file,
                    load_config, load_samples, model_settings, read_journal, source_hashes)
from ctc import ctc_log_probability, load_evidence
from local_regions import repeat_tail
from restore import encode, load_processor


def parse_reply(raw, plan, allow_proposals=True):
    """Only IDs can select existing text; model prose never becomes a page."""
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value).removesuffix("```").strip()
    data = json.loads(value)
    if not isinstance(data, dict) or set(data) != {"choices", "proposals"}:
        raise ValueError("Expected exactly choices and proposals JSON arrays")
    if not isinstance(data["choices"], list) or not isinstance(data["proposals"], list):
        raise ValueError("choices/proposals must be arrays")
    spans = {span["id"]: span for span in plan["spans"]}
    choices, proposals = {}, {}
    for item in data["choices"]:
        if not isinstance(item, dict) or set(item) != {"span_id", "candidate_id"}:
            raise ValueError("Invalid choice fields")
        sid, cid = item["span_id"], item["candidate_id"]
        if not isinstance(sid, str) or not isinstance(cid, str) or sid not in spans or sid in choices:
            raise ValueError("Unknown or duplicate choice span")
        if cid not in {candidate["id"] for candidate in spans[sid]["candidates"]}:
            raise ValueError("Unknown candidate_id")
        selected = next(candidate for candidate in spans[sid]["candidates"] if candidate["id"] == cid)
        if selected.get("origin") != "original" and selected.get("visual_support") is False:
            raise ValueError("Selected candidate cannot be aligned in any visual source")
        choices[sid] = cid
    if set(choices) != set(spans):
        raise ValueError("Every span must have exactly one choice")
    for item in data["proposals"]:
        if not isinstance(item, dict) or set(item) != {"span_id", "text"}:
            raise ValueError("Invalid proposal fields")
        sid, text = item["span_id"], item["text"]
        if not isinstance(sid, str) or sid not in spans or sid in proposals or not isinstance(text, str):
            raise ValueError("Unknown/duplicate proposal span or non-text proposal")
        proposals[sid] = text
    if proposals and not allow_proposals:
        raise ValueError("The final selection round cannot propose more text")
    return choices, proposals


def prompt_data(plan, sources, context, final=False):
    # Pass only the data needed for this target. Do not copy arbitrary manifest fields.
    spans = []
    for span in plan["spans"]:
        options = []
        for candidate in span["candidates"]:
            if candidate.get("origin") != "original" and candidate.get("visual_support") is False:
                continue
            options.append({key: candidate[key] for key in
                            ("id", "text", "scores", "origin", "edit_cost") if key in candidate})
        spans.append({"id": span["id"], "original": span["original"], "candidates": options})
    return {"round": "final_selection" if final else "selection_and_optional_proposal",
            "baseline": plan["base_text"], "spans": spans,
            "source_readings": [{"name": source["name"], "view": source["view"],
                                 "text": source["restricted_text"]} for source in sources],
            "neighboring_lines_context_only": context}


def add_proposals(plan, proposals, sources, alphabet, max_growth):
    accepted, rejected = [], []
    allowed = set(alphabet)
    for span in plan["spans"]:
        if span["id"] not in proposals:
            continue
        text = proposals[span["id"]]
        reason = None
        if not text or not text.strip() or text != text.strip() or any(char not in allowed for char in text):
            reason = "invalid_characters_or_whitespace"
        elif len(text) > max(2 * len(span["original"]), len(span["original"]) + max_growth):
            reason = "proposal_exceeds_local_span_budget"
        if reason:
            rejected.append({"span_id": span["id"], "text": text, "reason": reason})
            continue
        if text in {candidate["text"] for candidate in span["candidates"]}:
            continue
        try:
            candidate = score_candidate(plan, span, text, sources)
        except ValueError as error:
            rejected.append({"span_id": span["id"], "text": text, "reason": str(error)})
            continue
        candidate.update(id=f"c{len(span['candidates'])}", origin="qwen_proposal")
        if not any(value is not None for value in candidate["scores"].values()):
            rejected.append({"span_id": span["id"], "text": text, "reason": "no_ctc_alignment"})
            continue
        span["candidates"].append(candidate)
        accepted.append({"span_id": span["id"], "candidate_id": candidate["id"]})
    return accepted, rejected


class Qwen:
    def __init__(self, settings, config, offline):
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; Qwen will not be loaded on CPU")
        self.config = config
        self.processor = load_processor(settings, config, offline)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            settings.get("path", settings["id"]), revision=settings["revision"],
            local_files_only=offline, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0}).eval()

    def __call__(self, paths, prompt):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopRepeat(StoppingCriteria):
            def __init__(self, prefix):
                self.prefix, self.loop = prefix, False

            def __call__(self, input_ids, scores, **kwargs):
                count = input_ids.shape[1] - self.prefix
                if count >= 64 and count % 32 == 0:
                    self.loop = repeat_tail(input_ids[0, max(self.prefix, input_ids.shape[1] - 512):].tolist())
                return self.loop

        images = []
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB")
                scale = self.config["image_scale"]
                images.append(image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS))
        messages = [{"role": "user", "content": [
            *({"type": "image", "image": image} for image in images),
            {"type": "text", "text": prompt}]}]
        batch = generated = tokens = None
        try:
            batch = encode(self.processor, messages, True)
            length = batch["input_ids"].shape[1]
            budget = self.config["max_new_tokens"]
            if length + budget > self.config["max_seq_length"]:
                raise ValueError("Target line input plus output budget exceeds max_seq_length")
            stop = StopRepeat(length)
            with torch.inference_mode():
                generated = self.model.generate(**batch.to(self.model.device), do_sample=False,
                    max_new_tokens=budget, stopping_criteria=StoppingCriteriaList([stop]))
            tokens = generated[0, length:]
            raw = self.processor.decode(tokens, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False).strip()
            eos = self.model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else (eos or [])
            ended = bool(len(tokens) and int(tokens[-1]) in eos)
            status = "loop" if stop.loop else ("truncated" if len(tokens) >= budget and not ended else "ok")
            return {"raw": raw, "status": status, "generated_tokens": len(tokens), "input_tokens": length}
        finally:
            batch = generated = tokens = None


def recover_line(line, sources, plan, context, ask, prompt, settings, alphabet):
    replies = []
    result = {"base_text": plan["base_text"], "plan": plan, "replies": replies}
    if not plan["spans"]:
        return {**result, "text": plan["base_text"], "status": "error", "flags": ["no_word_candidates"],
                "error": "No restricted OCR reading; inspect line crops and OCR evidence"}

    def choose(final=False):
        response = ask(prompt + "\n\nEvidence (JSON data):\n" + json.dumps(
            prompt_data(plan, sources, context, final), ensure_ascii=True, allow_nan=False))
        replies.append(response)
        if response["status"] != "ok":
            raise ValueError(f"Qwen generation {response['status']}")
        return parse_reply(response["raw"], plan, allow_proposals=not final)

    try:
        choices, proposals = choose()
        accepted, rejected = add_proposals(plan, proposals, sources, alphabet, settings["proposal_max_growth"])
        flags = ["rejected_proposal"] if rejected else []
        if accepted:
            try:
                choices, _ = choose(final=True)
            except (ValueError, TypeError, KeyError) as error:
                # The first round is already a valid selection. Keep it instead
                # of discarding all useful edits when an optional second round fails.
                flags.append("final_selection_failed")
                result["final_selection_error"] = str(error)
        text = apply_choices(plan, choices)
        if any(char not in set(alphabet) for char in text):
            raise ValueError("Selected line violates the configured alphabet")
        final_scores = {}
        for source in sources:
            score = ctc_log_probability(source["probs"], source["alphabet"], text, source.get("blank_id", 0))
            final_scores[source["name"] + ":" + source["view"]] = score if score != float("-inf") else None
        if not any(value is not None for value in final_scores.values()):
            flags.append("assembled_line_has_no_ctc_alignment")
        # Process completion is distinct from accuracy; every edit retains its evidence.
        result.update(text=text, choices=choices, accepted_proposals=accepted,
                      rejected_proposals=rejected, final_scores=final_scores,
                      status="review" if flags else "ok", flags=flags)
    except (ValueError, TypeError, KeyError) as error:
        result.update(text=plan["base_text"], status="error", flags=["qwen_contract_failed"],
                      error=f"{type(error).__name__}: {error}")
    return result


def evidence_jobs(samples, evidence_path, config, settings, prompt, lexicon_identity):
    pages = {row["id"]: row for row in read_journal(evidence_path)}
    root = Path(evidence_path).resolve().parent
    code_root = Path(__file__).parent
    implementation = {name: hash_file(code_root / name) for name in
                      ("recover.py", "ctc.py", "candidates.py", "common.py", "restore.py", "local_regions.py",
                       "requirements-qwen.txt", "requirements-recovery.txt")}
    jobs, ordered_pages, line_ids = [], [], set()
    for sample in samples:
        page = pages.get(sample["id"])
        if not page or page.get("schema_version") != 2 or page.get("stage") != "ocr" or page.get("status") != "ok":
            raise ValueError(f"{sample['id']}: missing complete v2 OCR evidence; run ocr_lines.py")
        if page.get("source_hashes") != source_hashes(sample):
            raise ValueError(f"{sample['id']}: OCR input image hashes differ")
        if str(page.get("document_id")) != sample["document_id"]:
            raise ValueError(f"{sample['id']}: document identity mismatch")
        if set(page.get("alphabet", "")) != set(config["alphabet"]):
            raise ValueError(f"{sample['id']}: OCR alphabet changed; rerun ocr_lines.py before recovery")
        with Image.open(sample["blur"]) as blur, Image.open(sample["out"]) as out:
            if blur.size != out.size or list(out.size) != page["page_size"]:
                raise ValueError(f"{sample['id']}: page dimensions differ")
        lines = sorted(page["lines"], key=lambda item: item["order"])
        if not lines or len({line["line_id"] for line in lines}) != len(lines):
            raise ValueError(f"{sample['id']}: empty or duplicate line geometry")
        if len({line["order"] for line in lines}) != len(lines):
            raise ValueError(f"{sample['id']}: duplicate line reading order")
        ordered_pages.append({**page, "lines": lines})
        for index, line in enumerate(lines):
            if line["line_id"] in line_ids:
                raise ValueError(f"Duplicate line_id across pages: {line['line_id']}")
            line_ids.add(line["line_id"])
            width, height = page["page_size"]
            x0, y0, x1, y1 = line["box"]
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"Invalid line bounds: {line['line_id']}")
            assets = [line["crops"][view] for view in ("blur", "out")]
            assets += [source["probabilities"] for source in line["sources"]]
            hashes = {path: hash_file(root / path) for path in assets}
            for view in ("blur", "out"):
                if hashes[line["crops"][view]] != line.get("crop_hashes", {}).get(view):
                    raise ValueError(f"{line['line_id']}: crop hash mismatch; rerun ocr_lines.py")
            for source in line["sources"]:
                if hashes[source["probabilities"]] != source.get("probabilities_sha256"):
                    raise ValueError(f"{line['line_id']}: CTC cache hash mismatch; rerun ocr_lines.py")
            # Hash contents, not only filenames: changed NPZ/crops invalidate cached choices.
            radius = config["recovery"]["context_lines"]
            context = [{"line_id": other["line_id"], "readings": [source["restricted_text"] for source in other["sources"]]}
                       for other in lines[max(0, index - radius):index + radius + 1]
                       if other["line_id"] != line["line_id"] and other["column"] == line["column"]]
            key = fingerprint({"line": line, "images": page["source_hashes"], "assets": hashes,
                               "context": context, "config": {key: config[key] for key in
                               ("alphabet", "candidates", "recovery")}, "model": settings,
                               "prompt": prompt, "lexicon": lexicon_identity, "code": implementation})
            jobs.append({"id": page["id"], "line": line, "fingerprint": key, "context": context})
    return ordered_pages, jobs


def save_pages(output, pages, jobs, results, model):
    by_line = {job["line"]["line_id"]: results.get(job["fingerprint"]) for job in jobs}
    rows = []
    for page in pages:
        lines = []
        for line in page["lines"]:
            record = by_line[line["line_id"]]
            lines.append({key: line[key] for key in ("line_id", "box", "polygon", "column", "order")})
            lines[-1].update(text=record.get("text", "") if record else "", status=record["status"] if record else "pending",
                             fingerprint=record["fingerprint"] if record else None,
                             flags=record.get("flags", []) if record else [])
        statuses = {line["status"] for line in lines}
        status = "pending" if "pending" in statuses else ("error" if "error" in statuses else
                 ("review" if "review" in statuses else "ok"))
        flags = ["uncovered_ink_bands"] if any(
            view.get("uncovered_ink_bands") for view in page.get("coverage", {}).values()) else []
        if flags and status == "ok":
            status = "review"
        rows.append({"schema_version": 2, "stage": "recovery", "id": page["id"],
                     "document_id": page["document_id"], "source_hashes": page["source_hashes"],
                     "page_size": page["page_size"], "status": status, "lines": lines,
                     "model": model, "text": "\n".join(line["text"] for line in lines),
                     "ocr_fingerprint": page["fingerprint"], "coverage": page.get("coverage", {}), "flags": flags})
    atomic_write_jsonl(output, rows)
    return rows


def cached_lines(history, retry_review=False):
    latest = {row["fingerprint"]: row for row in history}
    reusable = {"ok"} if retry_review else {"ok", "review"}
    return {key: row for key, row in latest.items() if row.get("status") in reusable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", default="runs/v2/recovery.jsonl")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model-size", choices=("8b", "32b"))
    parser.add_argument("--model-path")
    parser.add_argument("--lexicon", help="Optional frequency dictionary: word count per line")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--retry-review", action="store_true", help="Also redo completed review records")
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix != ".jsonl":
        parser.error("--output must end in .jsonl")
    config = load_config(args.config)
    v2 = config["v2"]
    params = v2["recovery"]
    for key in ("image_scale", "max_new_tokens", "max_seq_length"):
        if not isinstance(params[key], int) or params[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if params["context_lines"] < 0 or params["proposal_max_growth"] < 0:
        parser.error("context_lines/proposal_max_growth must be nonnegative")
    settings = model_settings(config, args.model_size, args.model_path)
    lexicon = load_lexicon(args.lexicon)
    # The candidate module exposes the source hash so different dictionaries never share cache.
    lexicon_identity = getattr(lexicon, "identity", None)
    if lexicon_identity is None:
        raise ValueError("Lexicon must expose a deterministic identity")
    prompt = Path(__file__).with_name("RECOVERY_PROMPT.md").read_text(encoding="utf-8").strip()
    samples = load_samples(args.samples)
    pages, jobs = evidence_jobs(samples, args.evidence, v2, settings, prompt, lexicon_identity)
    journal = output.with_suffix(".lines.jsonl")
    history = read_journal(journal)
    results = cached_lines(history, args.retry_review)
    pending = [job for job in jobs if job["fingerprint"] not in results]
    print(f"v2: {len(pages)} regions, {len(jobs)} real lines, {len(pending)} pending; Qwen {settings['size']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(journal, history)  # Remove any interrupted tail before appending.
    save_pages(output, pages, jobs, results, settings)
    model = None
    root = Path(args.evidence).resolve().parent
    with journal.open("a", encoding="utf-8", newline="\n") as stream:
        for number, job in enumerate(pending, 1):
            line = job["line"]
            record = {"schema_version": 2, "stage": "line_recovery", "id": job["id"],
                      "line_id": line["line_id"], "fingerprint": job["fingerprint"], "model": settings}
            try:
                sources = [{**source, **load_evidence(root / source["probabilities"])} for source in line["sources"]]
                plan = build_candidates(sources, {**v2["candidates"], "alphabet": v2["alphabet"]}, lexicon)
                record["text"] = plan["base_text"]
                if model is None and plan["spans"]:
                    try:
                        model = Qwen(settings, {**config, **params}, args.offline)
                    except Exception as error:
                        raise SystemExit(f"Qwen setup failed: {type(error).__name__}: {error}") from error
                ask = lambda text: model([root / line["crops"][view] for view in ("blur", "out")], text)
                record.update(recover_line(line, sources, plan, job["context"], ask, prompt, params, v2["alphabet"]))
                if line.get("geometry_warning"):
                    record["flags"].append(line["geometry_warning"])
                if any(source.get("width_compressed") for source in sources):
                    record["flags"].append("recognizer_width_compressed")
                if record["flags"] and record["status"] == "ok":
                    record["status"] = "review"
            except Exception as error:
                record.update(status="error", error=f"{type(error).__name__}: {error}", flags=["processing_failed"])
                if type(error).__name__ == "OutOfMemoryError":
                    import torch
                    torch.cuda.empty_cache()
            append_record(stream, record)
            results[job["fingerprint"]] = record
            save_pages(output, pages, jobs, results, settings)
            print(f"[{number}/{len(pending)}] {line['line_id']}: {record['status']}" +
                  (f" ({record['error']})" if "error" in record else ""), flush=True)
    rows = save_pages(output, pages, jobs, results, settings)
    if any(row["status"] in {"error", "pending"} for row in rows):
        raise SystemExit("Some lines failed; see .lines.jsonl. Repeat the same command to retry only failures.")
    print(f"Saved {output}. 'ok' means valid processing, not verified transcription accuracy.", flush=True)


if __name__ == "__main__":
    main()
