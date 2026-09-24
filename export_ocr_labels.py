"""Export doc4 PDF text onto verified Out coordinates; run locally where references exist."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from common import hash_file, load_samples, write_jsonl


ASCII_REPLACEMENTS = {**dict.fromkeys("\u2018\u2019\u201a\u201b", "'"),
                      **dict.fromkeys("\u201c\u201d\u201e\u201f", '"'),
                      **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2212\u00ad", "-"),
                      "\u2026": "...", "\u00a0": " ", "\ufb00": "ff", "\ufb01": "fi",
                      "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


def normalize_word(text):
    text = "".join(ASCII_REPLACEMENTS.get(c, c) for c in text)
    return text if text and all(32 <= ord(c) <= 126 for c in text) else None


def ctc_steps(text):
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))


def mapped_box(box, pdf_width, registration, crop):
    sx, sy, tx, ty = (registration[k] for k in ("sx", "sy", "tx", "ty"))
    return [sx * (box[0] - .02 * pdf_width) / .96 + tx - crop[0],
            sy * box[1] / .96 + ty - crop[1],
            sx * (box[2] - .02 * pdf_width) / .96 + tx - crop[0],
            sy * box[3] / .96 + ty - crop[1]]


def supervised_phrases(words, pdf_size, reference, clear, audit, target_length=32):
    """Never join across an excluded word or crop a label through a word boundary."""
    width, height = clear.size
    registration, crop = reference["registration_parameters"], reference["original_512x768_crop_box"]
    stamps = reference["stamp_removal"].get("removed_logical_line_bounds_pdf", [])
    grouped, results = defaultdict(list), []
    for word in words:
        grouped[tuple(word[5:7])].append(word)

    def emit(chunk):
        if not chunk:
            return
        text = " ".join(w[0] for w in chunk)
        box = [min(w[1][0] for w in chunk), min(w[1][1] for w in chunk),
               max(w[1][2] for w in chunk), max(w[1][3] for w in chunk)]
        x0, y0, x1, y1 = math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])
        ink = np.asarray(clear.crop((x0, y0, x1, y1)).convert("L")) < 210
        rows = np.flatnonzero(ink.any(axis=1))
        if not len(rows):
            audit.append({"reason": "no_clear_ink", "text": text})
            return
        # Font ascender / descender rectangles are loose; Clear is used only for training crops.
        ink_y0, ink_y1 = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
        final = [max(0, x0 - 1), max(0, ink_y0 - 1), min(width, x1 + 1), min(height, ink_y1 + 1)]
        results.append({"box": final, "text": text, "complete": True})

    for key, line_words in grouped.items():
        line_words.sort(key=lambda w: w[7])
        joined = " ".join(w[4] for w in line_words)
        if any(marker in joined.lower() for marker in ("document shared on", "downloaded by")):
            audit.append({"reason": "watermark_line", "text": joined})
            continue
        chunk = []
        for word in line_words:
            cx, cy = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
            in_stamp = any(l <= cx <= r and pdf_size[1] - t <= cy <= pdf_size[1] - b
                           for l, b, r, t in stamps)
            text, box = normalize_word(word[4]), mapped_box(word[:4], pdf_size[0], registration, crop)
            reason = ("watermark_geometry" if in_stamp else "unsupported_character" if text is None
                      else "outside_or_partial_word" if not (0 <= box[0] < box[2] <= width
                                                               and 0 <= box[1] < box[3] <= height)
                      else "word_exceeds_ctc_capacity" if ctc_steps(text) > 40 else None)
            if reason:
                emit(chunk)
                chunk = []
                audit.append({"reason": reason, "text": word[4], "pdf_line": list(key)})
                continue
            if text != word[4]:
                audit.append({"reason": "normalized_punctuation_or_ligature", "original": word[4], "text": text})
            proposed = " ".join([w[0] for w in chunk] + [text])
            if chunk and (len(proposed) > target_length or ctc_steps(proposed) > 40):
                emit(chunk)
                chunk = []
            chunk.append((text, box))
        emit(chunk)
    return sorted(results, key=lambda line: (line["box"][1], line["box"][0]))


def export(samples_path, references_path, output):
    import fitz

    references_path, output = Path(references_path).resolve(), Path(output).resolve()
    references = json.loads(references_path.read_text(encoding="utf-8-sig"))["records"]
    selected = {(str(r["document_number"]), r["web_page"]): r for r in references
                if str(r["document_number"]) == "4"}
    samples = [r for r in load_samples(samples_path, views=("out",)) if r["document_id"] == "4"]
    if not samples:
        raise ValueError("No document 4 Out samples")
    documents, checked_hashes, labels, audit = {}, {}, [], []
    try:
        for sample in samples:
            reference = selected.get(("4", sample["page"]))
            if not reference or not reference.get("gt_exactly_matches_fresh_pdf_render") or not reference.get(
                    "stamp_removal", {}).get("body_text_preserved") or reference.get("outside_stamp_pixels_changed") != 0:
                raise ValueError(f"Missing verified reference geometry: {sample['id']}")
            crop = reference["original_512x768_crop_box"]
            if sample.get("blur_source", {}).get("box") != crop:
                raise ValueError(f"Sample / reference source crop mismatch: {sample['id']}")
            paths = {}
            for kind, key in (("pdf", "pdf_source"), ("gt", "gt_file")):
                path = Path(reference[key])
                path = path if path.is_absolute() else references_path.parent / path
                if path not in checked_hashes:
                    checked_hashes[path] = hash_file(path)
                if checked_hashes[path] != reference[f"{kind}_sha256"]:
                    raise ValueError(f"Changed {kind} reference: {sample['id']}")
                paths[kind] = path
            with Image.open(paths["gt"]) as source, Image.open(sample["out"]) as out:
                if list(source.size) != reference["size"] or out.size != (crop[2] - crop[0], crop[3] - crop[1]):
                    raise ValueError(f"Reference / Out dimensions mismatch: {sample['id']}")
                clear = source.convert("RGB").crop(crop)
                size = list(out.size)
            if paths["pdf"] not in documents:
                documents[paths["pdf"]] = fitz.open(paths["pdf"])
            page = documents[paths["pdf"]][reference["pdf_physical_page"] - 1]
            page_audit = []
            lines = supervised_phrases(page.get_text("words"), (page.rect.width, page.rect.height),
                                       reference, clear, page_audit)
            if not lines:
                raise ValueError(f"No usable reference phrases: {sample['id']}")
            labels.append({"id": sample["id"], "document_id": "4", "out_sha256": hash_file(sample["out"]),
                           "page_size": size, "label_status": "geometry_checked",
                           "provenance": {"method": "verified_pdf_word_geometry_v1", "source_sha256": reference["pdf_sha256"],
                                          "clear_sha256": reference["gt_sha256"], "pdf_physical_page": reference["pdf_physical_page"],
                                          "registration": reference["registration_parameters"], "source_crop": crop},
                           "lines": lines})
            audit.extend({"id": sample["id"], **event} for event in page_audit)
    finally:
        for document in documents.values():
            document.close()
    write_jsonl(output, labels)
    report = {"status": "geometry_checked; not a claim of human transcript review", "adaptation_document": "4",
              "references_sha256": hash_file(references_path), "pages": len(labels),
              "phrases": sum(len(r["lines"]) for r in labels), "event_counts": dict(Counter(e["reason"] for e in audit)),
              "events": audit}
    output.with_suffix(".audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--references", required=True, help="Verified source PDF / Clear alignment manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = export(args.samples, args.references, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
