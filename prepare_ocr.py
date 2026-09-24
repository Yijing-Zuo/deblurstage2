"""Prepare real Out / reference-text crops for the explicit stage2 adaptation split."""

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from common import fingerprint, hash_file, load_samples, read_jsonl, write_jsonl


def page_number(sample):
    """All versions/crops of the same original page stay in the same split."""
    value = sample.get("page")
    if value is None:
        match = re.fullmatch(r"[^_]+_(\d+)(?:_.*)?", sample["id"])
        value = int(match[1]) if match else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Sample requires a positive integer page: {sample['id']}")
    return value


def split_pages(samples, validation_fraction=0.2):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    pages = sorted({page_number(row) for row in samples if row["document_id"] == "4"})
    if len(pages) < 2:
        raise ValueError("Document 4 needs at least two original pages for train / validation")
    count = min(len(pages) - 1, max(1, math.ceil(len(pages) * validation_fraction)))
    validation = set(pages[-count:])
    return {row["id"]: ("evaluation" if row["document_id"] != "4" else
                         "validation" if page_number(row) in validation else "train")
            for row in samples}


def validate_labels(sample, label, image_size):
    """Fail before writing crops when a reference is stale, incomplete, or ambiguous."""
    sample_id = sample["id"]
    if str(label.get("document_id")) != "4":
        raise ValueError(f"Only document 4 can supply adaptation labels: {sample_id}")
    if label.get("out_sha256") != hash_file(sample["out"]):
        raise ValueError(f"Out hash mismatch: {sample_id}")
    if label.get("page_size") != list(image_size):
        raise ValueError(f"Label page_size does not match Out: {sample_id}")
    if label.get("label_status") not in {"geometry_checked", "human_checked"}:
        raise ValueError(f"Labels require geometry_checked or human_checked status: {sample_id}")
    provenance = label.get("provenance", {})
    if not isinstance(provenance, dict) or not provenance.get("method") or not re.fullmatch(
            r"[0-9a-f]{64}", str(provenance.get("source_sha256", ""))):
        raise ValueError(f"Labels require source method and SHA256 provenance: {sample_id}")
    lines = label.get("lines")
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"No complete supervised lines: {sample_id}")
    result, seen = [], set()
    for index, line in enumerate(lines):
        if not isinstance(line, dict) or line.get("complete") is not True:
            raise ValueError(f"Incomplete / partial-word label: {sample_id}:{index}")
        text, box = line.get("text"), line.get("box")
        if (not isinstance(text, str) or not text.strip() or text != text.strip()
                or any(not 32 <= ord(char) <= 126 for char in text)):
            raise ValueError(f"Label must be nonempty printable ASCII, without tabs/newlines: {sample_id}:{index}")
        # Official PPLCNetV4 training pooling emits 40 steps; repeats require blanks.
        if len(text) + sum(a == b for a, b in zip(text, text[1:])) > 40:
            raise ValueError(f"Label exceeds 40 CTC steps; provide shorter word-boundary boxes, "
                             f"never truncate text: {sample_id}:{index}")
        if (not isinstance(box, list) or len(box) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in box)):
            raise ValueError(f"Invalid line box: {sample_id}:{index}")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= image_size[0] and 0 <= y0 < y1 <= image_size[1]):
            raise ValueError(f"Line box outside Out / empty: {sample_id}:{index}")
        crop_box = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
        if min(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]) < 2:
            raise ValueError(f"Line box too small: {sample_id}:{index}")
        if crop_box in seen:
            raise ValueError(f"Duplicate supervised crop: {sample_id}:{index}")
        seen.add(crop_box)
        result.append({"box": list(crop_box), "text": text})
    return result


def prepare(samples_path, labels_path, output, validation_fraction=0.2):
    """Use only doc4 labels; original stage1 splits are retained unchanged."""
    output = Path(output).resolve()
    samples = load_samples(samples_path, views=("out",))
    if not samples:
        raise ValueError("Samples are empty")
    if len({row["deblur_run"] for row in samples}) != 1:
        raise ValueError("Adaptation requires one frozen deblur run")
    roles = split_pages(samples, validation_fraction)
    adaptations = {row["id"]: row for row in samples if row["document_id"] == "4"}
    labels = {}
    for row in read_jsonl(labels_path):
        sample_id = row.get("id")
        if sample_id in labels:
            raise ValueError(f"Duplicate label page: {sample_id}")
        if sample_id not in adaptations:
            raise ValueError(f"Labels must belong only to adaptation document 4: {sample_id}")
        labels[sample_id] = row
    missing = sorted(adaptations.keys() - labels.keys())
    if missing:
        raise ValueError(f"Missing reference labels for adaptation pages: {missing}")

    records, pages, hashes_by_role = [], [], {}
    for row in sorted(samples, key=lambda r: (r["document_id"], page_number(r), r["id"])):
        sample_id, role = row["id"], roles[row["id"]]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample_id):
            raise ValueError(f"Sample id is not safe as a crop filename: {sample_id}")
        out_hash = hash_file(row["out"])
        if out_hash in hashes_by_role and hashes_by_role[out_hash] != role:
            raise ValueError(f"Identical Out occurs across stage2 splits: {sample_id}")
        hashes_by_role[out_hash] = role
        page = {"id": sample_id, "document_id": row["document_id"], "page": page_number(row),
                "stage1_split": row["split"], "stage2_split": role, "out_sha256": out_hash}
        if role != "evaluation":
            label = labels[sample_id]
            with Image.open(row["out"]) as image:
                lines = validate_labels(row, label, image.size)
            page.update(label_status=label["label_status"], provenance=label["provenance"])
            for index, line in enumerate(lines):
                records.append({"id": sample_id, "stage2_split": role,
                                "image": f"images/{sample_id}_{index:04d}.png", **line})
        pages.append(page)

    stamp = fingerprint({"pages": pages, "records": records})
    metadata_path = output / "metadata.json"
    if output.exists() and any(output.iterdir()):
        if not metadata_path.exists() or json.loads(metadata_path.read_text(encoding="utf-8")).get("fingerprint") != stamp:
            raise ValueError("Output already contains a different dataset; choose a new output directory")

    (output / "images").mkdir(parents=True, exist_ok=True)
    records_by_page = {}
    for record in records:
        records_by_page.setdefault(record["id"], []).append(record)
    for sample_id, page_records in records_by_page.items():
        with Image.open(adaptations[sample_id]["out"]) as image:
            for record in page_records:
                image.convert("RGB").crop(record["box"]).save(output / record["image"])
    for role, filename in (("train", "train.txt"), ("validation", "val.txt")):
        (output / filename).write_text("".join(f"{r['image']}\t{r['text']}\n" for r in records
                                              if r["stage2_split"] == role), encoding="utf-8")
    portable = []
    for row in samples:
        row = {**row, "stage2_split": roles[row["id"]]}
        for view in ("out", "blur"):
            if row.get(view):
                path = Path(row[view])
                if not path.is_absolute():
                    path = Path(samples_path).resolve().parent / path
                row[view] = Path(os.path.relpath(path.resolve(), output)).as_posix()
        portable.append(row)
    write_jsonl(output / "stage2_samples.jsonl", portable)
    write_jsonl(output / "lines.jsonl", records)
    metadata = {"schema_version": 3, "fingerprint": stamp, "adaptation_document": "4",
                "training_ctc_steps": 40,
                "evaluation_documents": sorted({r["document_id"] for r in samples if r["document_id"] != "4"}),
                "validation_fraction": validation_fraction,
                "split_rule": "Final ordered original-page block of doc4 is validation; other documents are evaluation",
                "train_pages": [p["id"] for p in pages if p["stage2_split"] == "train"],
                "validation_pages": [p["id"] for p in pages if p["stage2_split"] == "validation"],
                "evaluation_pages": [p["id"] for p in pages if p["stage2_split"] == "evaluation"],
                "counts": dict(Counter(r["stage2_split"] for r in records)),
                "label_status_counts": dict(Counter(p["label_status"] for p in pages if "label_status" in p)),
                "max_text_length": max(len(r["text"]) for r in records),
                "max_aspect_ratio": max((r["box"][2] - r["box"][0]) / (r["box"][3] - r["box"][1]) for r in records),
                "pages": pages}
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--labels", required=True, help="Geometry-checked doc4 line labels; no evaluation references")
    parser.add_argument("--output", default="data/ocr_out")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    result = prepare(args.samples, args.labels, args.output, args.validation_fraction)
    print(f"Prepared {result['counts']}; max label length {result['max_text_length']}. "
          f"Stage1 split preserved; evaluation documents: {result['evaluation_documents']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
