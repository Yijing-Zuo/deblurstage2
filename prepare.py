"""Prepare aligned image pairs, or export reviewed labels for post-OCR LoRA."""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from common import (fingerprint, hash_file, load_samples, make_prompt, read_jsonl, source_hashes,
                    write_jsonl)


def crop_image(source, box, destination):
    with Image.open(source) as image:
        if box is not None:
            if (len(box) != 4 or any(type(v) is not int for v in box)
                    or not 0 <= box[0] < box[2] <= image.width
                    or not 0 <= box[1] < box[3] <= image.height):
                raise ValueError(f"Invalid crop {box} for {source} ({image.size})")
            image = image.crop(box)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(destination)
        return image.size


def import_pairs(args):
    """Use audited crop coordinates, never resize a full page to an Out crop."""
    manifest, output = Path(args.pairs).resolve(), Path(args.output).resolve()
    records = json.loads(manifest.read_text(encoding="utf-8-sig"))["records"]
    rows = []
    for record in records[:args.limit]:
        doc, page = record["document_number"], record["web_page"]
        sample_id = f"{doc}_{page:03d}"
        original = manifest.parent / record["blur_file"]
        if hash_file(original) != record["blur_sha256"]:
            raise ValueError(f"Audited Blur changed: {original}")
        out = Path(args.out_dir).resolve() / f"Out_{sample_id}.png"
        cropped = output.parent / "images" / f"Blur_{sample_id}.png"
        size = crop_image(original, record["original_512x768_crop_box"], cropped)
        with Image.open(out) as image:
            if image.size != size:
                raise ValueError(f"Blur/Out size mismatch for {sample_id}: {size}/{image.size}")
        stored_out = output.parent / "images" / out.name
        if out != stored_out:
            shutil.copy2(out, stored_out)
        rows.append({"id": sample_id, "document_id": str(doc),
                     "document_uid": record["docsity_id"], "page": page,
                     "deblur_run": args.deblur_run, "split": "test",
                     "blur": str(cropped), "out": str(stored_out),
                     "blur_source": {"path": str(original),
                                     "box": record["original_512x768_crop_box"]}})
    return rows


def import_manifest(args):
    rows = load_samples(args.manifest)[:args.limit]
    destination = Path(args.output).resolve().parent / "images"
    for row in rows:
        sizes = []
        for view in ("blur", "out"):
            box = row.pop(f"{view}_box", None)
            if box is not None:
                original = row[view]
                stamp = fingerprint([hash_file(original), box])[:16]
                cropped = destination / f"{stamp}_{view}.png"
                sizes.append(crop_image(original, box, cropped))
                row[view] = str(cropped)
                row[f"{view}_source"] = {"path": original, "box": box}
            else:
                with Image.open(row[view]) as image:
                    sizes.append(image.size)
        if sizes[0] != sizes[1]:
            raise ValueError(f"Paired images must cover the same region: {row['id']} {sizes}")
    return rows


def export_training(args):
    rows = load_samples(args.samples)
    runs = {r["deblur_run"] for r in rows if r["split"] in {"train", "validation"}}
    if len(runs) > 1:
        raise ValueError("Training and validation must use one frozen deblur run")
    labels = read_jsonl(args.labels)
    if len({r["id"] for r in labels}) != len(labels):
        raise ValueError("Duplicate label ids")
    labels = {r["id"]: r for r in labels}
    # Last attempt wins. An error following a success must not silently reuse it.
    candidates = {r["id"]: r for r in read_jsonl(args.candidates)}
    batches = {"train": [], "validation": []}
    for row in rows:
        if row["split"] not in batches:
            continue
        label, candidate = labels.get(row["id"], {}), candidates.get(row["id"], {})
        if label.get("verified") is not True or not label.get("text", "").strip():
            raise ValueError(f"Missing reviewed, nonempty ROI label: {row['id']}")
        hashes = source_hashes(row)
        if candidate.get("status") != "ok" or candidate.get("source_hashes") != hashes:
            raise ValueError(f"Missing, failed or stale OCR candidates: {row['id']}")
        prompt = make_prompt(candidate["blur_text"], candidate["out_text"])
        batches[row["split"]].append({
            "id": row["id"], "document_id": row["document_id"],
            "deblur_run": row["deblur_run"], "split": row["split"],
            "image": [row["blur"], row["out"]], "source_hashes": hashes,
            "blur_text": candidate["blur_text"], "out_text": candidate["out_text"],
            "conversations": [{"from": "human", "value": "<image>\n<image>\n" + prompt},
                              {"from": "gpt", "value": label["text"]}]
        })
    if not batches["train"]:
        raise ValueError("No training rows. Export real Out for training documents first.")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    for split, filename in (("train", "train.json"), ("validation", "eval.json")):
        (destination / filename).write_text(
            json.dumps(batches[split], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(batches['train'])} training / {len(batches['validation'])} validation rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pairs", help="Existing audited verified_4_14.json")
    mode.add_argument("--manifest", help="Paired JSONL, optionally with blur_box/out_box")
    mode.add_argument("--training", action="store_true")
    parser.add_argument("--out-dir", help="Existing directory containing Out_N_PPP.png")
    parser.add_argument("--deblur-run", default="docsity_20260915")
    parser.add_argument("--samples", default="data/samples.jsonl")
    parser.add_argument("--candidates", default="runs/candidates.jsonl")
    parser.add_argument("--labels", default="data/labels.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.training:
        if args.limit is not None:
            parser.error("--limit is for importing images, not training exports")
        export_training(args)
    else:
        if args.pairs and not args.out_dir:
            parser.error("--pairs requires --out-dir")
        rows = import_pairs(args) if args.pairs else import_manifest(args)
        # Store relative paths to allow transferring data/ and its images together.
        root = Path(args.output).resolve().parent
        for row in rows:
            for view in ("blur", "out"):
                image = Path(row[view])
                if image.is_relative_to(root):
                    row[view] = image.relative_to(root).as_posix()
        write_jsonl(args.output, rows)
        load_samples(args.output)
        print(f"Prepared {len(rows)} aligned samples: {args.output}")


if __name__ == "__main__":
    main()
