"""Read deblur Out with PaddleOCR; retain the legacy dual-view v2 explicitly."""

import argparse
import json
import os
import string
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from PIL import Image

from common import (atomic_write_jsonl, fingerprint, hash_file, load_config,
                    load_samples, read_journal, source_hashes)
from ctc import greedy_decode, prefix_beam_search
from paddle_ctc import DEFAULT_DETECTOR, DEFAULT_RECOGNIZERS, PaddleCTC, resolve_models


def _overlap(a, b, axis):
    return max(0, min(a[axis + 2], b[axis + 2]) - max(a[axis], b[axis]))


def merge_detections(detections, page_size):
    """Union matching detections, then join nearby same-baseline fragments.

    Geometry is deliberately a shared axis-aligned crop, not a guessed deskew.
    Original polygons are retained so skew and suspicious merges remain visible.
    """
    width, height = page_size
    lines = []
    for detection in detections:
        points = np.asarray(detection["polygon"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            continue
        x0, y0 = np.maximum(points.min(axis=0), [0, 0])
        x1, y1 = np.minimum(points.max(axis=0), [width, height])
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        lines.append({"box": [float(x0), float(y0), float(x1), float(y1)],
                      "detections": [detection]})
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            a = lines[i]["box"]
            for j in range(i + 1, len(lines)):
                b = lines[j]["box"]
                ah, bh = a[3] - a[1], b[3] - b[1]
                same_row = (_overlap(a, b, 1) >= 0.65 * min(ah, bh)
                            and abs((a[1] + a[3]) - (b[1] + b[3])) / 2 < 0.45 * max(ah, bh)
                            and max(ah, bh) < 1.8 * min(ah, bh))
                x_overlap = _overlap(a, b, 0)
                duplicate = x_overlap >= 0.6 * min(a[2] - a[0], b[2] - b[0])
                gap = max(a[0], b[0]) - min(a[2], b[2])
                fragment = -0.1 * min(a[2] - a[0], b[2] - b[0]) <= gap <= 0.6 * min(ah, bh)
                if same_row and (duplicate or fragment):
                    lines[i]["box"] = [min(a[0], b[0]), min(a[1], b[1]),
                                       max(a[2], b[2]), max(a[3], b[3])]
                    lines[i]["detections"].extend(lines[j]["detections"])
                    lines.pop(j)
                    changed = True
                    break
            if changed:
                break
    return lines


def _gaps(intervals):
    intervals = sorted(intervals)
    if not intervals:
        return []
    right, gaps = intervals[0][1], []
    for left, end in intervals[1:]:
        if left > right:
            gaps.append((right, left))
        right = max(right, end)
    return gaps


def reading_order(lines):
    """Read columns top-to-bottom, split into sections around spanning headings."""
    columns = {}

    def ordered(group, path=()):
        if not group:
            return []
        left = min(x["box"][0] for x in group)
        right = max(x["box"][2] for x in group)
        typical_height = float(np.median([x["box"][3] - x["box"][1] for x in group]))
        narrow = [x for x in group if x["box"][2] - x["box"][0] < 0.72 * (right - left)]
        gaps = [g for g in _gaps([(x["box"][0], x["box"][2]) for x in narrow])
                if g[1] - g[0] >= max(6, typical_height)]
        if gaps:
            start, end = max(gaps, key=lambda x: x[1] - x[0])
            cut = (start + end) / 2
            lft = [x for x in group if x["box"][2] <= cut]
            rgt = [x for x in group if x["box"][0] >= cut]
            spans = [x for x in group if x not in lft and x not in rgt]
            # Require evidence for two columns, not one short end-of-line word.
            if len(lft) >= 2 and len(rgt) >= 2:
                if not spans:
                    return ordered(lft, path + (0,)) + ordered(rgt, path + (1,))
                result, remaining = [], lft + rgt
                for span in sorted(spans, key=lambda x: x["box"][1]):
                    before = [x for x in remaining if sum(x["box"][1::2]) / 2 < sum(span["box"][1::2]) / 2]
                    remaining = [x for x in remaining if x not in before]
                    result.extend(ordered(before, path))
                    result.extend(ordered([span], path))
                return result + ordered(remaining, path)
        column = columns.setdefault(path, len(columns))
        return [{**x, "column": column} for x in sorted(group, key=lambda x: (x["box"][1], x["box"][0]))]

    return [{**line, "order": i} for i, line in enumerate(ordered(lines))]


def _bands(active):
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def split_multiline_boxes(lines, images):
    """Split unusually tall boxes only where both views show a real blank row gap."""
    if not lines:
        return []
    typical = float(np.median([x["box"][3] - x["box"][1] for x in lines]))
    result = []
    for line in lines:
        x0, y0, x1, y1 = [int(round(x)) for x in line["box"]]
        if y1 - y0 < max(12, 1.8 * typical):
            result.append(line)
            continue
        profiles = []
        for image in images:
            pixels = np.asarray(image.convert("L"))[y0:y1, x0:x1].astype(float)
            contrast = np.percentile(pixels, 95) - pixels
            profiles.append((contrast > max(12, np.percentile(contrast, 90) * 0.3)).mean(axis=1))
        active = np.maximum.reduce(profiles) > 0.06
        bands = [(a, b) for a, b in _bands(active) if b - a >= max(3, 0.35 * typical)]
        if (len(bands) > 1 and all(b - a <= 1.65 * typical for a, b in bands)
                and all(bands[i + 1][0] - bands[i][1] >= 2 for i in range(len(bands) - 1))):
            for a, b in bands:
                result.append({**line, "box": [x0, y0 + a, x1, y0 + b],
                               "split_from_multiline": True})
        else:
            result.append({**line, "geometry_warning": "tall_box_may_contain_multiple_lines"})
    return result


def out_geometry(detections, image):
    """Split, tighten vertical padding, then suppress duplicate Out line fragments.

    Never expand a partial line from neighboring text or a Clear reference.
    Suspicious uncovered regions remain explicit diagnostics.
    """
    lines = split_multiline_boxes(merge_detections(detections, image.size), [image])
    gray = np.asarray(image.convert("L"), dtype=float)
    for line in lines:
        if line.get("geometry_warning"):
            continue
        x0, y0, x1, y1 = crop_geometry(line["box"], image.size, 0)[0]
        pixels = gray[y0:y1, x0:x1]
        contrast = np.percentile(pixels, 95) - pixels
        ink = contrast > max(12, np.percentile(contrast, 90) * 0.3)
        active = np.flatnonzero(ink.sum(axis=1) >= max(2, pixels.shape[1] * 0.008))
        if len(active) and active[-1] - active[0] >= max(3, 0.4 * (y1 - y0)):
            line["before_vertical_trim"] = line["box"]
            line["box"] = [x0, y0 + int(active[0]), x1, y0 + int(active[-1]) + 1]
    kept, suppressed = [], []
    # Wider complete rows take precedence over same-baseline snippets.
    for line in sorted(lines, key=lambda x: (bool(x.get("geometry_warning")),
                                            -(x["box"][2] - x["box"][0]))):
        a = line["box"]
        ah = a[3] - a[1]
        duplicate = next((other for other in kept if
            _overlap(a, other["box"], 0) >= 0.9 * (a[2] - a[0])
            and _overlap(a, other["box"], 1) >= 0.8 * min(ah, other["box"][3] - other["box"][1])
            and abs(sum(a[1::2]) - sum(other["box"][1::2])) < 0.6 * ah
            and max(ah, other["box"][3] - other["box"][1]) < 1.8 * min(ah, other["box"][3] - other["box"][1])), None)
        if duplicate:
            suppressed.append({"box": a, "reason": "same_line_contained_fragment"})
            duplicate["detections"].extend(line["detections"])
        else:
            kept.append(line)
    return reading_order(kept), suppressed


def crop_geometry(box, page_size, padding=1):
    """Integer crop and its exact crop-to-page translation, shared between views."""
    width, height = page_size
    x0, y0 = np.maximum(np.floor(box[:2]).astype(int) - padding, [0, 0])
    x1, y1 = np.minimum(np.ceil(box[2:]).astype(int) + padding, [width, height])
    rect = [int(x0), int(y0), int(x1), int(y1)]
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Empty line crop")
    polygon = [[int(x0), int(y0)], [int(x1), int(y0)], [int(x1), int(y1)], [int(x0), int(y1)]]
    return rect, polygon, [[1, 0, int(x0)], [0, 1, int(y0)], [0, 0, 1]]


def coverage_diagnostics(images, lines, views=("blur", "out")):
    """Report visible ink outside boxes; flags are diagnostics, not invented lines."""
    width, height = images[0].size
    covered = np.zeros((height, width), dtype=bool)
    for line in lines:
        x0, y0, x1, y1 = crop_geometry(line["box"], (width, height))[0]
        covered[y0:y1, x0:x1] = True
    diagnostics = {}
    for view, image in zip(views, images):
        gray = np.asarray(image.convert("L"), dtype=float)
        ink = np.percentile(gray, 95) - gray > max(15, (np.percentile(gray, 95) - np.percentile(gray, 10)) * 0.3)
        uncovered = ink & ~covered
        diagnostics[view] = {
            "uncovered_ink_fraction": float(uncovered.sum() / max(1, ink.sum())),
            "uncovered_ink_bands": [[int(a), int(b)] for a, b in _bands(uncovered.sum(axis=1) > max(3, width * 0.03))
                                    if b - a >= 2],
        }
    return diagnostics


def _save_npz(path, values):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **values)
    temporary.replace(path)


def cache_valid(row, output):
    if (row.get("schema_version") not in (2, 3) or row.get("stage") != "ocr"
            or row.get("status") not in ("ok", "review")):
        return False
    try:
        for line in row["lines"]:
            for view, path in line["crops"].items():
                if hash_file(output.parent / path) != line["crop_hashes"][view]:
                    return False
            for source in line["sources"]:
                if hash_file(output.parent / source["probabilities"]) != source["probabilities_sha256"]:
                    return False
        return bool(row["lines"])
    except (OSError, KeyError):
        return False


def process_page(row, frontend, output, page_fingerprint, settings, cached_detections=None):
    views = settings.get("input_views", ("blur", "out"))
    out_only = list(views) == ["out"]
    images = []
    for view in views:
        with Image.open(row[view]) as image:
            images.append(image.convert("RGB"))
    if any(image.size != images[0].size for image in images):
        raise ValueError("Aligned Blur and Out must have the same dimensions")
    detections = cached_detections if cached_detections is not None else [
        dict(detection, view=view) for view in views for detection in frontend.detect(row[view])]
    suppressed = []
    if out_only:
        lines, suppressed = out_geometry(detections, images[0])
    else:
        lines = reading_order(split_multiline_boxes(merge_detections(detections, images[0].size), images))
    if not lines:
        raise ValueError("No lines detected; this page must not silently become empty text")
    coverage = coverage_diagnostics(images, lines, views)
    directory = output.parent / (output.stem + ".assets") / fingerprint(row["id"])[:16] / page_fingerprint[:16]
    directory.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(lines):
        line["line_id"] = f"{row['id']}:l{i:04d}"
        line["detected_box"] = line["box"]
        box, polygon, transform = crop_geometry(line["box"], images[0].size, settings.get("crop_padding", 1))
        line.update(box=box, polygon=polygon, crop_to_page=transform,
                    crop_method="out_axis_aligned" if out_only else "shared_axis_aligned",
                    crops={}, crop_hashes={}, sources=[])
        for view, image in zip(views, images):
            path = directory / f"l{i:04d}_{view}.png"
            temporary = path.with_suffix(".png.tmp")
            image.crop(box).save(temporary, format="PNG")
            temporary.replace(path)
            line["crops"][view] = path.relative_to(output.parent).as_posix()
            line["crop_hashes"][view] = hash_file(path)
            for source, values in frontend.recognize(path):
                npz = directory / f"l{i:04d}_{view}_{source['name']}.npz"
                _save_npz(npz, values)
                line["sources"].append({**source, "view": view,
                    "probabilities": npz.relative_to(output.parent).as_posix(),
                    "probabilities_sha256": hash_file(npz)})
                if out_only and len(line["sources"]) == 1:
                    text = greedy_decode(values["probs"], values["alphabet"])
                    if settings.get("decode", "greedy") == "beam":
                        beams = prefix_beam_search(values["probs"], values["alphabet"],
                                                   beam_width=8, top_k=1)
                        text = beams[0]["text"] if beams else text
                    line.update(text=text, raw_text=source.get("raw_text", ""),
                                status="review" if not text.strip() or line.get("geometry_warning") else "ok")
        if out_only and "text" not in line:
            raise ValueError("Recognizer returned no CTC evidence")
    return {"page_size": list(images[0].size), "lines": lines, "coverage": coverage,
            "detections": detections, "suppressed_boxes": suppressed,
            "detected_boxes": len(detections),
            "status": "review" if out_only and (any(x["status"] == "review" for x in lines)
                or coverage["out"]["uncovered_ink_bands"]) else "ok"}


def cached_out_detections(page, out_hash, detector, detection_settings):
    """Reuse detector geometry only; never relabel old probabilities as new weights."""
    if page.get("source_hashes", {}).get("out") != out_hash:
        raise ValueError("Detection cache belongs to a different Out image")
    if (not page.get("models") or page["models"][0] != detector
            or page.get("configuration", {}).get("settings", {}).get("detection", {}) != detection_settings):
        raise ValueError("Detection cache uses a different detector or detection configuration")
    originals = page.get("detections")
    if originals is None:
        originals = [d for line in page.get("lines", []) for d in line.get("detections", [])]
    unique = {fingerprint(d): d for d in originals if d.get("view") == "out"}
    if not unique:
        raise ValueError("No original Out detections in the supplied cache")
    return list(unique.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--samples")
    parser.add_argument("--output", default="runs/v3/predictions.jsonl")
    parser.add_argument("--pipeline", choices=("v3", "v2"), default="v3")
    parser.add_argument("--model-dir", help="Officially exported v3 recognition model with stage2_training.json")
    parser.add_argument("--documents", nargs="+", help="Only these document IDs, e.g. 14")
    parser.add_argument("--decode", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--detections", help="Optional existing evidence JSONL; reuse compatible Out detection polygons")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--device", help="gpu:0 (default) or cpu")
    args = parser.parse_args(argv)
    config = load_config(args.config).get(args.pipeline, {})
    settings = dict(config.get("ocr", {}))
    out_only = args.pipeline == "v3"
    views = ("out",) if out_only else ("blur", "out")
    if not out_only and (args.model_dir or args.detections):
        parser.error("--model-dir/--detections belong to v3")
    settings.update(input_views=list(views), decode=args.decode)
    recognizers = settings.get("recognizers", DEFAULT_RECOGNIZERS[:1] if out_only else DEFAULT_RECOGNIZERS)
    training = None
    if args.model_dir:
        provenance = Path(args.model_dir) / "stage2_training.json"
        training = json.loads(provenance.read_text(encoding="utf-8"))
        if training.get("schema_version") != 3 or "dataset" not in training:
            raise ValueError("Missing v3 training provenance; use train_ocr.py export")
        recognizers = [{"name": DEFAULT_RECOGNIZERS[0]["name"], "path": args.model_dir,
                        "training_provenance_sha256": hash_file(provenance)}]
    if out_only and len(recognizers) != 1:
        parser.error("v3 uses one recognizer per experiment; compare recognizers in separate runs")
    device = args.device or settings.get("device", "gpu:0")
    allowed = config.get("alphabet", string.ascii_letters + string.digits + string.punctuation + " ")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    models = resolve_models([settings.get("detector", DEFAULT_DETECTOR)]
                            + recognizers, args.offline)
    if training:
        validate_export_provenance(training, models[1])
    model_metadata = [{k: v for k, v in model.items() if k != "path"} for model in models]
    if args.download_only:
        for model in models:
            print(f"{model['name']}: {model['path']}", flush=True)
        return 0
    if not args.samples:
        parser.error("--samples is required unless --download-only is set")
    output = Path(args.output).resolve()
    rows = load_samples(args.samples, views=views)
    if args.documents:
        missing = set(args.documents) - {row["document_id"] for row in rows}
        if missing:
            parser.error(f"Unknown documents: {sorted(missing)}")
        rows = [row for row in rows if row["document_id"] in args.documents]
    if not rows:
        parser.error("No samples selected")
    if training:
        validate_training_roles(rows, training["dataset"])
    detections = {r["id"]: r for r in read_journal(args.detections)} if args.detections else {}
    previous = {r["id"]: r for r in read_journal(output)} if output.exists() else {}
    code = {name: hash_file(Path(__file__).with_name(name)) for name in ("ocr_lines.py", "paddle_ctc.py", "ctc.py")}
    runtime = {}
    for package in ("paddlex", "paddlepaddle-gpu", "paddlepaddle", "numpy", "Pillow", "opencv-contrib-python"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            pass
    schema = 3 if out_only else 2
    stage_key = {"schema_version": schema, "models": model_metadata, "settings": settings,
                 "alphabet": allowed, "code": code, "runtime": runtime, "device": device}
    frontend, results, failures = None, [], 0
    for row in rows:
        hashes = source_hashes(row, views)
        identity = {key: row[key] for key in ("id", "document_id", "split", "deblur_run")}
        if out_only:
            identity["stage2_split"] = row.get("stage2_split", "unassigned")
        reused = None
        if args.detections:
            if row["id"] not in detections:
                raise ValueError(f"Detection cache is missing {row['id']}")
            reused = cached_out_detections(detections[row["id"]], hashes["out"],
                                          model_metadata[0], settings.get("detection", {}))
        key = fingerprint({**stage_key, "sample": identity, "source_hashes": hashes, "detections": reused})
        old = previous.get(row["id"], {})
        if old.get("fingerprint") == key and cache_valid(old, output):
            results.append(old)
            print(f"{row['id']}: cached ({len(old['lines'])} lines)", flush=True)
            continue
        if frontend is None:
            # Model/setup failures are fatal; don't emit one identical error per page.
            frontend = PaddleCTC(models, allowed, device,
                                 settings.get("detection", {}))
        result = {**identity, "schema_version": schema, "stage": "ocr", "fingerprint": key,
                  "source_hashes": hashes, "models": model_metadata, "alphabet": allowed,
                  "configuration": stage_key}
        try:
            # Keep valid source geometry even if detection/recognition fails, so
            # --allow-incomplete can render an explicit failed-page diagnostic.
            with Image.open(row["out"]) as image:
                result["page_size"] = list(image.size)
            result.update(process_page(row, frontend, output, key, settings, reused))
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}", lines=[])
            failures += 1
        results.append(result)
        # Preserve not-yet-visited valid records when interrupted during a rerun.
        processed = {r["id"] for r in results}
        pending_old = [previous[r["id"]] for r in rows if r["id"] in previous and r["id"] not in processed]
        atomic_write_jsonl(output, results + pending_old)
        print(f"{row['id']}: {result['status']} ({len(result['lines'])} lines)"
              + (f" - {result['error']}" if result.get("error") else ""), flush=True)
    atomic_write_jsonl(output, results)
    return 1 if failures else 0


def validate_export_provenance(training, model):
    """A training record must describe these exported files, not an older model."""
    identity = training.get("training")
    if not isinstance(identity, dict) or fingerprint(identity) != training.get("training_fingerprint"):
        raise ValueError("Training provenance was changed; export again from the training run")
    if not training.get("export_hashes") or training["export_hashes"] != model["file_hashes"]:
        raise ValueError("Exported model files do not match their training provenance")


def validate_training_roles(rows, dataset):
    """Training images must never be advertised as evaluation after export."""
    role_by_id = {sample_id: role for role, key in (
        ("train", "train_pages"), ("validation", "validation_pages"), ("evaluation", "evaluation_pages"))
        for sample_id in dataset[key]}
    page_hashes = {p["id"]: p["out_sha256"] for p in dataset["pages"]}
    for row in rows:
        role = role_by_id.get(row["id"])
        if role is None or role != row.get("stage2_split"):
            raise ValueError(f"{row['id']}: use the stage2 manifest belonging to this trained model")
        if hash_file(row["out"]) != page_hashes[row["id"]]:
            raise ValueError(f"{row['id']}: Out changed since the training split was prepared")


if __name__ == "__main__":
    raise SystemExit(main())
