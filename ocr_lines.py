"""Detect shared Blur/Out lines and cache original CTC evidence for recovery."""

import argparse
import os
import string
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from PIL import Image

from common import (atomic_write_jsonl, fingerprint, hash_file, load_config,
                    load_samples, read_journal, source_hashes)
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
        active = np.maximum(*profiles) > 0.06
        bands = [(a, b) for a, b in _bands(active) if b - a >= max(3, 0.35 * typical)]
        if (len(bands) > 1 and all(b - a <= 1.65 * typical for a, b in bands)
                and all(bands[i + 1][0] - bands[i][1] >= 2 for i in range(len(bands) - 1))):
            for a, b in bands:
                result.append({**line, "box": [x0, y0 + a, x1, y0 + b],
                               "split_from_multiline": True})
        else:
            result.append({**line, "geometry_warning": "tall_box_may_contain_multiple_lines"})
    return result


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


def coverage_diagnostics(images, lines):
    """Report visible ink outside boxes; flags are diagnostics, not invented lines."""
    width, height = images[0].size
    covered = np.zeros((height, width), dtype=bool)
    for line in lines:
        x0, y0, x1, y1 = crop_geometry(line["box"], (width, height))[0]
        covered[y0:y1, x0:x1] = True
    diagnostics = {}
    for view, image in zip(("blur", "out"), images):
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
    if row.get("schema_version") != 2 or row.get("stage") != "ocr" or row.get("status") != "ok":
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


def process_page(row, frontend, output, page_fingerprint, settings):
    images = [Image.open(row[view]).convert("RGB") for view in ("blur", "out")]
    if images[0].size != images[1].size:
        raise ValueError("Aligned Blur and Out must have the same dimensions")
    detections = [dict(detection, view=view) for view in ("blur", "out")
                  for detection in frontend.detect(row[view])]
    lines = merge_detections(detections, images[0].size)
    lines = reading_order(split_multiline_boxes(lines, images))
    if not lines:
        raise ValueError("No lines detected; this page must not silently become empty text")
    coverage = coverage_diagnostics(images, lines)
    directory = output.parent / (output.stem + ".assets") / fingerprint(row["id"])[:16] / page_fingerprint[:16]
    directory.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(lines):
        line["line_id"] = f"{row['id']}:l{i:04d}"
        line["detected_box"] = line["box"]
        box, polygon, transform = crop_geometry(line["box"], images[0].size, settings.get("crop_padding", 1))
        line.update(box=box, polygon=polygon, crop_to_page=transform,
                    crop_method="shared_axis_aligned", crops={}, crop_hashes={}, sources=[])
        for view, image in zip(("blur", "out"), images):
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
    return {"page_size": list(images[0].size), "lines": lines, "coverage": coverage,
            "detected_boxes": len(detections), "status": "ok"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--samples")
    parser.add_argument("--output", default="runs/line_evidence.jsonl")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--device", help="gpu:0 (default) or cpu")
    args = parser.parse_args(argv)
    config = load_config(args.config).get("v2", {})
    settings = config.get("ocr", {})
    device = args.device or settings.get("device", "gpu:0")
    allowed = config.get("alphabet", string.ascii_letters + string.digits + string.punctuation + " ")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    models = resolve_models([settings.get("detector", DEFAULT_DETECTOR)]
                            + settings.get("recognizers", DEFAULT_RECOGNIZERS), args.offline)
    model_metadata = [{k: v for k, v in model.items() if k != "path"} for model in models]
    if args.download_only:
        for model in models:
            print(f"{model['name']}: {model['path']}", flush=True)
        return 0
    if not args.samples:
        parser.error("--samples is required unless --download-only is set")
    output = Path(args.output).resolve()
    rows = load_samples(args.samples)
    previous = {r["id"]: r for r in read_journal(output)} if output.exists() else {}
    code = {name: hash_file(Path(__file__).with_name(name)) for name in ("ocr_lines.py", "paddle_ctc.py")}
    runtime = {}
    for package in ("paddlex", "paddlepaddle-gpu", "paddlepaddle", "numpy", "Pillow", "opencv-contrib-python"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            pass
    stage_key = {"schema_version": 2, "models": model_metadata, "settings": settings,
                 "alphabet": allowed, "code": code, "runtime": runtime, "device": device}
    frontend, results, failures = None, [], 0
    for row in rows:
        hashes = source_hashes(row)
        identity = {key: row[key] for key in ("id", "document_id", "split", "deblur_run")}
        key = fingerprint({**stage_key, "sample": identity, "source_hashes": hashes})
        old = previous.get(row["id"], {})
        if old.get("fingerprint") == key and cache_valid(old, output):
            results.append(old)
            print(f"{row['id']}: cached ({len(old['lines'])} lines)", flush=True)
            continue
        if frontend is None:
            # Model/setup failures are fatal; don't emit one identical error per page.
            frontend = PaddleCTC(models, allowed, device,
                                 settings.get("detection", {}))
        result = {**identity, "schema_version": 2, "stage": "ocr", "fingerprint": key,
                  "source_hashes": hashes, "models": model_metadata, "alphabet": allowed,
                  "configuration": stage_key}
        try:
            result.update(process_page(row, frontend, output, key, settings))
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


if __name__ == "__main__":
    raise SystemExit(main())
