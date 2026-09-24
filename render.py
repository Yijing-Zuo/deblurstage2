"""Render direct OCR or v2 recovery as white pages and copyable comparison PDFs."""

import argparse
import io
import json
import math
import re
from pathlib import Path

import fitz
from PIL import Image

from common import fingerprint, hash_file, load_samples, read_jsonl


def body_text(value):
    """The renderer never silently removes or guesses an unsupported character."""
    if not isinstance(value, str) or any(not 32 <= ord(c) <= 126 for c in value):
        raise ValueError("Line text must contain printable ASCII, including digits and punctuation")
    return value


def filename(sample_id):
    name = str(sample_id)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name) and name.rstrip(". ") == name:
        if name.split(".")[0].upper() not in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(10)], *[f"LPT{i}" for i in range(10)]}:
            return name
    return "sample-" + fingerprint(name)[:16]


def clear_references(path, sample_ids):
    """Clear images enter here only; they are never used by either model stage."""
    references = {}
    if path:
        path = Path(path).resolve()
        for row in read_jsonl(path):
            sample_id = row["id"]
            if sample_id not in sample_ids or sample_id in references:
                raise ValueError(f"Unknown or duplicate Clear reference id: {sample_id}")
            image = Path(row["clear"])
            image = image if image.is_absolute() else path.parent / image
            if not image.is_file():
                raise FileNotFoundError(image)
            references[sample_id] = image
    return references


def ordered_lines(record):
    seen = set()
    lines = record.get("lines", [])
    for line in lines:
        if not line.get("line_id") or line["line_id"] in seen:
            raise ValueError(f"Missing or duplicate line id in {record['id']}")
        seen.add(line["line_id"])
        if not isinstance(line.get("order"), (int, float)) or not math.isfinite(line["order"]):
            raise ValueError(f"Line requires a finite reading order: {line['line_id']}")
    return sorted(lines, key=lambda line: (line["order"], line["line_id"]))


def line_rect(box, source_size, page_size):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in box):
        raise ValueError(f"Invalid line box: {box}")
    x0, y0, x1, y1 = box
    sw, sh = source_size
    if not (0 <= x0 < x1 <= sw and 0 <= y0 < y1 <= sh):
        raise ValueError(f"Line box outside source page: {box}")
    sx, sy = page_size[0] / sw, page_size[1] / sh
    return fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def fit_line(font, text, rect, minimum=6):
    """Keep a line in its own box; return None instead of clipping tiny text."""
    size = min(14.0, rect.height / (font.ascender - font.descender))
    if size < minimum:
        return None
    width = font.text_length(text, fontsize=size)
    if width > rect.width:
        size = max(minimum, size * rect.width / width)
        width = font.text_length(text, fontsize=size)
    scale = min(1.0, rect.width / width) if width else 1.0
    return (size, scale) if scale >= 0.8 else None


def placement(font, text, rect, page_size, other_boxes):
    """Give tiny labels a bounded readable box, without covering another line."""
    fitting = fit_line(font, text, rect)
    if fitting:
        return rect, fitting
    width = max(rect.width, font.text_length(text, fontsize=6) + 0.1)
    height = max(rect.height, 6 * (font.ascender - font.descender) + 0.1)
    if width > rect.width + 24 or height > rect.height + 12:
        return rect, None
    if width > page_size[0] or height > page_size[1]:
        return rect, None
    for left in (rect.x0, rect.x1 - width):
        for top in (rect.y0, rect.y1 - height):
            x = max(0, min(left, page_size[0] - width))
            y = max(0, min(top, page_size[1] - height))
            expanded = fitz.Rect(x, y, x + width, y + height)
            if all((expanded & box).get_area() <= (rect & box).get_area() + 0.01 for box in other_boxes):
                return expanded, fit_line(font, text, expanded)
    return rect, None


def put_line(page, font, font_path, text, rect, fitting):
    size, scale = fitting
    page.insert_font(fontname="body" if font_path else "helv", fontfile=font_path)
    baseline = fitz.Point(rect.x0, rect.y0 + font.ascender * size)
    page.insert_text(baseline, text, fontname="body" if font_path else "helv",
                     fontsize=size, color=(0, 0, 0),
                     morph=(baseline, fitz.Matrix(scale, 1)))


def wrap_text(font, text, width, size):
    """Wrap even an unbroken gibberish token, preserving every non-space character."""
    chunks = []
    while text:
        if font.text_length(text, fontsize=size) <= width:
            chunks.append(text)
            break
        lo, hi = 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.text_length(text[:mid], fontsize=size) <= width:
                lo = mid
            else:
                hi = mid - 1
        end = text.rfind(" ", 0, lo + 1)
        end = end if end > 0 else lo
        chunks.append(text[:end])
        text = text[end:].lstrip(" ")
    return chunks


def overflow_pages(document, items, font, font_path, page_size, sample_id):
    pages, page, cursor = [], None, 0
    width, height = page_size
    for item in items:
        for text in [f"Line {item['line_id']}", *wrap_text(font, item["text"], width - 48, 10), ""]:
            if page is None or cursor + 15 > height - 24:
                page = document.new_page(width=width, height=height)
                pages.append(page.number)
                page.insert_text((24, 22), "Recovered text - overflow", fontsize=10)
                # IDs are metadata and need not satisfy the body alphabet contract.
                page.insert_text((24, 37), filename(sample_id), fontsize=8)
                cursor = 55
            rect = fitz.Rect(24, cursor, width - 24, cursor + 15)
            put_line(page, font, font_path, text, rect, (10, 1))
            cursor += 15
    return pages


def render_record(document, record, font, font_path, page_size):
    page = document.new_page(width=page_size[0], height=page_size[1])
    primary = page.number
    report = {"id": record["id"], "status": record["status"], "page": primary + 1,
              "overflow": [], "missing_lines": [], "review_lines": [], "layout": [],
              "stage2_split": record.get("stage2_split", "unassigned"),
              "flags": record.get("flags", []), "coverage": record.get("coverage", {})}
    lines = ordered_lines(record)
    boxes = [line_rect(line["box"], record["page_size"], page_size) for line in lines]
    if not lines:
        page.insert_text((24, 40), "No recovered lines. See recovery.jsonl.", fontsize=12)
        report["status"] = "error"
    for index, line in enumerate(lines):
        rect = boxes[index]
        text = body_text(line.get("text", ""))
        if line.get("status") == "review":
            report["review_lines"].append(line["line_id"])
        if not text.strip() or line.get("status") == "error":
            report["missing_lines"].append(line["line_id"])
            text = text or "[unrecovered]"
        placed, fitting = placement(font, text, rect, page_size, boxes[:index] + boxes[index + 1:])
        if fitting:
            put_line(page, font, font_path, text, placed, fitting)
            layout = {"line_id": line["line_id"], "font_size": fitting[0], "horizontal_scale": fitting[1]}
            if placed != rect:
                layout.update({"adjustment": "minimum_readable_box", "original_box": list(rect), "placed_box": list(placed)})
                # Later tiny labels must also respect the already expanded box.
                boxes[index] = placed
            report["layout"].append(layout)
        else:
            report["overflow"].append({"line_id": line["line_id"], "text": text})
            # An explicit mark replaces a line only when its entire text is appended.
            page.draw_rect(rect, color=(0.55, 0.55, 0.55), width=0.5)
            marker = f"[overflow {len(report['overflow'])}]"
            marker_size = min(8, rect.width / max(font.text_length(marker), 1), rect.height / 1.4)
            if marker_size >= 4:
                put_line(page, font, font_path, marker, rect, (marker_size, 1))
    report["overflow_pages"] = [p + 1 for p in overflow_pages(document, report["overflow"], font, font_path, page_size, record["id"])]
    if report["overflow"] or report["missing_lines"] or report["review_lines"]:
        report["status"] = "review" if report["status"] != "error" else "error"
    return primary, report


def compare_page(comparison, recovered, primary, sample, clear, report, page_size):
    panels = ([("Clear", clear)] if clear else []) + [("Output", sample["out"]), ("Blur", sample.get("blur")), ("Recovered", None)]
    width, height = page_size
    page = comparison.new_page(width=24 + len(panels) * (width + 12), height=height + 64)
    summary = f"{filename(sample['id'])} | {report['status']}"
    if report["stage2_split"] != "unassigned":
        summary += f" | stage2: {report['stage2_split']}"
    if report["status"] != "ok" and report["flags"]:
        summary += " | " + ", ".join(report["flags"])
    page.insert_text((12, 13), summary, fontsize=9)
    for index, (label, image_path) in enumerate(panels):
        x = 12 + index * (width + 12)
        page.insert_text((x, 31), label, fontsize=12)
        rect = fitz.Rect(x, 40, x + width, 40 + height)
        if image_path:
            with Image.open(image_path) as image:
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="PNG")
            page.insert_image(rect, stream=buffer.getvalue())
        elif label == "Recovered":
            page.show_pdf_page(rect, recovered, primary)
        else:
            page.insert_text((x + 16, 65), "Blur not supplied (Out-only OCR)", fontsize=12)
    if report["overflow"]:
        page.insert_text((12, height + 56), "Full overflow text is on additional pages in recovered.pdf.", fontsize=9)
    elif not clear:
        page.insert_text((12, height + 56), "No matching Clear reference supplied: three-column comparison.", fontsize=9)


def render(samples_path, recovery_path, output, clear_path=None, font_path=None, page_size=(512, 768), allow_incomplete=False, documents=None):
    if min(page_size) < 128:
        raise ValueError("Render dimensions must be at least 128 pixels")
    rows = read_jsonl(recovery_path)
    direct_ocr = bool(rows) and all(row.get("schema_version") == 3 and row.get("stage") == "ocr" for row in rows)
    views = ("out",) if direct_ocr else ("blur", "out")
    samples = load_samples(samples_path, views=views)
    manifest_ids = {sample["id"] for sample in samples}
    if documents:
        documents = {str(doc) for doc in documents}
        samples = [sample for sample in samples if sample["document_id"] in documents]
        rows = [row for row in rows if str(row.get("document_id")) in documents]
        if documents != {sample["document_id"] for sample in samples}:
            raise ValueError("Requested render document is missing from the sample manifest")
    if not samples:
        raise ValueError("Sample manifest is empty")
    ids = {sample["id"] for sample in samples}
    records = {}
    for row in rows:
        if row.get("id") not in ids or row["id"] in records:
            raise ValueError(f"Unknown or duplicate recovery id: {row.get('id')}")
        valid_stage = (row.get("schema_version"), row.get("stage")) in {(2, "recovery"), (3, "ocr")}
        if not valid_stage or row.get("status") not in {"ok", "review", "error"}:
            raise ValueError(f"Not a v2 recovery or v3 OCR record: {row.get('id')}")
        if (row.get("stage") == "ocr") != direct_ocr:
            raise ValueError("Do not mix v2 recovery and v3 OCR records")
        records[row["id"]] = row
    if ids != set(records):
        raise ValueError(f"Recovery missing sample ids: {sorted(ids - set(records))}")
    references = {key: value for key, value in clear_references(clear_path, manifest_ids).items() if key in ids}
    if font_path:
        font_path = str(Path(font_path).resolve(strict=True))
    font = fitz.Font(fontfile=font_path) if font_path else fitz.Font("helv")
    # Validate every page before writing any artifacts, including reference alignment.
    for sample in samples:
        row = records[sample["id"]]
        if direct_ocr:
            role = row.get("stage2_split", "unassigned")
            if role not in {"train", "validation", "evaluation", "unassigned"} or role != sample.get("stage2_split", "unassigned"):
                raise ValueError(f"Stage2 split does not match the source sample: {sample['id']}")
            # Blur is an optional comparison image, never a v3 inference dependency.
            blur = sample.pop("blur", None)
            if blur:
                blur = Path(blur)
                blur = blur if blur.is_absolute() else Path(samples_path).resolve().parent / blur
                if blur.is_file():
                    sample["blur"] = str(blur)
        if not allow_incomplete and (row["status"] == "error" or not row.get("lines") or any(
                line.get("status") == "error" or not line.get("text", "").strip() for line in row.get("lines", []))):
            raise ValueError(f"Incomplete recovery for {sample['id']}; fix that record or explicitly use --allow-incomplete")
        hashes = {view: hash_file(sample[view]) for view in views}
        if row.get("source_hashes") != hashes or str(row.get("document_id")) != sample["document_id"]:
            raise ValueError(f"Recovery does not match the source sample: {sample['id']}")
        for path in [sample[view] for view in ("out", "blur") if view in sample] + ([references[sample["id"]]] if sample["id"] in references else []):
            with Image.open(path) as image:
                if list(image.size) != row.get("page_size"):
                    raise ValueError(f"Page size mismatch: {sample['id']} / {path}")
        for line in ordered_lines(row):
            line_rect(line["box"], row["page_size"], page_size)
            text = body_text(line.get("text", ""))
            if any(not font.has_glyph(ord(char)) for char in text):
                raise ValueError(f"Selected font lacks a required character: {line['line_id']}")
    output = Path(output)
    images = output / "recovered"
    images.mkdir(parents=True, exist_ok=True)
    reports, written_names = [], set()
    with fitz.open() as recovered:
        for sample in samples:
            primary, report = render_record(recovered, records[sample["id"]], font, font_path, page_size)
            report["comparison_columns"] = 4 if sample["id"] in references else 3
            report["images"] = []
            for number, page_index in enumerate([primary, *[p - 1 for p in report["overflow_pages"]]]):
                suffix = "" if number == 0 else f"-overflow-{number}"
                target = images / f"{filename(sample['id'])}{suffix}.png"
                if target.name.casefold() in written_names:
                    raise ValueError(f"Conflicting rendered filenames: {target.name}")
                written_names.add(target.name.casefold())
                recovered[page_index].get_pixmap(alpha=False).save(target)
                report["images"].append(target.relative_to(output).as_posix())
            reports.append(report)
        recovered.save(output / "recovered.pdf", garbage=4, deflate=True)
    # MuPDF caches source object IDs on first import: never grow that source PDF.
    # Reopen the finished document before composing copyable comparison panels.
    with fitz.open(output / "recovered.pdf") as recovered, fitz.open() as comparison:
        for sample, report in zip(samples, reports):
            compare_page(comparison, recovered, report["page"] - 1, sample,
                         references.get(sample["id"]), report, page_size)
        comparison.save(output / "comparison.pdf", garbage=4, deflate=True)
    report = {"schema_version": 3 if direct_ocr else 2, "source_stage": "ocr" if direct_ocr else "recovery",
              "page_size": list(page_size), "font": font_path or "Helvetica", "samples": reports}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--predictions", "--recovery", dest="recovery", required=True)
    parser.add_argument("--documents", nargs="+", help="Render only these document IDs; require all their samples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clear-references")
    parser.add_argument("--font")
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicitly render error/missing lines as a diagnostic artifact")
    parser.add_argument("--page-width", type=int, default=512)
    parser.add_argument("--page-height", type=int, default=768)
    args = parser.parse_args(argv)
    report = render(args.samples, args.recovery, args.output, args.clear_references, args.font,
                    (args.page_width, args.page_height), args.allow_incomplete, args.documents)
    reviews = sum(row["status"] != "ok" for row in report["samples"])
    print(f"Rendered {len(report['samples'])} samples; {reviews} require review. See {args.output}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
