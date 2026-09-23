"""Render selected v2 lines as white pages and copyable comparison PDFs (CPU)."""

import argparse
import io
import json
import math
import re
from pathlib import Path

import fitz
from PIL import Image

from common import fingerprint, load_samples, read_jsonl, source_hashes


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
              "flags": record.get("flags", []), "coverage": record.get("coverage", {})}
    lines = ordered_lines(record)
    if not lines:
        page.insert_text((24, 40), "No recovered lines. See recovery.jsonl.", fontsize=12)
        report["status"] = "error"
    for line in lines:
        rect = line_rect(line["box"], record["page_size"], page_size)
        text = body_text(line.get("text", ""))
        if line.get("status") == "review":
            report["review_lines"].append(line["line_id"])
        if not text.strip() or line.get("status") == "error":
            report["missing_lines"].append(line["line_id"])
            text = text or "[unrecovered]"
        fitting = fit_line(font, text, rect)
        if fitting:
            put_line(page, font, font_path, text, rect, fitting)
            report["layout"].append({"line_id": line["line_id"], "font_size": fitting[0], "horizontal_scale": fitting[1]})
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
    panels = ([("Clear", clear)] if clear else []) + [("Output", sample["out"]), ("Blur", sample["blur"]), ("Recovered", None)]
    width, height = page_size
    page = comparison.new_page(width=24 + len(panels) * (width + 12), height=height + 64)
    summary = f"{filename(sample['id'])} | {report['status']}"
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
        else:
            page.show_pdf_page(rect, recovered, primary)
    if report["overflow"]:
        page.insert_text((12, height + 56), "Full overflow text is on additional pages in recovered.pdf.", fontsize=9)
    elif not clear:
        page.insert_text((12, height + 56), "No matching Clear reference supplied: three-column comparison.", fontsize=9)


def render(samples_path, recovery_path, output, clear_path=None, font_path=None, page_size=(512, 768), allow_incomplete=False):
    if min(page_size) < 128:
        raise ValueError("Render dimensions must be at least 128 pixels")
    samples = load_samples(samples_path)
    if not samples:
        raise ValueError("Sample manifest is empty")
    ids = {sample["id"] for sample in samples}
    records = {}
    for row in read_jsonl(recovery_path):
        if row.get("id") not in ids or row["id"] in records:
            raise ValueError(f"Unknown or duplicate recovery id: {row.get('id')}")
        if row.get("schema_version") != 2 or row.get("stage") != "recovery" or row.get("status") not in {"ok", "review", "error"}:
            raise ValueError(f"Not a v2 recovery record: {row.get('id')}")
        records[row["id"]] = row
    if ids != set(records):
        raise ValueError(f"Recovery missing sample ids: {sorted(ids - set(records))}")
    references = clear_references(clear_path, ids)
    if font_path:
        font_path = str(Path(font_path).resolve(strict=True))
    font = fitz.Font(fontfile=font_path) if font_path else fitz.Font("helv")
    # Validate every page before writing any artifacts, including reference alignment.
    for sample in samples:
        row = records[sample["id"]]
        if not allow_incomplete and (row["status"] == "error" or not row.get("lines") or any(
                line.get("status") == "error" or not line.get("text", "").strip() for line in row.get("lines", []))):
            raise ValueError(f"Incomplete recovery for {sample['id']}; fix that record or explicitly use --allow-incomplete")
        if row.get("source_hashes") != source_hashes(sample) or str(row.get("document_id")) != sample["document_id"]:
            raise ValueError(f"Recovery does not match the source sample: {sample['id']}")
        for path in [sample["blur"], sample["out"], *([references[sample["id"]]] if sample["id"] in references else [])]:
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
    with fitz.open() as recovered, fitz.open() as comparison:
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
            compare_page(comparison, recovered, primary, sample, references.get(sample["id"]), report, page_size)
            reports.append(report)
        recovered.save(output / "recovered.pdf", garbage=4, deflate=True)
        comparison.save(output / "comparison.pdf", garbage=4, deflate=True)
    report = {"schema_version": 2, "page_size": list(page_size), "font": font_path or "Helvetica", "samples": reports}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--recovery", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clear-references")
    parser.add_argument("--font")
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicitly render error/missing lines as a diagnostic artifact")
    parser.add_argument("--page-width", type=int, default=512)
    parser.add_argument("--page-height", type=int, default=768)
    args = parser.parse_args(argv)
    report = render(args.samples, args.recovery, args.output, args.clear_references, args.font,
                    (args.page_width, args.page_height), args.allow_incomplete)
    reviews = sum(row["status"] != "ok" for row in report["samples"])
    print(f"Rendered {len(report['samples'])} samples; {reviews} require review. See {args.output}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
