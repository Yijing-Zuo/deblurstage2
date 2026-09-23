"""Local image geometry and output checks; only Pillow/NumPy, no model imports."""
import json
import re

import numpy as np
from PIL import Image, ImageDraw


def split_regions(out_image, core_height=192, context=32):
    """Partition EVERY source row; overlapping context is never another core."""
    if core_height < 32 or context < 0:
        raise ValueError("core_height must be >=32; context must be >=0")
    width, height = out_image.size
    gray = np.asarray(out_image.convert("L"))
    # Ignore small side margins and isolated dark pixels; no OCR/GT supplies boxes.
    margin = min(width // 16, 24)
    band = gray[:, margin:width - margin] if margin else gray
    ink = (band < 190).mean(axis=1)
    cuts = [0]
    radius = min(16, core_height // 8)
    for target in range(core_height, height, core_height):
        if height - target < core_height // 2:
            break
        candidates = range(max(cuts[-1] + 1, target - radius),
                           min(height, target + radius + 1))
        # Score a short stripe so a single white scanline inside a glyph is not preferred.
        cut = min(candidates, key=lambda y: (
            float(ink[max(0, y - 2):min(height, y + 3)].mean()), abs(y - target)))
        cuts.append(cut)
    cuts.append(height)
    return [{"index": index, "core_box": [0, start, width, end],
             "crop_box": [0, max(0, start - context), width, min(height, end + context)]}
            for index, (start, end) in enumerate(zip(cuts, cuts[1:]))]


def crop_pair(sample, region, scale=2):
    """Use identical coordinates; red brackets occupy added white margins only."""
    if scale < 1:
        raise ValueError("scale must be >=1")
    sources = []
    for view in ("blur", "out"):
        with Image.open(sample[view]) as source:
            sources.append(source.convert("RGB"))
    if sources[0].size != sources[1].size:
        raise ValueError("Blur/Out must have the same size and aligned coordinates")
    x0, y0, x1, y1 = region["crop_box"]
    cx0, cy0, cx1, cy1 = region["core_box"]
    width, height = sources[0].size
    if not (0 <= x0 <= cx0 < cx1 <= x1 <= width and 0 <= y0 <= cy0 < cy1 <= y1 <= height):
        raise ValueError("Invalid crop/core coordinates")
    images = []
    for source in sources:
        crop = source.crop(region["crop_box"])
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
        gutter = 24
        image = Image.new("RGB", (crop.width + 2 * gutter, crop.height), "white")
        image.paste(crop, (gutter, 0))
        draw = ImageDraw.Draw(image)
        y0 = (region["core_box"][1] - region["crop_box"][1]) * scale
        y1 = (region["core_box"][3] - region["crop_box"][1]) * scale - 1
        for x, direction in ((8, 1), (image.width - 9, -1)):
            draw.line([(x + 10 * direction, y0), (x, y0), (x, y1),
                       (x + 10 * direction, y1)], fill="red", width=3)
        images.append(image)
    return images


def parse_transcription(raw):
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, count=1).removesuffix("```").strip()
    data = json.loads(value)
    if not isinstance(data, dict) or set(data) != {"text"} or not isinstance(data["text"], str):
        raise ValueError('Expected exactly {"text": "transcribed text"}')
    return data["text"].strip()


def repeat_tail(tokens):
    """Conservative runaway detector: four identical cycles of >=12 tokens."""
    return any(tokens[-period:] * 4 == tokens[-4 * period:]
               for period in range(12, min(128, len(tokens) // 4) + 1))


def review_flags(text):
    # These are observable warnings, NOT a language/accuracy classifier.
    lower = text.lower()
    flags = []
    if any(phrase in lower for phrase in (
            "i cannot", "i can't", "i am unable", "i'm unable", "as an ai",
            "the image is too", "the text is too blurred", "here is the transcription")):
        flags.append("possible_explanation")
    if "[unclear]" in lower:
        flags.append("unclear_spans")
    if sum("\u4e00" <= char <= "\u9fff" for char in text) > 20:
        flags.append("unexpected_language")
    return flags
