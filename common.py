"""Shared data contract and prompt. No model imports or network access."""

import hashlib
import json
import os
import warnings
from pathlib import Path


def _json_scalar(value):
    """Keep NumPy metadata numeric; arrays belong in NPZ, not JSONL."""
    import numpy as np

    for scalar, native in ((np.integer, int), (np.floating, float), (np.bool_, bool)):
        if isinstance(value, scalar):
            return native(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False,
                                    default=_json_scalar) + "\n" for r in rows),
                    encoding="utf-8")


def atomic_write_jsonl(path, rows):
    """Replace a stage snapshot only after every record has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                    default=_json_scalar) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_journal(path):
    """Recover complete records; tolerate only an interrupted final write."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line.decode("utf-8-sig")))
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                warnings.warn(f"Ignoring interrupted final record in {path}")
                break
            raise ValueError(f"Corrupt JSONL record {index + 1} in {path}") from None
    return rows


def append_record(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                            default=_json_scalar) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False,
                                     default=_json_scalar).encode()).hexdigest()


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(row, views=("blur", "out")):
    return {view: hash_file(row[view]) for view in views}


def load_config(path="config.json"):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def model_settings(config, size=None, path=None):
    size = (size or config["model_size"]).lower()
    settings = {**config["models"][size], "size": size}
    if path:
        local = Path(path).resolve()
        if not local.is_dir():
            raise ValueError(f"Model directory does not exist: {local}")
        settings["path"] = str(local)
        settings["local_stamp"] = fingerprint([
            (p.relative_to(local).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(local.rglob("*")) if p.is_file()])
    return settings


def load_samples(path, views=("blur", "out")):
    path = Path(path).resolve()
    if not views or set(views) - {"blur", "out"}:
        raise ValueError("Sample views must be blur and/or out")
    rows, seen, documents = read_jsonl(path), set(), {}
    for row in rows:
        for field in ("id", "document_id", "deblur_run", "split", *views):
            if row.get(field) is None or str(row[field]).strip() == "":
                raise ValueError(f"Sample requires {field}: {row.get('id')}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        seen.add(row["id"])
        row["document_id"] = str(row["document_id"])
        split = row["split"]
        if split not in {"train", "validation", "test", "demo"}:
            raise ValueError(f"Unknown split: {split}")
        if row["document_id"] in {"0", "4", "14"} and split in {"train", "validation"}:
            raise ValueError("Current documents 0/4/14 are held out; do not train on them")
        # A document number groups all pages/editions; run/version cannot bypass split.
        doc = row["document_id"]
        if doc in documents and documents[doc] != split:
            raise ValueError(f"Document {doc} occurs in multiple splits")
        documents[doc] = split
        if row.get("stage2_split") not in {None, "train", "validation", "evaluation", "unassigned"}:
            raise ValueError(f"Unknown stage2 split: {row.get('stage2_split')}")
        for view in ("blur", "out"):
            if not row.get(view):
                continue
            image = Path(row[view])
            image = (path.parent / image).resolve() if not image.is_absolute() else image
            if view in views and not image.is_file():
                raise FileNotFoundError(image)
            row[view] = str(image)
    return rows


def make_prompt(blur_text, out_text):
    # The two images are supplied separately by the processor, in this order.
    template = Path(__file__).with_name("PROMPT.md").read_text(encoding="utf-8-sig").strip()
    return template + "\n\nOCR candidates (JSON data):\n" + json.dumps(
        {"blur_ocr": blur_text, "deblur_ocr": out_text}, ensure_ascii=False)
