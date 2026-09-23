"""Cross-stage CPU contracts with synthetic probabilities, no model inference."""

import json
import string
import tempfile
import unittest
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw

from candidates import build_candidates
from common import source_hashes, write_jsonl
from ctc import load_evidence
from ocr_lines import process_page
from recover import evidence_jobs, recover_line, save_pages
from render import render


ALPHABET = string.ascii_letters + string.digits + string.punctuation + " "


class FixtureOCR:
    """Deterministic adapter substitute used to exercise actual file boundaries."""
    def detect(self, path):
        return [{"polygon": [[30, y], [480, y], [480, y + 18], [30, y + 18]], "score": 1.0}
                for y in (40, 80)]

    def recognize(self, path):
        text = "English words 2026." if "l0000" in str(path) else "Keep 0 and O, 1 and l."
        chars = ["", *ALPHABET]
        matrix = np.zeros((2 * len(text) + 1, len(chars)), dtype=np.float32)
        matrix[::2, 0] = 1
        for index, char in enumerate(text):
            matrix[2 * index + 1, chars.index(char)] = 1
        yield {"name": "synthetic-recognizer", "revision": "fixture", "restricted_text": text,
               "raw_text": text, "width_compressed": False}, {
                   "probs": matrix, "alphabet": np.asarray(chars), "blank_id": np.asarray(0),
                   "excluded_mass": np.zeros(len(matrix), dtype=np.float32)}


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = Image.new("RGB", (512, 768), "white")
        draw = ImageDraw.Draw(image)
        for y in (45, 85):
            draw.rectangle((35, y, 450, y + 5), fill="black")
        for view in ("blur", "out"):
            image.save(self.root / f"{view}.png")
        self.sample = {"id": "4_004", "document_id": "4", "split": "test", "deblur_run": "synthetic",
                       "blur": str(self.root / "blur.png"), "out": str(self.root / "out.png")}
        self.sample_path = self.root / "samples.jsonl"
        write_jsonl(self.sample_path, [self.sample])
        self.evidence_path = self.root / "evidence.jsonl"
        self.page = {"schema_version": 2, "stage": "ocr", "id": "4_004", "document_id": "4",
                     "source_hashes": source_hashes(self.sample), "fingerprint": "fixture", "alphabet": ALPHABET,
                     **process_page(self.sample, FixtureOCR(), self.evidence_path, "fixture", {"crop_padding": 1})}
        write_jsonl(self.evidence_path, [self.page])
        self.config = {"alphabet": ALPHABET, "candidates": {"beam_width": 2, "beam_top_k": 2},
                       "recovery": {"context_lines": 1, "proposal_max_growth": 12}}
        self.model = {"id": "synthetic-model", "size": "32b", "revision": "fixture"}

    def test_npz_geometry_selection_assembly_and_copyable_pdf_are_compatible(self):
        pages, jobs = evidence_jobs([self.sample], self.evidence_path, self.config, self.model, "Test", "no-lexicon")
        results = {}
        for job in jobs:
            line = job["line"]
            sources = [{**source, **load_evidence(self.root / source["probabilities"])} for source in line["sources"]]
            plan = build_candidates(sources, {**self.config["candidates"], "alphabet": ALPHABET}, lexicon=None)

            def ask(prompt):
                self.assertNotIn("Clear", prompt)
                return {"status": "ok", "raw": json.dumps({"choices": [
                    {"span_id": span["id"], "candidate_id": next(candidate["id"] for candidate in span["candidates"]
                     if candidate["origin"] == "original")} for span in plan["spans"]], "proposals": []})}

            result = recover_line(line, sources, plan, job["context"], ask, "Test", self.config["recovery"], ALPHABET)
            self.assertEqual(result["status"], "ok")
            results[job["fingerprint"]] = {**result, "fingerprint": job["fingerprint"]}
        recovery_path = self.root / "recovery.jsonl"
        saved = save_pages(recovery_path, pages, jobs, results, self.model)
        self.assertEqual(len(saved[0]["lines"]), 2)
        report = render(self.sample_path, recovery_path, self.root / "render")
        self.assertEqual(report["samples"][0]["comparison_columns"], 3)
        with fitz.open(self.root / "render/recovered.pdf") as pdf:
            self.assertEqual(pdf[0].get_text().splitlines(), [line["text"] for line in saved[0]["lines"]])
        with Image.open(self.root / "render/recovered/4_004.png") as image:
            self.assertEqual(image.size, (512, 768))

    def test_modified_crop_cannot_reuse_old_ocr_metadata(self):
        crop = self.root / self.page["lines"][0]["crops"]["blur"]
        Image.new("RGB", (452, 20), "black").save(crop)
        with self.assertRaisesRegex(ValueError, "(?i)(hash|crop|asset|differ)"):
            evidence_jobs([self.sample], self.evidence_path, self.config, self.model, "Test", "none")

    def test_modified_probabilities_cannot_reuse_old_ocr_metadata(self):
        matrix = self.root / self.page["lines"][0]["sources"][0]["probabilities"]
        matrix.write_bytes(b"stale asset from another run")
        with self.assertRaisesRegex(ValueError, "(?i)(hash|probabilit|asset|differ)"):
            evidence_jobs([self.sample], self.evidence_path, self.config, self.model, "Test", "none")


if __name__ == "__main__":
    unittest.main()
