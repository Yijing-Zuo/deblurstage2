"""Out-only contracts: no Blur/GT reads, post-split deduplication and split lineage."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from common import fingerprint, hash_file, load_samples, read_jsonl, write_jsonl
from ocr_lines import (cache_valid, cached_out_detections, main, out_geometry,
                       process_page, validate_export_provenance, validate_training_roles)
from paddle_ctc import restrict_probabilities


def detection(box):
    x0, y0, x1, y1 = box
    return {"polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "view": "out", "score": .9}


class OutPipelineTests(unittest.TestCase):
    def test_out_only_manifest_and_output_never_require_blur(self):
        class OCR:
            def detect(self, path):
                self.detected = path
                return [detection([2, 4, 95, 17])]

            def recognize(self, path):
                values = restrict_probabilities(np.array([[.1, .9], [.9, .1]], np.float32),
                                                 ["blank", "a"], "a")
                yield {"name": "fake", "raw_text": "a"}, values

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (100, 30), "white").save(root / "out.png")
            sample = {"id": "14_001", "document_id": "14", "deblur_run": "frozen",
                      "split": "test", "stage2_split": "evaluation", "out": "out.png"}
            write_jsonl(root / "samples.jsonl", [sample])
            row = load_samples(root / "samples.jsonl", views=("out",))[0]
            frontend = OCR()
            output = root / "predictions.jsonl"
            page = process_page(row, frontend, output, "a" * 64, {"input_views": ["out"]})
            self.assertEqual(frontend.detected, row["out"])
            self.assertEqual(page["lines"][0]["text"], "a")
            self.assertEqual(set(page["lines"][0]["crops"]), {"out"})
            self.assertEqual(set(page["coverage"]), {"out"})
            page.update(schema_version=3, stage="ocr")
            self.assertTrue(cache_valid(page, output))

    def test_split_fragments_do_not_duplicate_existing_rows(self):
        image = Image.new("RGB", (150, 120), "white")
        draw = ImageDraw.Draw(image)
        for y in (12, 32, 62, 92):
            draw.rectangle([10, y, 130, y + 7], fill="black")
        rows, suppressed = out_geometry([detection(b) for b in (
            [5, 9, 65, 43], [5, 10, 135, 22], [5, 30, 135, 42],
            [5, 60, 135, 72], [5, 90, 135, 102])], image)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(suppressed), 2)
        self.assertTrue(all(row["box"][2] == 135 for row in rows))

    def test_detection_reuse_checks_pixels_and_detector_not_recognizer(self):
        detector = {"name": "det", "revision": "frozen"}
        d = detection([0, 0, 100, 10])
        old = {"source_hashes": {"out": "sha"}, "models": [detector, {"name": "old_rec"}],
               "configuration": {"settings": {"detection": {"thresh": .2}}},
               "lines": [{"detections": [d, d, {**d, "view": "blur"}]}]}
        self.assertEqual(cached_out_detections(old, "sha", detector, {"thresh": .2}), [d])
        with self.assertRaisesRegex(ValueError, "different Out"):
            cached_out_detections(old, "changed", detector, {"thresh": .2})
        with self.assertRaisesRegex(ValueError, "configuration"):
            cached_out_detections(old, "sha", detector, {"thresh": .3})

    def test_exported_training_pages_cannot_be_relabeled_as_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.png"
            Image.new("RGB", (30, 20), "white").save(path)
            dataset = {"train_pages": ["4_004"], "validation_pages": [], "evaluation_pages": [],
                       "pages": [{"id": "4_004", "out_sha256": hash_file(path)}]}
            row = {"id": "4_004", "out": str(path), "stage2_split": "train"}
            validate_training_roles([row], dataset)
            with self.assertRaisesRegex(ValueError, "stage2 manifest"):
                validate_training_roles([{**row, "stage2_split": "evaluation"}], dataset)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Out changed"):
                validate_training_roles([row], dataset)

    def test_export_provenance_rejects_changed_record_or_replaced_weights(self):
        identity = {"settings": {"epochs": 20}, "data_hash": "training-data"}
        hashes = {"inference.json": "graph", "inference.pdiparams": "weights", "inference.yml": "config"}
        record = {"training": identity, "training_fingerprint": fingerprint(identity), "export_hashes": hashes}
        validate_export_provenance(record, {"file_hashes": hashes})
        with self.assertRaisesRegex(ValueError, "model files"):
            validate_export_provenance(record, {"file_hashes": {**hashes, "inference.pdiparams": "other"}})
        with self.assertRaisesRegex(ValueError, "provenance was changed"):
            validate_export_provenance({**record, "training": {"settings": {"epochs": 1}}}, {"file_hashes": hashes})

    def test_failed_inference_keeps_source_size_for_diagnostic_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (512, 768), "white").save(root / "out.png")
            sample = {"id": "14_001", "document_id": "14", "deblur_run": "frozen",
                      "split": "test", "stage2_split": "evaluation", "out": "out.png"}
            write_jsonl(root / "samples.jsonl", [sample])
            (root / "config.json").write_text(json.dumps({"v3": {}}))
            output = root / "predictions.jsonl"
            with patch("ocr_lines.resolve_models", return_value=[{"name": "det"}, {"name": "rec"}]), \
                    patch("ocr_lines.PaddleCTC"), patch("ocr_lines.process_page", side_effect=ValueError("No lines detected")):
                status = main(["--samples", str(root / "samples.jsonl"), "--config", str(root / "config.json"),
                               "--output", str(output), "--offline"])
            self.assertEqual(status, 1)
            row = read_jsonl(output)[0]
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["page_size"], [512, 768])
            self.assertEqual(row["lines"], [])


if __name__ == "__main__":
    unittest.main()
