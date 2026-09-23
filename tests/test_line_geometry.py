"""CPU contracts: detection union, reading order, shared coordinates and CTC capture."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from ocr_lines import (cache_valid, crop_geometry, merge_detections, process_page,
                       reading_order, split_multiline_boxes)
from paddle_ctc import PaddleCTC, restrict_probabilities


def detection(box, view="out"):
    a, b, c, d = box
    return {"polygon": [[a, b], [c, b], [c, d], [a, d]], "score": 0.9, "view": view}


class GeometryTests(unittest.TestCase):
    def test_union_keeps_blur_only_line_and_joins_fragments_not_columns(self):
        detections = [detection([5, 10, 90, 20]), detection([6, 10, 91, 21], "blur"),
                      detection([5, 30, 40, 40]), detection([44, 30, 91, 40]),
                      detection([120, 10, 195, 20]), detection([5, 50, 90, 60], "blur")]
        lines = merge_detections(detections, (200, 100))
        self.assertEqual(len(lines), 4)
        self.assertTrue(any(x["box"] == [5, 30, 91, 40] for x in lines))
        self.assertTrue(any(x["detections"][0]["view"] == "blur" for x in lines))

    def test_columns_are_not_interleaved_and_heading_is_first(self):
        boxes = [[0, 0, 200, 10], [5, 25, 90, 35], [5, 50, 90, 60],
                 [120, 25, 195, 35], [120, 50, 195, 60], [0, 80, 200, 90]]
        result = reading_order([{"box": b, "tag": i} for i, b in enumerate(boxes)])
        self.assertEqual([x["tag"] for x in result], [0, 1, 2, 3, 4, 5])
        self.assertEqual(result[1]["column"], result[2]["column"])
        self.assertNotEqual(result[2]["column"], result[3]["column"])

    def test_crop_transform_is_exact_and_clipped(self):
        box, polygon, transform = crop_geometry([-2, 5.5, 110, 20.2], (100, 80), 1)
        self.assertEqual(box, [0, 4, 100, 22])
        self.assertEqual(polygon[-1], [0, 22])
        np.testing.assert_array_equal(np.asarray(transform) @ [10, 3, 1], [10, 7, 1])

    def test_multiline_split_requires_image_evidence(self):
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        for y in [12, 32]:
            draw.rectangle([10, y, 90, y + 7], fill="black")
        rows = [{"box": [5, 10, 95, 42]}, {"box": [5, 60, 95, 68]},
                {"box": [5, 80, 95, 88]}]
        split = split_multiline_boxes(rows, [image, image])
        self.assertEqual(len(split), 4)
        self.assertTrue(split[0]["split_from_multiline"])

    def test_page_evidence_hashes_and_shared_crop_survive_cache(self):
        class FakeOCR:
            def detect(self, path):
                return [detection([10, 10, 80, 20])]

            def recognize(self, path):
                values = restrict_probabilities(np.asarray([[0.2, 0.7, 0.1]]),
                                                ["blank", "a", "Ω"], "a")
                yield {"name": "fake", "raw_text": "a", "restricted_text": "a"}, values

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for view in ("blur", "out"):
                Image.new("RGB", (100, 50), "white").save(path / (view + ".png"))
            sample = {"id": "sample", "blur": str(path / "blur.png"), "out": str(path / "out.png")}
            output = path / "evidence.jsonl"
            page = process_page(sample, FakeOCR(), output, "a" * 64, {})
            page.update(schema_version=2, stage="ocr")
            self.assertTrue(cache_valid(page, output))
            line = page["lines"][0]
            self.assertEqual(line["crop_hashes"]["blur"], line["crop_hashes"]["out"])
            source = line["sources"][0]
            with np.load(path / source["probabilities"], allow_pickle=False) as evidence:
                self.assertAlmostEqual(float(evidence["probs"].sum()), 0.9, places=6)
            (path / line["crops"]["out"]).write_bytes(b"changed")
            self.assertFalse(cache_valid(page, output))


class CaptureTests(unittest.TestCase):
    def test_class_indices_keep_original_probability_mass(self):
        raw = np.array([[0.1, 0.3, 0.4, 0.2]], dtype=np.float32)
        result = restrict_probabilities(raw, ["blank", "Ω", "a", " "], "a ")
        np.testing.assert_array_equal(result["original_class_ids"], [0, 2, 3])
        np.testing.assert_array_equal(result["alphabet"], ["", "a", " "])
        np.testing.assert_allclose(result["probs"], [[0.1, 0.4, 0.2]])
        np.testing.assert_allclose(result["excluded_mass"], [0.3])
        # Unsupported requested characters are unavailable for this source;
        # they must not shift existing classes or become zero-probability columns.
        missing = restrict_probabilities(raw, ["blank", "Ω", "a", " "], "ab ")
        np.testing.assert_array_equal(missing["alphabet"], ["", "a", " "])
        with self.assertRaises(ValueError):
            restrict_probabilities(raw * 5, ["blank", "Ω", "a", " "], "a ")

    def test_actual_runner_boundary_not_global_capture_or_tail_trim(self):
        raw = np.zeros((1, 5, 3), dtype=np.float32)
        raw[0, :, 0] = 1
        raw[0, 0, :] = [0, 1, 0]
        raw[0, 4, :] = [0, 0, 1]
        class Resize:
            max_imgW = 3200
            def __call__(self, imgs):
                return [np.zeros((3, 48, 320), dtype=np.float32)]

        predictor = SimpleNamespace(pre_tfs={
            "Read": lambda imgs: [np.zeros((20, 30, 3), dtype=np.uint8)],
            "ReisizeNorm": Resize(), "ToBatch": lambda imgs: [np.stack(imgs)]},
            runner=lambda x: [raw])
        adapter = PaddleCTC.__new__(PaddleCTC)
        adapter.allowed = "ab"
        adapter.recognizers = [({"name": "fake"}, predictor, ["blank", "a", "b"])]
        metadata, arrays = list(adapter.recognize("line.png"))[0]
        self.assertEqual(metadata["raw_text"], "ab")
        self.assertEqual(arrays["probs"].shape, (5, 3))
        self.assertEqual(metadata["padding_right"], 248)
        self.assertEqual(metadata["time_steps"], 5)


if __name__ == "__main__":
    unittest.main()
