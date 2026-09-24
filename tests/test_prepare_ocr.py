"""Real-Out supervision must preserve page isolation, provenance and complete text."""

import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from common import hash_file, load_samples, write_jsonl
from prepare_ocr import prepare, split_pages


class PrepareOCRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples_path = self.root / "samples.jsonl"
        self.labels_path = self.root / "labels.jsonl"
        self.output = self.root / "prepared"
        self.samples, self.labels = [], []
        for index, (doc, page) in enumerate([(4, n) for n in range(1, 6)] + [(14, 1), (0, 1)]):
            sample_id = f"{doc}_{page:03d}"
            image = self.root / f"Out_{sample_id}.png"
            Image.new("RGB", (80, 60), (index * 20, 100, 100)).save(image)
            self.samples.append({"id": sample_id, "document_id": str(doc), "page": page,
                                 "split": "test", "deblur_run": "frozen", "out": image.name})
            if doc == 4:
                self.labels.append({"id": sample_id, "document_id": "4", "out_sha256": hash_file(image),
                    "page_size": [80, 60], "label_status": "geometry_checked",
                    "provenance": {"method": "pdf_text_geometry", "source_sha256": "a" * 64},
                    "lines": [{"box": [1, 2, 79, 20], "text": f"Original English page {page}.", "complete": True},
                              {"box": [3, 25, 70, 45], "text": "Numbers 1900; don't invent words!", "complete": True}]})
        self.save()

    def save(self):
        write_jsonl(self.samples_path, self.samples)
        write_jsonl(self.labels_path, self.labels)

    def run_prepare(self):
        return prepare(self.samples_path, self.labels_path, self.output)

    def test_page_block_split_outputs_and_portability(self):
        result = self.run_prepare()
        self.assertEqual(result["counts"], {"train": 8, "validation": 2})
        self.assertEqual(result["validation_pages"], ["4_005"])
        self.assertEqual(set(result["evaluation_documents"]), {"0", "14"})
        self.assertEqual(result["label_status_counts"], {"geometry_checked": 5})
        rows = load_samples(self.output / "stage2_samples.jsonl", views=("out",))
        self.assertEqual({r["split"] for r in rows}, {"test"})
        self.assertEqual(len([r for r in rows if r["stage2_split"] == "evaluation"]), 2)
        for role, filename in (("train", "train.txt"), ("validation", "val.txt")):
            for line in (self.output / filename).read_text().splitlines():
                image, text = line.split("\t")
                self.assertTrue(text)
                self.assertNotIn("14_", image)
                with Image.open(self.output / image) as crop:
                    self.assertIn(crop.size, {(78, 18), (67, 20)})
        self.assertEqual(result["fingerprint"], self.run_prepare()["fingerprint"])

    def test_sibling_crops_stay_on_original_page(self):
        samples = self.samples + [{**self.samples[4], "id": "4_005_secondcrop"}]
        roles = split_pages(samples)
        self.assertEqual(roles["4_005"], roles["4_005_secondcrop"])
        self.assertEqual(roles["4_004"], "train")
        self.assertEqual(roles["14_001"], "evaluation")

    def test_evaluation_labels_are_not_silently_accepted(self):
        self.labels.append({**self.labels[0], "id": "14_001", "document_id": "14"})
        self.save()
        with self.assertRaisesRegex(ValueError, "only to adaptation document 4"):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_stale_or_misregistered_labels_rejected_before_writes(self):
        good = copy.deepcopy(self.labels)
        for field, value, message in (("out_sha256", "0" * 64, "hash mismatch"),
                                      ("page_size", [160, 120], "page_size"),
                                      ("label_status", "unreviewed", "status"),
                                      ("provenance", {}, "provenance")):
            with self.subTest(field=field):
                self.labels = copy.deepcopy(good)
                self.labels[0][field] = value
                self.save()
                with self.assertRaisesRegex(ValueError, message):
                    self.run_prepare()
                self.assertFalse(self.output.exists())

    def test_partial_words_non_ascii_and_invalid_boxes_rejected(self):
        good = copy.deepcopy(self.labels)
        cases = [("complete", False, "partial-word"),
                 ("text", "cut\tword", "ASCII"), ("text", "caf\u00e9", "ASCII"),
                 ("text", " word ", "ASCII"), ("box", [-1, 2, 20, 20], "outside"),
                 ("box", [0, 0, 0, 20], "outside"), ("box", [0, 0, 90, 20], "outside")]
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                self.labels = copy.deepcopy(good)
                self.labels[0]["lines"][0][field] = value
                self.save()
                with self.assertRaisesRegex(ValueError, message):
                    self.run_prepare()
                self.assertFalse(self.output.exists())

    def test_missing_duplicate_labels_and_duplicate_crops_rejected(self):
        good = copy.deepcopy(self.labels)
        self.labels.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "Missing reference labels"):
            self.run_prepare()
        self.labels = good + [good[0]]
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate label page"):
            self.run_prepare()
        self.labels = good
        self.labels[0]["lines"].append(copy.deepcopy(good[0]["lines"][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate supervised crop"):
            self.run_prepare()

    def test_identical_pixels_cannot_cross_splits(self):
        self.samples[-2]["out"] = self.samples[0]["out"]
        self.save()
        with self.assertRaisesRegex(ValueError, "Identical Out occurs across"):
            self.run_prepare()

    def test_existing_dataset_is_not_overwritten_with_changed_labels(self):
        self.run_prepare()
        original = (self.output / "train.txt").read_bytes()
        self.labels[0]["lines"][0]["text"] = "Different supervision."
        self.save()
        with self.assertRaisesRegex(ValueError, "different dataset"):
            self.run_prepare()
        self.assertEqual((self.output / "train.txt").read_bytes(), original)

    def test_stage1_split_and_valid_labels_retained_exactly(self):
        text = "This full text remains unchanged."
        self.labels[0]["lines"][0]["text"] = text
        self.save()
        result = self.run_prepare()
        self.assertEqual(result["max_text_length"], len(text))
        self.assertIn(text, (self.output / "train.txt").read_text())
        self.assertEqual(result["training_ctc_steps"], 40)
        self.assertEqual({p["stage1_split"] for p in result["pages"]}, {"test"})

    def test_long_or_repeated_labels_rejected_without_truncation(self):
        for text in ("A long label must have its crop split at word boundaries.", "a" * 21):
            with self.subTest(text=text):
                self.labels[0]["lines"][0]["text"] = text
                self.save()
                with self.assertRaisesRegex(ValueError, "40 CTC steps.*word-boundary"):
                    self.run_prepare()
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
