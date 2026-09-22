"""CPU-only contracts: alignment, document isolation and supervised label separation."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from common import load_samples, make_prompt, source_hashes, write_jsonl
from prepare import crop_image, export_training, import_manifest, import_pairs
from train import read_training


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.blur, self.out = self.root / "blur.png", self.root / "out.png"
        Image.new("RGB", (12, 16), "white").save(self.blur)
        Image.new("RGB", (12, 16), "gray").save(self.out)
        self.sample = {"id": "1_001", "document_id": "1", "split": "train",
                       "deblur_run": "test-run", "blur": "blur.png", "out": "out.png"}
        self.samples = self.root / "samples.jsonl"
        write_jsonl(self.samples, [self.sample])

    def setup_training(self):
        row = load_samples(self.samples)[0]
        write_jsonl(self.root / "candidates.jsonl", [{"id": row["id"], "status": "ok",
                    "blur_text": "noisy A", "out_text": "noisy B",
                    "source_hashes": source_hashes(row)}])
        write_jsonl(self.root / "labels.jsonl", [{"id": row["id"], "verified": True,
                                                 "text": "REVIEWED SECRET TARGET"}])
        return argparse.Namespace(samples=self.samples, candidates=self.root / "candidates.jsonl",
                                  labels=self.root / "labels.jsonl", output=self.root / "training")

    def test_relative_paths_and_integer_zero(self):
        self.assertEqual(load_samples(self.samples)[0]["blur"], str(self.blur.resolve()))
        row = {**self.sample, "document_id": 0, "split": "demo"}
        write_jsonl(self.samples, [row])
        self.assertEqual(load_samples(self.samples)[0]["document_id"], "0")

    def test_duplicates_and_document_split_rejected(self):
        for second in [self.sample, {**self.sample, "id": "1_002", "split": "validation"}]:
            write_jsonl(self.samples, [self.sample, second])
            with self.assertRaises(ValueError):
                load_samples(self.samples)

    def test_holdouts_cannot_train(self):
        for doc in (0, 4, 14):
            write_jsonl(self.samples, [{**self.sample, "document_id": doc}])
            with self.assertRaisesRegex(ValueError, "held out"):
                load_samples(self.samples)

    def test_crop_uses_coordinates_and_rejects_padding(self):
        destination = self.root / "crop.png"
        self.assertEqual(crop_image(self.blur, [2, 3, 10, 15], destination), (8, 12))
        for box in ([0, 0, 13, 16], [2, 0, 1, 10], [0.5, 0, 2, 2]):
            with self.assertRaises(ValueError):
                crop_image(self.blur, box, destination)

    def test_manifest_pair_sizes_must_match(self):
        write_jsonl(self.samples, [{**self.sample, "blur_box": [0, 0, 6, 8]}])
        args = argparse.Namespace(manifest=self.samples, output=self.root / "prepared.jsonl", limit=None)
        with self.assertRaisesRegex(ValueError, "same region"):
            import_manifest(args)

    def test_audited_import_is_portable_and_does_not_recrop(self):
        from common import hash_file
        audited = self.root / "verified.json"
        audited.write_text(json.dumps({"records": [{"document_number": 4, "web_page": 3,
            "blur_file": "blur.png", "blur_sha256": hash_file(self.blur), "docsity_id": "test",
            "original_512x768_crop_box": [2, 3, 10, 15]}]}), encoding="utf-8")
        Image.new("RGB", (8, 12), "gray").save(self.root / "Out_4_003.png")
        args = argparse.Namespace(pairs=audited, output=self.root / "data/samples.jsonl",
            out_dir=self.root, limit=None, deblur_run="test-run")
        rows = import_pairs(args)
        self.assertNotIn("blur_box", rows[0])
        self.assertEqual(Path(rows[0]["out"]).parent, self.root / "data/images")
        write_jsonl(args.output, rows)
        load_samples(args.output)

    def test_gold_is_only_in_assistant_target(self):
        args = self.setup_training()
        export_training(args)
        row = read_training(args.output / "train.json", {"train"})[0]
        self.assertNotIn("REVIEWED SECRET TARGET", row["prompt"])
        self.assertEqual(row["conversations"][0]["value"], "<image>\n<image>\n" + make_prompt("noisy A", "noisy B"))
        self.assertEqual(row["conversations"][1]["value"], "REVIEWED SECRET TARGET")
        self.assertEqual(json.loads((args.output / "eval.json").read_text()), [])

    def test_stale_failed_and_unreviewed_inputs_rejected(self):
        args = self.setup_training()
        Image.new("RGB", (12, 16), "black").save(self.out)
        with self.assertRaisesRegex(ValueError, "stale OCR"):
            export_training(args)
        args = self.setup_training()
        with args.candidates.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": "1_001", "status": "error"}) + "\n")
        with self.assertRaisesRegex(ValueError, "failed or stale"):
            export_training(args)
        args = self.setup_training()
        write_jsonl(args.labels, [{"id": "1_001", "verified": False, "text": "guess"}])
        with self.assertRaisesRegex(ValueError, "reviewed"):
            export_training(args)

    def test_mixed_deblur_runs_rejected(self):
        args = self.setup_training()
        write_jsonl(self.samples, [self.sample, {**self.sample, "id": "2_001",
            "document_id": "2", "split": "validation", "deblur_run": "historical-run"}])
        with self.assertRaisesRegex(ValueError, "one frozen deblur run"):
            export_training(args)

    def test_changed_export_images_and_prompt_rejected(self):
        args = self.setup_training()
        export_training(args)
        training = args.output / "train.json"
        rows = json.loads(training.read_text())
        rows[0]["conversations"][0]["value"] += " and use the gold answer"
        training.write_text(json.dumps(rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "prompt differs"):
            read_training(training, {"train"})
        export_training(args)
        Image.new("RGB", (12, 16), "red").save(self.blur)
        with self.assertRaisesRegex(ValueError, "images changed"):
            read_training(training, {"train"})


if __name__ == "__main__":
    unittest.main()
