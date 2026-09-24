"""CPU checks for training data boundaries, official config and checkpoint reuse."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from train_ocr import (BASE_CONFIG, DEFAULT_CONFIG, MODEL_NAME, build_config,
                       checkpoint_prefix, inspect_data, main, upstream_directory)
from common import fingerprint


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / "data"
        (self.data / "images").mkdir(parents=True)
        self.meta = {"schema_version": 3, "adaptation_document": "4",
                     "train_pages": ["4_004"], "validation_pages": ["4_005"],
                     "evaluation_pages": ["14_004"]}
        (self.data / "metadata.json").write_text(json.dumps(self.meta), encoding="utf-8")
        self.characters = set(chr(value) for value in range(32, 127))
        for name, sample, color in (("train", "4_004", "white"), ("val", "4_005", "gray")):
            Image.new("RGB", (250, 24), color).save(self.data / f"images/{sample}_0000.png")
            (self.data / f"{name}.txt").write_text(
                f"images/{sample}_0000.png\tEnglish letters 1999.\n", encoding="utf-8")
        self.base = {
            "Global": {"character_dict_path": "dict.txt"},
            "Architecture": {"Backbone": {"name": "PPLCNetV4", "model_size": "medium"},
                             "Head": {"head_list": [{"NRTRHead": {"max_text_length": 25}}]}},
            "Optimizer": {"lr": {"name": "Cosine"}}, "Metric": {},
        }
        self.settings = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_shared_mount_owner_check_still_checks_revision_and_changes(self):
        upstream = self.root / "Paddle OCR"
        upstream.mkdir()
        subprocess.run(["git", "init", "-q", str(upstream)], check=True)
        (upstream / "tools").mkdir()
        script = upstream / "tools/train.py"
        script.write_text("# fixture\n", encoding="utf-8")
        git = ["git", "-C", str(upstream)]
        subprocess.run(git + ["config", "core.autocrlf", "false"], check=True)
        subprocess.run(git + ["add", "tools/train.py"], check=True)
        subprocess.run(git + ["-c", "user.name=OCR test", "-c", "user.email=ocr-test@example.invalid",
                             "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], check=True)
        revision = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        local_config = (upstream / ".git/config").read_bytes()
        isolated_global = self.root / "empty-global-config"
        with patch.dict(os.environ, {"GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
                                     "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": str(isolated_global)}), patch(
                "train_ocr.UPSTREAM_REVISION", revision):
            blocked = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("dubious ownership", blocked.stderr)
            self.assertEqual(upstream_directory(upstream), upstream)
            with patch("train_ocr.UPSTREAM_REVISION", "0" * 40), self.assertRaisesRegex(ValueError, "must be v3.7.0"):
                upstream_directory(upstream)
            script.write_text("# modified training code\n", encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError):
                upstream_directory(upstream)
        self.assertEqual((upstream / ".git/config").read_bytes(), local_config)
        self.assertFalse(isolated_global.exists())

    def test_full_labels_and_nrtr_special_tokens_fit(self):
        stats = inspect_data(self.data, self.characters)
        config = build_config(self.base, self.settings, self.data, stats,
                              self.root / "run", self.root / "dict.txt", self.root / "pre.pdparams")
        self.assertEqual(config["Global"]["max_text_length"], len("English letters 1999.") + 2)
        self.assertEqual(config["Train"]["dataset"]["ratio_list"], [1.0])
        self.assertFalse(config["Train"]["loader"]["drop_last"])
        self.assertEqual(config["Train"]["dataset"]["transforms"][2],
                         {"RecResizeImg": {"image_shape": [3, 48, 320], "padding": True}})
        self.assertNotIn("sampler", config["Train"])
        self.assertEqual(config["Global"]["model_name"], MODEL_NAME)

    def test_repeated_characters_count_against_ctc_capacity(self):
        (self.data / "train.txt").write_text("images/4_004_0000.png\t" + "a" * 21 + "\n")
        with self.assertRaisesRegex(ValueError, "41 CTC steps"):
            inspect_data(self.data, self.characters)

    def test_heldout_crop_cannot_be_inserted_in_training_list(self):
        (self.data / "train.txt").write_text("images/14_004_0000.png\tHello\n")
        with self.assertRaisesRegex(ValueError, "outside the declared train"):
            inspect_data(self.data, self.characters)

    def test_same_pixels_cannot_cross_train_validation(self):
        (self.data / "images/4_005_0000.png").write_bytes((self.data / "images/4_004_0000.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "Identical crop pixels"):
            inspect_data(self.data, self.characters)

    def test_unsupported_labels_are_not_silently_filtered(self):
        (self.data / "train.txt").write_text("images/4_004_0000.png\tcaf\u00e9\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported characters"):
            inspect_data(self.data, self.characters)

    def test_resume_requires_optimizer_and_epoch_state(self):
        run = self.root / "run"
        run.mkdir()
        (run / "latest.pdparams").write_bytes(b"weights")
        self.assertEqual(checkpoint_prefix("latest", run), run / "latest")
        with self.assertRaisesRegex(ValueError, "pdopt"):
            checkpoint_prefix("latest", run, resume=True)
        (run / "latest.pdopt").write_bytes(b"optimizer")
        (run / "latest.states").write_bytes(b"state")
        self.assertEqual(checkpoint_prefix("latest", run, resume=True), run / "latest")
        with self.assertRaisesRegex(ValueError, "belong to this training run"):
            checkpoint_prefix("../outside", run)

    def test_official_train_dispatch_and_resume_provenance(self):
        upstream = self.root / "PaddleOCR"
        (upstream / BASE_CONFIG).parent.mkdir(parents=True)
        (upstream / BASE_CONFIG).write_text(yaml.safe_dump(self.base), encoding="utf-8")
        (upstream / "dict.txt").write_text("\n".join(sorted(self.characters)), encoding="utf-8")
        pretrained = self.root / "pre.pdparams"
        pretrained.write_bytes(b"official training weights fixture")
        run = self.root / "run"
        args = ["train", "--paddleocr", str(upstream), "--data", str(self.data),
                "--output", str(run), "--pretrained", str(pretrained)]
        with patch("train_ocr.upstream_directory", return_value=upstream), patch("train_ocr.subprocess.run") as call:
            self.assertEqual(main(args), 0)
            self.assertIn(str(upstream / "tools/train.py"), call.call_args.args[0])
            record = json.loads((run / "stage2_training.json").read_text())
            self.assertEqual(record["dataset"]["evaluation_pages"], ["14_004"])
            for suffix in ("pdparams", "pdopt", "states"):
                (run / f"latest.{suffix}").write_bytes(b"checkpoint fixture")
            self.assertEqual(main(args + ["--resume"]), 0)
            config = yaml.safe_load((run / "resume.yml").read_text())
            self.assertIsNone(config["Global"]["pretrained_model"])
            self.assertEqual(config["Global"]["checkpoints"], str(run / "latest"))
            record = json.loads((run / "stage2_training.json").read_text())
            self.assertEqual(record["training_fingerprint"], fingerprint(record["training"]))
            self.assertIsNone(record["training"]["config"]["Global"]["checkpoints"])
            (self.data / "train.txt").write_text("images/4_004_0000.png\tChanged text.\n")
            with self.assertRaisesRegex(ValueError, "Resume data/config/code/weights differ"):
                main(args + ["--resume"])

    def test_export_dispatch_preserves_dictionary_and_split_provenance(self):
        from train_ocr import UPSTREAM_REVISION
        upstream, run, exported = self.root / "PaddleOCR", self.root / "run", self.root / "export"
        run.mkdir()
        config = {"Global": {}, "Architecture": {}}
        identity = {"config": config, "upstream_revision": UPSTREAM_REVISION}
        record = {"schema_version": 3, "dataset": self.meta, "training": identity,
                  "training_fingerprint": fingerprint(identity)}
        (run / "stage2_training.json").write_text(json.dumps(record))
        (run / "best_accuracy.pdparams").write_bytes(b"trained weights")

        def fake_export(command, **kwargs):
            self.assertIn(str(upstream / "tools/export_model.py"), command)
            export_config = yaml.safe_load((run / "export.yml").read_text())
            self.assertEqual(export_config["Global"]["checkpoints"], str(run / "best_accuracy"))
            exported.mkdir()
            for name in ("inference.json", "inference.pdiparams"):
                (exported / name).write_bytes(b"static model fixture")
            (exported / "inference.yml").write_text(yaml.safe_dump({
                "Global": {"model_name": MODEL_NAME}, "PostProcess": {"character_dict": ["a", "b"]}}))

        with patch("train_ocr.upstream_directory", return_value=upstream), patch(
                "train_ocr.subprocess.run", side_effect=fake_export):
            main(["export", "--paddleocr", str(upstream), "--run", str(run), "--output", str(exported)])
        saved = json.loads((exported / "stage2_training.json").read_text())
        self.assertEqual(saved["dataset"]["evaluation_pages"], ["14_004"])
        self.assertEqual(set(saved["export_hashes"]), {"inference.json", "inference.pdiparams", "inference.yml"})
        self.assertEqual(len(saved["checkpoint_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
