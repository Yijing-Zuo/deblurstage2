"""CPU layout checks: copyable text, evidence identity, and explicit overflow."""

import tempfile
import unittest
from pathlib import Path

import fitz
from PIL import Image

from common import source_hashes, write_jsonl
from render import body_text, filename, render, wrap_text


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("blur", "out", "clear"):
            Image.new("RGB", (512, 768), "white").save(self.root / f"{name}.png")
        self.sample = {"id": "4_004", "document_id": "4", "split": "test", "deblur_run": "fixture",
                       "blur": str(self.root / "blur.png"), "out": str(self.root / "out.png")}
        self.sample_path, self.recovery_path = self.root / "samples.jsonl", self.root / "recovery.jsonl"
        write_jsonl(self.sample_path, [self.sample])
        self.record = {"schema_version": 2, "stage": "recovery", "id": "4_004", "document_id": "4",
                       "source_hashes": source_hashes(self.sample), "status": "ok", "page_size": [512, 768],
                       "lines": [self.line("b", 1, 70, "Numbers 0-9, I/l and O/0 are retained."),
                                 self.line("a", 0, 40, "English letters, punctuation: (yes)!" )]}

    @staticmethod
    def line(line_id, order, y, text):
        return {"line_id": line_id, "order": order, "column": 0, "box": [30, y, 480, y + 18], "text": text, "status": "ok"}

    def run_render(self, **kwargs):
        write_jsonl(self.recovery_path, [self.record])
        return render(self.sample_path, self.recovery_path, self.root / "render", **kwargs)

    def test_copyable_text_reading_order_and_identical_png_dimensions(self):
        report = self.run_render()
        with fitz.open(self.root / "render/recovered.pdf") as document:
            self.assertEqual(len(document), 1)
            self.assertEqual(document[0].get_text().splitlines(), [self.record["lines"][1]["text"], self.record["lines"][0]["text"]])
            pixmap = document[0].get_pixmap(alpha=False)
            with Image.open(self.root / "render/recovered/4_004.png") as image:
                self.assertEqual(image.size, (512, 768))
                self.assertEqual(image.tobytes(), pixmap.samples)
        self.assertEqual(report["samples"][0]["comparison_columns"], 3)

    def test_matching_clear_is_fourth_panel_input_only(self):
        clear_path = self.root / "clear.jsonl"
        write_jsonl(clear_path, [{"id": "4_004", "clear": "clear.png"}])
        report = self.run_render(clear_path=clear_path)
        self.assertEqual(report["samples"][0]["comparison_columns"], 4)
        with fitz.open(self.root / "render/comparison.pdf") as document:
            text = document[0].get_text()
            for header in ("Clear", "Output", "Blur", "Recovered"):
                self.assertIn(header, text)
            self.assertIn(self.record["lines"][0]["text"], text)

    def test_multiple_samples_with_overflow_keep_comparison_page_mapping(self):
        samples, records = [], []
        for index in range(3):
            sample_id = f"4_{index:03d}"
            samples.append({**self.sample, "id": sample_id})
            lines = [self.line(f"{sample_id}:first", 0, 40, f"Unique sample number {index}.")]
            if index == 1:
                lines.append(self.line(f"{sample_id}:long", 1, 70, "longword " * 300))
            records.append({**self.record, "id": sample_id, "lines": lines})
        write_jsonl(self.sample_path, samples)
        write_jsonl(self.recovery_path, records)
        clear_path = self.root / "clear.jsonl"
        write_jsonl(clear_path, [{"id": s["id"], "clear": "clear.png"} for s in samples])
        for columns, references in ((3, None), (4, clear_path)):
            with self.subTest(columns=columns):
                output = self.root / f"render-{columns}"
                report = render(self.sample_path, self.recovery_path, output, clear_path=references)
                self.assertTrue(report["samples"][1]["overflow_pages"])
                with fitz.open(output / "recovered.pdf") as recovered, fitz.open(output / "comparison.pdf") as comparison:
                    self.assertEqual(len(comparison), len(samples))
                    self.assertGreater(len(recovered), len(samples))
                    for index, item in enumerate(report["samples"]):
                        primary = recovered[item["page"] - 1]
                        x = 12 + (columns - 1) * (512 + 12)
                        panel = fitz.Rect(x, 40, x + 512, 40 + 768)
                        text = comparison[index].get_text(clip=panel)
                        self.assertEqual(text, primary.get_text())
                        self.assertIn(f"Unique sample number {index}.", text)
                        with Image.open(output / item["images"][0]) as image:
                            self.assertEqual(image.tobytes(), primary.get_pixmap(alpha=False).samples)

    def test_mismatched_clear_id_or_dimensions_is_rejected(self):
        path = self.root / "clear.jsonl"
        write_jsonl(path, [{"id": "14_001", "clear": "clear.png"}])
        with self.assertRaisesRegex(ValueError, "Clear reference id"):
            self.run_render(clear_path=path)
        Image.new("RGB", (256, 256), "white").save(self.root / "clear.png")
        write_jsonl(path, [{"id": "4_004", "clear": "clear.png"}])
        with self.assertRaisesRegex(ValueError, "Page size mismatch"):
            self.run_render(clear_path=path)

    def test_hash_mismatch_or_legacy_schema_is_rejected(self):
        self.record["source_hashes"]["out"] = "different image"
        with self.assertRaisesRegex(ValueError, "source sample"):
            self.run_render()
        self.record["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "v2 recovery"):
            self.run_render()

    def test_non_ascii_is_never_silently_deleted(self):
        self.record["lines"][0]["text"] = "foo\u03b1bar"
        with self.assertRaisesRegex(ValueError, "ASCII"):
            self.run_render()
        self.assertEqual(body_text("A-Z 0123456789, !?;:()[]'\""), "A-Z 0123456789, !?;:()[]'\"")

    def test_long_text_goes_to_explicit_overflow_pages_without_dropping_letters(self):
        text = " ".join(f"word{i}" for i in range(300))
        self.record["lines"] = [self.line("long", 0, 40, text)]
        report = self.run_render()
        row = report["samples"][0]
        self.assertEqual(row["status"], "review")
        self.assertEqual(row["overflow"][0]["text"], text)
        self.assertGreater(len(row["overflow_pages"]), 0)
        with fitz.open(self.root / "render/recovered.pdf") as document:
            self.assertIn("overflow", document[0].get_text())
            recovered = " ".join(document[index - 1].get_text() for index in row["overflow_pages"])
            for word in text.split():
                self.assertIn(word, recovered.split())
        self.assertEqual(len(row["images"]), 1 + len(row["overflow_pages"]))

    def test_unbroken_gibberish_wrap_preserves_all_letters(self):
        text = "abcdefghijk" * 100
        lines = wrap_text(fitz.Font("helv"), text, 100, 10)
        self.assertEqual("".join(lines), text)
        self.assertTrue(all(fitz.Font("helv").text_length(line, fontsize=10) <= 100 for line in lines))

    def test_incomplete_is_refused_unless_explicitly_requested(self):
        self.record["status"] = "error"
        self.record["lines"] = []
        with self.assertRaisesRegex(ValueError, "Incomplete recovery"):
            self.run_render()
        report = self.run_render(allow_incomplete=True)
        self.assertEqual(report["samples"][0]["status"], "error")
        with fitz.open(self.root / "render/recovered.pdf") as document:
            self.assertIn("No recovered lines", document[0].get_text())

    def test_outside_box_and_duplicate_lines_are_rejected(self):
        self.record["lines"][0]["box"][2] = 999
        with self.assertRaisesRegex(ValueError, "outside source"):
            self.run_render()
        self.record["lines"][0]["box"][2] = 480
        self.record["lines"][0]["line_id"] = "a"
        with self.assertRaisesRegex(ValueError, "duplicate line"):
            self.run_render()

    def test_filename_never_escapes_output(self):
        for name in ("../x", "CON", "name.", "\u4e2d\u6587", "/tmp/file", "a\\b"):
            self.assertRegex(filename(name), r"^sample-[0-9a-f]{16}$")
        self.assertEqual(filename("4_004"), "4_004")


if __name__ == "__main__":
    unittest.main()
