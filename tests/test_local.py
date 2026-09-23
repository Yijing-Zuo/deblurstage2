"""CPU contracts for paired local evidence and structured generation output."""
import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from local_regions import crop_pair, parse_transcription, repeat_tail, split_regions
from restore_local import main as local_main, save_pages


class LocalRegionTests(unittest.TestCase):
    def test_cores_cover_page_once_and_context_stays_inside_page(self):
        page = Image.new("RGB", (512, 768), "white")
        regions = split_regions(page, core_height=192, context=32)
        self.assertGreater(len(regions), 1)
        cursor = 0
        for index, region in enumerate(regions):
            left, top, right, bottom = region["core_box"]
            crop_left, crop_top, crop_right, crop_bottom = region["crop_box"]
            self.assertEqual(region["index"], index)
            self.assertEqual((left, right, top), (0, page.width, cursor))
            self.assertGreater(bottom, top)
            self.assertEqual((crop_left, crop_right), (0, page.width))
            self.assertEqual(crop_top, max(0, top - 32))
            self.assertEqual(crop_bottom, min(page.height, bottom + 32))
            cursor = bottom
        self.assertEqual(cursor, page.height)

    def test_boundary_moves_out_of_text_line(self):
        page = Image.new("RGB", (512, 450), "white")
        # A nominal cut at y=192 would split this line in two.
        ImageDraw.Draw(page).rectangle((30, 183, 480, 199), fill="black")
        regions = split_regions(page, core_height=192, context=32)
        first_cut = regions[0]["core_box"][3]
        self.assertFalse(183 < first_cut < 200, first_cut)
        self.assertLessEqual(abs(first_cut - 192), 32)

    def test_short_page_is_not_padded_or_split(self):
        page = Image.new("RGB", (101, 37), "white")
        regions = split_regions(page, core_height=192, context=32)
        self.assertEqual(len(regions), 1)
        self.assertEqual(list(regions[0]["core_box"]), [0, 0, 101, 37])
        self.assertEqual(list(regions[0]["crop_box"]), [0, 0, 101, 37])

    def test_identical_paired_sources_receive_identical_crops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = Image.new("RGB", (64, 80), "white")
            ImageDraw.Draw(page).rectangle((8, 23, 56, 43), fill="black")
            paths = {key: str(root / f"{key}.png") for key in ("blur", "out")}
            for path in paths.values():
                page.save(path)
            region = {"index": 0, "core_box": [0, 24, 64, 48],
                      "crop_box": [0, 16, 64, 56]}
            pair = crop_pair(paths, region, scale=2)
            try:
                self.assertEqual(len(pair), 2)
                self.assertEqual(pair[0].size, pair[1].size)
                self.assertEqual(pair[0].tobytes(), pair[1].tobytes())
                self.assertGreaterEqual(pair[0].width, 128)
                self.assertGreaterEqual(pair[0].height, 80)
            finally:
                for image in pair:
                    image.close()

    def test_brackets_leave_aligned_visual_evidence_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, pages = {}, []
            for view, channel in (("blur", 17), ("out", 231)):
                page = Image.new("RGB", (40, 64))
                page.putdata([(x, y, channel) for y in range(64) for x in range(40)])
                paths[view] = str(root / f"{view}.png")
                page.save(paths[view])
                pages.append(page)
            region = {"index": 0, "core_box": [0, 24, 40, 40],
                      "crop_box": [0, 16, 40, 48]}
            pair = crop_pair(paths, region, scale=1)
            try:
                for source, image in zip(pages, pair):
                    gutter = (image.width - source.width) // 2
                    self.assertGreater(gutter, 0)
                    self.assertEqual(image.height, 32)
                    evidence = image.crop((gutter, 0, gutter + 40, 32))
                    self.assertEqual(evidence.tobytes(), source.crop((0, 16, 40, 48)).tobytes())
            finally:
                for image in pair + pages:
                    image.close()

    def test_pair_dimensions_and_crop_bounds_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {key: str(root / f"{key}.png") for key in ("blur", "out")}
            Image.new("RGB", (40, 64), "white").save(paths["blur"])
            Image.new("RGB", (40, 63), "white").save(paths["out"])
            region = {"index": 0, "core_box": [0, 24, 40, 40],
                      "crop_box": [0, 16, 40, 48]}
            with self.assertRaises(ValueError):
                crop_pair(paths, region, scale=1)
            Image.new("RGB", (40, 64), "white").save(paths["out"])
            for crop_box in ([0, -1, 40, 48], [0, 16, 40, 65], [0, 25, 40, 48]):
                with self.subTest(crop_box=crop_box), self.assertRaises(ValueError):
                    crop_pair(paths, {**region, "crop_box": crop_box}, scale=1)


class LocalOutputTests(unittest.TestCase):
    def test_transcription_keeps_lines_and_uncertain_spans(self):
        expected = "First line.\n[unclear] and a name: O'Neill."
        self.assertEqual(parse_transcription(json.dumps({"text": expected})), expected)
        fenced = '```json\n' + json.dumps({"text": expected}) + '\n```'
        self.assertEqual(parse_transcription(fenced), expected)

    def test_explanations_and_malformed_payloads_are_not_transcriptions(self):
        invalid = [
            'I cannot read this image. {"text": "[unclear]"}',
            '{"text": "First"}\n{"text": "Second"}',
            '{"text": ["First"]}',
            '{"text": null}',
            '{"text": "First", "explanation": "I guessed it."}',
            '[{"text": "First"}]',
            '{"transcription": "First"}',
            '```json\n{"text": "First"}\n```\nAn extra explanation.',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_transcription(raw)

    def test_repeated_long_tail_stops_but_ordinary_repetition_does_not(self):
        for length in (12, 37, 128):
            cycle = list(range(length))
            with self.subTest(length=length):
                self.assertTrue(repeat_tail([999] + cycle * 4))
                self.assertFalse(repeat_tail(cycle * 3))
                self.assertFalse(repeat_tail(cycle * 4 + [1000]))
        self.assertFalse(repeat_tail([]))
        self.assertFalse(repeat_tail([1, 2, 3] * 4))
        self.assertFalse(repeat_tail(list(range(600))))


class LocalPageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "pages.jsonl"
        self.samples = [{"id": "4_004"}]
        self.jobs = [{"id": "4_004", "index": i, "block_id": f"4_004:{i:03d}",
                      "fingerprint": f"key{i}", "core_box": [0, i * 100, 40, (i + 1) * 100]}
                     for i in range(4)]

    def row(self, index, status="ok", text="Repeated line."):
        return {**self.jobs[index], "status": status, "text": text, "flags": []}

    def read_page(self):
        return json.loads(self.output.read_text(encoding="utf-8").splitlines()[0])

    def test_spatial_merge_preserves_genuine_repeated_text(self):
        # Arrival order is deliberately reversed; identical text in different cores is real.
        results = {"key2": self.row(2, text="Last line."),
                   "key1": self.row(1), "key0": self.row(0)}
        save_pages(self.output, self.samples, self.jobs[:3], results)
        page = self.read_page()
        self.assertEqual(page["text"], "Repeated line.\n\nRepeated line.\n\nLast line.")
        self.assertEqual(page["status"], "ok")
        self.assertEqual(page["blocks_completed"], 3)
        self.assertEqual([b["block_id"] for b in page["blocks"]],
                         [job["block_id"] for job in self.jobs[:3]])

    def test_failed_and_pending_cores_keep_their_place_without_raw_essays(self):
        results = {"key0": self.row(0, text="First line.")}
        for index, status in ((1, "format_error"), (2, "loop")):
            results[f"key{index}"] = {**self.jobs[index], "status": status,
                                      "raw": "An invented essay which is not a transcription."}
        save_pages(self.output, self.samples, self.jobs, results)
        page = self.read_page()
        self.assertEqual(page["text"],
                         "First line.\n\n[unread block]\n\n[unread block]\n\n[pending block]")
        self.assertEqual(page["status"], "pending")
        self.assertEqual((page["blocks_completed"], page["blocks_total"]), (3, 4))
        self.assertNotIn("invented essay", self.output.with_suffix(".md").read_text(encoding="utf-8"))

    def test_resume_reuses_latest_ok_and_review_without_loading_model(self):
        history = [self.row(0, text="Obsolete result."),
                   self.row(0, status="review", text="Latest [unclear]."),
                   self.row(1, text="Next block.")]
        self.output.with_suffix(".blocks.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")
        argv = ["restore_local.py", "--samples", "unused.jsonl", "--output", str(self.output)]
        with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO), \
                patch("restore_local.load_config", return_value={"max_new_tokens": 32768}), \
                patch("restore_local.model_settings", return_value={}), \
                patch("restore_local.load_samples", return_value=self.samples), \
                patch("restore_local.plan_jobs", return_value=self.jobs[:2]), \
                patch.dict("sys.modules", {"torch": None}):
            local_main()
        page = self.read_page()
        self.assertEqual(page["text"], "Latest [unclear].\n\nNext block.")
        self.assertEqual(page["status"], "review")
        self.assertEqual(page["blocks_completed"], 2)


if __name__ == "__main__":
    unittest.main()
