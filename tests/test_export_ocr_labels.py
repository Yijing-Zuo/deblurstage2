"""Reference mapping must preserve complete words and exclude absent visual content."""

import unittest

from PIL import Image, ImageDraw

from export_ocr_labels import ctc_steps, mapped_box, normalize_word, supervised_phrases


class ExportOCRLabelTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 60), "white")
        ImageDraw.Draw(self.image).rectangle((0, 8, 199, 14), fill="black")
        self.reference = {"registration_parameters": {"sx": 1, "sy": 1, "tx": 0, "ty": 0},
                          "original_512x768_crop_box": [0, 0, 200, 60],
                          "stamp_removal": {"removed_logical_line_bounds_pdf": []}}
        self.audit = []

    def word(self, text, x0, x1, index=0):
        # Undo the known PDF wrapper for an intended native word at y=5..20.
        return (.96 * x0 + 4, .96 * 5, .96 * x1 + 4, .96 * 20, text, 0, 0, index)

    def phrases(self, words):
        return supervised_phrases(words, (200, 60), self.reference, self.image, self.audit)

    def test_affine_mapping_order_and_crop_offset(self):
        box = mapped_box([115.9389114, 70.6259308, 125.3852463, 82.1459274], 612,
                         {"sx": 1, "sy": .9997, "tx": .035, "ty": -.1}, [35, 4, 547, 772])
        self.assertAlmostEqual(box[0], 73.054699, places=5)
        self.assertAlmostEqual(box[1], 69.446606, places=5)

    def test_complete_words_and_tight_ink_bounds(self):
        rows = self.phrases([self.word("Hello", 10, 40), self.word("world!", 45, 80, 1)])
        self.assertEqual(rows[0]["text"], "Hello world!")
        self.assertEqual(rows[0]["box"], [9, 7, 81, 16])
        self.assertTrue(rows[0]["complete"])

    def test_partial_words_are_excluded_without_joining_across_them(self):
        rows = self.phrases([self.word("before", 5, 30), self.word("clipped", 190, 220, 1),
                             self.word("after", 40, 60, 2)])
        self.assertEqual([r["text"] for r in rows], ["before", "after"])
        self.assertEqual(self.audit[0]["reason"], "outside_or_partial_word")

    def test_unknown_letters_are_not_erased_or_silently_transliterated(self):
        self.assertIsNone(normalize_word("caf\u00e9"))
        self.assertIsNone(normalize_word("\ufffd"))
        self.assertEqual(normalize_word("\u201c\ufb01nish\u201d\u2014don't"), '"finish"-don\'t')
        rows = self.phrases([self.word("one", 5, 20), self.word("caf\u00e9", 25, 45, 1), self.word("two", 50, 70, 2)])
        self.assertEqual([r["text"] for r in rows], ["one", "two"])
        self.assertEqual(self.audit[0]["reason"], "unsupported_character")

    def test_shortphrases_use_word_boundaries_and_respect_ctc_capacity(self):
        rows = self.phrases([self.word("a" * 16, 5, 30), self.word("b" * 16, 35, 65, 1)])
        self.assertEqual([r["text"] for r in rows], ["a" * 16, "b" * 16])
        self.assertTrue(all(ctc_steps(r["text"]) <= 40 for r in rows))
        self.audit.clear()
        self.assertEqual(self.phrases([self.word("a" * 21, 5, 50)]), [])
        self.assertEqual(self.audit[0]["reason"], "word_exceeds_ctc_capacity")

    def test_watermarks_and_empty_clear_regions_never_become_labels(self):
        self.assertEqual(self.phrases([self.word("Downloaded", 5, 40), self.word("by", 45, 55, 1)]), [])
        self.assertEqual(self.audit[0]["reason"], "watermark_line")
        self.audit.clear()
        self.image = Image.new("RGB", (200, 60), "white")
        self.assertEqual(self.phrases([self.word("Invisible", 10, 60)]), [])
        self.assertEqual(self.audit[0]["reason"], "no_clear_ink")


if __name__ == "__main__":
    unittest.main()
