"""Word/span hypotheses remain local, selectable, and supported by separate views."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from candidates import (DEFAULT_ALPHABET, Lexicon, apply_choices, build_candidates,
                        score_candidate, weighted_edit)


def evidence(text, name="primary", view="out", soft=False):
    alphabet = [""] + list(DEFAULT_ALPHABET)
    probabilities = np.zeros((2 * len(text) + 1, len(alphabet)))
    probabilities[:, 0] = 1
    for i, char in enumerate(text):
        probabilities[2 * i + 1, :] = 0
        probabilities[2 * i + 1, alphabet.index(char)] = 1
    if soft:
        probabilities = .8 * probabilities + .2 / len(alphabet)
    return {"name": name, "view": view, "alphabet": alphabet, "probs": probabilities,
            "blank_id": 0, "restricted_text": text}


class FakeLexicon:
    words = {"in": 30, "the": 100, "inthe": 1, "modern": 20, "cat": 10}

    def lookup(self, word, limit=8, max_edit=3):
        return ["modern"] if word.lower() == "modem" else [word.lower()]


class CandidateTests(unittest.TestCase):
    config = {"beam_width": 2, "beam_top_k": 1, "token_top_k": 3}

    def test_every_word_has_original_even_when_ocr_agrees(self):
        plan = build_candidates([evidence("A 1999 book.")], self.config)
        self.assertEqual([span["original"] for span in plan["spans"]], ["A", "1999", "book."])
        year = plan["spans"][1]
        self.assertEqual([item["text"] for item in year["candidates"]], ["1999"])
        self.assertEqual(apply_choices(plan, {}), "A 1999 book.")

    def test_visual_space_merge_is_a_disjoint_span(self):
        sources = [evidence("in the city"), evidence("inthe city", name="english", view="blur")]
        plan = build_candidates(sources, self.config)
        span = plan["spans"][0]
        self.assertEqual(span["original"], "in the")
        candidate = next(item for item in span["candidates"] if item["text"] == "inthe")
        self.assertIsNone(candidate["scores"]["primary:out"])
        self.assertEqual(candidate["scores"]["english:blur"], 0)
        self.assertTrue(candidate["visual_support"])
        self.assertEqual(apply_choices(plan, {span["id"]: candidate["id"]}), "inthe city")

    def test_word_split_is_supported_and_keeps_punctuation(self):
        sources = [evidence("inthe city."), evidence("in the city.", name="english", view="blur")]
        plan = build_candidates(sources, self.config, FakeLexicon())
        span = plan["spans"][0]
        item = next(item for item in span["candidates"] if item["text"] == "in the")
        self.assertEqual(apply_choices(plan, {span["id"]: item["id"]}), "in the city.")

    def test_confusion_edit_is_cheaper_and_keeps_original(self):
        self.assertLess(weighted_edit("modem", "modern"), weighted_edit("modem", "model"))
        plan = build_candidates([evidence("modem", soft=True)], self.config, FakeLexicon())
        self.assertEqual(plan["spans"][0]["candidates"][0]["text"], "modem")
        self.assertIn("modern", [item["text"] for item in plan["spans"][0]["candidates"]])

    def test_proposal_can_make_more_than_two_edits(self):
        sources = [evidence("abxdzfgh", soft=True)]
        plan = build_candidates(sources, self.config)
        span = plan["spans"][0]
        candidate = score_candidate(plan, span, "recovery", sources)
        self.assertGreater(candidate["edit_cost"], 2)
        self.assertTrue(candidate["visual_support"])
        candidate["id"] = "proposal"
        span["candidates"].append(candidate)
        self.assertEqual(apply_choices(plan, {span["id"]: "proposal"}), "recovery")

    def test_invalid_characters_and_runaway_proposals_fail(self):
        sources = [evidence("cat", soft=True)]
        plan = build_candidates(sources, self.config)
        for text in ["café", "cat\ndog", "", "word " * 30]:
            with self.assertRaises(ValueError):
                score_candidate(plan, plan["spans"][0], text, sources)

    def test_overlaps_unknown_selections_and_modified_source_fail(self):
        plan = build_candidates([evidence("one two")], self.config)
        with self.assertRaises(ValueError):
            apply_choices(plan, {"unknown": "c0"})
        with self.assertRaises(ValueError):
            apply_choices(plan, {"s0": "unknown"})
        overlap = copy.deepcopy(plan)
        overlap["spans"][1]["start"] = 1
        with self.assertRaises(ValueError):
            apply_choices(overlap, {})
        changed = copy.deepcopy(plan)
        changed["spans"][0]["original"] = "fake"
        with self.assertRaises(ValueError):
            apply_choices(changed, {})

    def test_duplicate_source_keys_fail(self):
        with self.assertRaises(ValueError):
            build_candidates([evidence("cat"), evidence("dog")], self.config)

    def test_direct_blur_reading_survives_beam_budget(self):
        sources = [evidence("cat", soft=True), evidence("dog", name="english", view="blur", soft=True)]
        beams = [{"text": text, "score": -1} for text in ("cat", "bat", "car", "cab", "cap")]
        with patch("candidates.prefix_beam_search", return_value=beams):
            plan = build_candidates(sources, {**self.config, "max_candidates": 3})
        candidates = plan["spans"][0]["candidates"]
        self.assertIn("dog", [item["text"] for item in candidates])
        self.assertEqual(len(candidates), 3)

    def test_unsupported_change_cannot_be_assembled(self):
        sources = [evidence("cat")]
        plan = build_candidates(sources, self.config)
        span = plan["spans"][0]
        candidate = score_candidate(plan, span, "dog", sources)
        candidate["id"] = "guess"
        span["candidates"].append(candidate)
        self.assertFalse(candidate["visual_support"])
        with self.assertRaises(ValueError):
            apply_choices(plan, {span["id"]: "guess"})

    def test_real_word_lexicon_lookup_still_proposes_alternatives(self):
        lexicon = Lexicon.__new__(Lexicon)
        calls = []

        def lookup(word, verbosity, max_edit_distance):
            calls.append((word, verbosity, max_edit_distance))
            return [SimpleNamespace(term=term, count=10) for term in ("cat", "car", "bat")]

        lexicon.engine = SimpleNamespace(lookup=lookup)
        with patch.dict("sys.modules", {"symspellpy": SimpleNamespace(Verbosity=SimpleNamespace(ALL="all"))}):
            self.assertEqual(set(lexicon.lookup("cat")), {"car", "bat"})
        self.assertEqual(calls, [("cat", "all", 3)])


if __name__ == "__main__":
    unittest.main()
