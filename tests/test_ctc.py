"""CTC correctness against explicit path enumeration, without a model runtime."""
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ctc import (allowed_text, ctc_log_probabilities, ctc_log_probability,
                 greedy_decode, load_evidence, prefix_beam_search)


def enumerate_paths(probs, alphabet, blank):
    result = {}
    for path in itertools.product(range(len(alphabet)), repeat=len(probs)):
        text = "".join(alphabet[value] for i, value in enumerate(path)
                       if value != blank and (i == 0 or path[i - 1] != value))
        probability = float(np.prod([probs[i, value] for i, value in enumerate(path)]))
        result[text] = result.get(text, 0) + probability
    return result


class CTCTests(unittest.TestCase):
    def test_forward_matches_path_sums_with_nonzero_blank_index(self):
        probs = np.array([[.2, .6, .1], [.5, .2, .2], [.1, .3, .4], [.3, .4, .1]])
        alphabet = ["a", "", "b"]
        expected = enumerate_paths(probs, alphabet, 1)
        texts = list(expected) + ["aaaa", "c"]
        got = ctc_log_probabilities(probs, alphabet, texts, blank_id=1, batch_size=3)
        for text, score in zip(texts, got):
            self.assertAlmostEqual(np.exp(score), expected.get(text, 0), places=12, msg=text)

    def test_prefix_beam_sums_repeated_and_blank_paths(self):
        probs = np.array([[.2, .5, .3], [.3, .4, .3], [.1, .6, .3]])
        alphabet = ["", "a", "b"]
        expected = enumerate_paths(probs, alphabet, 0)
        beams = prefix_beam_search(probs, alphabet, beam_width=1000, top_k=1000, token_top_k=3)
        self.assertEqual({item["text"] for item in beams}, {text for text, prob in expected.items() if prob > 0})
        for item in beams:
            self.assertAlmostEqual(np.exp(item["score"]), expected[item["text"]], places=12)

    def test_zero_probabilities_stay_impossible(self):
        probs = np.array([[0, 1], [0, 1]], dtype=float)
        self.assertEqual(greedy_decode(probs, ["", "a"]), "a")
        self.assertEqual(ctc_log_probability(probs, ["", "a"], "a"), 0)
        self.assertEqual(ctc_log_probability(probs, ["", "a"], "aa"), -np.inf)
        self.assertEqual(ctc_log_probability(probs, ["", "a"], ""), -np.inf)
        separated = np.array([[0, 1], [1, 0], [0, 1]], dtype=float)
        self.assertEqual(greedy_decode(separated, ["", "a"]), "aa")
        self.assertEqual(ctc_log_probability(separated, ["", "a"], "aa"), 0)

    def test_empty_time_axis_and_forbidden_characters(self):
        probs = np.empty((0, 2))
        self.assertEqual(ctc_log_probabilities(probs, ["", "a"], ["", "a"]), [0, -np.inf])
        self.assertTrue(allowed_text("A1, 0.", "A10, ."))
        self.assertFalse(allowed_text("café", "cafe"))

    def test_load_checks_mass_without_renormalizing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.npz"
            np.savez_compressed(path, probs=np.array([[.2, .3]], dtype=np.float32),
                                alphabet=np.array(["", "a"]), blank_id=0, excluded_mass=np.array([.5]))
            loaded = load_evidence(path)
            self.assertAlmostEqual(loaded["probs"].sum(), .5)
            np.savez_compressed(path, probs=np.array([[.2, .3]]), alphabet=np.array(["", "a"]),
                                blank_id=0, excluded_mass=np.array([.1]))
            with self.assertRaises(ValueError):
                load_evidence(path)

    def test_invalid_probabilities_and_duplicate_classes_fail(self):
        for probs, alphabet in [(np.array([[1.2, -.2]]), ["", "a"]),
                                (np.array([[.6, .6]]), ["", "a"]),
                                (np.array([[.3, .3, .3]]), ["", "a", "a"])]:
            with self.assertRaises(ValueError):
                greedy_decode(probs, alphabet)


if __name__ == "__main__":
    unittest.main()
