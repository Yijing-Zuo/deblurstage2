"""A killed process may leave one partial tail, never corrupt earlier records."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common import append_record, atomic_write_jsonl, fingerprint, read_journal, write_jsonl


class JournalTests(unittest.TestCase):
    def test_interrupted_utf8_tail_and_safe_append(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage.jsonl"
            path.write_bytes(b'{"id":"complete"}\n{"text":"\xe4\xb8')
            with self.assertWarns(UserWarning):
                rows = read_journal(path)
            self.assertEqual(rows, [{"id": "complete"}])
            atomic_write_jsonl(path, rows)
            with path.open("a", encoding="utf-8") as stream:
                append_record(stream, {"id": "next"})
            self.assertEqual(read_journal(path), [{"id": "complete"}, {"id": "next"}])

    def test_corruption_before_tail_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage.jsonl"
            path.write_text('{"id":}\n{"id":"complete"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record 1"):
                read_journal(path)

    def test_nonfinite_values_do_not_replace_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage.jsonl"
            atomic_write_jsonl(path, [{"score": 1}])
            for value in (float("nan"), np.float32("nan"), np.float64("inf")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    atomic_write_jsonl(path, [{"score": value}])
                self.assertEqual(read_journal(path), [{"score": 1}])

    def test_numpy_metadata_stays_numeric_in_all_writers(self):
        row = {"box": [np.int64(12), np.int32(20)], "score": np.float32(0.5),
               "nested": {"split": np.bool_(True), "large": np.uint64(2 ** 63)}}
        expected = {"box": [12, 20], "score": 0.5,
                    "nested": {"split": True, "large": 2 ** 63}}
        self.assertEqual(fingerprint(row), fingerprint(expected))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage.jsonl"
            for writer in (write_jsonl, atomic_write_jsonl):
                with self.subTest(writer=writer.__name__):
                    writer(path, [row])
                    restored = read_journal(path)[0]
                    self.assertEqual(restored, expected)
                    self.assertIs(type(restored["box"][0]), int)
                    self.assertIs(type(restored["score"]), float)
                    self.assertIs(type(restored["nested"]["split"]), bool)
            with path.open("a", encoding="utf-8") as stream:
                append_record(stream, row)
            self.assertEqual(read_journal(path), [expected, expected])

    def test_unsupported_metadata_does_not_replace_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage.jsonl"
            atomic_write_jsonl(path, [{"id": "complete"}])
            for value in (np.array([1, 2]), object(), np.complex64(1 + 2j)):
                with self.subTest(value=value), self.assertRaises(TypeError):
                    atomic_write_jsonl(path, [{"bad": value}])
                self.assertEqual(read_journal(path), [{"id": "complete"}])


if __name__ == "__main__":
    unittest.main()
