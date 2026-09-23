"""A killed process may leave one partial tail, never corrupt earlier records."""
import tempfile
import unittest
from pathlib import Path

from common import append_record, atomic_write_jsonl, read_journal


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
            with self.assertRaises(ValueError):
                atomic_write_jsonl(path, [{"score": float("nan")}])
            self.assertEqual(read_journal(path), [{"score": 1}])


if __name__ == "__main__":
    unittest.main()
