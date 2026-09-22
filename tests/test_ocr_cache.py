"""Latest-attempt OCR cache semantics; no model or GPU imports required."""
import unittest

from ocr import completed_keys


class OCRCacheTests(unittest.TestCase):
    def test_success_then_other_config_failure_requires_retry(self):
        history = [{"id": "sample", "fingerprint": "A", "status": "ok"},
                   {"id": "sample", "fingerprint": "B", "status": "error"}]
        self.assertNotIn(("sample", "A"), completed_keys(history))
        history.append({"id": "sample", "fingerprint": "A", "status": "ok"})
        self.assertIn(("sample", "A"), completed_keys(history))

    def test_only_latest_configuration_is_reusable(self):
        history = [{"id": "sample", "fingerprint": "A", "status": "ok"},
                   {"id": "sample", "fingerprint": "B", "status": "ok"},
                   {"id": "other", "fingerprint": "C", "status": "ok"}]
        self.assertEqual(completed_keys(history), {("sample", "B"), ("other", "C")})


if __name__ == "__main__":
    unittest.main()
