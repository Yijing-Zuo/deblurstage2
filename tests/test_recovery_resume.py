import unittest

from recover import cached_lines


class RecoveryResumeTests(unittest.TestCase):
    def test_later_failure_does_not_reuse_older_review(self):
        history = [{"fingerprint": "same", "status": "review"},
                   {"fingerprint": "same", "status": "error"}]
        self.assertEqual(cached_lines(history), {})

    def test_configurations_are_separate_and_review_retry_is_explicit(self):
        history = [{"fingerprint": "old", "status": "ok"},
                   {"fingerprint": "new", "status": "review"}]
        self.assertEqual(set(cached_lines(history)), {"old", "new"})
        self.assertEqual(set(cached_lines(history, retry_review=True)), {"old"})


if __name__ == "__main__":
    unittest.main()
