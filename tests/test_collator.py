"""CPU checks of masking/control flow, not integration tests of Torch or Qwen."""
import sys
import types
import unittest
from unittest.mock import patch

from train import ParagraphCollator


class Tensor:
    """Only the five operations used by ParagraphCollator; no numerical model."""
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(rows), len(rows[0]))

    def clone(self):
        return Tensor(self.rows)

    def __getitem__(self, key):
        row_slice, column_slice = key
        return Tensor([row[column_slice] for row in self.rows[row_slice]])

    def __setitem__(self, key, value):
        row_slice, column_slice = key
        for row in self.rows[row_slice]:
            for index in range(*column_slice.indices(len(row))):
                row[index] = value


class Processor:
    def __init__(self, full=None, image_count=2):
        self.full = [10, 11, 12, 20, 21, 99] if full is None else full
        self.image_count = image_count
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        tokens = [10, 11, 12] if kwargs["add_generation_prompt"] else self.full
        return {"input_ids": Tensor([tokens]), "image_grid_thw": [[1, 1, 1]] * self.image_count}


class CollatorTests(unittest.TestCase):
    def setUp(self):
        self.torch = patch.dict(sys.modules, {"torch": types.SimpleNamespace(
            equal=lambda left, right: left.rows == right.rows)})
        self.images = patch("train.image_messages", return_value=[{"role": "user", "content": []}])
        self.torch.start()
        self.images.start()
        self.addCleanup(self.torch.stop)
        self.addCleanup(self.images.stop)
        self.row = {"id": "sample", "image": ["blur.png", "out.png"], "prompt": "read",
                    "conversations": [{"from": "human", "value": "read"},
                                      {"from": "gpt", "value": "truth"}]}

    def test_only_assistant_and_ending_tokens_have_loss(self):
        processor = Processor()
        batch = ParagraphCollator(processor, 20)([self.row])
        self.assertEqual(batch["labels"].rows, [[-100, -100, -100, 20, 21, 99]])
        self.assertEqual(batch["input_ids"].rows, [[10, 11, 12, 20, 21, 99]])
        self.assertTrue(all(call["truncation"] is False for call in processor.calls))

    def test_prefix_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "prefix mismatch"):
            ParagraphCollator(Processor([10, 88, 12, 20]), 20)([self.row])

    def test_overlength_fails_without_cutting(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            ParagraphCollator(Processor(), 5)([self.row])

    def test_empty_target_token_span_fails(self):
        with self.assertRaisesRegex(ValueError, "empty assistant"):
            ParagraphCollator(Processor([10, 11, 12]), 20)([self.row])

    def test_batch_above_one_fails(self):
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            ParagraphCollator(Processor(), 20)([self.row, self.row])

    def test_missing_second_image_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            ParagraphCollator(Processor(image_count=1), 20)([self.row])

    def test_extra_image_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            ParagraphCollator(Processor(image_count=3), 20)([self.row])


if __name__ == "__main__":
    unittest.main()
