"""Structured Qwen selection contracts; no model import or inference is needed."""
import json
import unittest

import numpy as np

from candidates import DEFAULT_ALPHABET, build_candidates, score_candidate
from recover import add_proposals, parse_reply, prompt_data, recover_line


def source(text, view="out", soft=True):
    alphabet = [""] + list(DEFAULT_ALPHABET)
    probs = np.zeros((2 * len(text) + 1, len(alphabet)))
    probs[:, 0] = 1
    for index, char in enumerate(text):
        probs[2 * index + 1, :] = 0
        probs[2 * index + 1, alphabet.index(char)] = 1
    if soft:
        probs = .8 * probs + .2 / len(alphabet)
    return {"name": "test_rec", "view": view, "restricted_text": text, "raw_text": text,
            "probs": probs, "alphabet": alphabet, "blank_id": 0}


def make_plan(sources):
    return build_candidates(sources, {"beam_width": 2, "beam_top_k": 1, "token_top_k": 3})


def reply(plan, choices=None, proposals=None):
    choices = choices or {span["id"]: "c0" for span in plan["spans"]}
    return json.dumps({"choices": [{"span_id": sid, "candidate_id": cid} for sid, cid in choices.items()],
                       "proposals": [{"span_id": sid, "text": text} for sid, text in (proposals or {}).items()]})


class RecoveryTests(unittest.TestCase):
    def run_line(self, sources, plan, responses):
        calls = []

        def ask(prompt):
            calls.append(prompt)
            return {"raw": responses[len(calls) - 1], "status": "ok", "generated_tokens": 20}

        result = recover_line({}, sources, plan, [], ask, "Recover only the target line.",
                              {"proposal_max_growth": 24}, DEFAULT_ALPHABET)
        return result, calls

    def test_parse_requires_complete_unique_known_choices(self):
        plan = make_plan([source("one two")])
        valid = json.loads(reply(plan))
        cases = ["Some explanation " + reply(plan), json.dumps({**valid, "text": "new paragraph"}),
                 json.dumps({**valid, "choices": valid["choices"][:1]}),
                 json.dumps({**valid, "choices": valid["choices"] + valid["choices"][:1]}),
                 json.dumps({**valid, "choices": [{"span_id": "s0", "candidate_id": "bad"}, valid["choices"][1]]})]
        for raw in cases:
            with self.assertRaises(ValueError):
                parse_reply(raw, plan)
        choices, proposals = parse_reply("```json\n" + reply(plan) + "\n```", plan)
        self.assertEqual(choices, {"s0": "c0", "s1": "c0"})
        self.assertEqual(proposals, {})

    def test_final_round_cannot_keep_proposing(self):
        plan = make_plan([source("cat")])
        with self.assertRaises(ValueError):
            parse_reply(reply(plan, proposals={"s0": "dog"}), plan, allow_proposals=False)

    def test_prompt_does_not_leak_clear_or_impossible_candidates(self):
        sources = [{**source("cat", soft=False), "clear": "SECRET_GROUND_TRUTH"}]
        plan = make_plan(sources)
        impossible = score_candidate(plan, plan["spans"][0], "dog", sources)
        impossible["id"] = "impossible"
        plan["spans"][0]["candidates"].append(impossible)
        data = prompt_data(plan, sources, [])
        self.assertNotIn("SECRET_GROUND_TRUTH", json.dumps(data))
        self.assertNotIn("impossible", [item["id"] for item in data["spans"][0]["candidates"]])

    def test_new_proposal_is_scored_then_chosen_in_one_final_round(self):
        sources = [source("cat")]
        plan = make_plan(sources)
        new_id = f"c{len(plan['spans'][0]['candidates'])}"
        responses = [reply(plan, proposals={"s0": "dog"}), reply(plan, choices={"s0": new_id})]
        result, calls = self.run_line(sources, plan, responses)
        self.assertEqual(result["text"], "dog")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(calls), 2)
        self.assertTrue(plan["spans"][0]["candidates"][-1]["visual_support"])

    def test_impossible_and_invalid_proposals_are_rejected_locally(self):
        sources = [source("cat", soft=False)]
        for proposal in ("dog", "café", "a b c d"):
            plan = make_plan(sources)
            result, calls = self.run_line(sources, plan, [reply(plan, proposals={"s0": proposal})])
            self.assertEqual(result["text"], "cat")
            self.assertEqual(result["status"], "review", proposal)
            self.assertEqual(len(result["rejected_proposals"]), 1)
            self.assertEqual(len(calls), 1)

    def test_final_reply_failure_preserves_valid_first_selection(self):
        sources = [source("cat"), source("bat", view="blur")]
        plan = make_plan(sources)
        alternate = next(item for item in plan["spans"][0]["candidates"] if item["text"] == "bat")
        responses = [reply(plan, choices={"s0": alternate["id"]}, proposals={"s0": "car"}), "not valid JSON"]
        result, calls = self.run_line(sources, plan, responses)
        self.assertEqual(result["text"], "bat")
        self.assertEqual(result["status"], "review")
        self.assertEqual(len(calls), 2)

    def test_bad_first_reply_falls_back_without_free_text(self):
        sources = [source("cat")]
        plan = make_plan(sources)
        result, calls = self.run_line(sources, plan, ["A newly invented paragraph."])
        self.assertEqual(result["text"], "cat")
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(calls), 1)

    def test_empty_line_never_calls_qwen(self):
        sources = [source("")]
        plan = make_plan(sources)
        result, calls = self.run_line(sources, plan, [])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["status"], "error")
        self.assertIn("no_word_candidates", result["flags"])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
