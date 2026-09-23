"""Finite visual/lexical alternatives anchored to disjoint source-string spans."""
import hashlib
import re
import string
from difflib import SequenceMatcher
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from ctc import allowed_text, ctc_log_probabilities, greedy_decode, prefix_beam_search

DEFAULT_ALPHABET = string.ascii_letters + string.digits + string.punctuation + " "
CONFUSIONS = (("rn", "m"), ("cl", "d"), ("vv", "w"), ("1", "l"), ("1", "I"), ("0", "O"))


class Lexicon:
    def __init__(self, path):
        from symspellpy import SymSpell

        self.path = str(Path(path).resolve())
        self.identity = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.engine = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        if not self.engine.load_dictionary(self.path, term_index=0, count_index=1):
            raise ValueError(f"Could not load word/count lexicon: {path}")
        self.words = self.engine.words

    @lru_cache(maxsize=8192)
    def lookup(self, word, limit=8, max_edit=3):
        from symspellpy import Verbosity

        found = self.engine.lookup(word.lower(), Verbosity.ALL, max_edit_distance=min(max_edit, 3))
        found = [item for item in found if item.term != word.lower()]
        return [item.term for item in sorted(found, key=lambda item: (weighted_edit(word.lower(), item.term), -item.count))[:limit]]


def load_lexicon(path=None):
    """Load once per run. identity is the content hash used by recovery caches."""
    if path is None:
        try:
            path = files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
        except ModuleNotFoundError as error:
            raise RuntimeError("Install requirements-recovery.txt in deblur-qwen to enable the English lexicon") from error
    return Lexicon(path)


def weighted_edit(source, target):
    """Character edit distance with explicit low-cost multi-character OCR confusions."""
    costs = np.zeros((len(source) + 1, len(target) + 1))
    costs[:, 0] = np.arange(len(source) + 1)
    costs[0, :] = np.arange(len(target) + 1)
    pairs = CONFUSIONS + tuple((b, a) for a, b in CONFUSIONS)
    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            a, b = source[i - 1], target[j - 1]
            substitution = 0 if a == b else (0.2 if a.lower() == b.lower() else 1)
            costs[i, j] = min(costs[i - 1, j] + 1, costs[i, j - 1] + 1, costs[i - 1, j - 1] + substitution)
            for before, after in pairs:
                if source[:i].endswith(before) and target[:j].endswith(after):
                    costs[i, j] = min(costs[i, j], costs[i - len(before), j - len(after)] + 0.35)
    return float(costs[-1, -1])


def source_key(source):
    return str(source.get("name", source.get("recognizer", "ocr"))) + ":" + str(source.get("view", "unknown"))


def _case_like(word, original):
    return word.upper() if original.isupper() else word.capitalize() if original.istitle() else word


def _word_alternatives(text, lexicon, config):
    result = []
    # Pure page numbers and years are not converted into letters.
    if not any(char.isalpha() for char in text):
        return result
    for before, after in CONFUSIONS + tuple((b, a) for a, b in CONFUSIONS):
        for match in re.finditer(re.escape(before), text):
            result.append((text[:match.start()] + after + text[match.end():], "character_confusion"))
    if lexicon is None:
        return result
    match = re.fullmatch(r"([^A-Za-z0-9]*)([A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*)([^A-Za-z0-9]*)", text)
    if match:
        left, word, right = match.groups()
        for item in lexicon.lookup(word, limit=config.get("lexicon_candidates", 8), max_edit=config.get("lexicon_max_edit", 3)):
            result.append((left + _case_like(item, word) + right, "lexicon"))
        for i in range(1, len(word)):
            if word[:i].lower() in lexicon.words and word[i:].lower() in lexicon.words:
                result.append((left + word[:i] + " " + word[i:] + right, "word_split"))
    elif " " in text and text.replace(" ", "").lower() in lexicon.words:
        result.append((text.replace(" ", ""), "word_merge"))
    return result


def _alignment(base, other):
    boundaries, changed = {}, []
    for tag, i, end, j, stop in SequenceMatcher(None, base, other, autojunk=False).get_opcodes():
        boundaries[i], boundaries[end] = j, stop
        if tag == "equal":
            boundaries.update((i + offset, j + offset) for offset in range(end - i + 1))
        else:
            changed.append((i, end))
    return boundaries, changed


def _validate_plan(plan):
    base = plan["base_text"]
    previous, ids = 0, set()
    for span in plan["spans"]:
        start, end = span["start"], span["end"]
        if not isinstance(start, int) or not isinstance(end, int) or not previous <= start < end <= len(base):
            raise ValueError("Candidate spans must be ordered, non-overlapping source intervals")
        if span["id"] in ids or span["original"] != base[start:end]:
            raise ValueError("Candidate span ID or source text does not match")
        ids.add(span["id"])
        previous = end


def _replacement_line(plan, span, text):
    if not allowed_text(text, plan["alphabet"]) or "\n" in text or "\r" in text:
        raise ValueError("Candidate contains characters outside the configured line alphabet")
    if not text.strip():
        raise ValueError("A local candidate cannot silently erase a source span")
    # A proposal is local, but may repair arbitrary edits inside the local span.
    max_length = max(24, 3 * len(span["original"]) + 12)
    if len(text) > max_length or len(text.split()) > max(3, len(span["original"].split()) + 2):
        raise ValueError("Candidate exceeds the local word/span budget")
    return plan["base_text"][:span["start"]] + text + plan["base_text"][span["end"]:]


def _score_entries(plan, entries, sources):
    lines = [_replacement_line(plan, span, candidate["text"]) for span, candidate in entries]
    unique = list(dict.fromkeys(lines))
    for source in sources:
        key = source_key(source)
        scores = dict(zip(unique, ctc_log_probabilities(source["probs"], source["alphabet"], unique, source.get("blank_id", 0))))
        for (_, candidate), line in zip(entries, lines):
            score = scores[line]
            candidate["scores"][key] = float(score) if np.isfinite(score) else None
    for _, candidate in entries:
        candidate["visual_support"] = any(score is not None for score in candidate["scores"].values())


def score_candidate(plan, span, text, sources):
    """Score a proposed local reading in its complete source line, with no hard veto per view."""
    _validate_plan(plan)
    if not any(item is span or item == span for item in plan["spans"]):
        raise ValueError("Unknown candidate span")
    candidate = {"text": text, "scores": {}, "edit_cost": weighted_edit(span["original"], text), "origin": "qwen_proposal"}
    _score_entries(plan, [(span, candidate)], sources)
    return candidate


def build_candidates(sources, config, lexicon=None):
    if not sources:
        raise ValueError("At least one CTC source is required")
    alphabet = config.get("alphabet", DEFAULT_ALPHABET)
    keys = [source_key(source) for source in sources]
    if len(set(keys)) != len(keys):
        raise ValueError("CTC source recognizer/view keys must be unique")
    texts, alternatives = [], []
    for source in sources:
        text = source.get("restricted_text")
        if text is None:
            text = greedy_decode(source["probs"], source["alphabet"], source.get("blank_id", 0))
        if not allowed_text(text, alphabet):
            raise ValueError("Restricted OCR text does not match the configured alphabet")
        texts.append(text)
        alternatives.append((text, source_key(source)))
        for beam in prefix_beam_search(source["probs"], source["alphabet"], source.get("blank_id", 0),
                                       beam_width=config.get("beam_width", 8), top_k=config.get("beam_top_k", 4),
                                       token_top_k=config.get("token_top_k", 12)):
            if allowed_text(beam["text"], alphabet):
                alternatives.append((beam["text"], "ctc_beam:" + source_key(source)))
    # Stable, explicit anchor; evidence from the other image is never discarded.
    base_index = next((i for i, source in enumerate(sources) if source.get("view") == "out" and texts[i].strip()), None)
    if base_index is None:
        base_index = next((i for i, text in enumerate(texts) if text.strip()), 0)
    base = texts[base_index]
    plan = {"base_text": base, "base_source": keys[base_index], "alphabet": alphabet, "spans": []}
    tokens = [(match.start(), match.end()) for match in re.finditer(r"\S+", base)]
    intervals = list(tokens)
    aligned = [(other, origin, *_alignment(base, other)) for other, origin in alternatives if other != base]
    max_words = config.get("span_max_words", 3)
    expansions = []
    for _, _, _, edits in aligned:
        for start, end in edits:
            touched = [(a, b) for a, b in tokens if a < end and b > start]
            if not touched:  # Inserting/deleting an existing word boundary.
                touched = [(a, b) for a, b in tokens if b == start or a == end or a < start < b]
            if 1 < len(touched) <= max_words:
                expansions.append((touched[0][0], touched[-1][1]))
    if lexicon is not None:
        for (a, b), (c, d) in zip(tokens, tokens[1:]):
            if base[a:b].isalpha() and base[c:d].isalpha() and (base[a:b] + base[c:d]).lower() in lexicon.words:
                expansions.append((a, d))
    for start, end in expansions:
        overlaps = [(a, b) for a, b in intervals if a < end and b > start]
        if not overlaps:
            continue
        left, right = min(start, overlaps[0][0]), max(end, overlaps[-1][1])
        if sum(a >= left and b <= right for a, b in tokens) <= max_words:
            intervals = sorted([item for item in intervals if item not in overlaps] + [(left, right)])
    budget = max(2, int(config.get("max_candidates", 6)))
    for start, end in intervals:
        original = base[start:end]
        span = {"id": f"s{len(plan['spans'])}", "start": start, "end": end, "original": original, "candidates": []}
        pool = {original: "original"}
        for other, origin, boundaries, _ in aligned:
            if start in boundaries and end in boundaries:
                value = other[boundaries[start]:boundaries[end]]
                if value.strip():
                    # Prefer a directly observed source reading over an earlier
                    # beam variant with the same text; preserve view diversity.
                    if value not in pool or (pool[value].startswith("ctc_beam:") and not origin.startswith("ctc_beam:")):
                        pool[value] = origin
        for value, origin in _word_alternatives(original, lexicon, config):
            pool.setdefault(value, origin)
        candidates = []
        for value, origin in pool.items():
            try:
                _replacement_line(plan, span, value)
            except ValueError:
                continue
            candidates.append({"text": value, "origin": origin, "edit_cost": weighted_edit(original, value), "scores": {}})
        # Keep directly observed readings from Blur and the other recognizer
        # before spending the remaining budget on near-identical beam variants.
        direct = sorted([item for item in candidates if ":" in item["origin"] and not item["origin"].startswith("ctc_beam:")], key=lambda item: item["edit_cost"])
        visual = sorted([item for item in candidates if item["origin"].startswith("ctc_beam:")], key=lambda item: item["edit_cost"])
        lexical = sorted([item for item in candidates if item["origin"] != "original" and ":" not in item["origin"]], key=lambda item: item["edit_cost"])
        selected = [item for item in candidates if item["origin"] == "original"]
        selected.extend(direct[:max(0, budget - len(selected))])
        while len(selected) < budget and (visual or lexical):
            for group in (lexical, visual):
                if group and len(selected) < budget:
                    selected.append(group.pop(0))
        for i, candidate in enumerate(selected):
            candidate["id"] = f"c{i}"
        span["candidates"] = selected
        plan["spans"].append(span)
    _validate_plan(plan)
    _score_entries(plan, [(span, item) for span in plan["spans"] for item in span["candidates"]], sources)
    return plan


def apply_choices(plan, choices):
    """Assemble edits once, by original offsets. Unchosen spans retain their source text."""
    _validate_plan(plan)
    known = {span["id"] for span in plan["spans"]}
    if set(choices) - known:
        raise ValueError("Selection references an unknown span")
    result, cursor = [], 0
    for span in plan["spans"]:
        result.append(plan["base_text"][cursor:span["start"]])
        value = span["original"]
        if span["id"] in choices:
            found = [item for item in span["candidates"] if item["id"] == choices[span["id"]]]
            if len(found) != 1:
                raise ValueError("Selection references an unknown or duplicate candidate")
            value = found[0]["text"]
            if value != span["original"] and not found[0].get("visual_support", False):
                raise ValueError("Selected change has no feasible CTC path in any source")
            _replacement_line(plan, span, value)
        result.append(value)
        cursor = span["end"]
    result.append(plan["base_text"][cursor:])
    text = "".join(result)
    if not allowed_text(text, plan["alphabet"]):
        raise ValueError("Assembled line violates the character alphabet")
    return text
