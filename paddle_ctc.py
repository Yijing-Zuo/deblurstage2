"""Small, version-pinned PaddleX adapter; no Paddle import until inference.

The 3.7.0 text recognizer runs pre_tfs -> runner -> CTCLabelDecode.
We retain runner outputs BEFORE argmax. These released heads already softmax.
Upstream: PaddlePaddle/PaddleX@v3.7.0/paddlex/inference/models/text_recognition/.
"""

import math
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np

from common import fingerprint, hash_file


DEFAULT_DETECTOR = {
    "name": "PP-OCRv6_medium_det", "repo": "PaddlePaddle/PP-OCRv6_medium_det",
    "revision": "8e0f56fb2ef86b461d99cfc7ac5c137738985f61",
}
DEFAULT_RECOGNIZERS = [
    {"name": "PP-OCRv6_medium_rec", "repo": "PaddlePaddle/PP-OCRv6_medium_rec",
     "revision": "e5a92bcbc5cc1b494628e458d267778f0704fd7c"},
    {"name": "en_PP-OCRv5_mobile_rec", "repo": "PaddlePaddle/en_PP-OCRv5_mobile_rec",
     "revision": "267c36e24c331595590fe7bd72bde2436fd286f2"},
]
MODEL_FILES = ("inference.json", "inference.pdiparams", "inference.yml")


def resolve_models(settings, offline=False):
    """Download exactly the static graph, weights and dictionary at immutable SHAs."""
    from huggingface_hub import snapshot_download

    resolved = []
    for spec in settings:
        if spec.get("path"):
            directory = Path(spec["path"]).resolve()
        else:
            revision = spec["revision"]
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("OCR model revisions must be full immutable commit SHAs")
            directory = Path(snapshot_download(
                spec["repo"], revision=revision, allow_patterns=list(MODEL_FILES),
                local_files_only=offline))
        hashes = {name: hash_file(directory / name) for name in MODEL_FILES}
        resolved.append({**spec, "path": str(directory), "file_hashes": hashes})
    return resolved


def restrict_probabilities(prediction, characters, allowed):
    """Keep original class probabilities/indices; never renormalize or softmax twice."""
    probabilities = np.asarray(prediction, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(characters):
        raise ValueError("CTC runner shape does not match the model's actual dictionary")
    if characters[0] != "blank":
        raise ValueError("PaddleX CTC blank must be class zero")
    if (not np.isfinite(probabilities).all() or probabilities.min() < 0
            or probabilities.max() > 1.0001
            or not np.allclose(probabilities.sum(axis=1), 1, atol=2e-4, rtol=0)):
        raise ValueError("Expected released CTC probabilities, not logits or NaNs")
    allowed = set(allowed)
    selected = [0] + [i for i, c in enumerate(characters) if i and c in allowed]
    alphabet = [""] + [characters[i] for i in selected[1:]]
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("Duplicate CTC labels require an explicit model-specific mapping")
    reduced = probabilities[:, selected].copy()
    omitted = np.ones(len(characters), dtype=bool)
    omitted[selected] = False
    # Sum excluded classes directly, avoiding cancellation of small probability mass.
    excluded = probabilities[:, omitted].sum(axis=1, dtype=np.float32)
    return {"probs": reduced, "alphabet": np.asarray(alphabet),
            "blank_id": np.asarray(0, dtype=np.int64), "excluded_mass": excluded,
            "original_class_ids": np.asarray(selected, dtype=np.int64)}


def _greedy(probabilities, characters):
    result, previous = [], -1
    for index in np.asarray(probabilities).argmax(axis=-1):
        if index != 0 and index != previous:
            result.append(characters[index])
        previous = index
    return "".join(result)


class PaddleCTC:
    """One detector and independent single-line recognizers, all on the same device."""

    def __init__(self, models, allowed, device="gpu:0", detection=None):
        if version("paddlex") != "3.7.0":
            raise RuntimeError("This internal CTC adapter requires paddlex==3.7.0")
        # Explicit model_dir already points to our pinned HF snapshot. Do not probe
        # PaddleX mirror availability or accidentally download a different revision.
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddlex import create_predictor

        self.allowed = allowed
        self.models = models
        self.detector = create_predictor(
            models[0]["name"], model_dir=models[0]["path"], device=device,
            engine="paddle_static", batch_size=1, **(detection or {}))
        self.recognizers = []
        for spec in models[1:]:
            predictor = create_predictor(
                spec["name"], model_dir=spec["path"], device=device,
                engine="paddle_static", batch_size=1)
            if not all(hasattr(predictor, attr) for attr in ("runner", "pre_tfs", "post_op")):
                raise RuntimeError("Unexpected PaddleX recognizer interface")
            characters = list(predictor.post_op.character)
            if not set(allowed).intersection(characters):
                raise ValueError(f"{spec['name']} has no configured text characters")
            self.recognizers.append((spec, predictor, characters))

    def detect(self, path):
        results = list(self.detector.predict(str(path)))
        if len(results) != 1:
            raise RuntimeError("Expected exactly one detection result per page")
        result = results[0]
        if len(result["dt_polys"]) != len(result["dt_scores"]):
            raise RuntimeError("Detector polygons and scores do not match")
        return [{"polygon": np.asarray(p).tolist(), "score": float(s)}
                for p, s in zip(result["dt_polys"], result["dt_scores"])]

    def recognize(self, path):
        for spec, predictor, characters in self.recognizers:
            # Read uses each model's own BGR/RGB config. Pass the path, not an
            # ambiguous numpy RGB array which PaddleX would interpret as BGR.
            image = predictor.pre_tfs["Read"](imgs=[str(path)])[0]
            normalized = predictor.pre_tfs["ReisizeNorm"](imgs=[image])
            inputs = predictor.pre_tfs["ToBatch"](imgs=normalized)
            outputs = predictor.runner(x=inputs)
            probabilities = np.asarray(outputs[0])
            if probabilities.ndim != 3 or probabilities.shape[0] != 1:
                raise RuntimeError("Expected one [1,T,C] CTC output; refusing implicit reordering")
            values = restrict_probabilities(probabilities[0], characters, self.allowed)
            shape = normalized[0].shape
            resize = predictor.pre_tfs["ReisizeNorm"]
            content_width = min(shape[2], math.ceil(shape[1] * image.shape[1] / image.shape[0]))
            yield {"name": spec["name"], "revision": spec.get("revision"),
                   "alphabet_hash": fingerprint(characters),
                   "missing_characters": sorted(set(self.allowed) - set(characters)),
                   "raw_text": _greedy(probabilities[0], characters),
                   "restricted_text": _greedy(values["probs"], values["alphabet"]),
                   "crop_shape": list(image.shape), "resized_height": shape[1],
                   "resized_content_width": content_width, "canvas_width": shape[2],
                   "padding_right": shape[2] - content_width,
                   "width_compressed": shape[1] * image.shape[1] / image.shape[0] > resize.max_imgW,
                   "time_steps": values["probs"].shape[0], "blank_id": 0,
                   "mean_excluded_mass": float(values["excluded_mass"].mean())}, values
