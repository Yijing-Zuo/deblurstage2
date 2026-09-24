"""Prepare, run and export the pinned official PaddleOCR recognizer trainer.

Only Out crops and ground-truth labels enter training. No detector or Qwen is
trained. The upstream trainer owns optimizer, validation and checkpoints.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

from PIL import Image
import yaml

from common import fingerprint, hash_file

UPSTREAM_REVISION = "b03f46425e8ff4442b268ce449e3eef758146cd4"  # v3.7.0
BASE_CONFIG = "configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml"
PRETRAINED_URL = ("https://paddle-model-ecology.bj.bcebos.com/paddlex/"
                  "official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams")
DEFAULT_CONFIG = Path(__file__).parent / "configs/ocr_out.yml"
MODEL_NAME = "PP-OCRv6_medium_rec"


def upstream_directory(path):
    path = Path(path).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"PaddleOCR must be v3.7.0 at {UPSTREAM_REVISION}")
    subprocess.run(["git", "-C", str(path), "diff", "--exit-code", "HEAD", "--",
                    "ppocr", "tools", "configs"], check=True, stdout=subprocess.DEVNULL)
    return path


def inspect_data(directory, characters):
    """Validate complete short-phrase labels; do not silently drop rows."""
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    page_sets = [set(metadata[key]) for key in
                 ("train_pages", "validation_pages", "evaluation_pages")]
    if any(page_sets[i] & page_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Training, validation and evaluation pages overlap")
    if not page_sets[0] or not page_sets[1]:
        raise ValueError("Both training and validation pages are required")
    if metadata.get("adaptation_document") != "4" or any(
            not str(page).startswith("4_") for page in page_sets[0] | page_sets[1]):
        raise ValueError("This experiment only adapts on document 4")
    result = {"counts": {}, "max_text_length": 0, "max_content_width": 0}
    rows, seen, split_hashes = [], set(), []
    for split in ("train", "val"):
        entries, image_hashes = [], set()
        for number, row in enumerate((directory / f"{split}.txt").read_text(
                encoding="utf-8").splitlines(), 1):
            pieces = row.split("\t")
            if len(pieces) != 2 or not pieces[1].strip():
                raise ValueError(f"Invalid {split}.txt label at line {number}")
            relative, label = pieces
            pages = page_sets[0 if split == "train" else 1]
            if not any(Path(relative).stem.startswith(page + "_") for page in pages):
                raise ValueError(f"Crop is outside the declared {split} pages: {relative}")
            path = (directory / relative).resolve()
            if Path(relative).is_absolute() or not path.is_relative_to(directory):
                raise ValueError(f"Crop must be inside the prepared dataset: {relative}")
            if path in seen:
                raise ValueError(f"Repeated training/validation crop: {relative}")
            seen.add(path)
            if any(ord(char) < 32 or ord(char) > 126 or char not in characters for char in label):
                raise ValueError(f"Label has unsupported characters: {relative}")
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            content_width = math.ceil(48 * width / height)
            ctc_steps = len(label) + sum(a == b for a, b in zip(label, label[1:]))
            # The released PPLCNetV4 training forward always pools to 40 steps.
            # Increasing input width alone cannot increase that CTC capacity.
            if ctc_steps > 40:
                raise ValueError(f"Label requires {ctc_steps} CTC steps: {relative}; "
                                 "prepare shorter complete phrases, never truncate labels")
            result["max_text_length"] = max(result["max_text_length"], len(label))
            result["max_content_width"] = max(result["max_content_width"], content_width)
            digest = hash_file(path)
            image_hashes.add(digest)
            entries.append([relative, label, digest])
        if not entries:
            raise ValueError(f"{split}.txt has no usable labels")
        result["counts"][split] = len(entries)
        rows.append(entries)
        split_hashes.append(image_hashes)
    if split_hashes[0] & split_hashes[1]:
        raise ValueError("Identical crop pixels appear in training and validation")
    result["data_hash"] = fingerprint({"metadata": metadata, "rows": rows})
    result["metadata"] = metadata
    return result


def build_config(base, settings, data, stats, output, dictionary, pretrained):
    config = copy.deepcopy(base)
    if config["Architecture"]["Backbone"] != {"name": "PPLCNetV4", "model_size": "medium"}:
        raise ValueError("Unexpected upstream architecture")
    for key in ("epochs", "batch_size", "workers"):
        if not isinstance(settings[key], int) or settings[key] < (0 if key == "workers" else 1):
            raise ValueError(f"Invalid {key}")
    if not math.isfinite(settings["learning_rate"]) or settings["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= settings["warmup_epochs"] < settings["epochs"]:
        raise ValueError("warmup_epochs must be smaller than epochs")
    width = 320  # Official short-phrase training canvas; inference stays dynamic.
    # NRTR adds BOS/EOS and rejects labels >= max_text_length - 1.
    max_length = stats["max_text_length"] + 2
    global_config = config["Global"]
    global_config.update({
        "model_name": MODEL_NAME, "use_gpu": True, "distributed": False,
        "epoch_num": settings["epochs"], "seed": settings["seed"],
        "save_model_dir": str(output), "save_epoch_step": 1,
        "eval_batch_step": [0, math.ceil(stats["counts"]["train"] / settings["batch_size"])],
        "pretrained_model": str(pretrained), "checkpoints": None,
        "character_dict_path": str(dictionary), "max_text_length": max_length,
        "d2s_train_image_shape": [3, 48, width], "use_space_char": True,
        "uniform_output_enabled": False, "export_with_pir": True,
    })
    config["Optimizer"]["lr"].update({"learning_rate": settings["learning_rate"],
                                      "warmup_epoch": settings["warmup_epochs"]})
    for head in config["Architecture"]["Head"]["head_list"]:
        if "NRTRHead" in head:
            head["NRTRHead"]["max_text_length"] = max_length
    config["Metric"]["main_indicator"] = "norm_edit_dis"
    transforms = [
        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
        {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
        {"RecResizeImg": {"image_shape": [3, 48, width], "padding": True}},
        {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}},
    ]
    for stage, split in (("Train", "train"), ("Eval", "val")):
        config[stage] = {
            "dataset": {"name": "SimpleDataSet", "data_dir": str(data),
                        "label_file_list": [str(data / f"{split}.txt")], "ratio_list": [1.0],
                        "transforms": copy.deepcopy(transforms)},
            "loader": {"shuffle": stage == "Train", "drop_last": False,
                       "batch_size_per_card": settings["batch_size"], "num_workers": settings["workers"]},
        }
    return config


def checkpoint_prefix(value, run, resume=False):
    path = Path(value)
    path = (run / path if not path.is_absolute() else path).resolve()
    if path.suffix == ".pdparams":
        path = path.with_suffix("")
    if not path.is_relative_to(run):
        raise ValueError("Checkpoint must belong to this training run")
    for suffix in ((".pdparams", ".pdopt", ".states") if resume else (".pdparams",)):
        if not Path(str(path) + suffix).is_file():
            raise ValueError(f"Missing checkpoint file: {path}{suffix}")
    return path


def train(args):
    upstream = upstream_directory(args.paddleocr)
    data, output = Path(args.data).resolve(), Path(args.output).resolve()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load((upstream / BASE_CONFIG).read_text(encoding="utf-8"))
    dictionary = upstream / base["Global"]["character_dict_path"]
    stats = inspect_data(data, set(dictionary.read_text(encoding="utf-8").splitlines()) | {" "})
    pretrained = Path(args.pretrained).resolve()
    if pretrained.suffix != ".pdparams" or not pretrained.is_file():
        raise ValueError(f"Supply official training .pdparams weights; download from {PRETRAINED_URL}")
    config = build_config(base, settings, data, stats, output, dictionary, pretrained)
    identity = {"upstream_revision": UPSTREAM_REVISION, "training_ctc_steps": 40,
                "data_hash": stats["data_hash"], "settings": settings,
                "pretrained_sha256": hash_file(pretrained), "dictionary_sha256": hash_file(dictionary),
                "wrapper_sha256": hash_file(__file__), "config": copy.deepcopy(config)}
    run_hash = fingerprint(identity)
    record_path = output / "stage2_training.json"
    if args.resume:
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old["training_fingerprint"] != run_hash:
            raise ValueError("Resume data/config/code/weights differ from the original run")
        config["Global"]["checkpoints"] = str(checkpoint_prefix(args.resume, output, resume=True))
        config["Global"]["pretrained_model"] = None
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; use --resume latest or a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    record = {**stats["metadata"], "dataset": stats["metadata"],
              "training_fingerprint": run_hash, "training": identity,
              "actual_training_counts": stats["counts"], "image_width": 320,
              "max_content_width": stats["max_content_width"],
              "max_text_length": stats["max_text_length"]}
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    config_path = output / ("resume.yml" if args.resume else "training.yml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Out OCR: {stats['counts']}, training width=320 / 40 CTC steps, "
          f"max label={stats['max_text_length']}, epochs={settings['epochs']}", flush=True)
    subprocess.run([sys.executable, str(upstream / "tools/train.py"), "-c", str(config_path)],
                   cwd=upstream, check=True)
    return 0


def export(args):
    upstream = upstream_directory(args.paddleocr)
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    record = json.loads((run / "stage2_training.json").read_text(encoding="utf-8"))
    if fingerprint(record["training"]) != record["training_fingerprint"]:
        raise ValueError("Training provenance was changed")
    if record["training"]["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("Export must use the training run's upstream revision")
    checkpoint = checkpoint_prefix(args.checkpoint, run)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export output is not empty; choose a new directory")
    config = copy.deepcopy(record["training"]["config"])
    config["Global"].update({"checkpoints": str(checkpoint), "pretrained_model": None,
                            "save_inference_dir": str(output)})
    path = run / "export.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    subprocess.run([sys.executable, str(upstream / "tools/export_model.py"), "-c", str(path)],
                   cwd=upstream, check=True)
    for name in ("inference.json", "inference.pdiparams", "inference.yml"):
        if not (output / name).is_file():
            raise ValueError(f"Expected Paddle 3.x PIR export file missing: {name}")
    infer = yaml.safe_load((output / "inference.yml").read_text(encoding="utf-8"))
    if infer.get("Global", {}).get("model_name") != MODEL_NAME or not infer["PostProcess"].get("character_dict"):
        raise ValueError("Export does not contain the expected model identity/dictionary")
    record.update({"checkpoint": checkpoint.name, "checkpoint_sha256": hash_file(str(checkpoint) + ".pdparams"),
                   "export_hashes": {name: hash_file(output / name) for name in
                                     ("inference.json", "inference.pdiparams", "inference.yml")}})
    (output / "stage2_training.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Exported Out recognizer: {output}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Run official recognizer fine-tuning")
    fit.add_argument("--paddleocr", required=True, help="Official PaddleOCR checkout at v3.7.0")
    fit.add_argument("--data", required=True, help="Prepared Out dataset directory")
    fit.add_argument("--output", required=True)
    fit.add_argument("--pretrained", required=True, help="Official .pdparams training checkpoint")
    fit.add_argument("--config", default=str(DEFAULT_CONFIG))
    fit.add_argument("--resume", nargs="?", const="latest", help="Checkpoint prefix inside --output")
    fit.set_defaults(function=train)
    save = commands.add_parser("export", help="Export best checkpoint for the existing PaddleX adapter")
    save.add_argument("--paddleocr", required=True)
    save.add_argument("--run", required=True)
    save.add_argument("--checkpoint", default="best_accuracy")
    save.add_argument("--output", required=True)
    save.set_defaults(function=export)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
