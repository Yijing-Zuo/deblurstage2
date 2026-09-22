"""Single-GPU Qwen3-VL language-only LoRA using Transformers Trainer and PEFT."""
import argparse
import json
from pathlib import Path

from common import (fingerprint, hash_file, load_config, make_prompt, model_settings,
                    read_jsonl, source_hashes)
from restore import base_identity, encode, image_messages, load_processor


def read_training(path, allowed_splits):
    path = Path(path).resolve()
    rows = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or (not rows and "train" in allowed_splits):
        raise ValueError(f"{path}: expected nonempty list of training examples")
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Duplicate example: {row['id']}")
        seen.add(row["id"])
        if row.get("split") not in allowed_splits or "document_id" not in row:
            raise ValueError(f"{row['id']}: invalid split or missing document_id")
        if not isinstance(row.get("deblur_run"), str) or not row["deblur_run"].strip():
            raise ValueError(f"{row['id']}: missing deblur_run provenance")
        if str(row["document_id"]) in {"0", "4", "14"}:
            raise ValueError(f"{row['id']}: documents 0/4/14 are held out")
        images, turns = row["image"], row["conversations"]
        if len(images) != 2 or any(not Path(p).is_absolute() for p in images):
            raise ValueError(f"{row['id']}: expected two absolute image paths")
        if len(turns) != 2 or [turn["from"] for turn in turns] != ["human", "gpt"]:
            raise ValueError(f"{row['id']}: expected one human/assistant pair")
        prompt = make_prompt(row["blur_text"], row["out_text"])
        if turns[0]["value"] != "<image>\n<image>\n" + prompt:
            raise ValueError(f"{row['id']}: training prompt differs from inference template")
        if not isinstance(turns[1]["value"], str) or not turns[1]["value"].strip():
            raise ValueError(f"{row['id']}: empty target transcription")
        if row.get("source_hashes") != source_hashes({"blur": images[0], "out": images[1]}):
            raise ValueError(f"{row['id']}: source images changed after export")
        row["prompt"] = prompt
    return rows


class ParagraphCollator:
    """Batch size one avoids lossy image/sequence padding and preserves exact masks."""
    def __init__(self, processor, max_length):
        self.processor, self.max_length = processor, max_length

    def __call__(self, rows):
        import torch
        if len(rows) != 1:
            raise ValueError("This collator requires per-device batch size 1")
        row = rows[0]
        messages = image_messages(row["image"], row["prompt"])
        prefix = encode(self.processor, messages, True)["input_ids"]
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": row["conversations"][1]["value"]}]})
        batch = encode(self.processor, messages)
        ids, prompt_length = batch["input_ids"], prefix.shape[1]
        if ids.shape[1] > self.max_length:
            raise ValueError(f"{row['id']}: sequence {ids.shape[1]} > {self.max_length}; no truncation allowed")
        if ids.shape[1] <= prompt_length or not torch.equal(ids[:, :prompt_length], prefix):
            raise ValueError(f"{row['id']}: chat template prefix mismatch or empty assistant tokens")
        labels = ids.clone()
        labels[:, :prompt_length] = -100
        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model-size", choices=("8b", "32b"))
    parser.add_argument("--model-path")
    parser.add_argument("--output", default="runs/adapter")
    parser.add_argument("--resume", help="Explicit Trainer checkpoint directory")
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value for an explicitly requested smoke run")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or positive")
    config = load_config(args.config)
    settings = model_settings(config, args.model_size, args.model_path)
    rows = read_training(args.train_file, {"train"})
    validation = (read_training(args.eval_file, {"val", "validation", "dev"}) or None) if args.eval_file else None
    if validation and {str(r["document_id"]) for r in rows} & {str(r["document_id"]) for r in validation}:
        raise ValueError("Train and validation contain the same document")
    if len({r["deblur_run"] for r in rows + (validation or [])}) != 1:
        raise ValueError("Train and validation must use one frozen deblur run")
    source = settings.get("path", settings["id"])
    metadata = {"base_model": base_identity(config, settings), "loaded_from": source,
                "prompt_template_hash": fingerprint(make_prompt("", ""))}
    metadata["signature"] = fingerprint({"config": config, "base": metadata, "settings": settings,
        "train": hash_file(args.train_file), "eval": hash_file(args.eval_file) if args.eval_file else None,
        "images": [(r["id"], r["source_hashes"]) for r in rows + (validation or [])],
        "max_steps": args.max_steps, "script": hash_file(__file__)})
    output = Path(args.output).resolve()
    meta_path = output / "training_metadata.json"
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        if checkpoint.parent != output or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("--resume must be a Trainer checkpoint inside --output")
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != metadata:
            raise ValueError("Resume data, model, prompt, code or training configuration changed")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; select a new --output or explicit --resume")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; training was not started")
    set_seed(42)
    processor = load_processor(settings, config, args.offline)
    collator = ParagraphCollator(processor, config["max_seq_length"])
    # Validate every template/length before allocating model weights or starting updates.
    for row in rows + (validation or []):
        collator([row])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source, revision=settings["revision"], local_files_only=args.offline,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.requires_grad_(False)
    targets = [name for name, _ in model.named_modules()
               if ".language_model." in name and name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj"))]
    if not targets or any("visual" in name for name in targets):
        raise ValueError("Could not identify language-only Qwen attention projections")
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=targets, bias="none", task_type=TaskType.CAUSAL_LM))
    if any(p.requires_grad and ("visual" in n or "lora_" not in n) for n, p in model.named_parameters()):
        raise ValueError("Unexpected trainable parameter outside language LoRA")
    model.enable_input_require_grads()
    train = config["training"]
    training_args = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=train["gradient_accumulation_steps"],
        learning_rate=train["learning_rate"], num_train_epochs=train["epochs"], max_steps=args.max_steps,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, label_names=["labels"], report_to="none", seed=42,
        logging_steps=1, save_strategy="epoch", save_total_limit=2,
        eval_strategy="epoch" if validation else "no", prediction_loss_only=True,
        dataloader_num_workers=0, optim="adamw_torch", warmup_ratio=0.03)
    trainer = Trainer(model=model, args=training_args, train_dataset=rows,
                      eval_dataset=validation, data_collator=collator, processing_class=processor)
    output.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output))
    trainer.save_state()
    processor.save_pretrained(str(output))
    print(f"Saved adapter and training_metadata.json to {output}")


if __name__ == "__main__":
    main()
