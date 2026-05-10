#!/usr/bin/env python3
"""Fine-tune Qwen2.5-3B-Instruct with QLoRA for one style type.

Requires CUDA + Unsloth. Run on ubuntu_t4 or equivalent.

Usage:
    python scripts/finetune.py --type type1
    python scripts/finetune.py --type type2 --env ubuntu_t4
    ENV_PROFILE=ubuntu_t4 python scripts/finetune.py --type type1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telegram_style_transfer.logging_utils import setup_logging
from telegram_style_transfer.paths import CONFIGS_DIR, MODELS_DIR, PROCESSED_DIR

log = logging.getLogger("finetune")


def _load_env(env_name: str) -> dict:
    with open(CONFIGS_DIR / "env.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["profiles"][env_name]


def _load_train_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _filter_by_input_source(records: list[dict], input_source: str | None) -> list[dict]:
    if not input_source:
        return records
    return [record for record in records if record.get("input_source") == input_source]


def _check_native_build_toolchain() -> None:
    """Fail early if Triton/Unsloth cannot compile native helpers via gcc."""
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        log.error(
            "C compiler not found. Install gcc/build-essential before running Unsloth fine-tuning."
        )
        sys.exit(1)

    test_src = "#include <stdlib.h>\nint main(void) { return 0; }\n"
    with TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "test.c"
        out_path = Path(tmpdir) / "test.o"
        src_path.write_text(test_src, encoding="utf-8")
        proc = subprocess.run(
            [cc, str(src_path), "-c", "-o", str(out_path)],
            capture_output=True,
            text=True,
        )

    if proc.returncode == 0:
        return

    stderr = (proc.stderr or "").strip()
    if "stdlib.h" in stderr or "fatal error" in stderr:
        log.error(
            "System C headers are missing; Triton cannot build its CUDA helpers.\n"
            "Install Ubuntu packages:\n"
            "  sudo apt update && sudo apt install -y build-essential libc6-dev "
            "linux-libc-dev python3-dev\n"
            "Compiler output:\n%s", stderr,
        )
        sys.exit(1)
    log.error(
        "Failed to compile a trivial C program required by Triton.\nCompiler output:\n%s",
        stderr or "(no stderr)",
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune for style transfer")
    parser.add_argument("--type", choices=["type1", "type2"], required=True, dest="style_type")
    parser.add_argument(
        "--env",
        default=os.environ.get("ENV_PROFILE", "ubuntu_t4"),
        help="Profile from configs/env.yaml",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIGS_DIR / "train.yaml"),
        help="Training config path (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--train-input-source",
        default="",
        help="Use only records with this input_source for train (e.g. llm_brief_v4)",
    )
    parser.add_argument(
        "--val-input-source",
        default="",
        help=(
            "Use only records with this input_source for val. "
            "If omitted, defaults to --train-input-source."
        ),
    )
    parser.add_argument(
        "--val-fallback",
        choices=["error", "all"],
        default="all",
        help=(
            "What to do if val becomes empty after input_source filtering: "
            "'error' or fallback to all val records (default: all)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Directory with type*_train.jsonl / type*_val.jsonl (default: data/processed/)",
    )
    args = parser.parse_args()
    setup_logging("finetune")

    env_cfg = _load_env(args.env)
    env_ft = env_cfg["finetune"]
    ft = _load_train_config(Path(args.config))

    if not env_ft["enabled"]:
        log.error("Fine-tuning disabled for env '%s': %s", args.env, env_ft.get("reason", ""))
        sys.exit(1)

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    _check_native_build_toolchain()

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    import torch

    base_model: str = ft["base_model"]
    max_seq_len: int = ft["max_seq_length"]
    style_type: str = args.style_type

    log.info("base_model=%s  style_type=%s  config=%s", base_model, style_type, args.config)
    log.info(
        "max_seq_length=%d  train_bs=%d  grad_accum=%d  eval_bs=%d",
        max_seq_len,
        ft["per_device_train_batch_size"],
        ft["gradient_accumulation_steps"],
        ft.get("per_device_eval_batch_size", ft["per_device_train_batch_size"]),
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=ft["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=ft["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=ft["lora_alpha"],
        lora_dropout=ft["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=ft["seed"],
    )

    data_dir = Path(args.data_dir) if args.data_dir else PROCESSED_DIR
    train_records_all = _read_jsonl(data_dir / f"{style_type}_train.jsonl")
    val_records_all = _read_jsonl(data_dir / f"{style_type}_val.jsonl")

    val_input_source = args.val_input_source or args.train_input_source
    train_records = _filter_by_input_source(train_records_all, args.train_input_source or None)
    val_records = _filter_by_input_source(val_records_all, val_input_source or None)

    if not train_records:
        log.error("No train records left after filtering input_source=%r", args.train_input_source)
        sys.exit(1)
    if not val_records:
        if args.val_fallback == "all":
            log.warning("Val filter produced 0 records; falling back to full validation split")
            val_records = val_records_all
        else:
            log.error("No val records left after filtering input_source=%r", val_input_source)
            sys.exit(1)

    log.info(
        "train=%d/%d  val=%d/%d  train_input_source=%s  val_input_source=%s",
        len(train_records), len(train_records_all),
        len(val_records), len(val_records_all),
        args.train_input_source or "all",
        val_input_source or "all",
    )

    EOS = tokenizer.eos_token

    def _build_text_dataset(records: list[dict], split_name: str):
        """Tokenize once and drop examples longer than max_seq_len.

        Unsloth's on-the-fly truncation collides with its fused CE loss
        (see https://github.com/unslothai/unsloth/issues — `Expected input
        batch_size to match target batch_size`). Pre-filtering avoids the
        crash. Truncating instead would chop the response, which is the
        learning target — so we drop oversize records.
        """
        rows: list[dict] = []
        oversize: list[int] = []
        for record in records:
            text = record["prompt"] + EOS
            n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if n_tokens > max_seq_len:
                oversize.append(n_tokens)
                continue
            rows.append({"text": text})
        if oversize:
            log.warning(
                "%s: dropped %d/%d records longer than max_seq_length=%d "
                "(token counts: max=%d, p95=%d)",
                split_name, len(oversize), len(records), max_seq_len,
                max(oversize), sorted(oversize)[int(len(oversize) * 0.95) - 1] if len(oversize) > 1 else oversize[0],
            )
        if not rows:
            log.error("%s: 0 records left after length filter; bump max_seq_length", split_name)
            sys.exit(1)
        return Dataset.from_list(rows), len(oversize)

    train_ds, train_dropped = _build_text_dataset(train_records, "train")
    val_ds, val_dropped = _build_text_dataset(val_records, "val")

    # Keep all fine-tune artifacts under models/<type>/<run_group>/...
    # The default group is "all" (i.e. no input_source filtering).
    run_group = "all"
    if args.data_dir and Path(args.data_dir).name == "synthetic_openrouter":
        run_group = "synthetic_openrouter"

    out_dir = MODELS_DIR / style_type / run_group
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    gpu = torch.cuda.get_device_properties(0)
    mem_before = torch.cuda.max_memory_reserved() / 1024**3
    log.info(
        "GPU=%s  total=%.1f GB  reserved_before=%.2f GB",
        gpu.name, gpu.total_memory / 1024**3, mem_before,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=ft["per_device_train_batch_size"],
            per_device_eval_batch_size=ft.get(
                "per_device_eval_batch_size",
                ft["per_device_train_batch_size"],
            ),
            eval_accumulation_steps=ft.get("eval_accumulation_steps"),
            gradient_accumulation_steps=ft["gradient_accumulation_steps"],
            warmup_steps=10,
            num_train_epochs=ft["num_train_epochs"],
            learning_rate=ft["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            fp16_full_eval=not is_bfloat16_supported(),
            bf16_full_eval=is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=ft["eval_steps"],
            save_steps=ft["save_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=ft["seed"],
            output_dir=str(ckpt_dir),
            report_to="none",
        ),
    )

    t0 = time.monotonic()
    stats = trainer.train()
    duration_s = time.monotonic() - t0

    mem_peak = torch.cuda.max_memory_reserved() / 1024**3
    log.info(
        "done in %.1f min  peak_mem=%.2f GB  train_loss=%s",
        duration_s / 60, mem_peak, stats.metrics.get("train_loss", "?"),
    )

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("adapter → %s", adapter_dir)

    manifest = {
        "style_type": style_type,
        "base_model": base_model,
        "train_config_path": str(Path(args.config)),
        "env": args.env,
        "train_input_source": args.train_input_source or None,
        "val_input_source": val_input_source or None,
        "val_fallback": args.val_fallback,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_s, 1),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_samples_unfiltered": len(train_records_all),
        "val_samples_unfiltered": len(val_records_all),
        "train_dropped_oversize": train_dropped,
        "val_dropped_oversize": val_dropped,
        "hyperparams": {k: ft[k] for k in [
            "lora_r", "lora_alpha", "lora_dropout", "max_seq_length",
            "per_device_train_batch_size", "per_device_eval_batch_size",
            "eval_accumulation_steps", "gradient_accumulation_steps",
            "learning_rate", "num_train_epochs", "seed",
        ]},
        "metrics": {
            "train_loss": stats.metrics.get("train_loss"),
            "eval_loss": stats.metrics.get("eval_loss"),
            "peak_vram_gb": round(mem_peak, 3),
        },
    }
    manifest_path = out_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("manifest → %s", manifest_path)


if __name__ == "__main__":
    main()
