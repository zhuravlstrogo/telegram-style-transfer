#!/usr/bin/env python3
"""Generate style-transferred posts using a FINE-TUNED LoRA adapter.

Reads one neutral brief per line from the input file, formats it as an
inference prompt, and writes one generated post per line to the output file.

On CUDA (ubuntu_t4): uses Unsloth's optimised inference (2x faster).
On MPS / CPU  (mac_m1): falls back to vanilla transformers + PEFT, float32.

Adapter is read from models/{type}/all/adapter/ by default (saved by finetune.py).

Usage:
    python scripts/generate.py --type type1
    python scripts/generate.py --type type2 \
        --input inputs_type2.txt \
        --output outputs_type2.txt \
        --env ubuntu_t4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telegram_style_transfer.logging_utils import setup_logging
from telegram_style_transfer.paths import CONFIGS_DIR, MODELS_DIR, OUTPUT_DIR
from telegram_style_transfer.prompts import format_inference_prompt

log = logging.getLogger("generate")
DATA_CONFIG_PATH = CONFIGS_DIR / "data.yaml"
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
TOKENIZER_CORE_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "spiece.model",
)
TOKENIZER_VOCAB_GROUPS = (
    ("vocab.json", "merges.txt"),
)


def _load_env(env_name: str) -> dict:
    with open(CONFIGS_DIR / "env.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["profiles"][env_name]


def _load_data_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_limit(cli_limit: int | None, use_config_limit: bool, data_config: dict) -> int | None:
    if cli_limit is not None:
        if cli_limit <= 0:
            raise SystemExit("--limit must be > 0")
        return cli_limit
    if not use_config_limit:
        return None
    configured = data_config.get("target_eval_samples")
    if configured is None:
        raise SystemExit("target_eval_samples is not set in configs/data.yaml")
    configured = int(configured)
    if configured <= 0:
        raise SystemExit("target_eval_samples in configs/data.yaml must be > 0")
    return configured


def _read_inputs(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _read_inputs_jsonl(path: Path, field: str) -> list[str]:
    import json
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)[field] for line in f if line.strip()]


def _extract_response(full_text: str) -> str:
    marker = "### Response:"
    idx = full_text.rfind(marker)
    if idx == -1:
        return full_text.strip()
    response = full_text[idx + len(marker):].strip()
    for eos in ["</s>", "<|endoftext|>", "<eos>", "<|im_end|>", "<|end|>"]:
        response = response.replace(eos, "")
    return response.strip()


def _has_adapter_weights(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and any((path / name).exists() for name in ADAPTER_WEIGHT_FILES)


def _has_tokenizer_files(path: Path) -> bool:
    if any((path / name).exists() for name in TOKENIZER_CORE_FILES):
        return True
    return any(all((path / name).exists() for name in group) for group in TOKENIZER_VOCAB_GROUPS)


def _checkpoint_step(path: Path) -> int:
    suffix = path.name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else -1


def _resolve_best_checkpoint(checkpoints_dir: Path) -> Path | None:
    if not checkpoints_dir.exists():
        return None

    checkpoint_dirs = sorted(
        [path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=_checkpoint_step,
    )
    if not checkpoint_dirs:
        return None

    for checkpoint_dir in reversed(checkpoint_dirs):
        state_path = checkpoint_dir / "trainer_state.json"
        if not state_path.exists():
            continue
        try:
            with open(state_path, encoding="utf-8") as f:
                trainer_state = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        best_checkpoint = trainer_state.get("best_model_checkpoint")
        if not best_checkpoint:
            continue
        candidate = checkpoints_dir / Path(best_checkpoint).name
        if _has_adapter_weights(candidate):
            return candidate

    for checkpoint_dir in reversed(checkpoint_dirs):
        if _has_adapter_weights(checkpoint_dir):
            return checkpoint_dir
    return None


def _resolve_adapter_source(adapter_dir: Path) -> Path:
    if _has_adapter_weights(adapter_dir):
        return adapter_dir

    checkpoint_dir = _resolve_best_checkpoint(adapter_dir.parent / "checkpoints")
    if checkpoint_dir is not None:
        log.warning(
            "adapter bundle is incomplete at %s; falling back to checkpoint %s",
            adapter_dir,
            checkpoint_dir,
        )
        return checkpoint_dir

    raise SystemExit(
        "Adapter bundle is incomplete: expected adapter_config.json and adapter_model.safetensors "
        f"under {adapter_dir}, and no usable checkpoint-* was found in {adapter_dir.parent / 'checkpoints'}."
    )


def _load_model_cuda(adapter_dir: Path, max_seq_len: int, load_in_4bit: bool):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _load_model_cpu_mps(adapter_dir: Path, base_model: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer_source = adapter_dir if _has_tokenizer_files(adapter_dir) else base_model
    if tokenizer_source == base_model:
        log.warning(
            "tokenizer files not found in %s; loading tokenizer from base model %s",
            adapter_dir,
            base_model,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.to(device).eval()
    return model, tokenizer


def _generate_one(
    model,
    tokenizer,
    prompt: str,
    device: str,
    inf_cfg: dict,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=inf_cfg["max_new_tokens"],
            do_sample=inf_cfg["do_sample"],
            temperature=inf_cfg["temperature"],
            top_p=inf_cfg["top_p"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return _extract_response(full_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuned model inference")
    parser.add_argument("--type", choices=["type1", "type2"], required=True, dest="style_type")
    parser.add_argument("--env", default=os.environ.get("ENV_PROFILE", "ubuntu_t4"))
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter dir (default: models/{type}/all/adapter)")
    parser.add_argument("--input", default=None,
                        help="Input txt file, one brief per line (default: inputs_{type}.txt)")
    parser.add_argument("--from-jsonl", default=None, metavar="FILE",
                        help="Read inputs from a JSONL file instead of --input txt")
    parser.add_argument("--field", default="input",
                        help="Field to extract from each JSONL record (default: input)")
    parser.add_argument("--output", default=None,
                        help="Output file (default: output/outputs_{type}.txt)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process only the first N inputs.")
    parser.add_argument("--limit-from-config", action="store_true",
                        help="Use target_eval_samples from configs/data.yaml as the input limit.")
    parser.add_argument("--data-config", default=str(DATA_CONFIG_PATH), metavar="FILE",
                        help=f"Dataset config for --limit-from-config (default: {DATA_CONFIG_PATH})")
    args = parser.parse_args()

    if args.from_jsonl and args.input:
        sys.exit("Use either --from-jsonl or --input, not both.")

    setup_logging("generate")

    cfg = _load_env(args.env)
    data_cfg = _load_data_config(Path(args.data_config))
    limit = _resolve_limit(args.limit, args.limit_from_config, data_cfg)
    inf_cfg = cfg["inference"]
    device: str = inf_cfg["device"]

    style_type = args.style_type
    adapter_dir = Path(args.adapter) if args.adapter else MODELS_DIR / style_type / "all" / "adapter"
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"outputs_{style_type}.txt"

    if not adapter_dir.exists():
        log.error("Adapter not found: %s\nRun: python scripts/finetune.py --type %s", adapter_dir, style_type)
        sys.exit(1)
    adapter_source = _resolve_adapter_source(adapter_dir)

    if args.from_jsonl:
        jsonl_path = Path(args.from_jsonl)
        if not jsonl_path.exists():
            log.error("JSONL file not found: %s", jsonl_path)
            sys.exit(1)
        inputs = _read_inputs_jsonl(jsonl_path, args.field)
        input_path = jsonl_path
    else:
        input_path = Path(args.input) if args.input else ROOT / f"inputs_{style_type}.txt"
        if not input_path.exists():
            log.error("Input file not found: %s", input_path)
            sys.exit(1)
        inputs = _read_inputs(input_path)
    if limit is not None:
        inputs = inputs[:limit]
        log.info("input limit=%d  source=%s", limit, args.data_config if args.limit_from_config else "cli")

    # resolve base_model: prefer run_manifest.json written by finetune.py
    manifest_candidates = [
        adapter_dir.parent / "run_manifest.json",
        MODELS_DIR / style_type / "all" / "run_manifest.json",
        MODELS_DIR / style_type / "synthetic_openrouter" / "run_manifest.json",
        MODELS_DIR / style_type / "run_manifest.json",  # backward-compat
    ]
    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    if manifest_path:
        with open(manifest_path, encoding="utf-8") as f:
            run_manifest = json.load(f)
        base_model = run_manifest["base_model"]
    else:
        ft_cfg = cfg.get("finetune", {})
        base_model = ft_cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct")

    max_seq_len: int = cfg.get("finetune", {}).get("max_seq_length", 1024)
    log.info(
        "style_type=%s  inputs=%d  adapter=%s  adapter_source=%s  device=%s",
        style_type,
        len(inputs),
        adapter_dir,
        adapter_source,
        device,
    )

    if device == "cuda":
        model, tokenizer = _load_model_cuda(adapter_source, max_seq_len, inf_cfg["load_in_4bit"])
    else:
        model, tokenizer = _load_model_cpu_mps(adapter_source, base_model, device)

    results = []
    for i, text in enumerate(inputs, 1):
        prompt = format_inference_prompt(style_type, text)
        response = _generate_one(model, tokenizer, prompt, device, inf_cfg)
        results.append(response)
        log.info("[%d/%d] %s…", i, len(inputs), response[:80].replace("\n", " "))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n===\n".join(results))
        f.write("\n")
    log.info("→ %s", output_path)

    inference_manifest = {
        "mode": "finetuned",
        "style_type": style_type,
        "base_model": base_model,
        "adapter": str(adapter_source),
        "adapter_requested": str(adapter_dir),
        "env": args.env,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "n_inputs": len(inputs),
        "input_limit": limit,
        "decoding": {k: inf_cfg[k] for k in ["temperature", "top_p", "max_new_tokens", "do_sample"]},
    }
    inf_manifest_path = output_path.with_suffix(".manifest.json")
    with open(inf_manifest_path, "w", encoding="utf-8") as f:
        json.dump(inference_manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
