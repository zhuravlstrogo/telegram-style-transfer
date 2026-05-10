#!/usr/bin/env python3
"""Generate style-transferred posts using the BASE model (no fine-tuning).

Reads one neutral brief per line from the input file, formats it as an
inference prompt, and writes one generated post per line to the output file.

On CUDA (ubuntu_t4): uses Unsloth's optimised inference (2x faster).
On MPS / CPU  (mac_m1): falls back to vanilla transformers, float32.

Usage:
    python scripts/generate_baseline.py --type type1
    python scripts/generate_baseline.py --type type2 \
        --input inputs_type2.txt \
        --output outputs_baseline_type2.txt \
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
from telegram_style_transfer.paths import CONFIGS_DIR, OUTPUT_DIR
from telegram_style_transfer.prompts import format_inference_prompt

log = logging.getLogger("generate_baseline")
DATA_CONFIG_PATH = CONFIGS_DIR / "data.yaml"


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


def _load_model_cuda(base_model: str, max_seq_len: int, load_in_4bit: bool):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _load_model_cpu_mps(base_model: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
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
    parser = argparse.ArgumentParser(description="Baseline inference (no fine-tuning)")
    parser.add_argument("--type", choices=["type1", "type2"], required=True, dest="style_type")
    parser.add_argument("--env", default=os.environ.get("ENV_PROFILE", "ubuntu_t4"))
    parser.add_argument("--input", default=None,
                        help="Input txt file, one brief per line (default: inputs_{type}.txt)")
    parser.add_argument("--from-jsonl", default=None, metavar="FILE",
                        help="Read inputs from a JSONL file instead of --input txt")
    parser.add_argument("--field", default="input",
                        help="Field to extract from each JSONL record (default: input)")
    parser.add_argument("--output", default=None,
                        help="Output file (default: output/outputs_baseline_{type}.txt)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process only the first N inputs.")
    parser.add_argument("--limit-from-config", action="store_true",
                        help="Use target_eval_samples from configs/data.yaml as the input limit.")
    parser.add_argument("--data-config", default=str(DATA_CONFIG_PATH), metavar="FILE",
                        help=f"Dataset config for --limit-from-config (default: {DATA_CONFIG_PATH})")
    args = parser.parse_args()

    if args.from_jsonl and args.input:
        sys.exit("Use either --from-jsonl or --input, not both.")

    setup_logging("generate_baseline")

    cfg = _load_env(args.env)
    data_cfg = _load_data_config(Path(args.data_config))
    limit = _resolve_limit(args.limit, args.limit_from_config, data_cfg)
    inf_cfg = cfg["inference"]
    device: str = inf_cfg["device"]

    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"outputs_baseline_{args.style_type}.txt"

    if args.from_jsonl:
        input_path = Path(args.from_jsonl)
        if not input_path.exists():
            log.error("JSONL file not found: %s", input_path)
            sys.exit(1)
        inputs = _read_inputs_jsonl(input_path, args.field)
    else:
        input_path = Path(args.input) if args.input else ROOT / f"inputs_{args.style_type}.txt"
        if not input_path.exists():
            log.error("Input file not found: %s", input_path)
            sys.exit(1)
        inputs = _read_inputs(input_path)
    if limit is not None:
        inputs = inputs[:limit]
        log.info("input limit=%d  source=%s", limit, args.data_config if args.limit_from_config else "cli")
    log.info("style_type=%s  inputs=%d  device=%s", args.style_type, len(inputs), device)

    ft_cfg = cfg.get("finetune", {})
    base_model: str = ft_cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
    max_seq_len: int = ft_cfg.get("max_seq_length", 1024)

    if device == "cuda":
        model, tokenizer = _load_model_cuda(base_model, max_seq_len, inf_cfg["load_in_4bit"])
    else:
        model, tokenizer = _load_model_cpu_mps(base_model, device)

    results = []
    for i, text in enumerate(inputs, 1):
        prompt = format_inference_prompt(args.style_type, text)
        response = _generate_one(model, tokenizer, prompt, device, inf_cfg)
        results.append(response)
        log.info("[%d/%d] %s…", i, len(inputs), response[:80].replace("\n", " "))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n===\n".join(results))
        f.write("\n")
    log.info("→ %s", output_path)

    manifest = {
        "mode": "baseline",
        "style_type": args.style_type,
        "base_model": base_model,
        "adapter": None,
        "env": args.env,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "n_inputs": len(inputs),
        "input_limit": limit,
        "decoding": {k: inf_cfg[k] for k in ["temperature", "top_p", "max_new_tokens", "do_sample"]},
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
