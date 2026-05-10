#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telegram_style_transfer.llm_neutralization import (  # noqa: E402
    DEFAULT_PRIMARY_MODEL,
    summarize_usage_and_cost,
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate OpenRouter token usage and cost from source JSONL files"
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="Explicit JSONL files. Defaults to pilot_300_manifest.jsonl if present.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_PRIMARY_MODEL,
        help="OpenRouter model name for price lookup",
    )
    parser.add_argument("--prompt-overhead", type=int, default=150)
    parser.add_argument("--chars-per-token", type=float, default=1.7)
    parser.add_argument("--completion-ratio", type=float, default=0.9)
    parser.add_argument("--min-completion-tokens", type=int, default=32)
    args = parser.parse_args()

    default_manifest = Path("data/processed/pilot_300_manifest.jsonl")
    paths = [Path(path) for path in (args.paths or ([str(default_manifest)] if default_manifest.exists() else []))]
    if not paths:
        raise SystemExit("No input files found. Pass --paths ...")

    summaries: list[dict] = []
    for path in paths:
        rows = load_jsonl(path)
        if not rows:
            continue
        summary = summarize_usage_and_cost(
            rows,
            model=args.model,
            estimate_missing=True,
            prompt_overhead_tokens=args.prompt_overhead,
            chars_per_token=args.chars_per_token,
            completion_ratio=args.completion_ratio,
            min_completion_tokens=args.min_completion_tokens,
        )
        summary["path"] = str(path)
        summary["n_records"] = len(rows)
        summaries.append(summary)

    if not summaries:
        raise SystemExit("No rows found in the provided JSONL files.")

    if len(summaries) > 1:
        total = {
            "path": "TOTAL",
            "n_records": sum(item["n_records"] for item in summaries),
            "usage_prompt_tokens": sum(item["usage_prompt_tokens"] for item in summaries),
            "usage_completion_tokens": sum(
                item["usage_completion_tokens"] for item in summaries
            ),
            "usage_total_tokens": sum(item["usage_total_tokens"] for item in summaries),
            "cost_usd_estimate": round(
                sum(float(item.get("cost_usd_estimate") or 0.0) for item in summaries),
                6,
            ),
        }
        summaries.append(total)

    columns = [
        "path",
        "n_records",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "usage_total_tokens",
        "cost_usd_estimate",
    ]
    print("\t".join(columns))
    for summary in summaries:
        print("\t".join(str(summary.get(column, "")) for column in columns))


if __name__ == "__main__":
    main()
