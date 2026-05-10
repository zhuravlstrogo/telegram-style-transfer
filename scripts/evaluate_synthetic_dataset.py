#!/usr/bin/env python3
"""
Evaluate a synthetic dataset produced by `generate_openrouter_synthetic.py`.

Usage:
    python scripts/evaluate_synthetic_dataset.py
    python scripts/evaluate_synthetic_dataset.py --data-dir data/processed/synthetic_openrouter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telegram_style_transfer.llm_neutralization import summarize_usage_and_cost
from telegram_style_transfer.synthetic import evaluate_quality_summary, summarize_quality


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic dataset quality from JSONL splits"
    )
    parser.add_argument("--data-dir", default="data/processed/synthetic_openrouter")
    parser.add_argument("--types", nargs="+", default=["type1", "type2"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--failed-column",
        default="synthetic_failed",
        help="Failure column name in synthetic JSONL",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for heuristic cost estimation when usage tokens are absent",
    )
    parser.add_argument("--out", default=None, help="Optional JSON summary path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    summaries: list[dict] = []
    for style_type in args.types:
        for split in args.splits:
            path = data_dir / f"{style_type}_{split}.jsonl"
            if not path.exists():
                print(f"[WARN] Missing {path}", file=sys.stderr)
                continue
            rows = load_jsonl(path)
            if not rows:
                print(f"[WARN] Empty {path}", file=sys.stderr)
                continue
            df = pd.DataFrame(rows)
            if args.failed_column not in df.columns:
                print(
                    f"[WARN] Missing failed column '{args.failed_column}' in {path}",
                    file=sys.stderr,
                )
                continue
            summary = summarize_quality(df, failed_column=args.failed_column)
            summary.update(evaluate_quality_summary(summary))
            summary.update(
                summarize_usage_and_cost(
                    rows,
                    model=args.model,
                    estimate_missing=bool(args.model),
                )
            )
            summary["style_type"] = style_type
            summary["split"] = split
            summary["path"] = str(path)
            summaries.append(summary)

    if not summaries:
        raise SystemExit("No synthetic datasets found.")

    frame = pd.DataFrame(summaries)
    print(frame.to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(summaries, handle, ensure_ascii=False, indent=2)
        print(f"\nSaved summary → {out_path}")


if __name__ == "__main__":
    main()
