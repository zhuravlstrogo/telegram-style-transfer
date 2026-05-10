#!/usr/bin/env python3
"""
Prepare style-transfer training datasets from Telegram Desktop exports.

Reads:  data/raw/{type}/telegram_export.json
Writes: data/interim/{type}_posts.csv
        data/processed/{type}_{train,val,test}.jsonl
        data/processed/split_report.json

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --types type1 --min-chars 80
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from telegram_style_transfer.dataset_inputs import load_brief_overrides
from telegram_style_transfer.data import (
    clean_for_input,
    deduplicate,
    filter_posts,
    normalized_text_hash,
    parse_telegram_export,
    sample_recent,
    temporal_split,
)
from telegram_style_transfer.logging_utils import setup_logging
from telegram_style_transfer.paths import CONFIGS_DIR, DATA_DIR
from telegram_style_transfer.prompts import build_input_heuristic, format_training_prompt

log = logging.getLogger("prepare_dataset")

DATA_CONFIG_PATH = CONFIGS_DIR / "data.yaml"


def _load_data_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_max_samples(
    style_type: str,
    cli_value: int | None,
    config: dict,
) -> int:
    """CLI flag wins; otherwise per-type from config; otherwise fallback default."""
    if cli_value is not None:
        return cli_value
    per_type = (config.get("max_samples") or {}).get(style_type)
    if per_type is not None:
        return int(per_type)
    return 10_000


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["char_len"] = df["text"].str.len()
    df["n_paragraphs"] = df["text"].str.count(r"\n\n") + 1
    df["has_emoji"] = df["text"].str.contains(
        r"[\U0001F600-\U0001FAFF]", regex=True, na=False
    )
    df["ends_with_question"] = df["text"].str.rstrip().str.endswith("?")
    df["source_text_hash"] = df["text"].apply(normalized_text_hash)
    df["group_id"] = df["source_text_hash"]
    return df


def _build_records(
    df: pd.DataFrame,
    style_type: str,
    input_overrides: dict[tuple[str, int], dict[str, str]] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    instruction = (
        f"Rewrite the input into a Telegram post in style type {style_type}. "
        "Preserve the facts, but match the tone, structure, pacing, "
        "and ending typical for this style."
    )
    records = []
    override_counts: Counter[str] = Counter()
    input_overrides = input_overrides or {}
    for _, row in df.iterrows():
        post_id = int(row["post_id"])
        response_raw = row["text"]
        response_clean = clean_for_input(response_raw)
        # Build heuristic input from the cleaned text so handles/emoji don't leak in
        heuristic_input = build_input_heuristic(response_clean)
        input_text = heuristic_input
        input_source = "heuristic"
        override = input_overrides.get((style_type, post_id))
        if override:
            input_text = override["input"]
            input_source = override["input_source"]
            override_counts["total"] += 1
            override_counts[str(row["split"])] += 1
        records.append(
            {
                "post_id": post_id,
                "date": str(row["date"]),
                "split": row["split"],
                "style_type": style_type,
                "channel": row["channel"],
                "instruction": instruction,
                "input": input_text,
                "input_source": input_source,
                "input_heuristic": heuristic_input,
                "response_raw": response_raw,
                "response_clean": response_clean,
                "group_id": row["group_id"],
                "source_text_hash": row["source_text_hash"],
                "split_strategy": "temporal_80_10_10_after_dedup",
                "char_len": int(row["char_len"]),
                "n_paragraphs": int(row["n_paragraphs"]),
                "has_emoji": bool(row["has_emoji"]),
                "ends_with_question": bool(row["ends_with_question"]),
                # prompt uses response_clean so style markers don't leak into training
                "prompt": format_training_prompt(style_type, input_text, response_clean),
            }
        )
    return records, dict(override_counts)


def process_type(
    style_type: str,
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    min_chars: int,
    max_samples: int,
    input_overrides: dict[tuple[str, int], dict[str, str]] | None = None,
) -> dict[str, int]:
    export_path = raw_dir / style_type / "telegram_export.json"
    if not export_path.exists():
        log.warning("Export not found: %s — skipping %s", export_path, style_type)
        return {}

    log.info("[%s] Reading %s", style_type, export_path)
    df = parse_telegram_export(export_path)
    log.info("[%s] Raw messages: %d", style_type, len(df))

    df = filter_posts(df, min_chars=min_chars)
    log.info("[%s] After filter (min_chars=%d): %d", style_type, min_chars, len(df))

    df = deduplicate(df)
    log.info("[%s] After dedup: %d", style_type, len(df))

    df = sample_recent(df, max_samples)
    log.info("[%s] After sampling (max=%d, newest first): %d", style_type, max_samples, len(df))

    df = _add_features(df)
    df = temporal_split(df)

    counts = df["split"].value_counts().sort_index().to_dict()
    log.info("[%s] Split sizes: %s", style_type, counts)

    interim_dir.mkdir(parents=True, exist_ok=True)
    interim_path = interim_dir / f"{style_type}_posts.csv"
    df.to_csv(interim_path, index=False)
    log.info("[%s] Saved interim CSV: %s", style_type, interim_path)

    records, override_counts = _build_records(
        df,
        style_type,
        input_overrides=input_overrides,
    )
    if override_counts:
        log.info("[%s] brief overrides applied: %s", style_type, override_counts)

    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        split_records = [r for r in records if r["split"] == split_name]
        out_path = processed_dir / f"{style_type}_{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in split_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info(
            "[%s] %s → %d records → %s",
            style_type,
            split_name,
            len(split_records),
            out_path,
        )

    return counts | {
        "total": int(len(df)),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "brief_overrides_total": int(override_counts.get("total", 0)),
        "brief_overrides_train": int(override_counts.get("train", 0)),
        "brief_overrides_val": int(override_counts.get("val", 0)),
        "brief_overrides_test": int(override_counts.get("test", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare style-transfer dataset from Telegram Desktop exports"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["type1", "type2"],
        metavar="TYPE",
        help="Style types to process (default: type1 type2)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        metavar="N",
        help="Minimum character length to keep a post (default: 50)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max posts per type; newest posts are kept. "
            "If omitted, per-type values from configs/data.yaml are used."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DATA_CONFIG_PATH),
        metavar="FILE",
        help=f"Dataset config (default: {DATA_CONFIG_PATH})",
    )
    parser.add_argument(
        "--brief-v4-path",
        default="",
        help=(
            "Optional JSONL with brief_v4 neutralization results; "
            "only successful records are used as input overrides"
        ),
    )
    parser.add_argument(
        "--brief-v4-max-jaccard",
        type=float,
        default=0.3,
        metavar="F",
        help=(
            "Accept brief_v4 overrides only when jaccard is below this threshold "
            "(default: 0.3)"
        ),
    )
    args = parser.parse_args()
    setup_logging("prepare_dataset")

    raw_dir = DATA_DIR / "raw"
    interim_dir = DATA_DIR / "interim"
    processed_dir = DATA_DIR / "processed"

    data_config = _load_data_config(Path(args.config))
    resolved_max_samples = {
        st: _resolve_max_samples(st, args.max_samples, data_config) for st in args.types
    }
    log.info("max_samples per type: %s", resolved_max_samples)

    input_overrides: dict[tuple[str, int], dict[str, str]] = {}
    override_stats: dict[str, int] | None = None
    if args.brief_v4_path:
        input_overrides, override_stats = load_brief_overrides(
            args.brief_v4_path,
            max_jaccard=args.brief_v4_max_jaccard,
        )
        log.info(
            "Loaded brief_v4 overrides from %s: %s",
            args.brief_v4_path,
            override_stats,
        )

    # Merge into existing split_report.json so partial runs (e.g. only --types type2)
    # don't drop the other type's entry.
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_path = processed_dir / "split_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {}

    report["_meta"] = {
        "min_chars": args.min_chars,
        "max_samples": resolved_max_samples,
        "target_train_ok": data_config.get("target_train_ok"),
        "target_eval_samples": data_config.get("target_eval_samples"),
        "split_strategy": "split_first_temporal_80_10_10",
        "input_source": (
            "heuristic_plus_successful_brief_v4"
            if args.brief_v4_path
            else "heuristic_brief"
        ),
        "brief_v4_path": args.brief_v4_path or None,
        "brief_v4_max_jaccard": (
            args.brief_v4_max_jaccard if args.brief_v4_path else None
        ),
    }
    if override_stats is not None:
        report["_meta"]["brief_v4_override_stats"] = override_stats
    for style_type in args.types:
        report[style_type] = process_type(
            style_type,
            raw_dir,
            interim_dir,
            processed_dir,
            args.min_chars,
            resolved_max_samples[style_type],
            input_overrides=input_overrides,
        )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Split report saved: %s", report_path)


if __name__ == "__main__":
    main()
