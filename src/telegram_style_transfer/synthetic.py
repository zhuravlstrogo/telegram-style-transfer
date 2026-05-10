from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from telegram_style_transfer.data import normalized_text_hash

NUMBER_PRESERVATION_THRESHOLD = 0.8
JACCARD_BINS = [0.0, 0.2, 0.3, 0.5, 0.7, 1.01]
JACCARD_LABELS = ["<0.2", "0.2-0.3", "0.3-0.5", "0.5-0.7", "≥0.7"]
QUALITY_TARGETS = {
    "jaccard_ge_0_5": ("<=", 0.05),
    "identity_rate": ("<=", 0.02),
    "number_preservation": (">=", 0.95),
    "length_collapse_rate": ("<=", 0.10),
    "good_neutral_rate": (">=", 0.70),
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [
        paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()
    ]
    return paragraphs or [text.strip()]


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a: str, b: str) -> float:
    words_a = words(a)
    words_b = words(b)
    if not words_a and not words_b:
        return 1.0
    return len(words_a & words_b) / len(words_a | words_b)


def numbers_preserved(
    original: str,
    neutralized: str,
    threshold: float = NUMBER_PRESERVATION_THRESHOLD,
) -> bool:
    orig_nums = extract_numbers(original)
    if not orig_nums:
        return True
    shared = orig_nums & extract_numbers(neutralized)
    return len(shared) / len(orig_nums) >= threshold


def quality_flags(
    response_clean: str,
    neutralized: str,
    failed: bool,
) -> dict[str, Any]:
    if failed:
        return {
            "numbers_ok": False,
            "identity": False,
            "jaccard_bin": None,
            "length_collapse": None,
        }

    score = jaccard(neutralized, response_clean)
    length_ratio = len(neutralized) / max(len(response_clean), 1)
    return {
        "numbers_ok": numbers_preserved(response_clean, neutralized),
        "identity": neutralized.strip() == response_clean.strip(),
        "jaccard_bin": jaccard_bin(score),
        "length_collapse": length_ratio < 0.4,
    }


def jaccard_bin(value: float) -> str:
    cut = pd.cut(
        pd.Series([value]),
        bins=JACCARD_BINS,
        labels=JACCARD_LABELS,
        right=False,
    )
    return str(cut.iloc[0])


def score_neutralization(
    response_clean: str,
    neutralized: str,
    failed: bool,
) -> dict[str, Any]:
    if failed:
        return {
            "jaccard": None,
            "is_identity": False,
            "numbers_preserved": False,
            "length_ratio": None,
            "quality_flags": quality_flags(response_clean, neutralized, failed=True),
        }

    overlap = jaccard(neutralized, response_clean)
    length_ratio = len(neutralized) / max(len(response_clean), 1)
    return {
        "jaccard": round(overlap, 4),
        "is_identity": neutralized.strip() == response_clean.strip(),
        "numbers_preserved": numbers_preserved(response_clean, neutralized),
        "length_ratio": round(length_ratio, 4),
        "quality_flags": quality_flags(response_clean, neutralized, failed=False),
    }


def enrich_synthetic_record(
    record: dict[str, Any],
    synthetic_input: str,
    method: str,
    model_name: str,
    model_config: dict[str, Any],
    created_at: str,
    failure_reason: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failed = bool(failure_reason) or not synthetic_input
    metrics = score_neutralization(
        response_clean=record["response_clean"],
        neutralized=synthetic_input,
        failed=failed,
    )
    if not failed and not metrics["numbers_preserved"]:
        failed = True
        failure_reason = "numbers_not_preserved"
        metrics = score_neutralization(
            response_clean=record["response_clean"],
            neutralized=synthetic_input,
            failed=True,
        )

    enriched = dict(record)
    if "input_heuristic" not in enriched:
        enriched["input_heuristic"] = enriched.get("input", "")

    enriched["synthetic_input"] = synthetic_input if not failed else ""
    enriched["input"] = synthetic_input if not failed else ""
    if not failed:
        # Mark input as LLM-neutralized so finetune.py --train-input-source
        # can filter on it. Format: "<method>_synthetic" (e.g. openrouter_synthetic).
        enriched["input_source"] = f"{method}_synthetic"
    enriched["synthetic_ok"] = not failed
    enriched["synthetic_failed"] = failed
    enriched["failure_reason"] = failure_reason or (
        "empty_after_clean" if not synthetic_input else None
    )
    enriched["synthetic_method"] = method
    enriched["synthetic_model"] = model_name
    enriched["synthetic_config"] = model_config
    enriched["synthetic_created_at"] = created_at
    enriched["synthetic_seed"] = None
    enriched["usable_for_training"] = not failed
    enriched.update(metrics)
    if extra_fields:
        enriched.update(extra_fields)
    return enriched


def add_sampling_columns(df: pd.DataFrame) -> pd.DataFrame:
    sampled = df.copy()
    sampled["has_numbers"] = sampled["response_clean"].str.contains(r"\d", regex=True)
    sampled["char_len_q"] = pd.qcut(
        sampled["char_len"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    sampled["multi_para"] = sampled["n_paragraphs"].ge(2)
    sampled["stratum"] = (
        sampled[["char_len_q", "multi_para", "has_numbers", "has_emoji"]]
        .astype(str)
        .agg("_".join, axis=1)
    )
    return sampled


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    sampled = add_sampling_columns(df)
    shuffled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    proportions = shuffled["stratum"].value_counts(normalize=True)

    selected_chunks: list[pd.DataFrame] = []

    for stratum, share in proportions.items():
        quota = max(1, int(round(share * n)))
        chunk = shuffled[shuffled["stratum"] == stratum].head(quota)
        selected_chunks.append(chunk)

    selected = (
        pd.concat(selected_chunks, ignore_index=False)
        .sort_index()
        .drop_duplicates(subset=["post_id", "style_type"])
    )

    if len(selected) < n:
        remainder = shuffled.loc[~shuffled.index.isin(selected.index)]
        selected = pd.concat(
            [selected, remainder.head(n - len(selected))], ignore_index=False
        )

    if len(selected) > n:
        selected = selected.head(n)

    return selected.reset_index(drop=True)


def build_pilot_manifest(
    processed_dir: Path,
    types: list[str],
    n_per_type: int,
    split: str = "val",
    seed: int = 42,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    created_at = utc_timestamp()
    for style_type in types:
        path = processed_dir / f"{style_type}_{split}.jsonl"
        if not path.exists():
            continue
        records = read_jsonl(path)
        frame = pd.DataFrame(records)
        if frame.empty:
            continue
        sampled = stratified_sample(frame, n=n_per_type, seed=seed)
        sampled["source_file"] = str(path)
        sampled["source_split"] = split
        sampled["manifest_seed"] = seed
        sampled["manifest_created_at"] = created_at
        sampled["pilot_id"] = sampled.apply(
            lambda row: f"{row['style_type']}:{row['post_id']}",
            axis=1,
        )
        if "group_id" not in sampled.columns:
            sampled["group_id"] = sampled["response_clean"].apply(normalized_text_hash)
        frames.append(sampled)

    if not frames:
        return pd.DataFrame()

    manifest = pd.concat(frames, ignore_index=True)
    columns = [
        "pilot_id",
        "post_id",
        "group_id",
        "style_type",
        "split",
        "source_split",
        "source_file",
        "date",
        "response_clean",
        "char_len",
        "n_paragraphs",
        "has_emoji",
        "has_numbers",
        "char_len_q",
        "multi_para",
        "stratum",
        "manifest_seed",
        "manifest_created_at",
    ]
    return manifest[columns].reset_index(drop=True)


def load_or_build_pilot_manifest(
    manifest_path: str | Path,
    processed_dir: Path,
    types: list[str],
    n_per_type: int,
    split: str = "val",
    seed: int = 42,
) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        return pd.DataFrame(read_jsonl(manifest_path))

    manifest = build_pilot_manifest(
        processed_dir=processed_dir,
        types=types,
        n_per_type=n_per_type,
        split=split,
        seed=seed,
    )
    if manifest.empty:
        return manifest
    write_jsonl(manifest_path, manifest.to_dict(orient="records"))
    return manifest


def summarize_quality(
    df: pd.DataFrame,
    failed_column: str,
) -> dict[str, Any]:
    total = len(df)
    if total == 0:
        return {"n_total": 0}

    active = df[~df[failed_column]]
    active_n = len(active)
    summary: dict[str, Any] = {
        "n_total": total,
        "n_active": active_n,
        "failed_rate": round(float(df[failed_column].mean()), 4),
    }

    if active_n == 0:
        return summary

    has_numbers_mask = active["response_clean"].str.contains(r"\d", regex=True)
    num_pres = (
        active.loc[has_numbers_mask, "numbers_preserved"].mean()
        if has_numbers_mask.any()
        else math.nan
    )

    summary.update(
        {
            "jaccard_ge_0_5": round(float(active["jaccard"].ge(0.5).mean()), 4),
            "identity_rate": round(float(active["is_identity"].mean()), 4),
            "number_preservation": (
                round(float(num_pres), 4) if not math.isnan(num_pres) else None
            ),
            "length_collapse_rate": round(
                float(active["length_ratio"].lt(0.4).mean()), 4
            ),
            "good_neutral_rate": round(
                float((active["jaccard"].lt(0.3) & ~active["is_identity"]).mean()),
                4,
            ),
            "mean_length_ratio": round(float(active["length_ratio"].mean()), 4),
        }
    )
    return summary


def evaluate_quality_summary(summary: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {"quality_gate_passed": True}
    for metric, (operator, threshold) in QUALITY_TARGETS.items():
        value = summary.get(metric)
        status_key = f"{metric}_pass"
        if value is None:
            results[status_key] = None
            results["quality_gate_passed"] = False
            continue
        if operator == "<=":
            passed = float(value) <= threshold
        elif operator == ">=":
            passed = float(value) >= threshold
        else:
            raise ValueError(f"Unsupported operator: {operator}")
        results[status_key] = passed
        if not passed:
            results["quality_gate_passed"] = False
    return results
