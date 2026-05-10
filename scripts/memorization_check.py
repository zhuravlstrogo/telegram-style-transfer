#!/usr/bin/env python3
"""
Memorization check for fine-tuned model outputs.

For each generated text finds the nearest neighbour in the corresponding
train split using rapidfuzz ratio and Jaccard on 5-word n-grams. Flags
suspicious cases (median fuzz_ratio >= 60, exact matches, long shared
fragments).

Usage:
    python scripts/memorization_check.py \\
        --generated-type1 bench_type1_test_finetuned.txt \\
        --generated-type2 bench_type2_test_finetuned.txt \\
        --out-dir reports/memorization
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telegram_style_transfer.logging_utils import setup_logging
from telegram_style_transfer.paths import PROCESSED_DIR

log = logging.getLogger("memorization_check")

try:
    from rapidfuzz import fuzz as rf_fuzz
except ImportError:
    logging.basicConfig()
    logging.getLogger("memorization_check").error("rapidfuzz not installed. Run: pip install rapidfuzz")
    sys.exit(1)

LONG_FRAGMENT_MIN = 30  # chars — flag if shared substring >= this length


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _ngrams5(text: str) -> set[str]:
    words = text.split()
    if len(words) < 5:
        return set(words)
    return {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _longest_common_substring(a: str, b: str) -> int:
    """Return length of longest common substring (char-level)."""
    if not a or not b:
        return 0
    # Use DP with rolling rows — acceptable for post-length strings (~300 chars).
    prev = [0] * (len(b) + 1)
    best = 0
    for ch_a in a:
        curr = [0] * (len(b) + 1)
        for j, ch_b in enumerate(b, 1):
            if ch_a == ch_b:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


def _response_text(record: dict) -> str:
    return record.get("response_clean", record.get("response", ""))


def _load_generated(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    blocks = content.split("\n===\n")
    return [b.strip() for b in blocks if b.strip()]


def _load_train(style_type: str) -> list[dict]:
    path = PROCESSED_DIR / f"{style_type}_train.jsonl"
    if not path.exists():
        log.warning("%s not found.", path)
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_nearest(gen_text: str, train_records: list[dict]) -> dict:
    norm_gen = _normalize(gen_text)
    gen_ng = _ngrams5(norm_gen)

    best_fuzz = -1.0
    best_rec = None
    best_jac = 0.0

    for rec in train_records:
        train_text = _response_text(rec)
        norm_train = _normalize(train_text)
        score = rf_fuzz.ratio(norm_gen, norm_train)
        if score > best_fuzz:
            best_fuzz = score
            best_rec = rec
            best_jac = _jaccard(gen_ng, _ngrams5(norm_train))

    nearest_text = _response_text(best_rec) if best_rec else ""
    lcs = _longest_common_substring(_normalize(gen_text), _normalize(nearest_text))

    return {
        "nearest_train_id": best_rec.get("post_id", "") if best_rec else "",
        "nearest_train_text": nearest_text,
        "fuzz_ratio": round(best_fuzz, 2),
        "jaccard_5gram": round(best_jac, 4),
        "longest_common_substr_chars": lcs,
    }


def _run(style_type: str, gen_path: str, out_dir: Path) -> pd.DataFrame | None:
    p = Path(gen_path)
    if not p.exists():
        log.warning("%s not found, skipping %s.", p, style_type)
        return None

    generated = _load_generated(gen_path)
    train_records = _load_train(style_type)
    if not train_records:
        return None

    log.info("%s: %d generated, %d train records", style_type, len(generated), len(train_records))

    rows = []
    for i, gen_text in enumerate(generated):
        nearest = _find_nearest(gen_text, train_records)
        rows.append({"generated": gen_text, **nearest})
        if (i + 1) % 20 == 0 or (i + 1) == len(generated):
            log.info("%d/%d processed", i + 1, len(generated))

    df = pd.DataFrame(rows)

    exact = (df["fuzz_ratio"] == 100).sum()
    long_frag = (df["longest_common_substr_chars"] >= LONG_FRAGMENT_MIN).sum()
    med_fuzz = df["fuzz_ratio"].median()
    med_jac = df["jaccard_5gram"].median()

    log.info("fuzz_ratio    : median=%.1f  max=%.1f", med_fuzz, df["fuzz_ratio"].max())
    log.info("jaccard_5gram : median=%.4f  max=%.4f", med_jac, df["jaccard_5gram"].max())
    log.info("longest_substr: median=%.0f  max=%d",
             df["longest_common_substr_chars"].median(), df["longest_common_substr_chars"].max())
    log.info("exact matches : %d", exact)
    log.info("long fragments: %d (>= %d chars)", long_frag, LONG_FRAGMENT_MIN)

    if med_fuzz >= 60:
        log.warning("Median fuzz_ratio %.1f >= 60 — possible memorization.", med_fuzz)
    else:
        log.info("OK: Median fuzz_ratio %.1f < 60.", med_fuzz)
    if exact:
        log.warning("%d exact match(es) found.", exact)
    if long_frag:
        log.warning("%d generated text(s) share >= %d-char fragment with train.", long_frag, LONG_FRAGMENT_MIN)

    csv_path = out_dir / f"memorization_{style_type}.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved → %s", csv_path)

    log.info("Top-5 closest to train (%s):", style_type)
    for _, row in df.nlargest(5, "fuzz_ratio").iterrows():
        log.info(
            "fuzz=%.1f  jaccard=%.4f  lcs=%d\n  Generated : %r\n  Train[%s]: %r",
            row["fuzz_ratio"], row["jaccard_5gram"], row["longest_common_substr_chars"],
            row["generated"][:150], row["nearest_train_id"], row["nearest_train_text"][:150],
        )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Memorization check for fine-tuned outputs.")
    parser.add_argument("--generated-type1", default=None, metavar="FILE")
    parser.add_argument("--generated-type2", default=None, metavar="FILE")
    parser.add_argument("--out-dir", default="reports/memorization", metavar="DIR")
    args = parser.parse_args()

    if not args.generated_type1 and not args.generated_type2:
        parser.error("Provide at least --generated-type1 or --generated-type2.")

    setup_logging("memorization_check")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_map = {"type1": args.generated_type1, "type2": args.generated_type2}
    flags: list[str] = []

    for style_type, gen_path in generated_map.items():
        if gen_path is None:
            continue
        df = _run(style_type, gen_path, out_dir)
        if df is None:
            continue
        med_fuzz = df["fuzz_ratio"].median()
        exact = (df["fuzz_ratio"] == 100).sum()
        long_frag = (df["longest_common_substr_chars"] >= LONG_FRAGMENT_MIN).sum()
        if med_fuzz >= 60 or exact or long_frag:
            flags.append(style_type)

    if not flags:
        log.info("PASS: No memorization signals detected.")
    else:
        log.error("FAIL: Memorization signals detected for: %s", ", ".join(flags))
        log.error("Review the CSV files in %s", out_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
