#!/usr/bin/env python3
"""
Evaluate style transfer quality.

Metrics:
  - length    : Length comparison (input vs reference vs generated)
  - style     : Style classifier accuracy + confidence (TF-IDF + LogReg)
  - crosstype : Cross-type confusion matrix ("не средний по больнице")
  - cosine    : Content preservation via LaBSE cosine similarity
  - mauve     : Distributional similarity vs target corpus

Usage:
    # Only length stats (no generated outputs required):
    python scripts/evaluate.py

    # Full evaluation with generated outputs:
    python scripts/evaluate.py \\
        --generated-type1 outputs_type1.txt \\
        --generated-type2 outputs_type2.txt \\
        --metrics all

    # Cross-type check only (requires both generated files):
    python scripts/evaluate.py \\
        --generated-type1 bench_type1_test_finetuned.txt \\
        --generated-type2 bench_type2_test_finetuned.txt \\
        --metrics crosstype
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telegram_style_transfer.logging_utils import setup_logging
from telegram_style_transfer.eval import (
    compare_lengths,
    cosine_similarity_scores,
    cross_type_confusion_matrix,
    length_report,
    load_embedder,
    load_jsonl,
    mauve_score,
    style_classifier_scores,
    train_style_classifier,
)
from telegram_style_transfer.paths import CONFIGS_DIR, PROCESSED_DIR

log = logging.getLogger("evaluate")

METRIC_CHOICES = ["length", "style", "crosstype", "cosine", "mauve"]
DATA_CONFIG_PATH = CONFIGS_DIR / "data.yaml"


def _response_text(record: dict) -> str:
    return record.get("response_clean", record.get("response", ""))


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


def load_generated(path: str | None) -> list[str] | None:
    if path is None:
        return None
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        content = f.read()
    blocks = content.split("\n===\n")
    texts = [b.strip() for b in blocks if b.strip()]
    return texts or None


def _load_split(base_dir: Path, style_type: str, split: str) -> list[dict]:
    path = base_dir / f"{style_type}_{split}.jsonl"
    if not path.exists():
        log.warning("%s not found.", path)
        return []
    return load_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate style transfer quality.")
    parser.add_argument("--types", nargs="+", default=["type1", "type2"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Evaluate only the first N records per selected split.")
    parser.add_argument("--limit-from-config", action="store_true",
                        help="Use target_eval_samples from configs/data.yaml as the eval limit.")
    parser.add_argument("--data-config", default=str(DATA_CONFIG_PATH), metavar="FILE",
                        help=f"Dataset config for --limit-from-config (default: {DATA_CONFIG_PATH})")
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Directory with type*_train.jsonl/type*_val.jsonl/type*_test.jsonl (default: data/processed/).",
    )
    parser.add_argument(
        "--split-jsonl-type1",
        default=None,
        metavar="FILE",
        help="Override the JSONL path for type1_<split>.jsonl for the selected --split.",
    )
    parser.add_argument(
        "--split-jsonl-type2",
        default=None,
        metavar="FILE",
        help="Override the JSONL path for type2_<split>.jsonl for the selected --split.",
    )
    parser.add_argument("--generated-type1", default=None, metavar="FILE")
    parser.add_argument("--generated-type2", default=None, metavar="FILE")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["length"],
        choices=METRIC_CHOICES + ["all"],
        help="Which metrics to compute. 'all' runs everything.",
    )
    parser.add_argument("--embedder", default="sentence-transformers/LaBSE",
                        help="Sentence-transformers model for cosine similarity.")
    parser.add_argument("--mauve-model", default="xlm-roberta-base",
                        help="HuggingFace model for MAUVE featurization.")
    parser.add_argument("--out-dir", default=None, metavar="DIR",
                        help="Save per-record CSVs and JSON summary here.")
    args = parser.parse_args()

    setup_logging("evaluate")

    data_cfg = _load_data_config(Path(args.data_config))
    limit = _resolve_limit(args.limit, args.limit_from_config, data_cfg)
    metrics = set(METRIC_CHOICES if "all" in args.metrics else args.metrics)
    generated_map = {"type1": args.generated_type1, "type2": args.generated_type2}
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(args.data_dir) if args.data_dir else PROCESSED_DIR

    split_overrides: dict[str, Path | None] = {
        "type1": Path(args.split_jsonl_type1) if args.split_jsonl_type1 else None,
        "type2": Path(args.split_jsonl_type2) if args.split_jsonl_type2 else None,
    }

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    records: dict[str, list[dict]] = {}
    generated: dict[str, list[str] | None] = {}

    for st in args.types:
        override = split_overrides.get(st)
        if override is not None:
            if not override.exists():
                log.error("Split override file not found for %s: %s", st, override)
                sys.exit(1)
            recs = load_jsonl(override)
        else:
            recs = _load_split(base_dir, st, args.split)
        if limit is not None:
            recs = recs[:limit]
        if not recs:
            continue
        records[st] = recs

        gen = load_generated(generated_map.get(st))
        if gen is not None and limit is not None:
            gen = gen[:limit]
        if gen is not None and len(gen) != len(recs):
            log.warning("%s: generated count (%d) != records (%d). Ignoring.", st, len(gen), len(recs))
            gen = None
        generated[st] = gen

    if not records:
        log.error("No data loaded.")
        sys.exit(1)

    if limit is not None:
        log.info("eval limit=%d  source=%s", limit, args.data_config if args.limit_from_config else "cli")

    summary: dict[str, dict] = {st: {} for st in records}

    # -----------------------------------------------------------------------
    # Length
    # -----------------------------------------------------------------------
    if "length" in metrics:
        log.info("--- [1/5] LENGTH COMPARISON ---")
        import pandas as pd
        dfs = []
        for st, recs in records.items():
            df = compare_lengths(recs, generated.get(st))
            dfs.append(df)
            if out_dir:
                df.to_csv(out_dir / f"length_{st}.csv", index=False)
        combined = pd.concat(dfs, ignore_index=True)
        log.info("Length report:\n%s", length_report(combined))

    # -----------------------------------------------------------------------
    # Style classifier — train once, reuse for both "style" and "crosstype"
    # -----------------------------------------------------------------------
    clf = None
    if "style" in metrics or "crosstype" in metrics:
        train1 = _load_split(base_dir, "type1", "train")
        train2 = _load_split(base_dir, "type2", "train")
        if not train1 or not train2:
            log.warning("Need both type1 and type2 train splits for style/crosstype metrics.")
        else:
            clf = train_style_classifier(
                [_response_text(r) for r in train1],
                [_response_text(r) for r in train2],
            )

    if "style" in metrics:
        log.info("--- [2/5] STYLE CLASSIFIER (TF-IDF + LogReg) ---")
        if clf is None:
            log.warning("Classifier not available. Skipping.")
        else:
            for st, recs in records.items():
                ref_texts = [_response_text(r) for r in recs]
                ref_scores = style_classifier_scores(clf, ref_texts, target_label=st)
                log.info("%s | Reference posts → accuracy=%.3f  confidence=%.3f",
                         st, ref_scores["style_accuracy"], ref_scores["style_confidence_mean"])
                summary[st]["style_ref"] = ref_scores

                gen = generated.get(st)
                if gen:
                    gen_scores = style_classifier_scores(clf, gen, target_label=st)
                    log.info("%s | Generated texts → accuracy=%.3f  confidence=%.3f",
                             st, gen_scores["style_accuracy"], gen_scores["style_confidence_mean"])
                    summary[st]["style_gen"] = gen_scores

    # -----------------------------------------------------------------------
    # Cross-type confusion matrix ("не средний по больнице")
    # -----------------------------------------------------------------------
    if "crosstype" in metrics:
        log.info("--- [3/5] CROSS-TYPE CONFUSION MATRIX ---")
        if clf is None:
            log.warning("Classifier not available. Skipping.")
        else:
            gen1 = generated.get("type1")
            gen2 = generated.get("type2")
            if not gen1 or not gen2:
                log.warning("Need --generated-type1 and --generated-type2. Skipping.")
            else:
                result = cross_type_confusion_matrix(clf, gen1, gen2)
                classes = result["classes"]
                lines = [" " * 18 + "  ".join(f"→ {c}" for c in classes)]
                for row_label, row_key in [("type1 adapter", "type1_gen"), ("type2 adapter", "type2_gen")]:
                    fracs = result[row_key]
                    n = result[f"n_{row_key.split('_')[0]}"]
                    vals = "  ".join(f"{fracs[c] * 100:6.1f}%" for c in classes)
                    lines.append(f"{row_label:<16}  {vals}   (n={n})")
                diag = result["diagonal_mean"] * 100
                lines.append(f"\nDiagonal mean (correct-class): {diag:.1f}%  |  random baseline: 50.0%")
                log.info("\n".join(lines))
                for st in records:
                    summary[st]["crosstype"] = result
                if out_dir:
                    conf_path = out_dir / "cross_type_confusion.json"
                    with open(conf_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    log.info("Saved → %s", conf_path)

    # -----------------------------------------------------------------------
    # Cosine similarity (content preservation)
    # -----------------------------------------------------------------------
    if "cosine" in metrics:
        log.info("--- [4/5] COSINE SIMILARITY (input vs generated) — model: %s ---", args.embedder)
        any_generated = any(generated.get(st) for st in records)
        if not any_generated:
            log.warning("No generated outputs provided. Skipping cosine metric.")
        else:
            embedder = load_embedder(args.embedder)
            for st, recs in records.items():
                gen = generated.get(st)
                if not gen:
                    log.warning("%s: no generated outputs, skipping.", st)
                    continue
                inputs = [r["input"] for r in recs]
                scores = cosine_similarity_scores(embedder, inputs, gen)
                log.info(
                    "%s | cosine mean=%.4f  median=%.4f  std=%.4f  [%.4f, %.4f]",
                    st, scores["cosine_mean"], scores["cosine_median"],
                    scores["cosine_std"], scores["cosine_min"], scores["cosine_max"],
                )
                summary[st]["cosine"] = {k: v for k, v in scores.items() if k != "per_sample"}
                if out_dir:
                    import pandas as pd
                    pd.DataFrame({"cosine": scores["per_sample"]}).to_csv(
                        out_dir / f"cosine_{st}.csv", index=False
                    )

    # -----------------------------------------------------------------------
    # MAUVE
    # -----------------------------------------------------------------------
    if "mauve" in metrics:
        log.info("--- [5/5] MAUVE (generated vs reference corpus) — model: %s ---", args.mauve_model)
        any_generated = any(generated.get(st) for st in records)
        if not any_generated:
            log.warning("No generated outputs provided. Skipping MAUVE.")
        else:
            for st, recs in records.items():
                gen = generated.get(st)
                if not gen:
                    log.warning("%s: no generated outputs, skipping.", st)
                    continue
                if len(gen) < 50:
                    log.warning("%s: only %d samples — MAUVE is unreliable below 50. Proceeding.", st, len(gen))
                ref_texts = [_response_text(r) for r in recs]
                scores = mauve_score(gen, ref_texts, featurize_model_name=args.mauve_model)
                log.info("%s | MAUVE=%.4f  frontier_integral=%.4f", st, scores["mauve"], scores["frontier_integral"])
                summary[st]["mauve"] = scores

    # -----------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------
    if out_dir and any(summary.values()):
        summary_path = out_dir / "eval_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log.info("Summary → %s", summary_path)


if __name__ == "__main__":
    main()
