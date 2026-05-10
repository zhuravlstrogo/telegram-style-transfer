#!/usr/bin/env python3
"""
Generate split-safe synthetic inputs via OpenRouter LLM neutralization.

Usage:
    python scripts/generate_openrouter_synthetic.py
    python scripts/generate_openrouter_synthetic.py --types type1 type2 --splits train val
    python scripts/generate_openrouter_synthetic.py --model openai/gpt-4.1-mini --fallback-model openai/gpt-4.1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telegram_style_transfer.llm_neutralization import (  # noqa: E402
    DEFAULT_MAX_JACCARD,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PRIMARY_MODEL,
    NEUTRALIZATION_RESPONSE_FORMAT,
    SYSTEM_PROMPT,
    build_user_prompt,
    estimate_cost_usd,
    parse_neutralization_payload,
    summarize_usage_and_cost,
    usage_from_response,
    validate_neutralization,
)
from telegram_style_transfer.paths import CONFIGS_DIR, PROCESSED_DIR  # noqa: E402
from telegram_style_transfer.prompts import format_training_prompt  # noqa: E402
from telegram_style_transfer.synthetic import (  # noqa: E402
    enrich_synthetic_record,
    evaluate_quality_summary,
    summarize_quality,
    utc_timestamp,
    write_jsonl,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_CONFIG_PATH = CONFIGS_DIR / "data.yaml"


def _load_data_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, min=2, max=60))
async def call_neutralizer(
    client: AsyncOpenAI,
    model: str,
    text: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    allow_plaintext_fallback: bool,
) -> dict[str, Any]:
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(text)},
            ],
            response_format=NEUTRALIZATION_RESPONSE_FORMAT,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            extra_body={"provider": {"require_parameters": True}},
        )

    content = response.choices[0].message.content or ""
    neutral_text = parse_neutralization_payload(
        content,
        allow_plaintext_fallback=allow_plaintext_fallback,
    )
    return {
        "neutral_text": neutral_text,
        "raw_content": content,
        **usage_from_response(response),
    }


async def neutralize_record(
    client: AsyncOpenAI,
    record: dict[str, Any],
    primary_model: str,
    fallback_model: str | None,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    max_jaccard: float,
    allow_plaintext_fallback: bool,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    models = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models.append(fallback_model)

    final_validation: dict[str, Any] | None = None
    final_model = primary_model
    for model_name in models:
        try:
            result = await call_neutralizer(
                client=client,
                model=model_name,
                text=record["response_clean"],
                semaphore=semaphore,
                max_tokens=max_tokens,
                allow_plaintext_fallback=allow_plaintext_fallback,
            )
            validation = validate_neutralization(
                response_clean=record["response_clean"],
                candidate=result["neutral_text"],
                max_jaccard=max_jaccard,
            )
            attempts.append(
                {
                    "model": model_name,
                    "raw_content": result["raw_content"],
                    "neutral_text": result["neutral_text"],
                    **validation,
                    "usage_prompt_tokens": result["usage_prompt_tokens"],
                    "usage_completion_tokens": result["usage_completion_tokens"],
                    "usage_total_tokens": result["usage_total_tokens"],
                }
            )
            final_validation = validation
            final_model = model_name
            if not validation["failed"]:
                break
        except Exception as exc:  # pragma: no cover - exercised only against live API
            attempts.append(
                {
                    "model": model_name,
                    "raw_content": "",
                    "neutral_text": "",
                    "failed": True,
                    "failure_reason": f"api_error:{type(exc).__name__}",
                    "usage_prompt_tokens": 0,
                    "usage_completion_tokens": 0,
                    "usage_total_tokens": 0,
                }
            )
            final_validation = {
                "neutralized_clean": "",
                "failed": True,
                "failure_reason": f"api_error:{type(exc).__name__}",
                "jaccard": None,
                "is_identity": False,
                "numbers_preserved": False,
                "length_ratio": None,
                "quality_flags": {
                    "numbers_ok": False,
                    "identity": False,
                    "jaccard_bin": None,
                    "length_collapse": None,
                },
            }
            final_model = model_name

    assert final_validation is not None

    total_prompt_tokens = sum(item["usage_prompt_tokens"] for item in attempts)
    total_completion_tokens = sum(item["usage_completion_tokens"] for item in attempts)
    cost = round(
        sum(
            estimate_cost_usd(
                item["usage_prompt_tokens"],
                item["usage_completion_tokens"],
                item["model"],
            )
            or 0.0
            for item in attempts
        ),
        6,
    )

    extra_fields = {
        "prompt": "",
        "usage_prompt_tokens": total_prompt_tokens,
        "usage_completion_tokens": total_completion_tokens,
        "usage_total_tokens": total_prompt_tokens + total_completion_tokens,
        "cost_usd_estimate": cost,
        "neutralizer_primary_model": primary_model,
        "neutralizer_final_model": final_model,
        "neutralizer_fallback_model": fallback_model,
        "neutralizer_attempts": len(attempts),
        "neutralizer_used_fallback": any(
            item["model"] != primary_model for item in attempts
        ),
        "neutralizer_attempt_log": attempts,
    }

    enriched = enrich_synthetic_record(
        record=record,
        synthetic_input=final_validation["neutralized_clean"],
        method="openrouter",
        model_name=final_model,
        model_config={
            "primary_model": primary_model,
            "fallback_model": fallback_model,
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": "json_schema",
            "provider_require_parameters": True,
            "max_tokens": max_tokens,
            "max_jaccard": max_jaccard,
        },
        created_at=utc_timestamp(),
        failure_reason=final_validation["failure_reason"],
        extra_fields=extra_fields,
    )
    if enriched["synthetic_ok"]:
        enriched["prompt"] = format_training_prompt(
            enriched["style_type"],
            enriched["input"],
            enriched["response_clean"],
        )
    return enriched


def _ok_count_in_records(
    records: list[dict[str, Any]],
    existing_by_post_id: dict[Any, dict[str, Any]],
) -> int:
    return sum(
        1
        for record in records
        if existing_by_post_id.get(record["post_id"], {}).get("synthetic_ok")
    )


async def process_file(
    client: AsyncOpenAI,
    records: list[dict[str, Any]],
    out_path: Path,
    primary_model: str,
    fallback_model: str | None,
    concurrency: int,
    max_tokens: int,
    max_jaccard: float,
    allow_plaintext_fallback: bool,
    save_every: int,
    target_ok: int = 0,
) -> list[dict[str, Any]]:
    existing_by_post_id: dict[Any, dict[str, Any]] = {}
    relabeled = 0
    if out_path.exists():
        with open(out_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Backfill: older runs left input_source="heuristic" on synthetic_ok rows.
                # finetune.py filters by input_source, so relabel without API calls.
                if row.get("synthetic_ok"):
                    method = row.get("synthetic_method") or "openrouter"
                    expected = f"{method}_synthetic"
                    if row.get("input_source") != expected:
                        row["input_source"] = expected
                        relabeled += 1
                existing_by_post_id[row["post_id"]] = row
    if relabeled:
        log.info("%s: relabeled input_source on %d cached rows", out_path.name, relabeled)

    pending = [record for record in records if record["post_id"] not in existing_by_post_id]
    # Newest-first: spend the API budget on recent posts. If we hit target_ok before
    # exhausting `pending`, the older tail is left unprocessed.
    pending.sort(key=lambda r: r.get("date", ""), reverse=True)

    semaphore = asyncio.Semaphore(concurrency)
    ok_cached = _ok_count_in_records(records, existing_by_post_id)
    log.info(
        "%s pending=%d cached=%d cached_ok=%d target_ok=%s",
        out_path.name,
        len(pending),
        len(existing_by_post_id),
        ok_cached,
        target_ok or "none",
    )

    if target_ok and ok_cached >= target_ok:
        log.info("%s target_ok already met (%d >= %d); skipping API calls",
                 out_path.name, ok_cached, target_ok)
        if relabeled:
            ordered = [
                existing_by_post_id[record["post_id"]]
                for record in records if record["post_id"] in existing_by_post_id
            ]
            write_jsonl(out_path, ordered)
    else:
        for start in range(0, len(pending), save_every):
            chunk = pending[start : start + save_every]
            tasks = [
                neutralize_record(
                    client=client,
                    record=record,
                    primary_model=primary_model,
                    fallback_model=fallback_model,
                    semaphore=semaphore,
                    max_tokens=max_tokens,
                    max_jaccard=max_jaccard,
                    allow_plaintext_fallback=allow_plaintext_fallback,
                )
                for record in chunk
            ]
            results = await atqdm.gather(
                *tasks,
                desc=f"{out_path.stem}[{start + 1}:{start + len(chunk)}]",
            )
            for result in results:
                existing_by_post_id[result["post_id"]] = result
            ordered = [existing_by_post_id[record["post_id"]] for record in records if record["post_id"] in existing_by_post_id]
            write_jsonl(out_path, ordered)

            if target_ok:
                ok_now = _ok_count_in_records(records, existing_by_post_id)
                if ok_now >= target_ok:
                    log.info(
                        "%s target_ok reached (%d >= %d) after %d / %d pending; stopping early",
                        out_path.name, ok_now, target_ok,
                        min(start + save_every, len(pending)), len(pending),
                    )
                    break

    return [existing_by_post_id[record["post_id"]] for record in records if record["post_id"] in existing_by_post_id]


async def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set — add it to .env or export it")

    client = make_client(api_key)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, Any]] = []

    for style_type in args.types:
        for split in args.splits:
            src_path = PROCESSED_DIR / f"{style_type}_{split}.jsonl"
            if not src_path.exists():
                log.warning("Not found, skipping: %s", src_path)
                continue

            with open(src_path, encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            if not records:
                log.warning("Empty file, skipping: %s", src_path)
                continue

            out_path = out_dir / f"{style_type}_{split}.jsonl"
            # target_ok applies only to train (val/test must cover the full split).
            target_ok = args.target_ok if split == "train" else 0
            enriched_records = await process_file(
                client=client,
                records=records,
                out_path=out_path,
                primary_model=args.model,
                fallback_model=args.fallback_model,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                max_jaccard=args.max_jaccard,
                allow_plaintext_fallback=args.allow_plaintext_fallback,
                save_every=args.save_every,
                target_ok=target_ok,
            )
            if args.drop_failed:
                enriched_records = [
                    record for record in enriched_records if record["synthetic_ok"]
                ]
                write_jsonl(out_path, enriched_records)

            summary = summarize_quality(
                pd.DataFrame(enriched_records),
                failed_column="synthetic_failed",
            )
            summary.update(evaluate_quality_summary(summary))
            summary.update(
                summarize_usage_and_cost(
                    enriched_records,
                    model=args.model,
                    estimate_missing=False,
                )
            )
            summary.update(
                {
                    "style_type": style_type,
                    "split": split,
                    "source_path": str(src_path),
                    "output_path": str(out_path),
                }
            )
            report_rows.append(summary)
            log.info(
                "[%s_%s] saved %d records → %s (quality_gate=%s, cost=$%.4f)",
                style_type,
                split,
                len(enriched_records),
                out_path,
                summary.get("quality_gate_passed"),
                summary.get("cost_usd_estimate", 0.0),
            )

    report = {
        "meta": {
            "method": "openrouter",
            "model": args.model,
            "fallback_model": args.fallback_model,
            "created_at": utc_timestamp(),
            "splits": args.splits,
            "types": args.types,
            "max_jaccard": args.max_jaccard,
            "concurrency": args.concurrency,
            "save_every": args.save_every,
        },
        "files": report_rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    log.info("Report saved → %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate split-safe synthetic inputs via OpenRouter neutralization"
    )
    parser.add_argument("--model", default=DEFAULT_PRIMARY_MODEL)
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument("--types", nargs="+", default=["type1", "type2"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--concurrency", type=int, default=8, metavar="N")
    parser.add_argument("--max-tokens", type=int, default=512, metavar="N")
    parser.add_argument("--max-jaccard", type=float, default=DEFAULT_MAX_JACCARD)
    parser.add_argument("--save-every", type=int, default=25, metavar="N")
    parser.add_argument(
        "--out-dir",
        default="data/processed/synthetic_openrouter",
        help="Directory for synthetic splits",
    )
    parser.add_argument(
        "--report",
        default="data/processed/synthetic_openrouter/report.json",
        help="Summary report path",
    )
    parser.add_argument("--drop-failed", action="store_true")
    parser.add_argument("--allow-plaintext-fallback", action="store_true")
    parser.add_argument(
        "--config",
        default=str(DATA_CONFIG_PATH),
        metavar="FILE",
        help=f"Dataset config; provides target_train_ok default (default: {DATA_CONFIG_PATH})",
    )
    parser.add_argument(
        "--target-ok",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Target number of synthetic_ok records per train file; "
            "process newest-first and early-stop once reached. "
            "Applies to train splits only. "
            "Defaults to target_train_ok from configs/data.yaml; 0 disables."
        ),
    )
    args = parser.parse_args()

    if args.target_ok is None:
        cfg = _load_data_config(Path(args.config))
        args.target_ok = int(cfg.get("target_train_ok") or 0)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
