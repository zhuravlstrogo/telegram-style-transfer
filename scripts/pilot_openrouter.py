#!/usr/bin/env python3
"""
Pilot: neutralize a fixed stratified val sample via OpenRouter.

Writes partial checkpoints every N records, can resume from an existing JSONL,
and treats provider-side content filters / 4xx request rejections as failed
records instead of crashing the whole run.

Usage:
    python scripts/pilot_openrouter.py
    python scripts/pilot_openrouter.py --types type1 type2 --n-per-type 150
    python scripts/pilot_openrouter.py --out data/processed/pilot_openrouter.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telegram_style_transfer.data import clean_for_input
from telegram_style_transfer.llm_neutralization import (
    build_brief_text,
    build_brief_v3_text,
    DEFAULT_MAX_JACCARD,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PRIMARY_MODEL,
    build_user_prompt,
    estimate_cost_usd,
    get_prompt_mode,
    parse_neutralization_payload,
    prompt_mode_names,
    response_format_for,
    validate_neutralization,
)
from telegram_style_transfer.synthetic import (
    add_sampling_columns,
    evaluate_quality_summary,
    score_neutralization,
    summarize_quality,
    utc_timestamp,
    write_jsonl,
)
from telegram_style_transfer.paths import PROCESSED_DIR

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 10
DEFAULT_SAVE_EVERY = 25
BRIEF_V4_BORDERLINE_MIN = 0.30
BRIEF_V4_BORDERLINE_MAX = 0.45
BRIEF_V4_MAX_SHARED_NGRAM = 5
BRIEF_V4_MAX_COPY_4GRAM_RATIO = 0.25
DEFAULT_OUT_PATH = "data/processed/pilot_openrouter.jsonl"
RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    APITimeoutError,
)
STRICT_FAILURE_REASONS = {"copied_span", "copied_4gram_ratio"}


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    sampled = add_sampling_columns(df)
    shuffled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    proportions = shuffled["stratum"].value_counts(normalize=True)
    selected_chunks: list[pd.DataFrame] = []

    for stratum, share in proportions.items():
        quota = max(1, round(share * n))
        chunk = shuffled[shuffled["stratum"] == stratum].head(quota)
        selected_chunks.append(chunk)

    selected = pd.concat(selected_chunks).drop_duplicates().head(n).reset_index(drop=True)
    return selected


def _make_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
async def _call_messages(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
) -> Any:
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0.0,
        top_p=1.0,
        extra_body={"provider": {"require_parameters": True}},
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
async def _call_once(client: AsyncOpenAI, model: str, text: str) -> Any:
    return await _call_once_with_mode(
        client,
        model,
        text,
        prompt_mode="rewrite",
        fact_limit=5,
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
async def _call_once_with_mode(
    client: AsyncOpenAI,
    model: str,
    text: str,
    prompt_mode: str,
    fact_limit: int,
) -> Any:
    mode = get_prompt_mode(prompt_mode)
    system_prompt = str(mode["system"]).format(
        fact_limit=fact_limit,
        entity_limit=fact_limit,
        number_limit=fact_limit,
    )
    return await _call_messages(
        client,
        model,
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_prompt(
                    text,
                    prompt_mode=prompt_mode,
                    fact_limit=fact_limit,
                ),
            },
        ],
        response_format_for(prompt_mode),
    )


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "usage_prompt_tokens": 0,
            "usage_completion_tokens": 0,
            "usage_total_tokens": 0,
        }
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    return {
        "usage_prompt_tokens": prompt_tokens,
        "usage_completion_tokens": completion_tokens,
        "usage_total_tokens": total_tokens,
    }


def _failed_record(
    row: pd.Series,
    reason: str,
    model: str,
    primary_model: str,
    fallback_model: str | None,
    prompt_mode: str,
    usage: dict[str, int] | None = None,
    cost_usd_estimate: float | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage = usage or {
        "usage_prompt_tokens": 0,
        "usage_completion_tokens": 0,
        "usage_total_tokens": 0,
    }
    record = {
        "post_id": row.get("post_id"),
        "style_type": row.get("style_type"),
        "source_split": row.get("split"),
        "response_clean": row["response_clean"],
        "neutral": "",
        "jaccard": None,
        "is_identity": False,
        "numbers_preserved": False,
        "length_ratio": None,
        "n_paragraphs": row.get("n_paragraphs"),
        "has_emoji": row.get("has_emoji"),
        "char_len": row.get("char_len"),
        "failed": True,
        "failure_reason": reason,
        "model": model,
        "primary_model": primary_model,
        "fallback_model": fallback_model,
        "prompt_mode": prompt_mode,
        "created_at": utc_timestamp(),
        "cost_usd_estimate": (
            cost_usd_estimate
            if cost_usd_estimate is not None
            else estimate_cost_usd(
                usage["usage_prompt_tokens"],
                usage["usage_completion_tokens"],
                model,
            )
        ),
        **usage,
    }
    if extra_fields:
        record.update(extra_fields)
    return record


def _empty_usage() -> dict[str, int]:
    return {
        "usage_prompt_tokens": 0,
        "usage_completion_tokens": 0,
        "usage_total_tokens": 0,
    }


def _merge_usage(total_usage: dict[str, int], usage: dict[str, int]) -> None:
    total_usage["usage_prompt_tokens"] += usage["usage_prompt_tokens"]
    total_usage["usage_completion_tokens"] += usage["usage_completion_tokens"]
    total_usage["usage_total_tokens"] += usage["usage_total_tokens"]


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _load_json_payload(raw: str) -> dict[str, Any]:
    payload = json.loads(_strip_json_fences(raw))
    if not isinstance(payload, dict):
        raise ValueError("Neutralization payload must be a JSON object")
    return payload


def _parse_brief_v3_payload(raw: str) -> dict[str, Any]:
    payload = _load_json_payload(raw)
    topic = str(payload.get("topic", "") or "").strip()
    entities = payload.get("entities", [])
    numbers = payload.get("numbers", [])
    facts = payload.get("facts", [])
    if not isinstance(entities, list):
        raise ValueError("brief_v3 payload.entities must be a list")
    if not isinstance(numbers, list):
        raise ValueError("brief_v3 payload.numbers must be a list")
    if not isinstance(facts, list):
        raise ValueError("brief_v3 payload.facts must be a list")
    return {
        "topic": topic,
        "entities": [str(item).strip() for item in entities if str(item).strip()],
        "numbers": [str(item).strip() for item in numbers if str(item).strip()],
        "facts": [str(item).strip() for item in facts if str(item).strip()],
    }


def _validate_candidate(
    response_clean: str,
    candidate: str,
    max_jaccard: float,
    prompt_mode: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"max_jaccard": max_jaccard}
    if prompt_mode == "brief_v4":
        kwargs.update(
            {
                "max_shared_ngram": BRIEF_V4_MAX_SHARED_NGRAM,
                "max_copy_ngram_ratio": BRIEF_V4_MAX_COPY_4GRAM_RATIO,
                "copy_ngram_size": 4,
            }
        )
    return validate_neutralization(
        response_clean=response_clean,
        candidate=candidate,
        **kwargs,
    )


def _is_brief_v4_borderline(validation: dict[str, Any]) -> bool:
    jaccard = validation.get("jaccard")
    if jaccard is None:
        return False
    return BRIEF_V4_BORDERLINE_MIN <= jaccard < BRIEF_V4_BORDERLINE_MAX


def _should_postedit_brief_v4(
    base_validation: dict[str, Any],
    strict_validation: dict[str, Any],
) -> bool:
    if not base_validation["failed"] and _is_brief_v4_borderline(base_validation):
        return True
    if strict_validation["failed"] and strict_validation["failure_reason"] in STRICT_FAILURE_REASONS:
        return True
    return False


def _validation_rank(validation: dict[str, Any]) -> tuple[float, float, float]:
    jaccard = float(validation.get("jaccard") or 1.0)
    copy_ratio = float(validation.get("copy_ngram_ratio") or 1.0)
    shared_ngram = float(validation.get("longest_common_ngram") or 0.0)
    return (jaccard, copy_ratio, shared_ngram)


def _pick_better_validation(
    current: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if current["failed"] and not candidate["failed"]:
        return candidate
    if not current["failed"] and candidate["failed"]:
        return current
    if _validation_rank(candidate) < _validation_rank(current):
        return candidate
    return current


async def _run_brief_v4_pipeline(
    client: AsyncOpenAI,
    current_model: str,
    response_clean: str,
    fact_limit: int,
    max_jaccard: float,
) -> dict[str, Any]:
    total_usage = _empty_usage()
    total_cost_usd = 0.0
    extracted: dict[str, Any]

    extraction_resp = await _call_once_with_mode(
        client,
        current_model,
        response_clean,
        prompt_mode="brief_v3",
        fact_limit=fact_limit,
    )
    extraction_usage = _usage_dict(extraction_resp)
    _merge_usage(total_usage, extraction_usage)
    total_cost_usd += estimate_cost_usd(
        extraction_usage["usage_prompt_tokens"],
        extraction_usage["usage_completion_tokens"],
        current_model,
    ) or 0.0
    extraction_raw = extraction_resp.choices[0].message.content or ""
    extracted = _parse_brief_v3_payload(extraction_raw)

    extracted_text = build_brief_v3_text(
        extracted["topic"],
        extracted["entities"],
        extracted["numbers"],
        extracted["facts"],
    )
    compression_messages = [
        {
            "role": "system",
            "content": (
                "Ты собираешь финальный factual brief для downstream-модели.\n"
                "Работай только по извлеченной структуре, не по исходному посту.\n"
                "Верни короткий topic и не более {fact_limit} коротких facts.\n"
                "Не делай отдельных секций entities и numbers: важные имена и числа "
                "встраивай внутрь facts, только если они нужны для фактической точности.\n"
                "Каждый fact должен быть кратким, телеграфным и не похожим на готовое "
                "предложение из поста.\n"
                "Ничего не объясняй и не комментируй. Верни только JSON по схеме."
            ).format(fact_limit=fact_limit),
        },
        {
            "role": "user",
            "content": (
                "Собери компактный final brief по структуре ниже.\n"
                f"Оставь не более {fact_limit} facts.\n"
                "<extracted_brief>\n"
                f"{extracted_text}\n"
                "</extracted_brief>"
            ),
        },
    ]
    compression_resp = await _call_messages(
        client,
        current_model,
        compression_messages,
        response_format_for("brief_v4"),
    )
    compression_usage = _usage_dict(compression_resp)
    _merge_usage(total_usage, compression_usage)
    total_cost_usd += estimate_cost_usd(
        compression_usage["usage_prompt_tokens"],
        compression_usage["usage_completion_tokens"],
        current_model,
    ) or 0.0
    compression_raw = compression_resp.choices[0].message.content or ""
    candidate = clean_for_input(
        parse_neutralization_payload(
            compression_raw,
            prompt_mode="brief_v4",
            allow_plaintext_fallback=False,
        )
    )

    base_validation = validate_neutralization(
        response_clean=response_clean,
        candidate=candidate,
        max_jaccard=max_jaccard,
    )
    strict_validation = _validate_candidate(
        response_clean=response_clean,
        candidate=candidate,
        max_jaccard=max_jaccard,
        prompt_mode="brief_v4",
    )
    selected_validation = strict_validation
    used_postedit = False

    if _should_postedit_brief_v4(base_validation, strict_validation):
        postedit_messages = [
            {
                "role": "system",
                "content": (
                    "Ты редактируешь factual brief так, чтобы он был дальше по лексике от "
                    "исходного Telegram-поста, но не терял факты.\n"
                    "Сохрани все важные числа, даты и имена собственные.\n"
                    "Не копируй длинные фразы из исходника. Старайся не повторять "
                    "последовательности длиннее 4 слов подряд.\n"
                    "Оставь только topic и короткие facts, максимум {fact_limit} facts.\n"
                    "Пиши телеграфно. Ничего не объясняй. Верни только JSON по схеме."
                ).format(fact_limit=fact_limit),
            },
            {
                "role": "user",
                "content": (
                    "Сделай brief менее похожим на исходный текст, не теряя факты.\n"
                    "Причина редакции: overlap_guard.\n"
                    "<source_text>\n"
                    f"{response_clean}\n"
                    "</source_text>\n"
                    "<extracted_brief>\n"
                    f"{extracted_text}\n"
                    "</extracted_brief>\n"
                    "<current_brief>\n"
                    f"{candidate}\n"
                    "</current_brief>"
                ),
            },
        ]
        postedit_resp = await _call_messages(
            client,
            current_model,
            postedit_messages,
            response_format_for("brief_v4"),
        )
        postedit_usage = _usage_dict(postedit_resp)
        _merge_usage(total_usage, postedit_usage)
        total_cost_usd += estimate_cost_usd(
            postedit_usage["usage_prompt_tokens"],
            postedit_usage["usage_completion_tokens"],
            current_model,
        ) or 0.0
        postedit_raw = postedit_resp.choices[0].message.content or ""
        postedit_candidate = clean_for_input(
            parse_neutralization_payload(
                postedit_raw,
                prompt_mode="brief_v4",
                allow_plaintext_fallback=False,
            )
        )
        postedit_validation = _validate_candidate(
            response_clean=response_clean,
            candidate=postedit_candidate,
            max_jaccard=max_jaccard,
            prompt_mode="brief_v4",
        )
        better_validation = _pick_better_validation(selected_validation, postedit_validation)
        if better_validation is postedit_validation:
            selected_validation = postedit_validation
            used_postedit = True
        elif not selected_validation["failed"]:
            used_postedit = True

    return {
        "validation": selected_validation,
        "usage": total_usage,
        "cost_usd_estimate": round(total_cost_usd, 6),
        "used_postedit": used_postedit,
        "used_two_pass": True,
    }


async def _call(
    client: AsyncOpenAI,
    row: pd.Series,
    sem: asyncio.Semaphore,
    model: str,
    fallback_model: str | None,
    prompt_mode: str,
    fact_limit: int,
    max_jaccard: float,
    sleep_min: float,
    sleep_max: float,
) -> dict[str, Any]:
    async with sem:
        if sleep_max > 0:
            await asyncio.sleep(random.uniform(sleep_min, sleep_max))
        models = [model]
        if fallback_model and fallback_model != model:
            models.append(fallback_model)
        last_failure: dict[str, Any] | None = None
        total_usage = _empty_usage()
        total_cost_usd = 0.0
        used_model = model
        last_validation: dict[str, Any] | None = None
        used_postedit = False
        used_two_pass = prompt_mode == "brief_v4"

        for current_model in models:
            used_model = current_model
            try:
                if prompt_mode == "brief_v4":
                    pipeline = await _run_brief_v4_pipeline(
                        client=client,
                        current_model=current_model,
                        response_clean=row["response_clean"],
                        fact_limit=fact_limit,
                        max_jaccard=max_jaccard,
                    )
                    _merge_usage(total_usage, pipeline["usage"])
                    total_cost_usd += float(pipeline["cost_usd_estimate"])
                    last_validation = dict(pipeline["validation"])
                    used_postedit = used_postedit or bool(pipeline["used_postedit"])
                else:
                    resp = await _call_once_with_mode(
                        client,
                        current_model,
                        row["response_clean"],
                        prompt_mode=prompt_mode,
                        fact_limit=fact_limit,
                    )
                    raw = resp.choices[0].message.content or ""
                    parsed = parse_neutralization_payload(
                        raw,
                        prompt_mode=prompt_mode,
                        allow_plaintext_fallback=False,
                    )
                    neutral = clean_for_input(parsed)
                    usage = _usage_dict(resp)
                    _merge_usage(total_usage, usage)
                    total_cost_usd += estimate_cost_usd(
                        usage["usage_prompt_tokens"],
                        usage["usage_completion_tokens"],
                        current_model,
                    ) or 0.0
                    last_validation = _validate_candidate(
                        response_clean=row["response_clean"],
                        candidate=neutral,
                        max_jaccard=max_jaccard,
                        prompt_mode=prompt_mode,
                    )
                assert last_validation is not None
                if not last_validation["failed"]:
                    break
                last_failure = _failed_record(
                    row,
                    reason=str(last_validation["failure_reason"]),
                    model=current_model,
                    primary_model=model,
                    fallback_model=fallback_model,
                    prompt_mode=prompt_mode,
                    usage=total_usage,
                    cost_usd_estimate=round(total_cost_usd, 6),
                    extra_fields={
                        "used_postedit": used_postedit,
                        "used_two_pass": used_two_pass,
                        "longest_common_ngram": last_validation.get("longest_common_ngram"),
                        "copy_ngram_ratio": last_validation.get("copy_ngram_ratio"),
                    },
                )
                continue
            except (BadRequestError, PermissionDeniedError) as exc:
                message = str(exc)
                if "content_filter" in message or "403" in message or "Forbidden" in message:
                    last_failure = _failed_record(
                        row,
                        reason=f"provider_rejected:{type(exc).__name__}",
                        model=current_model,
                        primary_model=model,
                        fallback_model=fallback_model,
                        prompt_mode=prompt_mode,
                        usage=total_usage,
                        cost_usd_estimate=round(total_cost_usd, 6),
                        extra_fields={"used_postedit": used_postedit, "used_two_pass": used_two_pass},
                    )
                    continue
                last_failure = _failed_record(
                    row,
                    reason=f"bad_request:{type(exc).__name__}",
                    model=current_model,
                    primary_model=model,
                    fallback_model=fallback_model,
                    prompt_mode=prompt_mode,
                    usage=total_usage,
                    cost_usd_estimate=round(total_cost_usd, 6),
                    extra_fields={"used_postedit": used_postedit, "used_two_pass": used_two_pass},
                )
                continue
            except RETRYABLE_EXCEPTIONS as exc:
                last_failure = _failed_record(
                    row,
                    reason=f"transient_error:{type(exc).__name__}",
                    model=current_model,
                    primary_model=model,
                    fallback_model=fallback_model,
                    prompt_mode=prompt_mode,
                    usage=total_usage,
                    cost_usd_estimate=round(total_cost_usd, 6),
                    extra_fields={"used_postedit": used_postedit, "used_two_pass": used_two_pass},
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive for live API only
                last_failure = _failed_record(
                    row,
                    reason=f"api_error:{type(exc).__name__}",
                    model=current_model,
                    primary_model=model,
                    fallback_model=fallback_model,
                    prompt_mode=prompt_mode,
                    usage=total_usage,
                    cost_usd_estimate=round(total_cost_usd, 6),
                    extra_fields={"used_postedit": used_postedit, "used_two_pass": used_two_pass},
                )
                continue
        else:
            assert last_failure is not None
            return last_failure

    assert last_validation is not None
    neutral = last_validation["neutralized_clean"]

    return {
        "post_id": row.get("post_id"),
        "style_type": row.get("style_type"),
        "source_split": row.get("split"),
        "response_clean": row["response_clean"],
        "neutral": neutral,
        "jaccard": last_validation["jaccard"],
        "is_identity": last_validation["is_identity"],
        "numbers_preserved": last_validation["numbers_preserved"],
        "length_ratio": last_validation["length_ratio"],
        "n_paragraphs": row.get("n_paragraphs"),
        "has_emoji": row.get("has_emoji"),
        "char_len": row.get("char_len"),
        "failed": False,
        "failure_reason": None,
        "model": used_model,
        "primary_model": model,
        "fallback_model": fallback_model,
        "prompt_mode": prompt_mode,
        "used_fallback": used_model != model,
        "used_postedit": used_postedit,
        "used_two_pass": used_two_pass,
        "longest_common_ngram": last_validation.get("longest_common_ngram"),
        "copy_ngram_ratio": last_validation.get("copy_ngram_ratio"),
        "created_at": utc_timestamp(),
        "cost_usd_estimate": round(total_cost_usd, 6),
        **total_usage,
    }


def _report(df_res: pd.DataFrame) -> None:
    if df_res.empty:
        print("No records to report.")
        return

    summary = summarize_quality(df_res, failed_column="failed")
    summary.update(evaluate_quality_summary(summary))

    def fmt_pct(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1%}"

    rows = [
        ("Jaccard ≥ 0.5 (share)", fmt_pct(summary.get("jaccard_ge_0_5")), "<5%"),
        ("Identity rate", fmt_pct(summary.get("identity_rate")), "<2%"),
        ("Number preservation", fmt_pct(summary.get("number_preservation")), "≥95%"),
        ("Length collapse", fmt_pct(summary.get("length_collapse_rate")), "<10%"),
        ("Good neutral", fmt_pct(summary.get("good_neutral_rate")), "≥70%"),
        ("Failed / rejected", fmt_pct(summary.get("failed_rate")), "<1%"),
    ]

    print(f"\n{'=' * 62}")
    print(f"PILOT REPORT — {df_res['primary_model'].iloc[0] if 'primary_model' in df_res.columns else DEFAULT_PRIMARY_MODEL}  (n={len(df_res)})")
    print(f"{'=' * 62}")
    print(f"{'Metric':<32} {'Fact':>10} {'Target':>10}")
    print("-" * 62)
    for name, fact, target in rows:
        print(f"{name:<32} {fact:>10} {target:>10}")

    if "style_type" in df_res.columns and df_res["style_type"].nunique() > 1:
        print("\nBy style_type:")
        grp = df_res.groupby("style_type").agg(
            n=("post_id", "count"),
            failed=("failed", "mean"),
            jaccard_ge_05=("jaccard", lambda x: x.dropna().ge(0.5).mean()),
            good_neutral=("jaccard", lambda x: x.dropna().lt(0.3).mean()),
        )
        print(grp.to_string())


def _load_existing(out_path: Path) -> dict[tuple[Any, Any], dict[str, Any]]:
    if not out_path.exists():
        return {}
    existing: dict[tuple[Any, Any], dict[str, Any]] = {}
    with open(out_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            existing[(row.get("style_type"), row.get("post_id"))] = row
    return existing


def _load_existing_many(paths: list[Path]) -> dict[tuple[Any, Any], dict[str, Any]]:
    merged: dict[tuple[Any, Any], dict[str, Any]] = {}
    for path in paths:
        rows = _load_existing(path)
        merged.update(rows)
    return merged


def _default_out_path(prompt_mode: str, split: str) -> Path:
    if prompt_mode == "brief_v4":
        return Path(f"data/processed/pilot_openrouter_brief_v4_{split}.jsonl")
    return Path(DEFAULT_OUT_PATH)


def _ordered_records(
    pilot: pd.DataFrame,
    by_key: dict[tuple[Any, Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for _, row in pilot.iterrows():
        key = (row.get("style_type"), row.get("post_id"))
        if key in by_key:
            ordered.append(by_key[key])
    return ordered


async def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set — add it to .env or export it")

    client = _make_client(api_key)

    dfs: list[pd.DataFrame] = []
    for style_type in args.types:
        path = PROCESSED_DIR / f"{style_type}_{args.split}.jsonl"
        if not path.exists():
            log.warning("Not found, skipping: %s", path)
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        sampled = stratified_sample(df, args.n_per_type, seed=args.seed)
        log.info(
            "[%s] sampled %d / %d %s records",
            style_type,
            len(sampled),
            len(df),
            args.split,
        )
        dfs.append(sampled)

    if not dfs:
        raise SystemExit(f"No {args.split} files found. Check data/processed/.")

    pilot = pd.concat(dfs, ignore_index=True)
    out_path = Path(args.out) if args.out else _default_out_path(args.prompt_mode, args.split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_paths: list[Path] = []
    if not args.overwrite:
        for raw_path in args.reuse_from:
            path = Path(raw_path)
            if path.exists():
                cache_paths.append(path)
            else:
                log.warning("Reuse cache not found, skipping: %s", path)
        if out_path.exists() and out_path not in cache_paths:
            cache_paths.append(out_path)
    existing = _load_existing_many(cache_paths) if cache_paths else {}
    if existing:
        log.info("Loaded %d cached neutralizations from %d file(s)", len(existing), len(cache_paths))
    pending_rows = [
        row
        for _, row in pilot.iterrows()
        if (row.get("style_type"), row.get("post_id")) not in existing
    ]

    log.info(
        "Total pilot: %d records — calling %s ... pending=%d cached=%d",
        len(pilot),
        args.model,
        len(pending_rows),
        len(existing),
    )

    sem = asyncio.Semaphore(args.concurrency)
    processed = existing.copy()

    for start in range(0, len(pending_rows), args.save_every):
        chunk = pending_rows[start : start + args.save_every]
        tasks = [
            _call(
                client,
                row,
                sem,
                model=args.model,
                fallback_model=args.fallback_model,
                prompt_mode=args.prompt_mode,
                fact_limit=args.fact_limit,
                max_jaccard=args.max_jaccard,
                sleep_min=args.sleep_min,
                sleep_max=args.sleep_max,
            )
            for row in chunk
        ]
        results_with_index = await atqdm.gather(
            *tasks,
            desc=f"neutralizing[{start + 1}:{start + len(chunk)}]",
        )
        for row, result in zip(chunk, results_with_index):
            processed[(row.get("style_type"), row.get("post_id"))] = result

        ordered = _ordered_records(pilot, processed)
        write_jsonl(out_path, ordered)
        log.info(
            "Checkpoint saved: %d / %d → %s",
            len(ordered),
            len(pilot),
            out_path,
        )

    records = _ordered_records(pilot, processed)
    log.info("Saved %d records → %s", len(records), out_path)
    _report(pd.DataFrame(records))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot neutralization via OpenRouter with periodic checkpointing"
    )
    parser.add_argument(
        "--model", default=DEFAULT_PRIMARY_MODEL,
        help="Primary OpenRouter model for pilot neutralization"
    )
    parser.add_argument(
        "--fallback-model", default=DEFAULT_FALLBACK_MODEL,
        help="Fallback model used when primary request is rejected or malformed"
    )
    parser.add_argument(
        "--prompt-mode",
        choices=prompt_mode_names(),
        default="rewrite",
        help="Target prompt/format for pilot output"
    )
    parser.add_argument(
        "--fact-limit", type=int, default=5,
        help="Maximum number of factual bullets for prompt-mode=brief"
    )
    parser.add_argument(
        "--max-jaccard", type=float, default=DEFAULT_MAX_JACCARD,
        help="If overlap with source is >= this threshold, rerun on fallback"
    )
    parser.add_argument(
        "--types", nargs="+", default=["type2"],
        help="Style types to pilot (default: type2)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Which processed split to sample from (default: val)",
    )
    parser.add_argument(
        "--n-per-type", type=int, default=150, metavar="N",
        help="Records per style type (default: 150; use 300 for single type)"
    )
    parser.add_argument(
        "--out",
        default="",
        help=(
            "Output JSONL path. If omitted and --prompt-mode=brief_v4, "
            "defaults to data/processed/pilot_openrouter_brief_v4_<split>.jsonl; "
            f"otherwise defaults to {DEFAULT_OUT_PATH}"
        ),
    )
    parser.add_argument(
        "--reuse-from",
        nargs="+",
        default=[],
        help=(
            "Optional JSONL cache files with previously neutralized records. "
            "Matching (style_type, post_id) rows are reused and skipped."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Ignore any existing output file and recompute the whole pilot"
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY, metavar="N",
        help=f"Max parallel API requests (default: {DEFAULT_CONCURRENCY})"
    )
    parser.add_argument(
        "--save-every", type=int, default=DEFAULT_SAVE_EVERY, metavar="N",
        help=f"Save checkpoint every N records (default: {DEFAULT_SAVE_EVERY})"
    )
    parser.add_argument(
        "--sleep-min", type=float, default=0.2,
        help="Minimum random sleep before each request in seconds"
    )
    parser.add_argument(
        "--sleep-max", type=float, default=1.0,
        help="Maximum random sleep before each request in seconds"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sleep_min < 0 or args.sleep_max < 0 or args.sleep_min > args.sleep_max:
        raise SystemExit("Require 0 <= --sleep-min <= --sleep-max")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
