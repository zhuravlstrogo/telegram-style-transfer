from __future__ import annotations

from pathlib import Path
from typing import Any

from telegram_style_transfer.data import clean_for_input
from telegram_style_transfer.synthetic import read_jsonl


def _input_source_name(record: dict[str, Any]) -> str:
    prompt_mode = str(record.get("prompt_mode", "") or "").strip()
    if prompt_mode:
        return f"llm_{prompt_mode}"
    return "llm_synthetic"


def load_brief_overrides_from_records(
    records: list[dict[str, Any]],
    max_jaccard: float | None = None,
) -> tuple[dict[tuple[str, int], dict[str, str]], dict[str, int]]:
    overrides: dict[tuple[str, int], dict[str, str]] = {}
    stats = {
        "loaded": 0,
        "accepted": 0,
        "rejected_failed": 0,
        "rejected_missing_input": 0,
        "rejected_jaccard": 0,
    }

    for record in records:
        stats["loaded"] += 1

        style_type = str(record.get("style_type", "") or "").strip()
        post_id = record.get("post_id")
        if not style_type or post_id is None:
            continue

        synthetic_ok = record.get("synthetic_ok")
        failed = record.get("failed")
        is_success = synthetic_ok is True or failed is False
        if not is_success:
            stats["rejected_failed"] += 1
            continue

        raw_input = (
            record.get("neutral")
            or record.get("synthetic_input")
            or record.get("input")
            or ""
        )
        input_text = clean_for_input(str(raw_input))
        if not input_text:
            stats["rejected_missing_input"] += 1
            continue

        if max_jaccard is not None:
            jaccard = record.get("jaccard")
            if jaccard is None or float(jaccard) >= max_jaccard:
                stats["rejected_jaccard"] += 1
                continue

        overrides[(style_type, int(post_id))] = {
            "input": input_text,
            "input_source": _input_source_name(record),
        }
        stats["accepted"] += 1

    return overrides, stats


def load_brief_overrides(
    path: str | Path,
    max_jaccard: float | None = None,
) -> tuple[dict[tuple[str, int], dict[str, str]], dict[str, int]]:
    records = read_jsonl(path)
    return load_brief_overrides_from_records(records, max_jaccard=max_jaccard)
