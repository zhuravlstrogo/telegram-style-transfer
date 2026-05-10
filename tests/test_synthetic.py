import json

import pandas as pd

from telegram_style_transfer.synthetic import (
    build_pilot_manifest,
    evaluate_quality_summary,
    score_neutralization,
    summarize_quality,
)


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_build_pilot_manifest_uses_val_splits(tmp_path):
    records_type1 = [
        {
            "post_id": idx,
            "style_type": "type1",
            "split": "val",
            "date": f"2026-01-0{idx} 10:00:00",
            "response_clean": f"Текст {idx} с числом {idx}",
            "char_len": 100 + idx,
            "n_paragraphs": 1,
            "has_emoji": False,
        }
        for idx in range(1, 5)
    ]
    records_type2 = [
        {
            "post_id": 100 + idx,
            "style_type": "type2",
            "split": "val",
            "date": f"2026-01-0{idx} 11:00:00",
            "response_clean": f"Другой текст {idx} без эмодзи",
            "char_len": 120 + idx,
            "n_paragraphs": 2,
            "has_emoji": False,
        }
        for idx in range(1, 5)
    ]
    _write_jsonl(tmp_path / "type1_val.jsonl", records_type1)
    _write_jsonl(tmp_path / "type2_val.jsonl", records_type2)

    manifest = build_pilot_manifest(
        processed_dir=tmp_path,
        types=["type1", "type2"],
        n_per_type=2,
        split="val",
        seed=42,
    )

    assert len(manifest) == 4
    assert set(manifest["style_type"]) == {"type1", "type2"}
    assert set(manifest["source_split"]) == {"val"}
    assert all(manifest["pilot_id"].str.contains(":"))


def test_summarize_quality_reports_expected_metrics():
    base = {
        "response_clean": "Текст 123",
        "synthetic_failed": False,
    }
    good = score_neutralization("Текст 123", "Нейтральный текст 123", failed=False)
    bad = score_neutralization("Текст 123", "", failed=True)
    frame = pd.DataFrame(
        [
            base | good,
            {"response_clean": "Текст 123", "synthetic_failed": True} | bad,
        ]
    )

    summary = summarize_quality(frame, failed_column="synthetic_failed")

    assert summary["n_total"] == 2
    assert summary["n_active"] == 1
    assert summary["failed_rate"] == 0.5


def test_evaluate_quality_summary_flags_gate_failures():
    summary = {
        "jaccard_ge_0_5": 0.04,
        "identity_rate": 0.01,
        "number_preservation": 0.98,
        "length_collapse_rate": 0.08,
        "good_neutral_rate": 0.55,
    }

    result = evaluate_quality_summary(summary)

    assert result["jaccard_ge_0_5_pass"] is True
    assert result["good_neutral_rate_pass"] is False
    assert result["quality_gate_passed"] is False
