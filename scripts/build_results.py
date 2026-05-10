#!/usr/bin/env python3
"""Build RESULTS.md from evaluation artefacts.

Reads:
  - reports/eval/baseline_test/eval_summary.json
  - reports/eval/finetuned_test/eval_summary.json
  - reports/eval/baseline_test/cross_type_confusion.json   (optional)
  - reports/eval/finetuned_test/cross_type_confusion.json  (optional)
  - reports/memorization/memorization_type1.csv            (optional)
  - reports/memorization/memorization_type2.csv            (optional)
  - data/processed/split_report.json                       (optional)
  - models/type1/all/run_manifest.json                     (optional)
  - models/type2/all/run_manifest.json                     (optional)

Writes RESULTS.md (or --out FILE).

Usage:
    python scripts/build_results.py
    python scripts/build_results.py --out RESULTS.md
    python scripts/build_results.py \\
        --baseline-dir reports/eval/baseline_test \\
        --finetuned-dir reports/eval/finetuned_test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telegram_style_transfer.logging_utils import setup_logging

log = logging.getLogger("build_results")

TYPES = ["type1", "type2"]
NA = "—"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        log.warning("not found: %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path):
    if not path.exists():
        log.warning("not found: %s", path)
        return None
    import pandas as pd
    return pd.read_csv(path)


def _fmt(v, decimals: int = 3, pct: bool = False) -> str:
    if v is None:
        return NA
    if pct:
        return f"{float(v) * 100:.1f}%"
    return f"{float(v):.{decimals}f}"


def _delta(base, fine, pct: bool = False) -> str:
    if base is None or fine is None:
        return NA
    d = float(fine) - float(base)
    sign = "+" if d >= 0 else ""
    if pct:
        return f"{sign}{d * 100:.1f}pp"
    return f"{sign}{d:.3f}"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_dataset(split_report: dict | None) -> str:
    lines = ["## 1. Датасет и разбивка\n"]
    if split_report is None:
        lines.append("_(данные недоступны — запустите `scripts/prepare_dataset.py`)_\n")
        return "\n".join(lines)

    meta = split_report.get("_meta", {})
    lines.append("| Сплит | type1 | type2 |")
    lines.append("|---|---|---|")
    for split in ["train", "val", "test"]:
        t1 = split_report.get("type1", {}).get(split, NA)
        t2 = split_report.get("type2", {}).get(split, NA)
        lines.append(f"| {split} | {t1} | {t2} |")
    lines.append("")

    strategy = meta.get("split_strategy", NA)
    min_chars = meta.get("min_chars", NA)
    max_samples = meta.get("max_samples", NA)
    input_source = meta.get("input_source", NA)
    lines.append(f"Стратегия разбивки: `{strategy}` (не random, по времени).")
    lines.append(f"Фильтры: min_chars={min_chars}, max_samples={max_samples}.")
    lines.append(f"Input source: `{input_source}`.")
    return "\n".join(lines)


def _section_training(manifests: dict[str, dict | None]) -> str:
    lines = ["## 2. Параметры обучения\n"]

    rows = [
        ("base_model",         lambda m: m.get("base_model", NA)),
        ("LoRA r",             lambda m: m.get("hyperparams", {}).get("lora_r", NA)),
        ("LoRA alpha",         lambda m: m.get("hyperparams", {}).get("lora_alpha", NA)),
        ("LoRA dropout",       lambda m: m.get("hyperparams", {}).get("lora_dropout", NA)),
        ("learning_rate",      lambda m: m.get("hyperparams", {}).get("learning_rate", NA)),
        ("epochs",             lambda m: m.get("hyperparams", {}).get("num_train_epochs", NA)),
        ("batch_size",         lambda m: m.get("hyperparams", {}).get("per_device_train_batch_size", NA)),
        ("grad_accum",         lambda m: m.get("hyperparams", {}).get("gradient_accumulation_steps", NA)),
        ("max_seq_length",     lambda m: m.get("hyperparams", {}).get("max_seq_length", NA)),
        ("train_samples",      lambda m: m.get("train_samples", NA)),
        ("val_samples",        lambda m: m.get("val_samples", NA)),
        ("train_loss",         lambda m: _fmt(m.get("metrics", {}).get("train_loss"), 4)),
        ("eval_loss",          lambda m: _fmt(m.get("metrics", {}).get("eval_loss"), 4)),
        ("peak_vram_gb",       lambda m: _fmt(m.get("metrics", {}).get("peak_vram_gb"), 2)),
        ("duration",           lambda m: _fmt_duration(m.get("duration_seconds"))),
        ("timestamp_utc",      lambda m: m.get("timestamp_utc", NA)[:19] if m.get("timestamp_utc") else NA),
    ]

    if all(v is None for v in manifests.values()):
        lines.append("_(данные недоступны — запустите `scripts/finetune.py` для обоих типов)_\n")
        return "\n".join(lines)

    lines.append("| Параметр | type1 | type2 |")
    lines.append("|---|---|---|")
    for label, getter in rows:
        vals = []
        for st in TYPES:
            m = manifests.get(st)
            vals.append(str(getter(m)) if m else NA)
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    return "\n".join(lines)


def _fmt_duration(seconds) -> str:
    if seconds is None:
        return NA
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _section_before_after(baseline: dict | None, finetuned: dict | None) -> str:
    lines = ["## 3. Сравнение до / после (test split)\n"]

    any_data = baseline or finetuned
    if not any_data:
        lines.append("_(данные недоступны — запустите шаги B2–B5)_\n")
        return "\n".join(lines)

    def _get(summary: dict | None, st: str, *keys):
        if summary is None:
            return None
        node = summary.get(st, {})
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        return node

    header = "| Метрика | baseline type1 | finetuned type1 | Δ | baseline type2 | finetuned type2 | Δ |"
    sep    = "|---|---|---|---|---|---|---|"
    lines += [header, sep]

    metrics = [
        ("style_accuracy",        ("style_gen", "style_accuracy"),        False),
        ("style_confidence_mean", ("style_gen", "style_confidence_mean"), False),
        ("cosine_mean",           ("cosine", "cosine_mean"),              False),
        ("MAUVE",                 ("mauve", "mauve"),                     False),
    ]

    for label, path, is_pct in metrics:
        row = [f"| {label}"]
        for st in TYPES:
            b = _get(baseline, st, *path)
            f = _get(finetuned, st, *path)
            row.append(_fmt(b) if not is_pct else _fmt(b, pct=True))
            row.append(_fmt(f) if not is_pct else _fmt(f, pct=True))
            row.append(_delta(b, f, pct=is_pct))
        lines.append(" | ".join(row) + " |")

    lines.append("")
    lines.append("**Критерий победы**: `style_accuracy ↑` **И** `cosine_mean ≈` **И** `MAUVE ↑`.")
    lines.append("")
    lines.append("Если `style_accuracy` вырос, а `cosine_mean` сильно упал — модель имитирует форму, теряя смысл.")
    return "\n".join(lines)


def _section_crosstype(baseline_conf: dict | None, finetuned_conf: dict | None) -> str:
    lines = ["## 4. Cross-type separability («не средний по больнице»)\n"]

    def _conf_table(conf: dict, label: str) -> list[str]:
        t = [f"### {label}\n"]
        classes = conf.get("classes", ["type1", "type2"])
        t.append("| | " + " | ".join(f"→ {c}" for c in classes) + " |")
        t.append("|---|" + "---|" * len(classes))
        for row_key, row_label in [("type1_gen", "type1 adapter"), ("type2_gen", "type2 adapter")]:
            fracs = conf.get(row_key, {})
            vals = " | ".join(f"{fracs.get(c, 0) * 100:.1f}%" for c in classes)
            n_key = f"n_{row_key.split('_')[0]}"
            n = conf.get(n_key, "?")
            t.append(f"| {row_label} (n={n}) | {vals} |")
        diag = conf.get("diagonal_mean", 0) * 100
        t.append(f"\nDiagonal mean: **{diag:.1f}%** (random baseline: 50.0%)")
        return t

    if baseline_conf is None and finetuned_conf is None:
        lines.append("_(данные недоступны — запустите `evaluate.py --metrics crosstype`)_\n")
        return "\n".join(lines)

    if baseline_conf:
        lines += _conf_table(baseline_conf, "Baseline")
        lines.append("")
    else:
        lines += ["### Baseline\n", "_(нет данных)_\n"]

    if finetuned_conf:
        lines += _conf_table(finetuned_conf, "Fine-tuned")
        lines.append("")
    else:
        lines += ["### Fine-tuned\n", "_(нет данных)_\n"]

    return "\n".join(lines)


def _section_memorization(mem_dfs: dict) -> str:
    lines = ["## 5. Memorization check\n"]

    any_df = any(v is not None for v in mem_dfs.values())
    if not any_df:
        lines.append("_(данные недоступны — запустите `scripts/memorization_check.py`)_\n")
        return "\n".join(lines)

    for st in TYPES:
        df = mem_dfs.get(st)
        if df is None:
            lines.append(f"### {st}\n_(нет данных)_\n")
            continue

        lines.append(f"### {st}\n")
        top5 = df.nlargest(5, "fuzz_ratio")
        lines.append("| generated (фрагмент) | nearest_train_id | fuzz_ratio | jaccard_5gram | lcs_chars |")
        lines.append("|---|---|---|---|---|")
        for _, row in top5.iterrows():
            snippet = str(row["generated"])[:60].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {snippet}… | {row['nearest_train_id']} "
                f"| {row['fuzz_ratio']:.1f} | {row['jaccard_5gram']:.3f} "
                f"| {int(row['longest_common_substr_chars'])} |"
            )
        lines.append("")

        med = df["fuzz_ratio"].median()
        mx = df["fuzz_ratio"].max()
        lcs_max = df["longest_common_substr_chars"].max()
        pass_med = med < 60
        pass_exact = mx < 100
        pass_lcs = lcs_max < 30
        status = "✓ PASS" if (pass_med and pass_exact and pass_lcs) else "✗ FAIL"
        lines.append(f"Median fuzz_ratio: {med:.1f} (target < 60) {'✓' if pass_med else '✗'}")
        lines.append(f"Max fuzz_ratio: {mx:.1f} (target < 100) {'✓' if pass_exact else '✗'}")
        lines.append(f"Max LCS chars: {lcs_max:.0f} (target < 30) {'✓' if pass_lcs else '✗'}")
        lines.append(f"\n**{status}**\n")

    return "\n".join(lines)


def _section_simplifications() -> str:
    return """\
## 6. Упрощения

Перечислены в соответствии с требованием `task_description.md` ("упрощения нужно обозначить").

- **Модель 3B вместо 7B** (Qwen2.5-3B-Instruct): снижает требования к GPU — помещается на T4 (16 GB) с 4-bit квантизацией. Ожидаемо уступает 7B по качеству генерации.
- **1 эпоха обучения**: LoRA с малым датасетом (~1600 примеров) склонна к быстрой переобучению; 1 эпохи достаточно для стабильного сигнала без memorization.
- **Heuristic input (не full brief_v4)**: часть train-примеров использует heuristic-нейтрализацию вместо LLM-brief. Производительность на примерах с LLM-brief ожидаемо выше.
- **Нет QA-цикла по содержанию**: synthetic inputs не проходили ручную проверку сохранения фактов. Возможны случаи потери числовых данных или имён в нейтральных версиях.
- **TF-IDF + LogReg классификатор**: вместо нейронного классификатора стиля. Быстрее, но менее точен на коротких или ambiguous текстах.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build RESULTS.md from evaluation artefacts.")
    parser.add_argument("--out", default="RESULTS.md", metavar="FILE",
                        help="Output file (default: RESULTS.md)")
    parser.add_argument("--baseline-dir", default="reports/eval/baseline_test", metavar="DIR")
    parser.add_argument("--finetuned-dir", default="reports/eval/finetuned_test", metavar="DIR")
    parser.add_argument("--memorization-dir", default="reports/memorization", metavar="DIR")
    parser.add_argument("--split-report", default="data/processed/split_report.json", metavar="FILE")
    args = parser.parse_args()
    setup_logging("build_results")

    baseline_dir   = ROOT / args.baseline_dir
    finetuned_dir  = ROOT / args.finetuned_dir
    mem_dir        = ROOT / args.memorization_dir

    # Load all artefacts
    split_report   = _load_json(ROOT / args.split_report)
    baseline_eval  = _load_json(baseline_dir / "eval_summary.json")
    finetuned_eval = _load_json(finetuned_dir / "eval_summary.json")
    baseline_conf  = _load_json(baseline_dir / "cross_type_confusion.json")
    finetuned_conf = _load_json(finetuned_dir / "cross_type_confusion.json")
    def _load_manifest(style_type: str) -> dict | None:
        candidates = [
            ROOT / "models" / style_type / "all" / "run_manifest.json",
            ROOT / "models" / style_type / "synthetic_openrouter" / "run_manifest.json",
            ROOT / "models" / style_type / "run_manifest.json",  # backward-compat
        ]
        for p in candidates:
            if p.exists():
                return _load_json(p)
        return None

    manifests      = {st: _load_manifest(st) for st in TYPES}

    mem_dfs = {}
    try:
        import pandas as pd
        for st in TYPES:
            mem_dfs[st] = _load_csv(mem_dir / f"memorization_{st}.csv")
    except ImportError:
        log.warning("pandas not available — skipping memorization section.")
        mem_dfs = {st: None for st in TYPES}

    # Build sections
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        f"# RESULTS\n\n_Сгенерировано: {now}_\n",
        _section_dataset(split_report),
        _section_training(manifests),
        _section_before_after(baseline_eval, finetuned_eval),
        _section_crosstype(baseline_conf, finetuned_conf),
        _section_memorization(mem_dfs),
        _section_simplifications(),
    ]

    content = "\n\n".join(s.rstrip() for s in sections) + "\n"

    out_path = ROOT / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info("→ %s", out_path)


if __name__ == "__main__":
    main()
