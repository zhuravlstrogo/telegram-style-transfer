from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _char_len(text: str) -> int:
    return len(text.strip())


def _word_len(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Length comparison
# ---------------------------------------------------------------------------


def length_stats(texts: Iterable[str]) -> dict:
    chars = [_char_len(t) for t in texts]
    words = [_word_len(t) for t in texts]
    s = pd.Series(chars)
    w = pd.Series(words)
    return {
        "char_mean": round(s.mean(), 1),
        "char_median": round(s.median(), 1),
        "char_std": round(s.std(), 1),
        "char_min": int(s.min()),
        "char_max": int(s.max()),
        "word_mean": round(w.mean(), 1),
        "word_median": round(w.median(), 1),
    }


def compare_lengths(
    records: list[dict],
    generated_texts: list[str] | None = None,
) -> pd.DataFrame:
    """Compare input vs reference (and optionally generated) text lengths."""
    rows = []
    for i, rec in enumerate(records):
        inp = rec["input"]
        ref = rec.get("response_clean", rec.get("response", ""))
        row = {
            "post_id": rec.get("post_id"),
            "style_type": rec.get("style_type"),
            "input_chars": _char_len(inp),
            "input_words": _word_len(inp),
            "ref_chars": _char_len(ref),
            "ref_words": _word_len(ref),
            "ref_input_char_ratio": round(_char_len(ref) / max(_char_len(inp), 1), 3),
        }
        if generated_texts is not None:
            gen = generated_texts[i]
            row["gen_chars"] = _char_len(gen)
            row["gen_words"] = _word_len(gen)
            row["gen_input_char_ratio"] = round(
                _char_len(gen) / max(_char_len(inp), 1), 3
            )
            row["gen_ref_char_ratio"] = round(
                _char_len(gen) / max(_char_len(ref), 1), 3
            )
        rows.append(row)
    return pd.DataFrame(rows)


def length_report(df: pd.DataFrame) -> str:
    lines = []

    def _fmt_group(label: str, sub: pd.DataFrame) -> None:
        lines.append(f"\n=== {label} (n={len(sub)}) ===")
        for col_prefix, name in [("input", "Input"), ("ref", "Reference")]:
            c = sub[f"{col_prefix}_chars"]
            lines.append(
                f"  {name}: chars mean={c.mean():.0f}  median={c.median():.0f}"
                f"  std={c.std():.0f}  [{c.min()}, {c.max()}]"
            )
        r = sub["ref_input_char_ratio"]
        lines.append(
            f"  Ref/Input char ratio: mean={r.mean():.2f}  median={r.median():.2f}"
        )
        if "gen_chars" in sub.columns:
            g = sub["gen_chars"]
            lines.append(
                f"  Generated: chars mean={g.mean():.0f}  median={g.median():.0f}"
                f"  std={g.std():.0f}  [{g.min()}, {g.max()}]"
            )
            gr = sub["gen_input_char_ratio"]
            lines.append(
                f"  Gen/Input char ratio: mean={gr.mean():.2f}  median={gr.median():.2f}"
            )
            gref = sub["gen_ref_char_ratio"]
            lines.append(
                f"  Gen/Ref char ratio: mean={gref.mean():.2f}  median={gref.median():.2f}"
            )

    if "style_type" in df.columns and df["style_type"].nunique() > 1:
        for st, grp in df.groupby("style_type"):
            _fmt_group(st, grp)
    else:
        _fmt_group("all", df)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Style classifier (TF-IDF + LogReg)
# ---------------------------------------------------------------------------


def train_style_classifier(
    type1_texts: list[str],
    type2_texts: list[str],
) -> object:
    """Train a TF-IDF + LogisticRegression binary style classifier.

    Returns a fitted sklearn Pipeline with classes ['type1', 'type2'].
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    texts = type1_texts + type2_texts
    labels = ["type1"] * len(type1_texts) + ["type2"] * len(type2_texts)

    clf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            ("lr", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )
    clf.fit(texts, labels)
    return clf


def style_classifier_scores(
    clf,
    texts: list[str],
    target_label: str,
) -> dict:
    """Evaluate style classifier on a list of texts.

    Returns accuracy (fraction classified as target_label) and mean confidence.
    """
    classes = list(clf.classes_)
    target_idx = classes.index(target_label)
    proba = clf.predict_proba(texts)
    preds = clf.predict(texts)

    accuracy = float(np.mean(preds == target_label))
    confidence = float(np.mean(proba[:, target_idx]))
    return {
        "style_accuracy": round(accuracy, 4),
        "style_confidence_mean": round(confidence, 4),
        "n": len(texts),
        "target_label": target_label,
    }


def cross_type_confusion_matrix(
    clf,
    type1_texts: list[str],
    type2_texts: list[str],
) -> dict:
    """Compute a 2×2 confusion matrix for cross-type separability.

    Runs both type1 and type2 generated texts through the same classifier and
    returns row-wise predicted-class fractions.  High diagonal = adapters produce
    clearly distinct styles; low diagonal = "средний по больнице".

    Returns:
        {
          "classes": ["type1", "type2"],
          "type1_gen": {"type1": 0.85, "type2": 0.15},
          "type2_gen": {"type1": 0.12, "type2": 0.88},
          "diagonal_mean": 0.865,   # (type1→type1 + type2→type2) / 2
          "n_type1": 120,
          "n_type2": 115,
        }
    """
    classes = [
        str(c) for c in clf.classes_
    ]  # guaranteed ['type1', 'type2'] by train_style_classifier

    def _fractions(texts: list[str]) -> dict[str, float]:
        preds = clf.predict(texts)
        total = len(preds)
        return {c: round(float(np.sum(preds == c)) / total, 4) for c in classes}

    type1_fracs = _fractions(type1_texts)
    type2_fracs = _fractions(type2_texts)

    diagonal_mean = round(
        (type1_fracs.get("type1", 0.0) + type2_fracs.get("type2", 0.0)) / 2,
        4,
    )

    return {
        "classes": classes,
        "type1_gen": type1_fracs,
        "type2_gen": type2_fracs,
        "diagonal_mean": diagonal_mean,
        "n_type1": len(type1_texts),
        "n_type2": len(type2_texts),
    }


# ---------------------------------------------------------------------------
# Content preservation — cosine similarity via sentence-transformers (LaBSE)
# ---------------------------------------------------------------------------


def load_embedder(model_name: str = "sentence-transformers/LaBSE"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def cosine_similarity_scores(
    embedder,
    texts_a: list[str],
    texts_b: list[str],
    batch_size: int = 64,
) -> dict:
    """Compute per-pair cosine similarity between two aligned lists of texts.

    Typical use: texts_a = inputs, texts_b = generated outputs.
    """
    import torch

    emb_a = embedder.encode(
        texts_a, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True
    )
    emb_b = embedder.encode(
        texts_b, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True
    )

    # normalise then dot product = cosine similarity
    emb_a = torch.nn.functional.normalize(emb_a, dim=1)
    emb_b = torch.nn.functional.normalize(emb_b, dim=1)
    sims = (emb_a * emb_b).sum(dim=1).cpu().numpy()

    return {
        "cosine_mean": round(float(sims.mean()), 4),
        "cosine_median": round(float(np.median(sims)), 4),
        "cosine_std": round(float(sims.std()), 4),
        "cosine_min": round(float(sims.min()), 4),
        "cosine_max": round(float(sims.max()), 4),
        "n": len(sims),
        "per_sample": sims.tolist(),
    }


# ---------------------------------------------------------------------------
# Distributional similarity — MAUVE
# ---------------------------------------------------------------------------


def mauve_score(
    generated_texts: list[str],
    reference_texts: list[str],
    featurize_model_name: str = "xlm-roberta-base",
    max_text_length: int = 512,
    batch_size: int = 8,
    seed: int = 42,
) -> dict:
    """Compute MAUVE between generated and reference corpus.

    Uses xlm-roberta-base as the ru-compatible backbone instead of GPT-2.
    Higher MAUVE = generated distribution is closer to reference distribution.

    Args:
        generated_texts: outputs from baseline or fine-tuned model.
        reference_texts: real posts from the target channel (test split).
        featurize_model_name: HuggingFace model used for featurization.
        max_text_length: token length cap (MAUVE truncates internally).
        seed: random seed for k-means quantization.
    """
    import os

    import mauve as mauve_lib

    # Prevent segfault from OpenMP on macOS; safe to set on Linux too.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    result = mauve_lib.compute_mauve(
        p_text=generated_texts,
        q_text=reference_texts,
        featurize_model_name=featurize_model_name,
        max_text_length=max_text_length,
        batch_size=batch_size,
        verbose=False,
        seed=seed,
        device_id=0 if _cuda_available() else -1,
    )
    return {
        "mauve": round(float(result.mauve), 4),
        "frontier_integral": round(float(result.frontier_integral), 4),
        "n_generated": len(generated_texts),
        "n_reference": len(reference_texts),
        "featurize_model": featurize_model_name,
    }


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
