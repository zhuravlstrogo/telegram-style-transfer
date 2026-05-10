from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def flatten_text(message_text) -> str:
    """Flatten a Telegram message text field (str or list) into a plain string."""
    if isinstance(message_text, str):
        return message_text
    parts = []
    for fragment in message_text:
        if isinstance(fragment, str):
            parts.append(fragment)
        elif isinstance(fragment, dict) and "text" in fragment:
            parts.append(fragment["text"])
    return "".join(parts)


def parse_telegram_export(path: str | Path) -> pd.DataFrame:
    """Parse a Telegram Desktop JSON export into a DataFrame.

    Expected format: result of "Export chat history" → JSON in Telegram Desktop.
    Each message with type='message' is included; service messages are skipped.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    channel_name = data.get("name", path.parent.name)
    rows = []
    for msg in data["messages"]:
        if msg.get("type") != "message":
            continue
        text = flatten_text(msg.get("text", ""))
        rows.append(
            {
                "post_id": msg["id"],
                "date": msg.get("date", ""),
                "text": text,
                "channel": channel_name,
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    return df


def filter_posts(df: pd.DataFrame, min_chars: int = 50) -> pd.DataFrame:
    """Remove empty and too-short posts."""
    df = df.copy()
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() >= min_chars]
    return df.reset_index(drop=True)


def _normalize_for_dedup(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalized_text_hash(text: str) -> str:
    return hashlib.md5(_normalize_for_dedup(text).encode()).hexdigest()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates (after normalization), keeping the earliest post."""
    df = df.copy()
    df["_norm"] = df["text"].apply(_normalize_for_dedup)
    df["_hash"] = df["_norm"].apply(lambda t: hashlib.md5(t.encode()).hexdigest())
    df = df.sort_values("date").drop_duplicates(subset="_hash", keep="first")
    df = df.drop(columns=["_norm", "_hash"])
    return df.reset_index(drop=True)


_CLEAN_PATTERNS: list[tuple[str, str]] = [
    (r"@[A-Za-z0-9_]{4,}", ""),
    (r"https?://t(?:elegram)?\.me/\S+", ""),
    (r"https?://\S+", ""),
    (r"#[\w_]+", ""),
    (r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", ""),
    (r"(?im)^.*(подпис(ывайтесь|ка)|ставь\s+лайк|жми\s+👇).*$", ""),
]


def clean_for_input(text: str) -> str:
    """Strip Telegram-specific artefacts: handles, URLs, hashtags, emoji, CTA lines."""
    for pattern, repl in _CLEAN_PATTERNS:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def sample_recent(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep the N most recent posts; if n >= len(df), return df unchanged."""
    if n >= len(df):
        return df.reset_index(drop=True)
    df = df.sort_values("date", ascending=False).head(n)
    return df.reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> pd.DataFrame:
    """Add a 'split' column using a chronological 80 / 10 / 10 partition.

    Splits by time rather than randomly to avoid leakage from adjacent posts
    that share topics or phrasing, and to simulate real deployment conditions.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    splits = (
        ["train"] * train_end
        + ["val"] * (val_end - train_end)
        + ["test"] * (n - val_end)
    )
    df["split"] = splits
    return df
