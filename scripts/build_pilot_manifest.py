#!/usr/bin/env python3
"""
Create a shared pilot manifest for offline neutralization experiments.

The manifest is built strictly from an existing split (`val` by default), so all
pilot generators evaluate on the same fixed subset.

Usage:
    python scripts/build_pilot_manifest.py
    python scripts/build_pilot_manifest.py --types type1 type2 --n-per-type 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telegram_style_transfer.paths import PROCESSED_DIR
from telegram_style_transfer.synthetic import build_pilot_manifest, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fixed pilot manifest from processed validation splits"
    )
    parser.add_argument("--types", nargs="+", default=["type1", "type2"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n-per-type", type=int, default=150, metavar="N")
    parser.add_argument("--seed", type=int, default=42, metavar="N")
    parser.add_argument(
        "--out",
        default="data/processed/pilot_300_manifest.jsonl",
        help="Output manifest path",
    )
    args = parser.parse_args()

    manifest = build_pilot_manifest(
        processed_dir=PROCESSED_DIR,
        types=args.types,
        n_per_type=args.n_per_type,
        split=args.split,
        seed=args.seed,
    )
    if manifest.empty:
        raise SystemExit("Pilot manifest is empty. Run prepare_dataset.py first.")

    out_path = Path(args.out)
    write_jsonl(out_path, manifest.to_dict(orient="records"))
    print(
        f"Saved pilot manifest: {out_path} "
        f"(n={len(manifest)}, types={','.join(args.types)}, split={args.split}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
