#!/usr/bin/env python3
"""Fail when any supplied dataset overlaps by task ID or normalized requirement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_isolation import load_task_catalog, require_isolated, task_ids

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--product-corpus",
        type=Path,
        default=ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=JSONL",
        help="repeat once per SFT/GRPO/development/evaluation artifact",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    named = {}
    for value in args.dataset:
        if "=" not in value:
            raise SystemExit("--dataset must use NAME=JSONL")
        name, raw_path = value.split("=", 1)
        if not name or name in named:
            raise SystemExit(f"invalid or duplicate dataset name: {name!r}")
        named[name] = task_ids(Path(raw_path))
    report = require_isolated(named, load_task_catalog(args.product_corpus))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
