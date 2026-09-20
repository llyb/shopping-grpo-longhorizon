#!/usr/bin/env python3
"""Build task/content-disjoint SFT, GRPO, development and evaluation pools."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dataset_isolation import (
    load_task_catalog,
    require_isolated,
    sha256_file,
    task_ids,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "data/evaluation/tasks.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sft-candidates", type=int, default=3000)
    parser.add_argument("--grpo-train", type=int, default=1000)
    parser.add_argument("--grpo-validation", type=int, default=100)
    parser.add_argument("--development", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = {
        "sft_candidates": args.sft_candidates,
        "grpo_train": args.grpo_train,
        "grpo_validation": args.grpo_validation,
        "development": args.development,
    }
    if any(value < 1 for value in requested.values()):
        raise SystemExit("all requested split sizes must be positive")
    catalog = load_task_catalog(args.product_corpus)
    evaluation_ids = task_ids(args.evaluation)
    by_id = {row["task_id"]: row for row in catalog}
    evaluation_fingerprints = {
        by_id[value]["requirement_fingerprint"]
        for value in evaluation_ids
        if value in by_id
    }
    candidates_by_fingerprint = {}
    for row in catalog:
        if not row.get("contract_admissible"):
            continue
        if row["task_id"] in set(evaluation_ids):
            continue
        fingerprint = row["requirement_fingerprint"]
        if fingerprint in evaluation_fingerprints:
            continue
        candidates_by_fingerprint.setdefault(fingerprint, row["task_id"])
    candidates = list(candidates_by_fingerprint.values())
    random.Random(args.seed).shuffle(candidates)
    if sum(requested.values()) > len(candidates):
        raise SystemExit(
            f"requested {sum(requested.values())} unique tasks but only {len(candidates)} are available"
        )
    cursor = 0
    splits = {}
    for name, size in requested.items():
        splits[name] = candidates[cursor : cursor + size]
        cursor += size
    report = require_isolated({**splits, "evaluation": evaluation_ids}, catalog)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, ids in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, ({"task_id": value} for value in ids))
        paths[name] = {"path": str(path), "rows": len(ids), "sha256": sha256_file(path)}
    metadata = {
        "schema_version": "shopping-data-splits-v1",
        "seed": args.seed,
        "product_corpus": str(args.product_corpus),
        "evaluation": str(args.evaluation),
        "splits": paths,
        "isolation": report,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
