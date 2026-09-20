#!/usr/bin/env python3
"""Materialize veRL parquet from a frozen task-ID split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_isolation import load_task_catalog, sha256_file, task_ids, write_jsonl
from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument(
        "--product-corpus",
        type=Path,
        default=ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to build GRPO parquet; run this command in the GRPO environment"
        ) from exc
    catalog = {row["task_id"]: row for row in load_task_catalog(args.product_corpus)}
    ids = task_ids(args.tasks)
    if len(ids) != len(set(ids)):
        raise SystemExit("task split contains duplicate task IDs")
    missing = sorted(set(ids) - set(catalog))
    if missing:
        raise SystemExit(f"task IDs are outside the environment catalog: {missing[:20]}")
    records = []
    for index, task_id in enumerate(ids):
        instruction = catalog[task_id]["instruction"]
        records.append(
            {
                "data_source": "shopping_reward_v4",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                "ability": "shopping",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "task_id": task_id,
                    "index": index,
                    "split": args.split,
                    "reward_version": "shopsimulator-reward-v4",
                },
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.output_dir / f"{args.split}.jsonl"
    parquet = args.output_dir / f"{args.split}.parquet"
    write_jsonl(jsonl, ({"task_id": value} for value in ids))
    pq.write_table(pa.Table.from_pylist(records), parquet, compression="zstd")
    metadata = {
        "schema_version": "shopping-grpo-dataset-v2",
        "environment": "shopsimulator-environment-v2.1",
        "reward": "shopsimulator-reward-v4",
        "split": args.split,
        "tasks": len(records),
        "jsonl": str(jsonl),
        "jsonl_sha256": sha256_file(jsonl),
        "parquet": str(parquet),
        "parquet_sha256": sha256_file(parquet),
    }
    (args.output_dir / f"{args.split}.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
