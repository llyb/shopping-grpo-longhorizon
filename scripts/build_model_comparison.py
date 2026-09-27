#!/usr/bin/env python3
"""Write the paired Baseline/SFT/GRPO comparison product for one eval root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.artifacts import (  # noqa: E402
    iter_jsonl,
    write_json_atomic,
)
from shopping_grpo.evaluation.comparison import (  # noqa: E402
    compare_evaluation_runs,
)
from shopping_grpo.evaluation.rollout import load_tasks  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="按 task_id 配对比较 Baseline/SFT/GRPO 的四面板评测产物"
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=ROOT / "outputs" / "evaluation",
        help="包含各模型目录的评测根目录",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "evaluation" / "tasks.jsonl",
        help="固定分母的 benchmark 清单",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def collect_runs(evaluation_dir: Path) -> dict[str, list[dict]]:
    """Index every model run that produced a per-task evaluations.jsonl."""

    runs = {}
    for run_dir in sorted(
        path for path in evaluation_dir.iterdir() if path.is_dir()
    ):
        evaluations = run_dir / "evaluations.jsonl"
        if evaluations.is_file():
            runs[run_dir.name] = list(iter_jsonl(evaluations))
    return runs


def main(argv=None):
    args = parse_args(argv)
    output = args.output or args.evaluation_dir / "model_comparison.json"
    expected_task_ids = [row["task_id"] for row in load_tasks(args.benchmark)]
    runs = collect_runs(args.evaluation_dir)
    if len(runs) < 2:
        print(
            f"{args.evaluation_dir} 下只有 {len(runs)} 个含 evaluations.jsonl 的"
            "模型目录，配对比较需要至少两个，跳过。"
        )
        return
    comparison = compare_evaluation_runs(
        expected_task_ids=expected_task_ids,
        runs=runs,
    )
    write_json_atomic(output, comparison, force=True)
    print(output)


if __name__ == "__main__":
    main()
