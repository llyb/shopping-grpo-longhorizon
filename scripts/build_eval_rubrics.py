#!/usr/bin/env python3
"""教师模型采集阶段：用 curator API 构建并缓存评测 Rubric（无需 GPU）。

读取 ``export_eval_task_facts.py`` 已导出的 TaskFacts，调用 curator 教师模型
（OpenAI-compatible）把候选约束冻结成 Rubric，写入 shared 目录。已存在且
curator_model / prompt version 匹配的 Rubric 会被复用，不会重复调用教师模型。

运行本脚本前需先完成 TaskFacts 导出，参见 ``scripts/prepare_eval.sh``。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.artifacts import (
    append_jsonl_fsync,
    index_jsonl,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.model_client import OpenAIJSONClient
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_PROMPT_VERSION,
    build_rubric_curator_messages,
)
from shopping_grpo.evaluation.rollout import load_tasks
from shopping_grpo.evaluation.rubric import (
    extract_rubric_candidates,
    materialize_rubric_bundle,
    reusable_cached_rubric,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="教师模型采集：构建并缓存评测 Rubric（无需 GPU）"
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--task-facts",
        type=Path,
        required=True,
        help="export_eval_task_facts.py 导出的 TaskFacts JSONL",
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        required=True,
        help="shared 目录，写入 rubrics.jsonl 与 rubric_candidates.jsonl",
    )
    parser.add_argument("--curator-base-url", required=True)
    parser.add_argument("--curator-model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--curator-api-key", default=None)
    parser.add_argument("--curator-max-tokens", type=int, default=16384,
                        help="单次教师模型输出上限；deepseek-v4 长程思考会占用大量 token，需留足空间给 JSON 正文。")
    parser.add_argument("--curator-timeout", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    task_ids = [task["task_id"] for task in load_tasks(args.benchmark)]
    facts = index_jsonl(args.task_facts, key="task_id", allowed_keys=set(task_ids))
    shared_dir = args.shared_dir
    shared_dir.mkdir(parents=True, exist_ok=True)
    api_key = args.curator_api_key or "EMPTY"
    curator = OpenAIJSONClient(
        model=args.curator_model,
        base_url=args.curator_base_url,
        api_key=api_key,
        max_tokens=args.curator_max_tokens,
        timeout=args.curator_timeout,
        response_format_json=True,
    )
    rubrics_path = shared_dir / "rubrics.jsonl"
    cached_rubrics = {}
    if rubrics_path.exists():
        cached_rubrics = index_jsonl(rubrics_path, key="task_id")
    # 只保留与本次 curator_model/prompt version 匹配的缓存行，避免追加后出现重复 task_id。
    reusable_rows = [
        row
        for row in cached_rubrics.values()
        if reusable_cached_rubric(
            cached_rubrics,
            task_id=row["task_id"],
            curator_model=args.curator_model,
            curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
        )
        is not None
    ]
    if len(reusable_rows) != len(cached_rubrics):
        write_jsonl_atomic(rubrics_path, reusable_rows, force=True)
        cached_rubrics = {row["task_id"]: row for row in reusable_rows}
    rubric_rows = []
    candidate_rows = []
    reused = 0
    for task_id in task_ids:
        task_facts = facts[task_id]
        candidates = extract_rubric_candidates(task_facts)
        candidate_rows.append(candidates)
        cached = reusable_cached_rubric(
            cached_rubrics,
            task_id=task_id,
            curator_model=args.curator_model,
            curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
        )
        if cached is not None:
            rubric_rows.append(cached)
            reused += 1
            continue
        response = curator.complete_json(
            build_rubric_curator_messages(
                task_id=task_id,
                query=task_facts["query"],
                candidates=candidates["candidates"],
            )
        )["result"]
        row = materialize_rubric_bundle(
            task_facts=task_facts,
            candidates=candidates,
            curator_response=response,
            curator_model=args.curator_model,
            curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
            rubric_version=RUBRIC_CURATOR_PROMPT_VERSION,
        )
        rubric_rows.append(row)
        # 断点续存：每条采集完成立即 fsync 落盘，中断后重跑自动跳过已完成任务。
        append_jsonl_fsync(rubrics_path, row)
    write_jsonl_atomic(rubrics_path, rubric_rows, force=True)
    write_jsonl_atomic(shared_dir / "rubric_candidates.jsonl", candidate_rows, force=True)
    print(
        f"Rubric 采集完成：共 {len(rubric_rows)} 条（复用缓存 {reused} 条，"
        f"本次新采 {len(rubric_rows) - reused} 条），已写入 {rubrics_path}"
    )


if __name__ == "__main__":
    main()
