#!/usr/bin/env python3
"""Collect resumable Teacher rollouts and build leak-free SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import signal
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from shopping_grpo.collection.sft import (
    acceptance_reasons,
    build_collection_artifacts,
    task_ids_from_jsonl,
)
from shopping_grpo.evaluation.rollout import (
    CollectionInfrastructureError,
    OpenAIChatClient,
    _is_infrastructure_failure,
    append_jsonl,
    collect_for_task,
    collect_tasks,
    completed_task_attempts,
    load_tasks,
    rollout_interrupted,
)


def batch_paths(output_dir: Path) -> dict[str, Path]:
    """Keep raw source data and every reproducible derivative in one directory."""

    return {
        "raw": output_dir / "raw.jsonl",
        "accepted": output_dir / "accepted.jsonl",
        "rejected": output_dir / "rejected.jsonl",
        "stats": output_dir / "reject_stats.json",
        "sft": output_dir / "sft.jsonl",
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "metadata": output_dir / "metadata.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sft-collection"),
    )
    parser.add_argument(
        "--held-out-tasks",
        type=Path,
        default=Path("data/evaluation/tasks.jsonl"),
        help="These task IDs are never collected or written to SFT outputs.",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-accepted", type=int, default=None)
    parser.add_argument("--attempts-per-task", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5700"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "deepseek/deepseek-v4-flash"),
    )
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("high", "max"), default="high")
    parser.add_argument("--context-window", type=int, default=0)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument("--context-compaction", action="store_true")
    parser.add_argument("--observation-token-budget", type=int, default=0)
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    return parser.parse_args()


def collect_until_target(
    *,
    tasks,
    target_accepted,
    client,
    client_factory=None,
    output_path,
    base_url,
    max_steps,
    attempts_per_task,
    workers=1,
    excluded_task_ids=(),
):
    """Collect concurrently without scheduling more possible successes than needed."""

    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    accepted = _accepted_count(output_path, excluded_task_ids)
    completed = completed_task_attempts(output_path)
    candidates = [
        (task, attempt_index)
        for task in tasks
        for attempt_index in range(int(attempts_per_task))
        if (int(task["task_id"]), attempt_index) not in completed
    ]
    candidate_iter = iter(candidates)
    pending = {}
    written = []
    infrastructure_failed = False

    def submit_available(executor):
        remaining = int(target_accepted) - accepted
        max_pending = min(workers, max(remaining, 0))
        while len(pending) < max_pending:
            try:
                task, attempt_index = next(candidate_iter)
            except StopIteration:
                return
            future = executor.submit(
                collect_for_task,
                task,
                client=client_factory() if client_factory else client,
                base_url=base_url,
                max_steps=max_steps,
                attempt_index=attempt_index,
            )
            pending[future] = (task, attempt_index)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        submit_available(executor)
        while pending:
            completed_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed_futures:
                pending.pop(future)
                trajectory = future.result()
                append_jsonl(output_path, [trajectory])
                written.append(trajectory)
                accepted += acceptance_reasons(trajectory)[0]
                infrastructure_failed |= _is_infrastructure_failure(trajectory)
            if not infrastructure_failed:
                submit_available(executor)

    if infrastructure_failed:
        raise CollectionInfrastructureError(
            "collection infrastructure failure; stopped before the next task"
        )
    return written, accepted


def _accepted_count(raw_path: Path, excluded_task_ids=()) -> int:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        return 0
    excluded = {int(task_id) for task_id in excluded_task_ids}
    with raw_path.open(encoding="utf-8") as handle:
        accepted = 0
        for line in handle:
            if not line.strip():
                continue
            trajectory = json.loads(line)
            if int(trajectory["task_id"]) in excluded:
                continue
            accepted += acceptance_reasons(trajectory)[0]
        return accepted


def _validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.target_accepted is not None and args.target_accepted < 1:
        raise SystemExit("--target-accepted must be at least 1")
    if args.attempts_per_task < 1:
        raise SystemExit("--attempts-per-task must be at least 1")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.workers > 1 and args.target_accepted is None:
        raise SystemExit("--workers > 1 requires --target-accepted")
    if not 0 <= args.validation_ratio < 1:
        raise SystemExit("--validation-ratio must be in [0, 1)")
    if not args.build_only and not args.llm_base_url:
        raise SystemExit("--llm-base-url or OPENAI_BASE_URL is required")
    if not args.build_only and not args.api_key:
        raise SystemExit("--api-key or OPENAI_API_KEY is required")
    if not args.build_only and args.tasks is None:
        raise SystemExit("--tasks is required unless --build-only is used")


def _make_client(args: argparse.Namespace) -> OpenAIChatClient:
    return OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        context_window=args.context_window or None,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget or None,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
    )


def _collection_config(args: argparse.Namespace) -> dict:
    """Record reproducibility settings without ever serializing the API key."""

    return {
        "tasks": str(args.tasks),
        "held_out_tasks": str(args.held_out_tasks),
        "model": args.model,
        "llm_base_url": args.llm_base_url,
        "shopsim_base_url": args.base_url,
        "limit": args.limit,
        "target_accepted": args.target_accepted,
        "attempts_per_task": args.attempts_per_task,
        "workers": args.workers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "context_window": args.context_window,
        "context_safety_margin": args.context_safety_margin,
        "context_compaction": args.context_compaction,
        "observation_token_budget": args.observation_token_budget,
        "observation_detail_token_budget": args.observation_detail_token_budget,
        "observation_generic_token_budget": args.observation_generic_token_budget,
        "observation_search_top_k": args.observation_search_top_k,
    }


def main() -> int:
    args = parse_args()
    _validate_args(args)
    paths = batch_paths(args.output_dir)
    held_out_ids = task_ids_from_jsonl(args.held_out_tasks)
    exit_code = 0
    collection_config = None

    if not args.build_only:
        collection_config = _collection_config(args)
        signal.signal(signal.SIGTERM, rollout_interrupted)
        signal.signal(signal.SIGINT, rollout_interrupted)
        tasks = [
            task
            for task in load_tasks(args.tasks)
            if int(task["task_id"]) not in held_out_ids
        ]
        if args.limit is not None:
            tasks = tasks[: args.limit]
        try:
            if args.target_accepted is None:
                client = _make_client(args)
                written = collect_tasks(
                    tasks,
                    client=client,
                    output_path=paths["raw"],
                    base_url=args.base_url,
                    max_steps=args.max_steps,
                    attempts_per_task=args.attempts_per_task,
                )
                print(f"collected_raw={len(written)}")
            else:
                written, accepted = collect_until_target(
                    tasks=tasks,
                    target_accepted=args.target_accepted,
                    client=None,
                    client_factory=lambda: _make_client(args),
                    output_path=paths["raw"],
                    base_url=args.base_url,
                    max_steps=args.max_steps,
                    attempts_per_task=args.attempts_per_task,
                    workers=args.workers,
                    excluded_task_ids=held_out_ids,
                )
                print(f"collected_raw={len(written)} accepted_total={accepted}")
        except CollectionInfrastructureError as exc:
            print(f"collection paused: {exc}")
            exit_code = 2

    if not paths["raw"].exists():
        raise SystemExit(f"raw trajectory file does not exist: {paths['raw']}")
    summary = build_collection_artifacts(
        raw_path=paths["raw"],
        output_dir=args.output_dir,
        held_out_task_ids=held_out_ids,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        collection_config=collection_config,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
