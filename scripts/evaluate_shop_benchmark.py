#!/usr/bin/env python3
"""在固定 ShopSimulator benchmark 上评测 OpenAI-compatible 本地或远端模型。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.summary import summarize_trajectories
from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_tasks, load_tasks
from shopping_grpo.evaluation.artifacts import index_jsonl, write_json_atomic, write_jsonl_atomic
from shopping_grpo.evaluation.contracts import ContractValidationError
from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.model_client import OpenAIJSONClient
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_PROMPT_VERSION, TRAJECTORY_JUDGE_PROMPT_VERSION,
    build_rubric_curator_messages, build_trajectory_judge_messages,
)
from shopping_grpo.evaluation.results import (
    assemble_task_evaluation, build_not_judged_result, summarize_evaluations,
)
from shopping_grpo.evaluation.rubric import (
    extract_rubric_candidates,
    materialize_rubric_bundle,
    reusable_cached_rubric,
)
from shopping_grpo.evaluation.trajectory import normalize_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description="评测 Base、SFT 或 GRPO Shopping Agent")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="原始评测轨迹 JSONL")
    parser.add_argument("--summary", type=Path, required=True, help="汇总指标 JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--api-key", required=True, help="本地 vLLM 可传 EMPTY")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="单次模型生成上限；防止未调用工具时耗尽完整上下文。",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=24576,
        help="上下文窗口；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument(
        "--context-compaction",
        action="store_true",
        help="上下文接近上限时压缩较早的交互；默认关闭。",
    )
    parser.add_argument(
        "--observation-token-budget",
        type=int,
        default=1536,
        help="Observation token 预算；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument("--task-facts", type=Path, required=True)
    parser.add_argument("--curator-base-url", default=None)
    parser.add_argument("--curator-model", default="deepseek/deepseek-v4-flash")
    parser.add_argument(
        "--skip-rubric-curation",
        action="store_true",
        help="只从 shared/rubrics.jsonl 读取教师模型已采集的 Rubric，不再调用 curator。",
    )
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=16384,
                        help="单次 Judge 输出上限；deepseek-v4 长程思考占用大量 token，需留足空间给 JSON 正文。")
    parser.add_argument("--judge-timeout", type=int, default=180)
    return parser.parse_args()


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps 必须为正数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须为正数")
    if args.context_window < 0:
        raise SystemExit("--context-window 不能为负数")
    if args.context_window and args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("--context-window 必须大于 --max-tokens 与安全余量之和")
    if args.observation_token_budget < 0:
        raise SystemExit("--observation-token-budget 不能为负数")
    tasks = load_tasks(args.benchmark)
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
    )
    collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
    )
    task_ids = [task["task_id"] for task in tasks]
    raw_trajectories = _read_jsonl(args.output)
    shared_dir = args.output.parent.parent / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    api_key = args.judge_api_key or args.api_key
    rubrics_path = shared_dir / "rubrics.jsonl"
    if args.skip_rubric_curation:
        if not rubrics_path.exists():
            raise SystemExit(
                "未找到缓存 Rubric：请先在无 GPU 环境运行 scripts/prepare_eval.sh "
                "完成教师模型采集"
            )
        cached_rubrics = index_jsonl(rubrics_path, key="task_id")
        rubric_rows = []
        for task_id in task_ids:
            cached = reusable_cached_rubric(
                cached_rubrics,
                task_id=task_id,
                curator_model=args.curator_model,
                curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
            )
            if cached is None:
                raise SystemExit(
                    f"缓存 Rubric 缺少 task {task_id} 或与 curator_model="
                    f"{args.curator_model} 不匹配；请重新运行 scripts/prepare_eval.sh"
                )
            rubric_rows.append(cached)
    else:
        if not args.curator_base_url:
            raise SystemExit("未跳过 Rubric 采集时必须提供 --curator-base-url")
        facts = index_jsonl(args.task_facts, key="task_id", allowed_keys=set(task_ids))
        curator = OpenAIJSONClient(model=args.curator_model, base_url=args.curator_base_url,
                                   api_key=api_key, max_tokens=args.judge_max_tokens,
                                   timeout=args.judge_timeout, response_format_json=True)
        rubric_rows = []
        candidate_rows = []
        cached_rubrics = {}
        if rubrics_path.exists():
            cached_rubrics = index_jsonl(rubrics_path, key="task_id")
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
                continue
            response = curator.complete_json(build_rubric_curator_messages(
                task_id=task_id, query=task_facts["query"], candidates=candidates["candidates"]
            ))["result"]
            rubric_rows.append(materialize_rubric_bundle(
                task_facts=task_facts, candidates=candidates, curator_response=response,
                curator_model=args.curator_model, curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
                rubric_version=RUBRIC_CURATOR_PROMPT_VERSION,
            ))
        write_jsonl_atomic(rubrics_path, rubric_rows, force=True)
        write_jsonl_atomic(shared_dir / "rubric_candidates.jsonl", candidate_rows, force=True)
    rubrics = {int(row["task_id"]): row for row in rubric_rows}
    judge = OpenAIJSONClient(model=args.judge_model, base_url=args.judge_base_url,
                             api_key=api_key, max_tokens=args.judge_max_tokens,
                             timeout=args.judge_timeout, response_format_json=True)
    evaluations = []
    normalized_rows = []
    metrics_rows = []
    judge_requests = []
    judge_results = []
    for raw in raw_trajectories:
        normalized = normalize_trajectory(raw)
        metrics = compute_deterministic_metrics(normalized)
        normalized_rows.append(normalized)
        metrics_rows.append(metrics)
        if metrics["validity"]["infrastructure_invalid"]:
            judge_result = build_not_judged_result(
                task_id=normalized["task_id"], trajectory_id=normalized["trajectory_id"],
                reason="infrastructure_invalid")
        else:
            messages = build_trajectory_judge_messages(
                normalized=normalized, rubric_bundle=rubrics[int(normalized["task_id"])],
                deterministic_metrics=metrics)
            judge_requests.append({"task_id": normalized["task_id"],
                                   "trajectory_id": normalized["trajectory_id"],
                                   "messages": messages})
            try:
                judge_result = judge.complete_json(messages)["result"]
            except Exception:
                judge_result = {"schema_version": "shopping-trajectory-judge-v1",
                                "task_id": normalized["task_id"],
                                "trajectory_id": normalized["trajectory_id"],
                                "judge_status": "invalid", "rubric_assessments": [],
                                "dimension_scores": {},
                                "errors": {"primary": "other", "secondary": [],
                                           "evidence_event_ids": []},
                                "overall_diagnosis": "Judge response failed contract validation."}
        judge_results.append(judge_result)
        try:
            evaluations.append(assemble_task_evaluation(
                actor={"label": args.model, "model": args.model},
                normalized_trajectory=normalized, deterministic_metrics=metrics,
                rubric_bundle=rubrics[int(normalized["task_id"])], judge_result=judge_result))
        except ContractValidationError:
            # A provider can return parseable JSON that violates the frozen schema.
            # Preserve the task in the denominator without manufacturing scores.
            invalid = {"schema_version": "shopping-trajectory-judge-v1",
                       "task_id": normalized["task_id"],
                       "trajectory_id": normalized["trajectory_id"],
                       "judge_status": "invalid", "rubric_assessments": [],
                       "dimension_scores": {},
                       "errors": {"primary": "other", "secondary": [],
                                  "evidence_event_ids": []},
                       "overall_diagnosis": "Judge response violated the frozen schema."}
            judge_results[-1] = invalid
            evaluations.append(assemble_task_evaluation(
                actor={"label": args.model, "model": args.model},
                normalized_trajectory=normalized, deterministic_metrics=metrics,
                rubric_bundle=rubrics[int(normalized["task_id"])], judge_result=invalid))
    write_jsonl_atomic(args.output.parent / "evaluations.jsonl", evaluations, force=True)
    write_jsonl_atomic(args.output.parent / "normalized.jsonl", normalized_rows, force=True)
    write_jsonl_atomic(args.output.parent / "preprocessed.jsonl", metrics_rows, force=True)
    write_jsonl_atomic(args.output.parent / "judge_requests.jsonl", judge_requests, force=True)
    write_jsonl_atomic(args.output.parent / "judges.jsonl", judge_results, force=True)
    evaluation_summary = summarize_evaluations(expected_task_ids=task_ids, evaluations=evaluations)
    write_json_atomic(args.output.parent / "evaluation_summary.json", evaluation_summary, force=True)
    summary = summarize_trajectories(task_ids, raw_trajectories)
    summary["llm_as_judge"] = evaluation_summary
    summary["protocol"] = {
        "benchmark": str(args.benchmark),
        "model": args.model,
        "reward_contract": "shopsimulator-reward-v3",
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "context_window": args.context_window,
        "context_safety_margin": args.context_safety_margin,
        "context_compaction": args.context_compaction,
        "observation_token_budget": args.observation_token_budget,
        "observation_detail_token_budget": args.observation_detail_token_budget,
        "observation_generic_token_budget": args.observation_generic_token_budget,
        "observation_search_top_k": args.observation_search_top_k,
        "llm_as_judge": True,
        "curator_model": args.curator_model,
        "judge_model": args.judge_model,
        "rubric_prompt_version": RUBRIC_CURATOR_PROMPT_VERSION,
        "judge_prompt_version": TRAJECTORY_JUDGE_PROMPT_VERSION,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
