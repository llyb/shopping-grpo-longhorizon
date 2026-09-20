#!/usr/bin/env python3
"""在本地 CPU 上用 transformers 加载 merged 权重，对 ShopSimulator 跑评测。

与 ``evaluate_shop_benchmark.py`` 的区别：模型推理用进程内
``LocalTransformersClient``（``transformers`` + CPU ``model.generate``），
不依赖 vLLM HTTP 服务。ShopSimulator Flask 服务仍需单独启动。
"""

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.local_client import LocalTransformersClient
from shopping_grpo.evaluation.rollout import collect_tasks, load_tasks
from shopping_grpo.evaluation.summary import summarize_trajectories


def parse_args():
    parser = argparse.ArgumentParser(
        description="本地 CPU 评测 Base、SFT 或 GRPO Shopping Agent"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="本地 merged 权重目录（含 tokenizer/processor + 权重 + chat template）。",
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/evaluation/tasks.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True, help="原始评测轨迹 JSONL")
    parser.add_argument("--summary", type=Path, required=True, help="汇总指标 JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
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
        help="上下文窗口；传 0 禁用本地 tokenizer 计数依赖。",
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
        help="Observation token 预算；传 0 禁用本地 tokenizer 计数依赖。",
    )
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="模型精度；auto 在 CUDA 上优先 bf16，CPU 使用 fp32。",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("auto", "sdpa"),
        default="sdpa",
        help="注意力实现；CPU 上 sdpa 即可。",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="模型加载设备；默认 cpu，CUDA 可传 cuda。",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="可选模型 revision；本地路径通常不需要。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑前 N 个任务做冒烟；不传则跑全集。",
    )
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
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须为正数")

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit("缺少推理依赖。请执行：uv sync --extra sft") from exc

    tasks = load_tasks(args.benchmark)
    if args.limit is not None:
        tasks = tasks[: args.limit]
        print(f"  --limit {args.limit}：只跑前 {len(tasks)} 个任务做冒烟")

    client = LocalTransformersClient(
        str(args.model),
        revision=args.revision,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
        device=args.device,
    )
    collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
    )
    summary = summarize_trajectories(
        [task["task_id"] for task in tasks], _read_jsonl(args.output)
    )
    summary["protocol"] = {
        "benchmark": str(args.benchmark),
        "model": str(args.model),
        "backend": "transformers-cpu",
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
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "device": args.device,
        "limit": args.limit,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
