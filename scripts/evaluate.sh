#!/usr/bin/env bash
set -euo pipefail

# 阶段 B：GPU 上执行 rollout 采集与 Judge 评测。
# 依赖阶段 A（scripts/prepare_eval.sh）已生成 shared/task_facts.jsonl 与
# shared/rubrics.jsonl；本脚本只复用教师模型采集的 Rubric，不再调用 curator。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-model}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:?JUDGE_BASE_URL must point to the V4 Flash/Pro OpenAI-compatible service}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${LLM_API_KEY}}"
CURATOR_MODEL="${CURATOR_MODEL:-deepseek/deepseek-v4-flash}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek/deepseek-v4-pro}"
TASK_FACTS="$OUTPUT_DIR/../shared/task_facts.jsonl"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --summary "$OUTPUT_DIR/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --task-facts "$TASK_FACTS" \
  --curator-model "$CURATOR_MODEL" \
  --judge-base-url "$JUDGE_BASE_URL" \
  --judge-model "$JUDGE_MODEL" \
  --judge-api-key "$JUDGE_API_KEY" \
  --skip-rubric-curation

"$ROOT/.venv/bin/python" scripts/build_eval_report.py --run-dir "$OUTPUT_DIR"
