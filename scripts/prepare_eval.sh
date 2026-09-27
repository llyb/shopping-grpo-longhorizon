#!/usr/bin/env bash
set -euo pipefail

# 阶段 A：教师模型采集（无需 GPU）。
# 先导出私有 TaskFacts（ShopSimulator runtime），再用 curator 教师模型构建 Rubric，
# 缓存到 shared 目录。完成后即可在 GPU 环境执行 scripts/evaluate.sh。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-model}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"
SHARED_DIR="$OUTPUT_DIR/../shared"
TASK_FACTS="$SHARED_DIR/task_facts.jsonl"

JUDGE_BASE_URL="${JUDGE_BASE_URL:?JUDGE_BASE_URL must point to the V4 Flash/Pro OpenAI-compatible service}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
CURATOR_MODEL="${CURATOR_MODEL:-deepseek/deepseek-v4-flash}"

mkdir -p "$SHARED_DIR"
cd "$ROOT"

"$ROOT/environments/ShopSimulator/.venv-shopsim/bin/python" scripts/export_eval_task_facts.py \
  --benchmark data/evaluation/tasks.jsonl \
  --output "$TASK_FACTS"

"$ROOT/.venv/bin/python" scripts/build_eval_rubrics.py \
  --benchmark data/evaluation/tasks.jsonl \
  --task-facts "$TASK_FACTS" \
  --shared-dir "$SHARED_DIR" \
  --curator-base-url "$JUDGE_BASE_URL" \
  --curator-model "$CURATOR_MODEL" \
  --curator-api-key "$JUDGE_API_KEY"
