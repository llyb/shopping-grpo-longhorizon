# Reward v4 完整重跑手册

本手册覆盖仓库唯一流程：`Baseline → SFT → GRPO → Evaluation`。所有新产物写入
`outputs/reward-v4-*`，不会覆盖现有模型或数据。SFT、GRPO train、GRPO validation、
开发集和 Final-200 同时按 `task_id` 与规范化需求文本隔离。

## 1. 一次性环境变量

以下命令在 Linux / WSL 的仓库根目录执行（Windows 主机请从 WSL 执行，原生 PowerShell
不满足 `uv`/CUDA/veRL 的启动约定）。API key 只通过当前 shell 的环境变量提供，
不会写入 metadata 或仓库文件；不要把它硬编码进脚本。`OPENAI_BASE_URL` 可以传 `/v1`
根地址，也兼容完整的
`/v1/chat/completions` 地址。如果教师服务没有 vLLM 专用的 `/tokenize` 接口，采集器会在
第一次探测失败后切换到保守的本地字符计数器；这不会把 token 计数请求当作聊天请求，
也不会记录 API key。

结构化商品详情和功能子页如果超过详情预算会 fail-closed，不会截断后继续把后台字段
当作 Agent 已见证据；停止候选的规格、合法 SKU 和价格也必须来自同一已观察规格指纹。

```bash
export ROOT="$(pwd)"
export RUN_ROOT="$ROOT/outputs/reward-v4-$(date +%Y%m%d-%H%M%S)"
export BASE_MODEL="$ROOT/models/Qwen3.5-2B" 
export OPENAI_BASE_URL="http://10.128.202.100:3010/v1"
export OPENAI_MODEL="qwen3.7-max"
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY && echo
export OPENAI_API_KEY

# If you copied the full /v1/chat/completions URL instead, the collector accepts
# it too.  The probe below normalizes both forms before appending the endpoint.
if [[ "$OPENAI_BASE_URL" == */chat/completions ]]; then
  export OPENAI_CHAT_URL="$OPENAI_BASE_URL"
else
  export OPENAI_CHAT_URL="${OPENAI_BASE_URL%/}/chat/completions"
fi

# OPENAI_MODEL 是教师/采集接口模型；BASE_MODEL 必须是本地可训练、且与
# scripts/serve_model.sh 所用工具调用解析器兼容的 Qwen3.5 系列或同等模型。
# glm-5.2 可以作为教师 API，不会自动变成 SFT/GRPO 的 BASE_MODEL。

bash scripts/setup.sh
mkdir -p "$RUN_ROOT"

export PYTHONPATH="$ROOT/src:$ROOT"
export SHOPPING_ENVIRONMENT_VERSION="shopsimulator-environment-v2.1"
export SHOPPING_ENV_MANIFEST="$ROOT/data/environment.json"
export SHOPPING_TOOL_CONFIG="$ROOT/configs/tools.json"
"$ROOT/.venv/bin/python" -c 'from scripts.check_grpo_runtime import validate_environment_contract; validate_environment_contract()'
```

可先验证教师 API；不要把真实 key 写进脚本或日志：

```bash
curl --fail-with-body --request POST \
  --url "$OPENAI_CHAT_URL" \
  --header "Authorization: Bearer $OPENAI_API_KEY" \
  --header 'Content-Type: application/json' \
  --data "{\"model\":\"$OPENAI_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你是什么模型？\"}],\"temperature\":0.0,\"max_tokens\":128,\"stream\":false}"
```

该探针会把认证 header 交给 `curl`；在共享机器上请改用受保护的 shell、避免开启
命令审计，并在结束后执行 `unset OPENAI_API_KEY`。采集器和评测器不会把 key 写入
轨迹或 metadata。

## 2. 冻结互斥任务池

```bash
"$ROOT/.venv/bin/python" scripts/build_data_splits.py \
  --output-dir "$RUN_ROOT/splits" \
  --sft-candidates 3000 \
  --grpo-train 1000 \
  --grpo-validation 100 \
  --development 200 \
  --seed 20260917
```

生成的 `metadata.json` 必须显示 `isolation.valid=true`。这一阶段会排除 Final-200 的
task ID、与 Final-200 需求文本相同的任务，以及 Reward v4 合同中规格轴无法唯一解析的任务。

## 3. 启动 ShopSimulator

另开一个终端，保持进程运行：

```bash
cd "$ROOT"
export SHOPSIM_ENV_SLOTS=8
bash scripts/start_environment.sh
```

主终端检查服务：

```bash
"$ROOT/.venv/bin/python" scripts/smoke_shop_env.py --base-url http://127.0.0.1:5700
```

## 4. 重新采集并构建 SFT 数据

```bash
"$ROOT/.venv/bin/python" scripts/collect_sft_data.py \
  --tasks "$RUN_ROOT/splits/sft_candidates.jsonl" \
  --held-out-tasks "$ROOT/data/evaluation/tasks.jsonl" \
  --output-dir "$RUN_ROOT/sft-collection" \
  --target-accepted 1200 \
  --attempts-per-task 2 \
  --workers 8 \
  --validation-ratio 0.10 \
  --seed 20260917 \
  --base-url http://127.0.0.1:5700 \
  --llm-base-url "$OPENAI_BASE_URL" \
  --model "$OPENAI_MODEL" \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 512 \
  --max-steps 35 \
  --timeout 180 \
  --context-window 24576 \
  --observation-token-budget 1536 \
  --observation-detail-token-budget 4096 \
  --observation-generic-token-budget 768
```

采集器可断点续跑；再次执行同一命令会跳过已经完成的 `(task_id, attempt_index)`。
主 SFT 集只接收 `reward_version=shopsimulator-reward-v4`、合法完整的
`gold_purchase`、`reward_valid=true`、`sampling_invalid=false`、`strict_success=true` 且关键证据覆盖率为 1 的轨迹；基础设施失败和不可还原终局也会拒绝。

确认目标数量确实达到后再进入训练（采集接口若提前耗尽候选会返回不足数量）：

```bash
"$ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["RUN_ROOT"]) / "sft-collection" / "metadata.json"
m = json.loads(p.read_text(encoding="utf-8"))
assert m.get("accepted", 0) >= 1200, m
assert m.get("train", 0) > 0 and m.get("validation", 0) > 0, m
print("SFT acceptance gate passed:", m["accepted"], "accepted")
PY
```

采集后执行隔离门禁：

```bash
"$ROOT/.venv/bin/python" scripts/verify_dataset_isolation.py \
  --dataset sft_train="$RUN_ROOT/sft-collection/train.jsonl" \
  --dataset sft_validation="$RUN_ROOT/sft-collection/validation.jsonl" \
  --dataset grpo_train="$RUN_ROOT/splits/grpo_train.jsonl" \
  --dataset grpo_validation="$RUN_ROOT/splits/grpo_validation.jsonl" \
  --dataset development="$RUN_ROOT/splits/development.jsonl" \
  --dataset evaluation="$ROOT/data/evaluation/tasks.jsonl" \
  --report "$RUN_ROOT/dataset-isolation.json"
```

任何 task ID 或规范化需求文本重合都会以非零状态退出，不能继续训练。

## 5. SFT 训练与合并

```bash
"$ROOT/.venv/bin/python" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train "$RUN_ROOT/sft-collection/train.jsonl" \
  --validation "$RUN_ROOT/sft-collection/validation.jsonl" \
  --output "$RUN_ROOT/models/sft-adapter" \
  --max-length 24576 \
  --epochs 2 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.03 \
  --lora-r 16 \
  --lora-alpha 32 \
  --dtype bf16 \
  --attention-implementation sdpa \
  --gradient-checkpointing \
  --seed 20260917

"$ROOT/.venv/bin/python" scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$RUN_ROOT/models/sft-adapter" \
  --output "$RUN_ROOT/models/sft-merged" \
  --bf16
```

输出目录必须不存在或为空；脚本拒绝覆盖已有模型。

## 6. 构建 GRPO 数据并训练

GRPO 数据只包含互斥任务池和公开 prompt；Reward 合同及 Gold 审计字段不进入 Actor prompt。

```bash
"$ROOT/.venv/bin/python" scripts/build_grpo_dataset.py \
  --tasks "$RUN_ROOT/splits/grpo_train.jsonl" \
  --output-dir "$RUN_ROOT/grpo-data" \
  --split train

"$ROOT/.venv/bin/python" scripts/build_grpo_dataset.py \
  --tasks "$RUN_ROOT/splits/grpo_validation.jsonl" \
  --output-dir "$RUN_ROOT/grpo-data" \
  --split validation

"$ROOT/.venv/bin/python" scripts/train_grpo.py \
  --model "$RUN_ROOT/models/sft-merged" \
  --train-data "$RUN_ROOT/grpo-data/train.parquet" \
  --val-data "$RUN_ROOT/grpo-data/validation.parquet" \
  --env-url http://127.0.0.1:5700 \
  --output "$RUN_ROOT/models/grpo" \
  --logger console \
  --experiment-name shopping-reward-v4
```

GRPO 直接使用环境返回的 v4 `terminal_utility`。`reward_unverifiable` 和基础设施失败会标记
`sampling_invalid=true`；训练侧不会把其占位值 0 当作普通负样本，也不会再叠加旧的长度惩罚。

训练完成后选择要导出的 checkpoint（通常使用验证指标最好的 `global_step_*`，不要仅按最后一步）：

```bash
export GRPO_ACTOR="$RUN_ROOT/models/grpo/global_step_<STEP>/actor"
bash scripts/export_grpo.sh "$GRPO_ACTOR" "$RUN_ROOT/models/grpo-merged"
```

## 7. 独立开发集评测与模型冻结

先只使用与 SFT、GRPO 和 Final-200 都互斥的 200 题开发集选择 GRPO checkpoint、检查退化，
不要根据 Final-200 结果回头调参。为 Base、SFT、GRPO 分别建立目录：

```bash
mkdir -p "$RUN_ROOT/development/base" "$RUN_ROOT/development/sft" "$RUN_ROOT/development/grpo"
```

每次在独立终端启动一个模型服务：

```bash
cd "$ROOT"
export SERVED_MODEL_NAME="<SERVED_NAME>"
export LLM_PORT=8000
bash scripts/serve_model.sh "<MODEL_PATH>"
```

主终端执行开发集评测。`<RUN_NAME>` 只使用 `base`、`sft`、`grpo`，分别对应
`$BASE_MODEL`、`$RUN_ROOT/models/sft-merged`、`$RUN_ROOT/models/grpo-merged`；
`<SERVED_NAME>` 可以分别使用 `base-agent`、`sft-agent`、`grpo-agent`：

```bash
"$ROOT/.venv/bin/python" scripts/evaluate_shop_benchmark.py \
  --benchmark "$RUN_ROOT/splits/development.jsonl" \
  --output "$RUN_ROOT/development/<RUN_NAME>/trajectories.jsonl" \
  --summary "$RUN_ROOT/development/<RUN_NAME>/summary.json" \
  --base-url http://127.0.0.1:5700 \
  --model "<SERVED_NAME>" \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --max-steps 35 \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 512 \
  --context-window 24576 \
  --observation-token-budget 1536 \
  --observation-detail-token-budget 4096 \
  --observation-generic-token-budget 768

"$ROOT/.venv/bin/python" scripts/build_eval_report.py \
  --run-dir "$RUN_ROOT/development/<RUN_NAME>"
```

用 GRPO 训练期间的 validation 指标选择并导出 checkpoint，再用这里的独立开发集做阶段对比和
一次性退化检查。确认运行与指标合理后冻结三个模型；不得用 Final-200 选择 step 或回头调参。

## 8. 固定 Final-200 评测

先建立三个结果目录：

```bash
mkdir -p "$RUN_ROOT/evaluation/base" "$RUN_ROOT/evaluation/sft" "$RUN_ROOT/evaluation/grpo"
```

每个模型都按下面两步执行：先在一个独立终端启动 vLLM，再在主终端跑评测。完成后停止 vLLM，
换下一个模型。三个模型路径分别为：

```text
Base: $BASE_MODEL
SFT:  $RUN_ROOT/models/sft-merged
GRPO: $RUN_ROOT/models/grpo-merged
```

启动示例（将 `<MODEL_PATH>` 和 `<SERVED_NAME>` 替换成当前模型）：

```bash
cd "$ROOT"
export SERVED_MODEL_NAME="<SERVED_NAME>"
export LLM_PORT=8000
bash scripts/serve_model.sh "<MODEL_PATH>"
```

主终端评测示例（将 `<RUN_NAME>` 替换为 `base`、`sft` 或 `grpo`）：

```bash
"$ROOT/.venv/bin/python" scripts/evaluate_shop_benchmark.py \
  --benchmark "$ROOT/data/evaluation/tasks.jsonl" \
  --output "$RUN_ROOT/evaluation/<RUN_NAME>/trajectories.jsonl" \
  --summary "$RUN_ROOT/evaluation/<RUN_NAME>/summary.json" \
  --base-url http://127.0.0.1:5700 \
  --model "<SERVED_NAME>" \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --max-steps 35 \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 512 \
  --context-window 24576 \
  --observation-token-budget 1536 \
  --observation-detail-token-budget 4096 \
  --observation-generic-token-budget 768
```

生成逐模型和对比报告：

```bash
for name in base sft grpo; do
  "$ROOT/.venv/bin/python" scripts/build_eval_report.py \
    --run-dir "$RUN_ROOT/evaluation/$name"
done

"$ROOT/.venv/bin/python" scripts/build_comparison_report.py \
  --evaluation-dir "$RUN_ROOT/evaluation" \
  --output "$RUN_ROOT/evaluation/comparison.html"
```

最终同时检查严格 Gold 成功率、完整需求满足率、可接受购买率、错误购买率、未核验购买率、
各停止类型、Reward 有效率、固定分母平均奖励、有效样本平均效用和平均操作成本。无效任务仍保留
在固定 200 题分母中。

## 9. 恢复与清理原则

- 所有采集 JSONL 都可续跑；不要删除 `raw.jsonl` 后只保留派生文件。
- SFT、GRPO 和评测使用不同目录；模型合并输出必须是新目录。
- 如果隔离检查失败，重新生成 split，不要手工删几行后继续。
- 如果 Reward v4 合同、运行时文件或环境配置变化，重新生成数据；环境 manifest 的哈希门禁会阻止
  旧运行时与新配置混用。
- 结构化商品详情若超过 4096-token 详情预算会 fail-closed；该轨迹不得用后台未展示字段取得正奖励。
- API key 只通过环境变量传入；metadata 会记录 endpoint 和模型名，但明确拒绝序列化密钥。
