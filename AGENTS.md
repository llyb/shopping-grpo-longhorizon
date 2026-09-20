# Repository contract

This repository supports one workflow only:

```text
Baseline → SFT → GRPO → Evaluation
```

The runtime contract is ShopSimulator Environment v2.1, Reward v4, observation
v2 and tool schema v2. Do not add compatibility launchers, historical datasets,
old benchmarks, machine-specific paths or experiment journals.

Training data must never overlap `data/evaluation/tasks.jsonl`. Strict success
requires a complete `gold_purchase` terminal result with `reward_valid=true`.

Do not start training, merge models or run the 200-task evaluation unless the
user explicitly requests execution.
