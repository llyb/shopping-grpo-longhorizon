# Documentation

Follow the guides in workflow order:

1. [Data collection](data-collection.md) explains how the checked-in SFT data
   was produced and audited.
2. [SFT](sft.md) trains the first useful shopping agent.
3. [GRPO](grpo.md) improves that model with online environment reward.
4. [Evaluation](evaluation.md) compares baseline, SFT and GRPO fairly.
5. [Final-200 Clean evaluation dataset](evaluation-dataset.md) defines the current
   curated benchmark and its update record.

[Reward v4](reward-v4-design.md) is the active specification shared by
collection, GRPO and evaluation. Use the
[complete Reward v4 rerun workflow](reward-v4-workflow.md) for fresh SFT data,
GRPO data/training, strict split isolation and Final-200 evaluation. The
[Reward v3 document](reward-v3.md) is retained as historical context only.
