"""验证 benchmark 清单与评测入口的最小 CLI 行为。"""

import sys
import unittest
from unittest.mock import patch

from scripts.evaluate_shop_benchmark import parse_args, reusable_cached_rubric


class BenchmarkCliTest(unittest.TestCase):
    def test_evaluation_defaults_match_frozen_protocol(self):
        """Base、SFT、GRPO 必须默认使用同一 35 步上限。"""
        with patch.object(
            sys,
            "argv",
            [
                "evaluate_shop_benchmark.py",
                "--benchmark",
                "data/evaluation/tasks.jsonl",
                "--output",
                "outputs/eval/base/raw.jsonl",
                "--summary",
                "outputs/eval/base/summary.json",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--llm-base-url",
                "http://127.0.0.1:8000/v1",
                "--api-key",
                "EMPTY",
                "--task-facts",
                "outputs/evaluation/shared/task_facts.jsonl",
                "--curator-base-url",
                "http://127.0.0.1:8100/v1",
                "--judge-base-url",
                "http://127.0.0.1:8100/v1",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.max_steps, 35)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.context_window, 24576)
        self.assertEqual(args.context_safety_margin, 512)
        self.assertFalse(args.context_compaction)
        self.assertEqual(args.observation_token_budget, 1536)
        self.assertEqual(args.observation_detail_token_budget, 4096)
        self.assertEqual(args.observation_generic_token_budget, 768)
        self.assertEqual(args.observation_search_top_k, 20)


class RubricCacheTest(unittest.TestCase):
    def test_rejects_cache_from_another_prompt_or_model(self):
        cached = {
            1: {
                "schema_version": "shopping-requirement-rubric-v1",
                "task_id": 1,
                "generation": {
                    "curator_model": "deepseek/deepseek-v4-flash",
                    "curator_prompt_version": "rubric-curator-v1-draft-r3",
                },
            }
        }

        self.assertIsNone(
            reusable_cached_rubric(
                cached,
                task_id=1,
                curator_model="deepseek/deepseek-v4-flash",
                curator_prompt_version="rubric-curator-v1-draft-r4",
            )
        )
        self.assertIsNone(
            reusable_cached_rubric(
                cached,
                task_id=1,
                curator_model="other-model",
                curator_prompt_version="rubric-curator-v1-draft-r3",
            )
        )

    def test_ignores_absent_task(self):
        self.assertIsNone(
            reusable_cached_rubric(
                {},
                task_id=7,
                curator_model="deepseek/deepseek-v4-flash",
                curator_prompt_version="rubric-curator-v1-draft-r4",
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
