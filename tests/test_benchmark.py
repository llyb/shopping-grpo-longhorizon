"""验证固定评测集的统计口径。"""

import unittest

from shopping_grpo.evaluation.summary import summarize_trajectories


def _trajectory(task_id, strict=False, steps=3, status="done", blocked=None):
    reward_detail = {
        "reward_version": "shopsimulator-reward-v4",
        "reward_type": "gold_purchase" if strict else "wrong_purchase",
        "reward_valid": True,
        "purchase_success": strict,
        "acceptable_purchase": strict,
        "strict_success": strict,
        "evidence_coverage": 1.0,
        "termination_reason": "gold_purchase" if strict else "wrong_purchase",
        "terminal_utility": 1.0 if strict else -0.85,
        "weighted_score": 1.0 if strict else 0.0,
    }
    return {
        "task_id": task_id,
        "status": status,
        "done": status == "done",
        "final_reward": 1.0 if strict else -0.85,
        "steps": [{"tool_name": "search_products"}] * steps,
        "blocked_tool_calls": blocked or [],
        "terminal_result": {"done": status == "done", "over": status == "done", "reward_detail": reward_detail},
    }


class BenchmarkTest(unittest.TestCase):
    def test_summary_uses_expected_tasks_as_v4_strict_success_denominator(self):
        """缺失或非严格成功 task 都计入失败，避免只统计已跑完的容易样本。"""
        summary = summarize_trajectories(
            expected_task_ids=[10, 11, 12],
            trajectories=[
                _trajectory(10, strict=True, steps=4),
                _trajectory(
                    11,
                    strict=False,
                    steps=6,
                    blocked=[{"reason": "schema_extra_arguments:asin"}],
                ),
            ],
        )

        self.assertEqual(summary["expected_tasks"], 3)
        self.assertEqual(summary["completed_tasks"], 2)
        self.assertEqual(summary["strict_successes"], 1)
        self.assertAlmostEqual(summary["strict_success_rate"], 1 / 3)
        self.assertEqual(summary["gold_purchases"], 1)
        self.assertAlmostEqual(summary["gold_purchase_rate"], 1 / 3)
        self.assertEqual(summary["reward_contract"], "shopsimulator-reward-v4")
        self.assertEqual(summary["reward_type_counts"]["wrong_purchase"], 1)
        self.assertAlmostEqual(summary["mean_final_reward"], (1.0 - 0.85) / 2)
        self.assertEqual(summary["missing_tasks"], [12])
        self.assertAlmostEqual(summary["average_steps"], 5.0)
        self.assertEqual(summary["guard_reason_counts"]["schema_extra_arguments:asin"], 1)

    def test_summary_separates_successes_by_projection_bucket(self):
        projected = _trajectory(10, strict=True, steps=1)
        projected["steps"][0]["projection"] = {
            "truncated": True,
            "raw_tokens": 1000,
            "visible_tokens": 700,
            "visible_asin_count": 10,
            "visible_button_count": 12,
            "critical_footer_preserved": True,
        }
        projected["context_turn_tokens"] = [{"input_tokens": 17000}]
        projected["blocked_tool_calls"] = [
            {"reason": "click", "latest_observation_truncated": True}
        ]
        plain = _trajectory(11, strict=False, steps=1)

        summary = summarize_trajectories([10, 11], [projected, plain])
        projection = summary["context_projection"]

        self.assertEqual(projection["truncated_tool_observations"], 1)
        self.assertEqual(projection["guard_rejections_after_truncation"], 1)
        self.assertEqual(projection["max_context_input_tokens"], 17000)
        self.assertEqual(
            projection["success_by_truncation_bucket"]["any"],
            {"tasks": 1, "strict_successes": 1},
        )
        self.assertEqual(
            projection["success_by_truncation_bucket"]["none"],
            {"tasks": 1, "strict_successes": 0},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
