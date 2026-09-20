from __future__ import annotations

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    apply_reward_length_shaping,
    make_runtime_state,
    reward_breakdown,
    validate_reward,
)


def detail(reward_type="gold_purchase", utility=1.0):
    valid = reward_type != "reward_unverifiable"
    purchase_success = reward_type in {"gold_purchase", "valid_alternative_purchase"}
    acceptable = reward_type in {
        "gold_purchase",
        "valid_alternative_purchase",
        "acceptable_compromise_purchase",
    }
    return {
        "reward_version": "shopsimulator-reward-v4",
        "reward_type": reward_type,
        "termination_reason": reward_type,
        "reward_valid": valid,
        "sampling_invalid": not valid,
        "terminal_utility": utility,
        "purchase_success": purchase_success,
        "acceptable_purchase": acceptable,
        "strict_success": reward_type == "gold_purchase" and valid,
        "target_asin_match": reward_type == "gold_purchase",
        "hard_gates": {},
        "evidence_coverage": 1.0 if valid else 0.0,
        "preference_satisfaction": {"aggregate": 1.0 if acceptable else 0.0},
        "price_utility": 1.0,
        "reward_components": {"total": utility},
        "waste_cost": 0.0,
        "violation_severity": 0.0,
    }


class GrpoRewardV4AdapterTest(unittest.TestCase):
    def test_v4_terminal_utility_is_used_without_extra_shaping(self):
        public = validate_reward(detail())
        state = make_runtime_state(task_id=1, max_steps=35)
        state.update(
            {
                "done": True,
                "terminal_result": {"done": True, "over": True},
                "final_reward": 1.0,
                "reward_version": public["reward_version"],
                "reward_type": public["reward_type"],
                "reward_valid": public["reward_valid"],
                "reward_detail": public,
                "steps": [{}] * 30,
            }
        )
        reward = reward_breakdown(state)
        shaped = apply_reward_length_shaping(reward, state, enabled=True)
        self.assertEqual(reward["total"], 1.0)
        self.assertEqual(shaped["total"], 1.0)
        self.assertFalse(shaped["sampling_invalid"])

    def test_unverifiable_is_sampling_invalid(self):
        public = validate_reward(detail("reward_unverifiable", 0.0))
        self.assertFalse(public["reward_valid"])
        self.assertTrue(public["sampling_invalid"])


if __name__ == "__main__":
    unittest.main()
