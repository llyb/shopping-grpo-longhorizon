"""汇总固定 Shopping Agent 评测集上的确定性指标。"""

from collections import Counter


REWARD_V3 = "shopsimulator-reward-v3"
REWARD_V4 = "shopsimulator-reward-v4"
REWARD_V3_TYPES = (
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
    "wrong_purchase",
    "graceful_stop",
    "early_abstain",
    "repeat_loop",
    "max_steps",
    "reward_unverifiable",
)
REWARD_V4_TYPES = (
    "gold_purchase",
    "valid_alternative_purchase",
    "acceptable_compromise_purchase",
    "unverified_purchase",
    "wrong_purchase",
    "stop_with_acceptable_candidate",
    "graceful_stop",
    "early_abstain",
    "repeat_loop",
    "max_steps",
    "reward_unverifiable",
)


def summarize_trajectories(expected_task_ids, trajectories):
    """以固定 benchmark 全体 task 为分母汇总严格购物成功率。"""
    expected_ids = [int(task_id) for task_id in expected_task_ids]
    expected_set = set(expected_ids)
    by_task = {}
    for trajectory in trajectories:
        task_id = trajectory.get("task_id")
        if task_id is not None and int(task_id) in expected_set:
            by_task[int(task_id)] = trajectory

    completed_ids = sorted(by_task)
    missing_ids = sorted(expected_set - set(completed_ids))
    strict_successes = [task_id for task_id, item in by_task.items() if _is_strict_success(item)]
    done_tasks = [task_id for task_id, item in by_task.items() if item.get("done")]
    reward_details = [_reward_detail(item) for item in by_task.values()]
    reward_type_counts = Counter(
        detail.get("reward_type", "unknown") for detail in reward_details
    )
    reward_version_counts = Counter(
        detail.get("reward_version", "unknown") for detail in reward_details
    )
    purchase_successes = sum(
        detail.get("purchase_success") is True for detail in reward_details
    )
    gold_purchases = sum(
        detail.get("reward_version") in {REWARD_V3, REWARD_V4}
        and detail.get("reward_type") == "gold_purchase"
        for detail in reward_details
    )
    reward_valid_tasks = sum(
        detail.get("reward_version") in {REWARD_V3, REWARD_V4}
        and detail.get("reward_valid") is True
        for detail in reward_details
    )
    final_rewards = [float(item.get("final_reward", 0.0)) for item in by_task.values()]
    terminal_utilities = [
        float(detail.get("terminal_utility", item.get("final_reward", 0.0)))
        for item, detail in zip(by_task.values(), reward_details)
    ]
    weighted_scores = [
        float(detail.get("weighted_score", 0.0)) for detail in reward_details
    ]
    steps = [len(item.get("steps") or []) for item in by_task.values()]
    guard_reasons = Counter(
        blocked.get("reason", "unknown")
        for item in by_task.values()
        for blocked in item.get("blocked_tool_calls") or []
    )
    statuses = Counter(item.get("status", "unknown") for item in by_task.values())
    projections = [
        step["projection"]
        for item in by_task.values()
        for step in item.get("steps") or []
        if isinstance(step.get("projection"), dict)
    ]
    truncated_task_ids = {
        task_id
        for task_id, item in by_task.items()
        if any(
            bool((step.get("projection") or {}).get("truncated"))
            for step in item.get("steps") or []
        )
    }
    guard_rejections = [
        blocked
        for item in by_task.values()
        for blocked in item.get("blocked_tool_calls") or []
    ]
    executed_after_truncation = sum(
        bool(((steps[index - 1].get("projection") or {}).get("truncated")))
        for item in by_task.values()
        for steps in [item.get("steps") or []]
        for index in range(1, len(steps))
    )
    guard_rejections_after_truncation = sum(
        bool(item.get("latest_observation_truncated")) for item in guard_rejections
    )
    context_overflows = sum(
        (item.get("error") or {}).get("type") == "ContextBudgetError"
        for item in by_task.values()
    )
    denominator = len(expected_ids)
    acceptable_purchases = sum(
        detail.get("acceptable_purchase") is True for detail in reward_details
    )
    compromise_purchases = reward_type_counts.get(
        "acceptable_compromise_purchase", 0
    )
    valid_terminal_utilities = [
        utility
        for utility, detail in zip(terminal_utilities, reward_details)
        if detail.get("reward_valid") is True
    ]
    waste_costs = [float(detail.get("waste_cost", 0.0)) for detail in reward_details]
    return {
        "expected_tasks": denominator,
        "completed_tasks": len(completed_ids),
        "missing_tasks": missing_ids,
        "done_tasks": len(done_tasks),
        "done_rate": len(done_tasks) / denominator if denominator else 0.0,
        "strict_successes": len(strict_successes),
        "strict_success_task_ids": sorted(strict_successes),
        "strict_success_rate": len(strict_successes) / denominator if denominator else 0.0,
        "reward_contract": REWARD_V4,
        "reward_version_counts": dict(sorted(reward_version_counts.items())),
        "reward_type_counts": dict(sorted(reward_type_counts.items())),
        "reward_type_rates": {
            reward_type: reward_type_counts.get(reward_type, 0) / denominator
            if denominator
            else 0.0
            for reward_type in REWARD_V4_TYPES
        },
        "purchase_successes": purchase_successes,
        "purchase_success_rate": purchase_successes / denominator if denominator else 0.0,
        "acceptable_purchases": acceptable_purchases,
        "acceptable_purchase_rate": acceptable_purchases / denominator if denominator else 0.0,
        "acceptable_compromise_purchases": compromise_purchases,
        "acceptable_compromise_rate": compromise_purchases / denominator if denominator else 0.0,
        "gold_purchases": gold_purchases,
        "gold_purchase_rate": gold_purchases / denominator if denominator else 0.0,
        "reward_valid_tasks": reward_valid_tasks,
        "reward_valid_rate": reward_valid_tasks / denominator if denominator else 0.0,
        "total_final_reward": sum(final_rewards),
        "mean_final_reward": (
            sum(final_rewards) / len(by_task)
            if by_task
            else 0.0
        ),
        "mean_terminal_utility": (
            sum(terminal_utilities) / len(terminal_utilities)
            if terminal_utilities
            else 0.0
        ),
        "mean_valid_terminal_utility": (
            sum(valid_terminal_utilities) / len(valid_terminal_utilities)
            if valid_terminal_utilities
            else 0.0
        ),
        "mean_waste_cost": sum(waste_costs) / len(waste_costs) if waste_costs else 0.0,
        "mean_weighted_score": (
            sum(weighted_scores) / len(weighted_scores)
            if weighted_scores
            else 0.0
        ),
        "average_steps": sum(steps) / len(steps) if steps else 0.0,
        "status_counts": dict(sorted(statuses.items())),
        "guard_reason_counts": dict(sorted(guard_reasons.items())),
        "context_projection": {
            "projected_tool_observations": len(projections),
            "truncated_tool_observations": sum(
                bool(item.get("truncated")) for item in projections
            ),
            "raw_tokens": sum(int(item.get("raw_tokens", 0)) for item in projections),
            "visible_tokens": sum(
                int(item.get("visible_tokens", 0)) for item in projections
            ),
            "critical_footer_failures": sum(
                not bool(item.get("critical_footer_preserved", False))
                for item in projections
            ),
            "visible_asin_count": sum(
                int(item.get("visible_asin_count", 0)) for item in projections
            ),
            "visible_button_count": sum(
                int(item.get("visible_button_count", 0)) for item in projections
            ),
            "guard_rejections": len(guard_rejections),
            "guard_rejections_after_truncation": guard_rejections_after_truncation,
            "action_attempts_after_truncation": (
                executed_after_truncation + guard_rejections_after_truncation
            ),
            "guard_rejection_rate_after_truncation": (
                guard_rejections_after_truncation
                / (executed_after_truncation + guard_rejections_after_truncation)
                if executed_after_truncation + guard_rejections_after_truncation
                else 0.0
            ),
            "context_overflow_tasks": context_overflows,
            "max_context_input_tokens": max(
                (
                    int(turn.get("input_tokens", 0))
                    for item in by_task.values()
                    for turn in item.get("context_turn_tokens") or []
                ),
                default=0,
            ),
            "success_by_truncation_bucket": {
                "none": {
                    "tasks": len(set(by_task) - truncated_task_ids),
                    "strict_successes": sum(
                        _is_strict_success(item)
                        for task_id, item in by_task.items()
                        if task_id not in truncated_task_ids
                    ),
                },
                "any": {
                    "tasks": len(truncated_task_ids),
                    "strict_successes": sum(
                        _is_strict_success(item)
                        for task_id, item in by_task.items()
                        if task_id in truncated_task_ids
                    ),
                },
            },
        },
    }


def _reward_detail(trajectory):
    terminal = trajectory.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    return detail if isinstance(detail, dict) else {}


def _is_strict_success(trajectory):
    terminal = trajectory.get("terminal_result") or {}
    detail = _reward_detail(trajectory)
    return (
        detail.get("reward_version") in {REWARD_V3, REWARD_V4}
        and trajectory.get("status") == "done"
        and trajectory.get("done") is True
        and terminal.get("done") is True
        and terminal.get("over") is True
        and detail.get("reward_type") == "gold_purchase"
        and detail.get("reward_valid") is True
        and detail.get("purchase_success") is True
        and detail.get("termination_reason") == "gold_purchase"
        and (
            detail.get("reward_version") == REWARD_V3
            or detail.get("strict_success") is True
        )
    )
