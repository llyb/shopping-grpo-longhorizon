import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_glm_report import build_data


class EvaluationReportTest(unittest.TestCase):
    def test_report_reads_any_evaluation_directory_and_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary = {
                "completed_tasks": 1,
                "done_tasks": 1,
                "done_rate": 1.0,
                "strict_successes": 1,
                "strict_success_rate": 1.0,
                "strict_success_task_ids": [42],
                "purchase_successes": 1,
                "purchase_success_rate": 1.0,
                "mean_final_reward": 1.0,
                "mean_weighted_score": 1.0,
                "average_steps": 1.0,
                "protocol": {"model": "new-model-1"},
            }
            trajectory = {
                "task_id": 42,
                "status": "done",
                "done": True,
                "final_reward": 1.0,
                "steps": [],
                "blocked_tool_calls": [],
                "initial_result": {"instruction": "test"},
                "terminal_result": {
                    "reward_detail": {"reward_type": "gold_purchase", "purchase_success": True},
                    "termination_reason": "gold_purchase",
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "trajectories.jsonl").write_text(
                json.dumps(trajectory) + "\n", encoding="utf-8"
            )

            data = build_data(run_dir)

            self.assertEqual(data["meta"]["model"], "new-model-1")
            self.assertEqual(data["summary"]["total"], 1)

    def test_report_includes_early_abstain_and_guard_per_task(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary = {
                "completed_tasks": 1,
                "done_tasks": 1,
                "done_rate": 1.0,
                "strict_successes": 0,
                "strict_success_rate": 0.0,
                "strict_success_task_ids": [],
                "purchase_successes": 0,
                "purchase_success_rate": 0.0,
                "mean_final_reward": 0.0,
                "mean_weighted_score": 0.0,
                "average_steps": 1.0,
                "protocol": {"model": "new-model-1"},
            }
            trajectory = {
                "task_id": 42,
                "status": "done",
                "done": True,
                "final_reward": 0.0,
                "steps": [],
                "blocked_tool_calls": [{"reason": "test_guard"}],
                "initial_result": {"instruction": "test"},
                "terminal_result": {
                    "reward_detail": {
                        "reward_type": "early_abstain",
                        "purchase_success": False,
                    },
                    "termination_reason": "early_abstain",
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "trajectories.jsonl").write_text(
                json.dumps(trajectory) + "\n", encoding="utf-8"
            )

            data = build_data(run_dir)

            early_abstain = next(
                item for item in data["charts"]["outcomes"]
                if item["key"] == "early_abstain"
            )
            self.assertEqual(early_abstain["value"], 1)
            self.assertEqual(data["summary"]["guard_per_task"], 1.0)

    def test_report_renders_judge_panel_when_summary_present(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_run(run_dir, _judge_summary())

            data = build_data(run_dir)

            self.assertTrue(data["meta"]["judge"])
            self.assertEqual(data["judge"]["coverage"], 1.0)
            self.assertEqual(data["judge"]["primary_errors"], {"repeat_loop": 1})
            self.assertEqual(len(data["judge"]["dimensions"]), 5)

    def test_report_without_judge_summary_has_no_panel_data(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_run(run_dir)

            data = build_data(run_dir)

            self.assertIsNone(data["judge"])
            self.assertFalse(data["meta"]["judge"])


def _write_run(run_dir, evaluation_summary=None):
    summary = {
        "completed_tasks": 1,
        "done_tasks": 1,
        "done_rate": 1.0,
        "strict_successes": 0,
        "strict_success_rate": 0.0,
        "strict_success_task_ids": [],
        "purchase_successes": 0,
        "purchase_success_rate": 0.0,
        "mean_final_reward": 0.0,
        "mean_weighted_score": 0.0,
        "average_steps": 1.0,
        "protocol": {"model": "new-model-1"},
    }
    trajectory = {
        "task_id": 42,
        "status": "done",
        "done": True,
        "final_reward": 0.0,
        "steps": [],
        "blocked_tool_calls": [],
        "initial_result": {"instruction": "test"},
        "terminal_result": {
            "reward_detail": {"reward_type": "early_abstain", "purchase_success": False},
            "termination_reason": "early_abstain",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "trajectories.jsonl").write_text(
        json.dumps(trajectory) + "\n", encoding="utf-8"
    )
    if evaluation_summary is not None:
        (run_dir / "evaluation_summary.json").write_text(
            json.dumps(evaluation_summary), encoding="utf-8"
        )


def _judge_summary():
    return {
        "expected_tasks": 1,
        "completed_evaluations": 1,
        "trajectory_quality": {
            "judge_status_counts": {"valid": 1},
            "judge_coverage_rate": 1.0,
            "dimensions": {
                name: {
                    "score_counts": {"0": 0, "1": 1, "2": 0},
                    "mean_score_among_valid_judges": 1.0,
                }
                for name in (
                    "search_strategy",
                    "candidate_utilization",
                    "evidence_verification",
                    "decision_quality",
                    "termination_efficiency",
                )
            },
            "primary_error_counts": {"repeat_loop": 1},
            "secondary_error_counts": {},
        },
        "requirement_rubric": {
            "status_counts": {"satisfied": 1},
            "status_counts_by_hardness": {"hard": {"satisfied": 1}},
            "reward_rubric_disagreement_tasks": 0,
        },
    }


if __name__ == "__main__":
    unittest.main()
