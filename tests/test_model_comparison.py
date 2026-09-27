import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.evaluation.contracts import CONTRACT_VERSION, JUDGE_DIMENSIONS
from shopping_grpo.evaluation.results import EVALUATION_RESULT_VERSION

from scripts.build_model_comparison import collect_runs, main


def _evaluation(task_id, *, strict):
    return {
        "schema_version": EVALUATION_RESULT_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "task_id": task_id,
        "trajectory_id": f"trajectory-{task_id}",
        "reward_and_terminal": {
            "metrics": {
                "strict_gold_success": strict,
                "reward_type": "gold_purchase" if strict else "wrong_purchase",
            }
        },
        "requirement_rubric": {
            "rubrics": [],
            "assessments": [],
            "reward_rubric_disagreement": False,
        },
        "trajectory_quality": {
            "judge_status": "valid",
            "dimension_scores": {
                name: {"score": 1} for name in JUDGE_DIMENSIONS
            },
        },
        "deterministic": {
            "actions_and_efficiency": {"executed_tool_steps": 3},
            "legality": {"guard_rejection_count": 0},
            "repetition": {"duplicate_canonical_action_count": 0},
        },
    }


class ModelComparisonTest(unittest.TestCase):
    def _write_run(self, root, label, *, strict):
        run = root / label
        run.mkdir()
        (run / "evaluations.jsonl").write_text(
            json.dumps(_evaluation(1, strict=strict)) + "\n",
            encoding="utf-8",
        )

    def _write_benchmark(self, root):
        benchmark = root / "tasks.jsonl"
        benchmark.write_text(json.dumps({"task_id": 1}) + "\n", encoding="utf-8")
        return benchmark

    def test_writes_paired_comparison_from_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = self._write_benchmark(root)
            self._write_run(root, "baseline", strict=False)
            self._write_run(root, "sft", strict=True)
            output = root / "model_comparison.json"

            main(
                [
                    "--evaluation-dir", str(root),
                    "--benchmark", str(benchmark),
                    "--output", str(output),
                ]
            )

            comparison = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(comparison["pairwise"]), {"baseline_to_sft"})
            pairwise = comparison["pairwise"]["baseline_to_sft"]
            self.assertEqual(pairwise["paired_tasks"], 1)
            self.assertEqual(
                pairwise["reward_and_terminal"]["strict_success_transitions"],
                {"failure_to_success": 1},
            )

    def test_skips_without_two_comparable_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = self._write_benchmark(root)
            self._write_run(root, "solo", strict=True)
            output = root / "model_comparison.json"

            main(
                [
                    "--evaluation-dir", str(root),
                    "--benchmark", str(benchmark),
                    "--output", str(output),
                ]
            )

            self.assertEqual(set(collect_runs(root)), {"solo"})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
