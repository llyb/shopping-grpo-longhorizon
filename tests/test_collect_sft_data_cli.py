import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.collect_sft_data import (
    _validate_args,
    batch_paths,
    collect_until_target,
    parse_args,
)


class CollectSftDataCliTests(unittest.TestCase):
    def test_defaults_match_the_current_teacher_collection_contract(self):
        with patch.object(sys, "argv", ["collect_sft_data.py", "--tasks", "tasks.jsonl"]):
            args = parse_args()

        self.assertEqual(args.model, "deepseek/deepseek-v4-flash")
        self.assertEqual(args.base_url, "http://127.0.0.1:5700")
        self.assertEqual(args.max_steps, 35)
        self.assertEqual(args.output_dir, Path("outputs/sft-collection"))

    def test_batch_paths_keep_raw_and_derived_files_together(self):
        paths = batch_paths(Path("outputs/example"))

        self.assertEqual(paths["raw"], Path("outputs/example/raw.jsonl"))
        self.assertEqual(paths["accepted"], Path("outputs/example/accepted.jsonl"))
        self.assertEqual(paths["train"], Path("outputs/example/train.jsonl"))
        self.assertEqual(
            paths["validation"], Path("outputs/example/validation.jsonl")
        )

    def test_build_only_needs_neither_tasks_nor_model_credentials(self):
        with patch.object(
            sys,
            "argv",
            ["collect_sft_data.py", "--build-only"],
        ):
            args = parse_args()

        _validate_args(args)
        self.assertIsNone(args.tasks)

    def test_collection_stops_at_the_accepted_target(self):
        rows = [
            {"trajectory_id": "one", "task_id": 1},
            {"trajectory_id": "two", "task_id": 2},
        ]
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "scripts.collect_sft_data.collect_for_task", side_effect=rows
        ) as collect, patch(
            "scripts.collect_sft_data.acceptance_reasons", return_value=(True, [])
        ):
            written, accepted = collect_until_target(
                tasks=[{"task_id": 1}, {"task_id": 2}, {"task_id": 3}],
                target_accepted=2,
                client=object(),
                output_path=Path(tmpdir) / "raw.jsonl",
                base_url="http://shop.test",
                max_steps=35,
                attempts_per_task=1,
                workers=1,
            )

        self.assertEqual([row["trajectory_id"] for row in written], ["one", "two"])
        self.assertEqual(accepted, 2)
        self.assertEqual(collect.call_count, 2)

    def test_accepted_target_ignores_held_out_rows_already_in_raw(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = Path(tmpdir) / "raw.jsonl"
            raw.write_text(json.dumps({"task_id": 1}) + "\n", encoding="utf-8")
            with patch(
                "scripts.collect_sft_data.collect_for_task",
                return_value={"trajectory_id": "two", "task_id": 2},
            ) as collect, patch(
                "scripts.collect_sft_data.acceptance_reasons",
                return_value=(True, []),
            ):
                written, accepted = collect_until_target(
                    tasks=[{"task_id": 2}],
                    target_accepted=1,
                    client=object(),
                    output_path=raw,
                    base_url="http://shop.test",
                    max_steps=35,
                    attempts_per_task=1,
                    workers=1,
                    excluded_task_ids={1},
                )

        self.assertEqual([row["task_id"] for row in written], [2])
        self.assertEqual(accepted, 1)
        self.assertEqual(collect.call_count, 1)

    def test_parallel_collection_uses_an_independent_client_per_trajectory(self):
        clients = []

        def client_factory():
            client = object()
            clients.append(client)
            return client

        def collect(task, *, client, **kwargs):
            return {
                "trajectory_id": f"trajectory-{task['task_id']}",
                "task_id": task["task_id"],
                "client_id": id(client),
            }

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "scripts.collect_sft_data.collect_for_task", side_effect=collect
        ), patch(
            "scripts.collect_sft_data.acceptance_reasons", return_value=(True, [])
        ):
            written, _ = collect_until_target(
                tasks=[{"task_id": 1}, {"task_id": 2}],
                target_accepted=2,
                client=None,
                client_factory=client_factory,
                output_path=Path(tmpdir) / "raw.jsonl",
                base_url="http://shop.test",
                max_steps=35,
                attempts_per_task=1,
                workers=2,
            )

        self.assertEqual(len(clients), 2)
        self.assertEqual(len({row["client_id"] for row in written}), 2)


if __name__ == "__main__":
    unittest.main()
