import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from shopping_grpo.collection.sft import (
    acceptance_reasons,
    build_collection_artifacts,
    build_sft_row,
)


def _assistant_tool(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _tool_message(call_id, name, content):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def _accepted_trajectory(task_id=1):
    asin = "100000000001"
    messages = [
        {"role": "system", "content": "use tools"},
        {"role": "user", "content": "帮我买乳胶枕"},
        _assistant_tool("search_products", {"query": "乳胶枕"}, "call-1"),
        _tool_message(
            "call-1",
            "search_products",
            f"1|{asin}|99.0|brand|category|attr|乳胶枕\n"
            f'可点击的按钮: ["{asin}"]',
        ),
        _assistant_tool("open_product", {"asin": asin}, "call-2"),
        _tool_message(
            "call-2",
            "open_product",
            '详情\n可点击的按钮: ["满天星", "Buy Now"]',
        ),
        _assistant_tool("select_option", {"value": "满天星"}, "call-3"),
        _tool_message(
            "call-3",
            "select_option",
            '已选择\n可点击的按钮: ["Buy Now"]',
        ),
        _assistant_tool("buy_now", {}, "call-4"),
        _tool_message("call-4", "buy_now", "Reward: 1.0; Gold purchase"),
    ]
    messages[2]["reasoning_content"] = "Teacher private reasoning"
    return {
        "trajectory_id": f"trajectory-{task_id}",
        "task_id": task_id,
        "attempt_index": 0,
        "status": "done",
        "done": True,
        "error": None,
        "release_error": None,
        "initial_result": {
            "environment_version": "shopsimulator-environment-v2.1"
        },
        "messages": messages,
        "steps": [
            {
                "tool_name": "search_products",
                "tool_call": messages[2]["tool_calls"][0],
                "env_action": "search[乳胶枕]",
                "done": False,
            },
            {
                "tool_name": "open_product",
                "tool_call": messages[4]["tool_calls"][0],
                "env_action": f"click[{asin}]",
                "done": False,
            },
            {
                "tool_name": "select_option",
                "tool_call": messages[6]["tool_calls"][0],
                "env_action": "click[满天星]",
                "done": False,
            },
            {
                "tool_name": "buy_now",
                "tool_call": messages[8]["tool_calls"][0],
                "env_action": "click[Buy Now]",
                "done": True,
            },
        ],
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v4",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "sampling_invalid": False,
                "purchase_success": True,
                "strict_success": True,
                "evidence_coverage": 1.0,
                "termination_reason": "gold_purchase",
            },
        },
    }


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class SftCollectionTests(unittest.TestCase):
    def test_accepts_only_strict_reward_v4_gold_purchase(self):
        accepted, reasons = acceptance_reasons(_accepted_trajectory())
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])

        invalid = _accepted_trajectory()
        invalid["terminal_result"]["reward_detail"]["reward_valid"] = False
        accepted, reasons = acceptance_reasons(invalid)
        self.assertFalse(accepted)
        self.assertIn("reward_v4_invalid", reasons)

        invalid = _accepted_trajectory()
        invalid["terminal_result"]["reward_detail"]["sampling_invalid"] = True
        accepted, reasons = acceptance_reasons(invalid)
        self.assertFalse(accepted)
        self.assertIn("sampling_invalid", reasons)

    def test_sft_row_removes_reasoning_and_terminal_reward(self):
        row = build_sft_row(_accepted_trajectory())
        payload = json.dumps(row, ensure_ascii=False)

        self.assertNotIn("Teacher private reasoning", payload)
        self.assertNotIn("reasoning_content", payload)
        self.assertNotIn("Reward: 1.0", payload)
        self.assertEqual(row["messages"][-1]["content"], "购买已完成。")
        self.assertTrue(row["tools"])

    def test_build_artifacts_excludes_held_out_tasks_and_splits_by_task(self):
        rejected = deepcopy(_accepted_trajectory(4))
        rejected["status"] = "error"
        rejected["error"] = {"type": "RuntimeError", "message": "boom"}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "raw.jsonl"
            output = root / "derived"
            _write_jsonl(
                raw,
                [
                    _accepted_trajectory(1),
                    _accepted_trajectory(2),
                    _accepted_trajectory(3),
                    rejected,
                ],
            )

            summary = build_collection_artifacts(
                raw_path=raw,
                output_dir=output,
                held_out_task_ids={3},
                validation_ratio=0.5,
                seed=42,
                collection_config={"model": "teacher-test"},
            )

            train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            validation = [
                json.loads(line)
                for line in (output / "validation.jsonl").read_text().splitlines()
            ]
            rejected_rows = [
                json.loads(line)
                for line in (output / "rejected.jsonl").read_text().splitlines()
            ]
            metadata_exists = (output / "metadata.json").exists()
            metadata = json.loads((output / "metadata.json").read_text())

        train_ids = {row["task_id"] for row in train}
        validation_ids = {row["task_id"] for row in validation}
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(train_ids | validation_ids, {1, 2})
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["held_out_excluded"], 1)
        self.assertEqual(summary["rejected"], 1)
        self.assertIn("held_out_task", rejected_rows[0]["reject_reasons"])
        self.assertTrue(metadata_exists)
        self.assertEqual(metadata["collection_config"]["model"], "teacher-test")


if __name__ == "__main__":
    unittest.main()
