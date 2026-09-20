"""Deterministic context compaction must preserve complete recent tool interactions."""

import unittest

from shopping_grpo.environment.context import (
    ContextBudgetError,
    FallbackTokenCounter,
    HeuristicChatTokenCounter,
    HeuristicTextTokenCounter,
    VllmChatTokenCounter,
    compact_chat_messages,
    compact_token_trajectory,
)


def assistant(call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "search_products", "arguments": "{}"},
            }
        ],
    }


def tool(call_id, content):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "search_products",
        "content": content,
    }


class ChatContextWindowTest(unittest.TestCase):
    def test_keeps_fixed_prompt_and_largest_recent_complete_group_suffix(self):
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "task"},
            assistant("old"),
            tool("old", "old page"),
            assistant("middle"),
            tool("middle", "middle page"),
            assistant("latest"),
            tool("latest", "latest page"),
        ]

        compacted, stats = compact_chat_messages(
            messages,
            tools=[],
            count_tokens=lambda candidate, tools: len(candidate),
            max_input_tokens=6,
        )

        self.assertEqual([message["role"] for message in compacted], ["system", "user", "assistant", "tool", "assistant", "tool"])
        self.assertNotIn("old page", str(compacted))
        self.assertIn("middle page", str(compacted))
        self.assertIn("latest page", str(compacted))
        self.assertEqual(stats.removed_groups, 1)
        self.assertEqual(stats.removed_messages, 2)
        self.assertEqual(messages[3]["content"], "old page")

    def test_latest_tool_observation_is_never_dropped(self):
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "task"},
            assistant("latest"),
            tool("latest", "latest page"),
        ]
        with self.assertRaisesRegex(ContextBudgetError, "latest tool observation"):
            compact_chat_messages(
                messages,
                tools=[],
                count_tokens=lambda candidate, tools: len(candidate),
                max_input_tokens=3,
            )

    def test_rejects_broken_tool_call_pair(self):
        messages = [
            {"role": "user", "content": "task"},
            assistant("call-1"),
            tool("other-call", "page"),
        ]
        with self.assertRaisesRegex(ContextBudgetError, "does not match"):
            compact_chat_messages(
                messages,
                tools=[],
                count_tokens=lambda candidate, tools: 100,
                max_input_tokens=10,
            )

    def test_vllm_counter_uses_server_root_tokenize_endpoint(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"url": url, "payload": payload, "headers": headers})
            return {"count": 17}

        counter = VllmChatTokenCounter(
            model="shopping",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            transport=transport,
        )
        count = counter([{"role": "user", "content": "task"}], [{"type": "function"}])

        self.assertEqual(count, 17)
        self.assertEqual(captured["url"], "http://127.0.0.1:8000/tokenize")
        self.assertEqual(captured["payload"]["model"], "shopping")
        self.assertTrue(captured["payload"]["add_generation_prompt"])

    def test_vllm_counter_normalizes_full_chat_completions_endpoint(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured["url"] = url
            return {"count": 3}

        counter = VllmChatTokenCounter(
            model="shopping",
            base_url="http://127.0.0.1:3010/v1/chat/completions",
            api_key="EMPTY",
            transport=transport,
        )

        self.assertEqual(counter([], []), 3)
        self.assertEqual(captured["url"], "http://127.0.0.1:3010/tokenize")

    def test_chat_only_provider_can_fall_back_without_tokenize_endpoint(self):
        calls = []

        def unavailable(*args, **kwargs):
            calls.append(1)
            raise ValueError("404 /tokenize")

        counter = FallbackTokenCounter(unavailable, HeuristicChatTokenCounter())
        first = counter([{"role": "user", "content": "任务"}], [])
        second = counter([{"role": "user", "content": "任务"}], [])

        self.assertGreater(first, 0)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertTrue(counter.fallback_active)

    def test_text_fallback_counts_unicode_without_provider_tokenizer(self):
        counter = HeuristicTextTokenCounter()
        self.assertEqual(counter("白色手机"), 4)


class TokenTrajectoryWindowTest(unittest.TestCase):
    def test_drops_old_complete_groups_and_aligns_rollout_logprobs(self):
        initial = [10, 11, 12]
        response = [20, 21, 22, 23, 30, 31, 32, 40, 41]
        mask = [1, 1, 0, 0, 1, 0, 0, 1, 0]
        logprobs = [float(index) for index in range(len(mask))]

        prompt, compacted_mask, compacted_logprobs, stats = compact_token_trajectory(
            initial + response,
            mask,
            logprobs,
            max_input_tokens=9,
        )

        self.assertEqual(prompt, initial + response[4:])
        self.assertEqual(compacted_mask, mask[4:])
        self.assertEqual(compacted_logprobs, logprobs[4:])
        self.assertEqual(stats.removed_tokens, 4)
        self.assertEqual(stats.removed_groups, 1)

    def test_never_drops_the_latest_complete_group(self):
        with self.assertRaisesRegex(ContextBudgetError, "protected recent tool group"):
            compact_token_trajectory(
                prompt_ids=[10, 11, 20, 21, 22],
                response_mask=[1, 0, 0],
                response_logprobs=[-1.0, 0.0, 0.0],
                max_input_tokens=4,
            )

    def test_requires_aligned_rollout_logprobs(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            compact_token_trajectory(
                prompt_ids=[10, 20, 21],
                response_mask=[1, 0],
                response_logprobs=[-1.0],
                max_input_tokens=2,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
