"""验证本地 transformers 客户端的工具调用解析与 token 计数。

真正的模型加载需要 merged 权重与 torch，留给端到端冒烟；这里只覆盖纯 Python
解析逻辑与基于最小 tokenizer mock 的计数器，保证它们在 CPU-only 环境也可运行。
"""

import json
import unittest

from shopping_grpo.evaluation.local_client import (
    TokenizerChatTokenCounter,
    TokenizerTextTokenCounter,
    _find_matching_brace,
    _parse_assistant_message,
    _parse_tool_call_object,
    _scan_tool_call_objects,
    _strip_spans,
    _strip_think_blocks,
)


THINK_OPEN = chr(0x3C) + "think" + chr(0x3E)
THINK_CLOSE = chr(0x3C) + "/think" + chr(0x3E)


class CharacterTokenizer:
    """无需 transformers 的最小 chat-template tokenizer，用于验证计数器。"""

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        del tools
        text = ""
        for message in messages:
            text += f"<{message['role']}>"
            text += message.get("content") or ""
            text += f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return {"input_ids": [ord(ch) for ch in text]}
        return text

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]


class ToolCallParseTest(unittest.TestCase):
    def test_bare_json_object_becomes_openai_tool_call(self):
        text = '先搜索。\n{"name": "search_products", "arguments": {"query": "水彩"}}\n'
        message = _parse_assistant_message(text, 0)

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "先搜索。")
        calls = message["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "search_products")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"query": "水彩"}
        )
        self.assertRegex(calls[0]["id"], r"^call_00_[0-9a-f]{24}$")

    def test_arguments_as_json_string_is_normalized_to_object(self):
        candidate = '{"name": "open_product", "arguments": "{\\"asin\\": \\"123\\"}"}'
        call = _parse_tool_call_object(candidate)

        self.assertIsNotNone(call)
        self.assertEqual(call["function"]["name"], "open_product")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"asin": "123"})

    def test_brace_inside_string_does_not_break_matching(self):
        text = 'noise {"name": "select_option", "arguments": {"value": "a}b"}} trailing'
        calls, residual = _scan_tool_call_objects(text)

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"value": "a}b"}
        )
        self.assertIn("noise", residual)
        self.assertIn("trailing", residual)

    def test_multiple_top_level_objects_are_all_extracted(self):
        text = '{"name":"a","arguments":{}} then {"name":"b","arguments":{"x":1}}'
        calls, _ = _scan_tool_call_objects(text)

        self.assertEqual(
            [call["function"]["name"] for call in calls], ["a", "b"]
        )

    def test_unbalanced_brace_yields_no_call(self):
        text = '{"name": "search_products", "arguments": {"query": "broken"'
        calls, _ = _scan_tool_call_objects(text)
        self.assertEqual(calls, [])

    def test_non_tool_json_object_is_ignored(self):
        text = '{"summary": "not a tool call", "count": 3}'
        message = _parse_assistant_message(text, 0)
        self.assertNotIn("tool_calls", message)

    def test_think_block_is_stripped_before_scanning(self):
        text = (
            THINK_OPEN
            + "reasoning about watercolor"
            + THINK_CLOSE
            + ' {"name": "search_products", "arguments": {"query": "watercolor"}}'
        )
        message = _parse_assistant_message(text, 3)

        calls = message.get("tool_calls") or []
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "search_products")
        self.assertNotIn("reasoning", message["content"] or "")

    def test_no_tool_call_returns_assistant_final_message(self):
        text = "这是最终答复，没有工具调用。"
        message = _parse_assistant_message(text, 2)

        self.assertNotIn("tool_calls", message)
        self.assertEqual(message["content"], text)

    def test_empty_content_becomes_none(self):
        text = '{"name": "buy_now", "arguments": {}}'
        message = _parse_assistant_message(text, 1)

        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "buy_now")

    def test_step_index_distinguishes_call_ids(self):
        text = '{"name": "search_products", "arguments": {}}'
        first = _parse_assistant_message(text, 0)["tool_calls"][0]["id"]
        second = _parse_assistant_message(text, 1)["tool_calls"][0]["id"]

        self.assertTrue(first.startswith("call_00_"))
        self.assertTrue(second.startswith("call_01_"))
        self.assertNotEqual(first, second)


class TokenCounterTest(unittest.TestCase):
    def test_chat_counter_counts_rendered_input_ids(self):
        tokenizer = CharacterTokenizer()
        counter = TokenizerChatTokenCounter(tokenizer, tokenizer)
        messages = [
            {"role": "system", "content": "rule"},
            {"role": "user", "content": "buy"},
        ]

        count = counter(messages, tools=[])

        self.assertEqual(count, len("<system>rule</system><user>buy</user><assistant>"))

    def test_chat_counter_normalizes_openai_arguments_string(self):
        tokenizer = CharacterTokenizer()
        counter = TokenizerChatTokenCounter(tokenizer, tokenizer)
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query": "pillow"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "search_products", "content": "result"},
        ]

        count = counter(messages, tools=[])

        self.assertGreater(count, 0)
        # Qwen 模板期望 arguments 已是 dict；normalize 后字符序列应包含
        # query/pillow 文本而非原始 JSON 字符串。
        self.assertGreater(count, len("pillow"))

    def test_text_counter_counts_plain_tokens(self):
        tokenizer = CharacterTokenizer()
        counter = TokenizerTextTokenCounter(tokenizer)

        # 计数器返回的是 token 个数，而非 ord 之和。
        self.assertEqual(counter("ab"), 2)
        self.assertEqual(counter(""), 0)

    def test_matching_brace_returns_position_or_none(self):
        self.assertEqual(_find_matching_brace("a{b}c", 1), 3)
        self.assertIsNone(_find_matching_brace("a{b", 1))
        self.assertIsNone(_find_matching_brace("abc", 0))


class MatchingBraceTest(unittest.TestCase):
    def test_matching_brace_returns_position_or_none(self):
        self.assertEqual(_find_matching_brace("a{b}c", 1), 3)
        self.assertIsNone(_find_matching_brace("a{b", 1))
        self.assertIsNone(_find_matching_brace("abc", 0))


class StripSpansTest(unittest.TestCase):
    def test_strip_spans_removes_listed_intervals(self):
        self.assertEqual(_strip_spans("abcdef", [(1, 3), (4, 5)]), "adf")

    def test_strip_spans_passthrough_when_empty(self):
        self.assertEqual(_strip_spans("abcdef", []), "abcdef")


class StripThinkTest(unittest.TestCase):
    def test_think_block_removed(self):
        text = THINK_OPEN + "inner" + THINK_CLOSE + "after"
        self.assertEqual(_strip_think_blocks(text), "after")

    def test_think_block_dotall_across_newlines(self):
        text = THINK_OPEN + "line1\nline2" + THINK_CLOSE + "after"
        self.assertEqual(_strip_think_blocks(text), "after")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
