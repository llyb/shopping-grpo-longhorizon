from __future__ import annotations

import unittest
from urllib.error import HTTPError

from shopping_grpo.evaluation.model_client import (
    ModelResponseError,
    OpenAIJSONClient,
)


def _client(model="glm-5.2", transport=None, **overrides):
    options = {
        "model": model,
        "base_url": "http://127.0.0.1:3010/v1",
        "api_key": "secret",
        "max_tokens": 2048,
        "transport": transport,
    }
    options.update(overrides)
    return OpenAIJSONClient(**options)


def _reply(content, *, finish_reason="stop", reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-1",
        "model": "glm-5.2",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"total_tokens": 12},
    }


class ThinkingHintTests(unittest.TestCase):
    def test_glm_disables_thinking_explicitly(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured["payload"] = payload
            return _reply('{"ok": true}')

        _client(transport=transport, response_format_json=True).complete_json(
            [{"role": "user", "content": "hi"}]
        )

        # 网关对 glm 系列默认开启思考：必须显式下发 disabled。
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(
            captured["payload"]["response_format"], {"type": "json_object"}
        )
        self.assertIn("temperature", captured["payload"])

    def test_glm_thinking_enabled_keeps_reasoning_budget_contract(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured["payload"] = payload
            return _reply('{"ok": true}')

        _client(
            transport=transport,
            thinking=True,
            reasoning_effort="high",
        ).complete_json([{"role": "user", "content": "hi"}])

        self.assertEqual(captured["payload"]["thinking"], {"type": "enabled"})
        self.assertEqual(captured["payload"]["reasoning_effort"], "high")
        self.assertNotIn("temperature", captured["payload"])
        self.assertNotIn("top_p", captured["payload"])

    def test_other_model_families_receive_no_provider_specific_field(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured["payload"] = payload
            return _reply('{"ok": true}')

        _client(model="qwen3.5-27b", transport=transport).complete_json(
            [{"role": "user", "content": "hi"}]
        )

        self.assertNotIn("thinking", captured["payload"])
        self.assertIn("temperature", captured["payload"])

    def test_rejected_thinking_hint_is_dropped_instead_of_failing(self):
        seen = []

        def transport(url, payload, headers, timeout):
            seen.append(payload)
            if "thinking" in payload:
                raise HTTPError(url, 400, "unsupported parameter", None, None)
            return _reply('{"ok": true}')

        result = _client(transport=transport).complete_json(
            [{"role": "user", "content": "hi"}]
        )

        self.assertEqual(len(seen), 2)
        self.assertEqual(result["result"], {"ok": True})
        self.assertTrue(result["metadata"]["thinking_hint_dropped"])

    def test_rejected_json_mode_is_dropped_after_the_hint(self):
        seen = []

        def transport(url, payload, headers, timeout):
            seen.append(payload)
            if "thinking" in payload or "response_format" in payload:
                raise HTTPError(url, 400, "unsupported parameter", None, None)
            return _reply('{"ok": true}')

        result = _client(
            transport=transport,
            response_format_json=True,
        ).complete_json([{"role": "user", "content": "hi"}])

        self.assertEqual(len(seen), 3)
        self.assertNotIn("thinking", seen[-1])
        self.assertNotIn("response_format", seen[-1])
        self.assertEqual(result["result"], {"ok": True})
        self.assertTrue(result["metadata"]["thinking_hint_dropped"])
        self.assertTrue(result["metadata"]["response_format_degraded"])


class ResponseShapeTests(unittest.TestCase):
    def test_empty_content_with_reasoning_names_the_cause(self):
        def transport(url, payload, headers, timeout):
            return _reply(
                None,
                finish_reason="length",
                reasoning="先核对规格，再搜索。",
            )

        with self.assertRaises(ModelResponseError) as caught:
            _client(transport=transport).complete_json(
                [{"role": "user", "content": "hi"}]
            )

        message = str(caught.exception)
        self.assertIn("non-empty JSON text", message)
        self.assertIn("finish_reason='length'", message)
        self.assertIn("reasoning", message)

    def test_json_mode_degradation_recovers_from_empty_output(self):
        seen = []

        def transport(url, payload, headers, timeout):
            seen.append(payload)
            if "response_format" in payload:
                return _reply("")
            return _reply("```json\n{\"ok\": true}\n```")

        result = _client(
            transport=transport,
            response_format_json=True,
        ).complete_json([{"role": "user", "content": "hi"}])

        self.assertEqual(result["result"], {"ok": True})
        self.assertIn("response_format", seen[0])
        self.assertNotIn("response_format", seen[1])
        self.assertTrue(result["metadata"]["response_format_degraded"])

    def test_part_list_content_is_flattened(self):
        def transport(url, payload, headers, timeout):
            return _reply([{"type": "text", "text": '{"ok":'}, {"type": "text", "text": " true}"}])

        result = _client(transport=transport).complete_json(
            [{"role": "user", "content": "hi"}]
        )

        self.assertEqual(result["result"], {"ok": True})

    def test_truncated_json_reports_finish_reason(self):
        def transport(url, payload, headers, timeout):
            return _reply('{"ok": tr', finish_reason="length")

        with self.assertRaises(ModelResponseError) as caught:
            _client(transport=transport).complete_json(
                [{"role": "user", "content": "hi"}]
            )

        self.assertIn("not strict JSON", str(caught.exception))
        self.assertIn("finish_reason='length'", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
