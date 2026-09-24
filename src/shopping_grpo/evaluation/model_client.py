"""OpenAI-compatible JSON client for frozen Rubric and Judge prompts."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import RemoteDisconnected
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_FLASH_MODEL = "deepseek-v4-flash"
DEFAULT_PRO_MODEL = "deepseek-v4-pro"
RETRYABLE_HTTP_STATUSES = frozenset(
    {408, 409, 429, 500, 502, 503, 504}
)

# 教师网关（OpenCode Go 风格）对下列模型族默认开启思考：省略 thinking 字段并不等于
# 关闭。思考会先吃满 max_tokens，让 message.content 变成空串，调用方只看到
# "content must be non-empty JSON text"。本地 vLLM 不认识这个字段，所以只在已知
# 模型族上显式下发。（与 rollout.py 对 DeepSeek V4 的处理保持一致。）
THINKING_DEFAULT_ENABLED_PREFIXES = ("deepseek-v4", "glm-")
_CODE_FENCE_PATTERN = re.compile(
    r"^```[A-Za-z0-9_+-]*[ \t]*\r?\n?(?P<body>.*?)\r?\n?```$",
    re.DOTALL,
)


class ModelResponseError(ValueError):
    """Raised when a provider response cannot satisfy the JSON contract."""


def _content_text(content: Any) -> str:
    """Flatten plain-text or OpenAI-style part-list content into a string."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _strip_code_fence(text: str) -> str:
    """Drop one surrounding Markdown code fence, if the provider added it."""

    stripped = text.strip()
    match = _CODE_FENCE_PATTERN.match(stripped)
    return match.group("body").strip() if match else stripped


def _empty_content_error(
    choice: Any,
    message: Any,
    content: Any,
) -> ModelResponseError:
    """Explain *why* content was unusable instead of failing opaquely."""

    finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    reasoning = message.get("reasoning_content") if isinstance(message, Mapping) else None
    reasoning_chars = len(reasoning) if isinstance(reasoning, str) else None
    details = (
        f"finish_reason={finish_reason!r}, "
        f"content_type={type(content).__name__}, "
        f"reasoning_content_chars={reasoning_chars}"
    )
    if reasoning_chars:
        hint = (
            "provider returned reasoning only: this gateway defaults thinking to "
            "enabled, so the thinking budget consumed max_tokens. Send an explicit "
            "thinking=disabled hint or raise max_tokens"
        )
    elif finish_reason == "length":
        hint = "output hit max_tokens; raise max_tokens"
    else:
        hint = "provider returned no usable text; check response_format support and max_tokens"
    return ModelResponseError(
        "provider response message.content must be non-empty JSON text "
        f"({details}); {hint}"
    )


def _retry_after_seconds(
    error: HTTPError,
    *,
    now: datetime | None = None,
) -> float | None:
    value = error.headers.get("Retry-After") if error.headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0.0, (retry_at - current).total_seconds())


class OpenAIJSONClient:
    """Call one chat-completions model without serializing credentials."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int = 4096,
        timeout: float = 120,
        retries: int = 2,
        retry_delay_seconds: float = 2,
        response_format_json: bool = False,
        thinking: bool = False,
        reasoning_effort: str = "high",
        transport: Callable | None = None,
    ):
        if not str(model).strip():
            raise ValueError("model is required")
        if not str(base_url).strip():
            raise ValueError("base_url is required")
        if not str(api_key):
            raise ValueError("api_key is required")
        if int(max_tokens) < 1:
            raise ValueError("max_tokens must be positive")
        if int(retries) < 0:
            raise ValueError("retries cannot be negative")
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.response_format_json = bool(response_format_json)
        self.thinking = bool(thinking)
        self.reasoning_effort = str(reasoning_effort)
        self.transport = transport

    def _supports_thinking_hint(self) -> bool:
        """Whether this model family accepts an explicit thinking switch."""

        return self.model.casefold().startswith(
            THINKING_DEFAULT_ENABLED_PREFIXES
        )

    def _request_payload(
        self,
        messages: list[Mapping],
        *,
        thinking_hint: bool = True,
        response_format: bool | None = None,
    ) -> tuple[dict, bool]:
        """Build one request body and report whether a thinking hint was sent."""

        json_mode = (
            self.response_format_json
            if response_format is None
            else bool(response_format)
        )
        payload = {
            "model": self.model,
            "messages": deepcopy(messages),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
        }
        hint_sent = False
        if thinking_hint and self._supports_thinking_hint():
            hint_sent = True
            if self.thinking:
                payload["thinking"] = {"type": "enabled"}
                payload["reasoning_effort"] = self.reasoning_effort
                payload.pop("temperature", None)
                payload.pop("top_p", None)
            else:
                payload["thinking"] = {"type": "disabled"}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload, hint_sent

    def _post(
        self,
        url: str,
        payload: dict,
        headers: Mapping,
    ) -> tuple[Any, int, list[int], float]:
        """Send one payload, retrying only transient transport failures."""

        attempts = 0
        retry_http_statuses: list[int] = []
        retry_wait_seconds = 0.0
        for attempt in range(self.retries + 1):
            attempts = attempt + 1
            try:
                if self.transport is not None:
                    response = self.transport(
                        url,
                        payload,
                        headers,
                        self.timeout,
                    )
                else:
                    request = Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urlopen(request, timeout=self.timeout) as raw:
                        response = json.loads(raw.read().decode("utf-8"))
                return response, attempts, retry_http_statuses, retry_wait_seconds
            except HTTPError as exc:
                status = int(exc.code)
                if (
                    status not in RETRYABLE_HTTP_STATUSES
                    or attempt >= self.retries
                ):
                    raise
                retry_http_statuses.append(status)
                exponential = self.retry_delay_seconds * (2**attempt)
                retry_after = _retry_after_seconds(exc)
                delay = max(exponential, retry_after or 0.0)
                retry_wait_seconds += delay
                if delay > 0:
                    time.sleep(delay)
            except (RemoteDisconnected, TimeoutError, URLError):
                if attempt >= self.retries:
                    raise
                delay = self.retry_delay_seconds * (2**attempt)
                retry_wait_seconds += delay
                if delay > 0:
                    time.sleep(delay)
        raise ModelResponseError("provider request produced no response")

    def complete_json(self, messages: list[Mapping]) -> dict:
        """Return parsed JSON plus non-secret request metadata."""

        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "shopping-grpo-longhorizon-evaluator/0.1",
        }
        url = f"{self.base_url}/chat/completions"
        started = time.monotonic()
        attempts = 0
        retry_http_statuses: list[int] = []
        retry_wait_seconds = 0.0
        thinking_hint = True
        thinking_hint_dropped = False
        json_mode = self.response_format_json
        response_format_degraded = False
        while True:
            payload, hint_sent = self._request_payload(
                messages,
                thinking_hint=thinking_hint,
                response_format=json_mode,
            )
            try:
                response, used, statuses, waited = self._post(
                    url, payload, headers
                )
            except HTTPError as exc:
                # 网关不认显式 thinking 或 json_object 参数时逐项降级重试，
                # 而不是把参数不兼容升级成构建失败。两个开关各只翻转一次。
                if int(exc.code) == 400:
                    if hint_sent:
                        thinking_hint = False
                        thinking_hint_dropped = True
                        continue
                    if json_mode:
                        json_mode = False
                        response_format_degraded = True
                        continue
                raise
            attempts += used
            retry_http_statuses.extend(statuses)
            retry_wait_seconds += waited
            if not isinstance(response, Mapping):
                raise ModelResponseError("provider response must be an object")
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ModelResponseError("provider response is missing choices")
            first = choices[0]
            message = (
                first.get("message") if isinstance(first, Mapping) else None
            )
            content = (
                message.get("content") if isinstance(message, Mapping) else None
            )
            text = _strip_code_fence(_content_text(content))
            if not text and json_mode:
                # 个别兼容网关在 json_object 约束下会回空输出：去掉约束再试一次。
                json_mode = False
                response_format_degraded = True
                continue
            if not text:
                raise _empty_content_error(first, message, content)
            break
        latency = time.monotonic() - started
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            finish_reason = (
                first.get("finish_reason")
                if isinstance(first, Mapping)
                else None
            )
            raise ModelResponseError(
                "provider response content is not strict JSON "
                f"(finish_reason={finish_reason!r}, chars={len(text)})"
            ) from exc
        if not isinstance(result, dict):
            raise ModelResponseError(
                "provider response JSON root must be an object"
            )
        usage = response.get("usage")
        usage = deepcopy(dict(usage)) if isinstance(usage, Mapping) else {}
        return {
            "result": result,
            "metadata": {
                "provider_request_id": response.get("id"),
                "provider_model": response.get("model") or self.model,
                "requested_model": self.model,
                "requested_thinking": self.thinking,
                "requested_reasoning_effort": (
                    self.reasoning_effort if self.thinking else None
                ),
                "requested_response_format_json": self.response_format_json,
                "response_format_degraded": response_format_degraded,
                "thinking_hint_dropped": thinking_hint_dropped,
                "finish_reason": (
                    first.get("finish_reason")
                    if isinstance(first, Mapping)
                    else None
                ),
                "attempts": attempts,
                "retry_http_statuses": retry_http_statuses,
                "retry_wait_seconds": retry_wait_seconds,
                "latency_seconds": latency,
                "usage": usage,
            },
        }


def client_from_environment(
    *,
    model: str,
    max_tokens: int,
    timeout: float = 120,
    retries: int = 2,
    response_format_json: bool = False,
    thinking: bool = False,
    reasoning_effort: str = "high",
) -> OpenAIJSONClient:
    """Use the same environment-variable convention as Teacher collection."""

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is required")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    return OpenAIJSONClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        response_format_json=response_format_json,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
