"""进程内 transformers 推理客户端，替代评测链路里的 OpenAI/vLLM HTTP 后端。

本地 Windows 主机无法安装 vLLM/verl（无 Windows wheel）。训练在云端 GPU 完成后，
导出 merged 权重，即可用本客户端在 CPU 上对 ShopSimulator 跑评测，复用
``collect_for_task`` / ``collect_tasks`` / ``summarize_trajectories`` 整套循环，
仅把“模型推理后端”从 HTTP 换成 ``transformers`` + ``model.generate``。

本类镜像 ``OpenAIChatClient``（``rollout.py``）的四个公开成员——
``complete``、``project_observation``、``last_context_tokens``、``last_context_event``，
因此 ``collect_for_task`` 无需任何改动即可直接接入。
"""

from __future__ import annotations

import json
import re
from uuid import uuid4

from shopping_grpo.environment.context import ContextBudgetError, compact_chat_messages
from shopping_grpo.environment.projection import project_observation
from shopping_grpo.training.sft.dataset import normalize_messages_for_chat_template


__all__ = ["LocalTransformersClient"]


# Qwen3 思维链块：模型在 thinking 模式下把推理放进这对 marker 之间。
# skip_special_tokens=True 有时仍保留其文本形式，因此用文本扫描兜底。
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _load_inference_components(
    model_name,
    torch,
    auto_config,
    auto_tokenizer,
    auto_processor,
    auto_model_causal,
    auto_model_multimodal,
    *,
    revision=None,
    torch_dtype=None,
    attention_implementation=None,
):
    """按模型配置选择 chat template 持有者与模型类，照搬 train_lora_sft 范式。"""

    load_kwargs = {"trust_remote_code": True}
    if revision:
        load_kwargs["revision"] = revision
    config = auto_config.from_pretrained(model_name, **load_kwargs)
    is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    model_load_kwargs = {"trust_remote_code": True}
    if torch_dtype is not None:
        model_load_kwargs["torch_dtype"] = torch_dtype
    if revision:
        model_load_kwargs["revision"] = revision
    if attention_implementation and attention_implementation != "auto":
        model_load_kwargs["attn_implementation"] = attention_implementation
    if is_multimodal:
        processor = auto_processor.from_pretrained(model_name, **load_kwargs)
        model = auto_model_multimodal.from_pretrained(model_name, **model_load_kwargs)
        return processor.tokenizer, processor, model, True
    tokenizer = auto_tokenizer.from_pretrained(model_name, **load_kwargs)
    model = auto_model_causal.from_pretrained(model_name, **model_load_kwargs)
    return tokenizer, tokenizer, model, False


def _resolve_dtype(name, torch):
    """auto 在 CPU 上落到 fp32，与 train_lora_sft._resolve_dtype 一致。"""

    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unknown dtype: {name!r}")
    return mapping[name]


class TokenizerChatTokenCounter:
    """本地替换 VllmChatTokenCounter：用 tokenizer 渲染并计数 chat token。"""

    def __init__(self, tokenizer, chat_template):
        self._tokenizer = tokenizer
        self._chat_template = chat_template

    def __call__(self, messages, tools):
        normalized = normalize_messages_for_chat_template(messages)
        if normalized is None:
            normalized = messages
        encoded = self._chat_template.apply_chat_template(
            normalized,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded, dict):
            input_ids = encoded.get("input_ids", [])
        else:
            input_ids = encoded
        return len(input_ids)


class TokenizerTextTokenCounter:
    """本地替换 VllmTextTokenCounter：用 tokenizer 计数纯文本 token。"""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, text):
        return len(self._tokenizer.encode(str(text), add_special_tokens=False))


def _strip_think_blocks(text):
    """移除 Qwen3 思维链块；marker 若被 tokenizer 保留为文本也能兜底。"""

    return _THINK_BLOCK.sub("", text)


def _scan_tool_call_objects(text):
    """扫描文本中的顶层 JSON 对象，返回 (tool_calls, 残留文本)。"""

    tool_calls = []
    spans = []  # 需要从残留文本里剔除的区间
    index = 0
    length = len(text)
    while index < length:
        brace = text.find("{", index)
        if brace == -1:
            break
        end = _find_matching_brace(text, brace)
        if end is None:
            break
        candidate = text[brace : end + 1]
        call = _parse_tool_call_object(candidate)
        if call is not None:
            tool_calls.append(call)
            spans.append((brace, end + 1))
        index = end + 1
    residual = _strip_spans(text, spans)
    return tool_calls, residual


def _find_matching_brace(text, start):
    """返回与 start 处左括号配对的右括号位置；不配对返回 None。"""

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _parse_tool_call_object(candidate):
    """把一个 JSON 对象解析成 OpenAI tool_call；不合法返回 None。"""

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    arguments = obj.get("arguments")
    if not isinstance(name, str):
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None
    # id 由 _parse_assistant_message 统一按 (step, offset) 分配，这里不生成。
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _strip_spans(text, spans):
    """删除 spans 列出的区间，返回拼接后的剩余文本。"""

    if not spans:
        return text
    pieces = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _parse_assistant_message(text, step_index):
    """把生成文本解析成 OpenAI assistant 消息，与 vLLM 版语义一致。"""

    cleaned = _strip_think_blocks(text)
    tool_calls, residual = _scan_tool_call_objects(cleaned)
    content = residual.strip() or None
    if tool_calls:
        for call in tool_calls:
            call["id"] = "call_{:02d}_{}".format(step_index, uuid4().hex[:24])
        message = {"role": "assistant", "content": content}
        message["tool_calls"] = tool_calls
        return message
    # 没有工具调用：把原始生成文本作为最终回复交给 collect_for_task 的
    # assistant_final 分支，与 HTTP 版“模型未调工具”语义一致。
    return {"role": "assistant", "content": text.strip() or None}


class LocalTransformersClient:
    """进程内 transformers 推理客户端，镜像 OpenAIChatClient 的公开接口。"""

    def __init__(
        self,
        model,
        *,
        revision=None,
        dtype="auto",
        attention_implementation="sdpa",
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        context_window=24576,
        context_safety_margin=512,
        context_compaction_enable=False,
        observation_token_budget=1536,
        observation_detail_token_budget=4096,
        observation_generic_token_budget=768,
        observation_search_top_k=20,
        device=None,
    ):
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForMultimodalLM,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - 依赖检查由脚本前置完成
            raise ImportError(
                "缺少推理依赖。请执行：uv sync --extra sft"
            ) from exc

        self.model_name = model
        self.revision = revision
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.context_window = int(context_window) if context_window else None
        self.context_safety_margin = int(context_safety_margin)
        self.context_compaction_enable = bool(context_compaction_enable)
        self.observation_token_budget = (
            int(observation_token_budget) if observation_token_budget else None
        )
        self.observation_detail_token_budget = int(observation_detail_token_budget)
        self.observation_generic_token_budget = int(observation_generic_token_budget)
        self.observation_search_top_k = int(observation_search_top_k)
        self._device = device

        if self.context_window is not None:
            if self.context_window <= self.max_tokens + self.context_safety_margin:
                raise ValueError(
                    "context_window must exceed max_tokens plus context_safety_margin"
                )
        if self.observation_token_budget is not None:
            if self.observation_token_budget < 64:
                raise ValueError("observation_token_budget must be at least 64")

        resolved_dtype = dtype
        if resolved_dtype == "auto":
            resolved_dtype = "bf16" if torch.cuda.is_available() else "fp32"
        torch_dtype = _resolve_dtype(resolved_dtype, torch)
        self._dtype_name = resolved_dtype

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device_str = resolved_device

        self._torch = torch
        self._tokenizer, self._chat_template, self._model, self._is_multimodal = (
            _load_inference_components(
                model,
                torch,
                AutoConfig,
                AutoTokenizer,
                AutoProcessor,
                AutoModelForCausalLM,
                AutoModelForMultimodalLM,
                revision=revision,
                torch_dtype=torch_dtype,
                attention_implementation=attention_implementation,
            )
        )
        self._model = self._model.to(self._device_str)
        self._model.eval()

        if self.context_window is not None:
            self.token_counter = TokenizerChatTokenCounter(
                self._tokenizer, self._chat_template
            )
        else:
            self.token_counter = None
        if self.observation_token_budget is not None:
            self.observation_token_counter = TokenizerTextTokenCounter(
                self._tokenizer
            )
        else:
            self.observation_token_counter = None

        self.last_context_event = None
        self.last_context_tokens = None
        self._step_counter = -1

    def complete(self, messages, tools):
        """请求模型下一轮回复，并在上下文超限时按配置压缩历史。"""

        self.last_context_event = None
        self.last_context_tokens = None
        request_messages = messages
        if self.context_window is not None:
            input_budget = (
                self.context_window - self.max_tokens - self.context_safety_margin
            )
            original_tokens = int(self.token_counter(messages, tools))
            self.last_context_tokens = original_tokens
            if original_tokens > input_budget:
                if not self.context_compaction_enable:
                    raise ContextBudgetError(
                        f"prompt uses {original_tokens} tokens, "
                        f"above input budget {input_budget}"
                    )
                request_messages, stats = compact_chat_messages(
                    messages,
                    tools,
                    count_tokens=self.token_counter,
                    max_input_tokens=input_budget,
                )
                if stats.removed_groups:
                    self.last_context_event = stats.to_dict()

        normalized = normalize_messages_for_chat_template(request_messages)
        if normalized is None:
            normalized = request_messages
        prompt_text = self._chat_template.apply_chat_template(
            normalized,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        torch = self._torch
        device = self._device_str
        enc = self._tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id
        do_sample = self.temperature > 0
        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else 1.0,
                top_p=self.top_p if do_sample else 1.0,
                pad_token_id=pad_token_id,
            )
        new_tokens = output[0][input_ids.shape[-1] :]
        generated = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        step_index = self._next_step_index()
        return _parse_assistant_message(generated, step_index)

    def project_observation(self, tool_name, observation, parameters=None):
        if self.observation_token_budget is None:
            return str(observation), None
        visible, meta = project_observation(
            tool_name=tool_name,
            observation=observation,
            parameters=parameters,
            count_tokens=self.observation_token_counter,
            token_budget=self.observation_token_budget,
            detail_token_budget=self.observation_detail_token_budget,
            generic_token_budget=self.observation_generic_token_budget,
            search_top_k=self.observation_search_top_k,
        )
        return visible, meta.to_dict()

    def _next_step_index(self):
        """为 tool_call id 提供稳定序号；collect_for_task 每轮调用一次。"""

        self._step_counter += 1
        return self._step_counter
