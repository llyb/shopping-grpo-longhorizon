"""Deterministic Rubric candidates and code-owned Flash materialization."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy

from shopping_grpo.evaluation.contracts import (
    RUBRIC_SCHEMA_VERSION,
    ContractValidationError,
    validate_curator_response,
    validate_rubric_bundle,
)

TASK_FACTS_VERSION = "shopping-task-facts-v1"
RUBRIC_CANDIDATE_VERSION = "shopping-rubric-candidates-v1"
RUBRIC_EXTRACTOR_VERSION = "shopping-rubric-extractor-v3"

_NUMBER = r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?"
_UPPER_PREFIX = re.compile(
    rf"(?:预算|价格)?\s*"
    rf"(?:控制在?|要|得|需|需要)?\s*"
    rf"(?:不超过|不要超过|别超|不能超过|不得超过|不可超过|"
    rf"不高于|最高|至多|封顶(?:在)?)\s*"
    rf"{_NUMBER}\s*元?",
)
_UPPER_SUFFIX = re.compile(rf"{_NUMBER}\s*元?\s*(?:以内|以下|之内)")
_LOWER_PREFIX = re.compile(
    rf"(?:预算|价格)?\s*"
    rf"(?:不低于|至少|最低|不少于|高于|超过)\s*"
    rf"{_NUMBER}\s*元?",
)
_LOWER_SUFFIX = re.compile(rf"{_NUMBER}\s*元?\s*(?:以上|起)")
_PLUS_LOWER = re.compile(rf"{_NUMBER}\s*元?\s*\+")
_RANGE = re.compile(
    rf"(?:预算|价格)?(?:控制)?在?\s*{_NUMBER}\s*元?\s*"
    rf"(?:-|~|～|至|到)\s*{_NUMBER}\s*元?(?:之间)?"
)
_AROUND = re.compile(
    rf"(?:预算|价格)?(?:控制)?在?\s*{_NUMBER}\s*元?\s*(?:左右|上下)"
)
_WAN_SHORTHAND = r"(\d+(?:\.\d+)?)\s*万\s*(\d+(?:\.\d+)?)\s*(千)?"
_WAN_UPPER_PREFIX = re.compile(
    rf"(?:预算|价格)?\s*"
    rf"(?:控制在?|要|得|需|需要)?\s*"
    rf"(?:不超过|不要超过|别超|不能超过|不得超过|不可超过|"
    rf"不高于|最高|至多|封顶(?:在)?)\s*"
    rf"{_WAN_SHORTHAND}\s*元?"
)
_WAN_UPPER_SUFFIX = re.compile(
    rf"{_WAN_SHORTHAND}\s*元?\s*(?:以内|以下|之内)"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in result:
            result.append(text)
    return result


def _scaled(number: str, unit: str | None) -> float:
    value = float(number)
    normalized = str(unit or "").casefold()
    if normalized == "万":
        value *= 10000
    elif normalized in {"千", "k"}:
        value *= 1000
    return value


def _scaled_wan_shorthand(
    major: str,
    tail: str,
    tail_unit: str | None,
) -> float:
    """Parse colloquial forms such as ``1万2`` and ``1万2千`` as 12,000."""

    value = float(major) * 10000
    remainder = float(tail)
    if tail_unit == "千" or remainder < 10:
        remainder *= 1000
    return value + remainder


def _span(query: str, start: int, end: int) -> dict:
    return {"text": query[start:end], "start": start, "end": end}


def _exact_spans(query: str, value: str) -> list[dict]:
    if not value:
        return []
    return [
        _span(query, match.start(), match.end())
        for match in re.finditer(re.escape(value), query, flags=re.IGNORECASE)
    ]


def extract_price_candidates(query: str) -> list[dict]:
    """Extract explicit price semantics for Rubric candidates, not Reward."""

    query = str(query or "")
    candidates = []
    occupied = []

    def overlaps(match: re.Match) -> bool:
        return any(
            match.start() < previous_end and previous_start < match.end()
            for previous_start, previous_end in occupied
        )

    for pattern in (_WAN_UPPER_PREFIX, _WAN_UPPER_SUFFIX):
        for match in pattern.finditer(query):
            if overlaps(match):
                continue
            value = _scaled_wan_shorthand(
                match.group(1),
                match.group(2),
                match.group(3),
            )
            if value <= 0:
                continue
            occupied.append((match.start(), match.end()))
            candidates.append(
                {
                    "constraint_type": "budget_upper",
                    "description_hint": f"价格不应超过{value:g}元",
                    "field_path": "purchase.price",
                    "operator": "lte",
                    "expected_value": {
                        "value": value,
                        "currency": "CNY",
                    },
                    "hardness_hint": "hard",
                    "hardness_source": "explicit_upper_budget_rule",
                    "query_spans": [
                        _span(query, match.start(), match.end())
                    ],
                    "data_sources": ["query"],
                }
            )
    for match in _RANGE.finditer(query):
        low = _scaled(match.group(1), match.group(2))
        high = _scaled(match.group(3), match.group(4))
        if low <= 0 or high < low:
            continue
        occupied.append((match.start(), match.end()))
        candidates.append(
            {
                "constraint_type": "price_range",
                "description_hint": (
                    f"价格应在{low:g}元到{high:g}元之间"
                ),
                "field_path": "purchase.price",
                "operator": "between",
                "expected_value": {"min": low, "max": high, "currency": "CNY"},
                "hardness_hint": "hard",
                "hardness_source": "explicit_price_range_rule",
                "query_spans": [_span(query, match.start(), match.end())],
                "data_sources": ["query"],
            }
        )
    for pattern, operator, constraint_type, hardness, source in (
        (
            _UPPER_PREFIX,
            "lte",
            "budget_upper",
            "hard",
            "explicit_upper_budget_rule",
        ),
        (
            _UPPER_SUFFIX,
            "lte",
            "budget_upper",
            "hard",
            "explicit_upper_budget_rule",
        ),
        (
            _LOWER_PREFIX,
            "gte",
            "price_lower",
            "hard",
            "explicit_lower_price_rule",
        ),
        (
            _LOWER_SUFFIX,
            "gte",
            "price_lower",
            "hard",
            "explicit_lower_price_rule",
        ),
        (
            _PLUS_LOWER,
            "gte",
            "price_lower",
            "hard",
            "explicit_plus_lower_rule",
        ),
        (
            _AROUND,
            "approximately",
            "price_preference",
            "soft",
            "approximate_price_preference_rule",
        ),
    ):
        for match in pattern.finditer(query):
            if overlaps(match):
                continue
            occupied.append((match.start(), match.end()))
            value = _scaled(match.group(1), match.group(2))
            if operator == "lte":
                description = f"价格不应超过{value:g}元"
            elif operator == "gte":
                description = f"价格不应低于{value:g}元"
            else:
                description = f"价格倾向于{value:g}元左右"
            candidates.append(
                {
                    "constraint_type": constraint_type,
                    "description_hint": description,
                    "field_path": "purchase.price",
                    "operator": operator,
                    "expected_value": {
                        "value": value,
                        "currency": "CNY",
                    },
                    "hardness_hint": hardness,
                    "hardness_source": source,
                    "query_spans": [_span(query, match.start(), match.end())],
                    "data_sources": ["query"],
                }
            )
    return sorted(
        candidates,
        key=lambda item: item["query_spans"][0]["start"],
    )


def build_task_facts(
    *,
    task_id: int,
    query: str,
    target_product: Mapping,
    instruction_record: Mapping,
    reward_goal: Mapping | None = None,
) -> dict:
    """Build a private task snapshot without exposing it to the Actor/Judge."""

    product = target_product if isinstance(target_product, Mapping) else {}
    instruction = (
        instruction_record if isinstance(instruction_record, Mapping) else {}
    )
    goal = reward_goal if isinstance(reward_goal, Mapping) else {}
    target = {
        "asin": product.get("asin"),
        "category": product.get("category"),
        "title": product.get("title"),
        "brand": product.get("brand"),
        "shop_name": product.get("shop_name"),
        "pricing": deepcopy(product.get("pricing")),
        "attributes": deepcopy(product.get("attribute")),
        "customization_options": deepcopy(
            product.get("customization_options")
        ),
    }
    instruction_view = {
        "instruction": str(query or instruction.get("instruction") or ""),
        "attributes": _clean_list(instruction.get("attributes")),
        "instruction_options": _clean_list(
            instruction.get("instruction_options")
        ),
    }
    reward_features = {
        key: deepcopy(goal.get(key))
        for key in (
            "category",
            "expected_brand",
            "expected_model",
            "expected_core_functions",
            "required_options_by_key",
            "unresolved_option_requirements",
            "price_upper",
            "reward_feature_version",
        )
        if key in goal
    }
    payload = {
        "schema_version": TASK_FACTS_VERSION,
        "task_id": int(task_id),
        "query": instruction_view["instruction"],
        "instruction": instruction_view,
        "target_product": target,
        "reward_features": reward_features,
    }
    payload["task_data_hash"] = stable_hash(payload)
    payload["query_hash"] = stable_hash(payload["query"])
    return payload


def reusable_cached_rubric(
    cached: Mapping,
    *,
    task_id: int,
    curator_model: str,
    curator_prompt_version: str,
):
    """Reuse a frozen Rubric only when its generation matches this run."""

    bundle = cached.get(task_id)
    if bundle is None:
        return None
    generation = bundle.get("generation")
    generation = generation if isinstance(generation, dict) else {}
    if generation.get("curator_model") != curator_model:
        return None
    if generation.get("curator_prompt_version") != curator_prompt_version:
        return None
    return validate_rubric_bundle(bundle, expected_task_id=task_id)


def extract_rubric_candidates(task_facts: object) -> dict:
    """Extract a superset of constraints; Flash must decide which are requested."""

    if not isinstance(task_facts, Mapping):
        raise TypeError("task_facts must be an object")
    if task_facts.get("schema_version") != TASK_FACTS_VERSION:
        raise ValueError("unsupported task facts schema")
    query = str(task_facts.get("query") or "")
    instruction = task_facts.get("instruction")
    instruction = instruction if isinstance(instruction, Mapping) else {}
    product = task_facts.get("target_product")
    product = product if isinstance(product, Mapping) else {}
    reward = task_facts.get("reward_features")
    reward = reward if isinstance(reward, Mapping) else {}
    raw_candidates = []

    category = reward.get("category") or product.get("category")
    if category:
        leaf = str(category).split("›")[-1].strip()
        raw_candidates.append(
            {
                "constraint_type": "category",
                "description_hint": f"商品品类应为{leaf}",
                "field_path": "product.category",
                "operator": "in_category",
                "expected_value": str(category),
                "hardness_hint": "hard",
                "hardness_source": "explicit_category_rule",
                "query_spans": _exact_spans(query, leaf),
                "data_sources": ["task.target_product.category", "query"],
                "selection_guidance": (
                    "Query 明确请求该商品类型或无歧义同义品类时应选。"
                ),
            }
        )

    for value in _clean_list(reward.get("expected_brand")):
        raw_candidates.append(
            {
                "constraint_type": "brand",
                "description_hint": f"品牌应为{value}",
                "field_path": "product.brand",
                "operator": "eq",
                "expected_value": value,
                "hardness_hint": "hard",
                "hardness_source": "explicit_brand_rule",
                "query_spans": _exact_spans(query, value),
                "data_sources": ["reward_features.expected_brand", "query"],
                "selection_guidance": (
                    "仅当 Query 要求所购商品本身属于该品牌时选择；兼容/适用于"
                    "某品牌、或与品牌值同名的商品品类词都不是品牌要求。"
                ),
            }
        )
    for value in _clean_list(reward.get("expected_model")):
        raw_candidates.append(
            {
                "constraint_type": "model",
                "description_hint": f"型号应为{value}",
                "field_path": "product.model",
                "operator": "eq",
                "expected_value": value,
                "hardness_hint": "hard",
                "hardness_source": "explicit_model_rule",
                "query_spans": _exact_spans(query, value),
                "data_sources": ["reward_features.expected_model", "query"],
            }
        )

    attributes = _clean_list(
        reward.get("expected_core_functions")
        or instruction.get("attributes")
    )
    for value in attributes:
        raw_candidates.append(
            {
                "constraint_type": "core_function",
                "description_hint": f"商品应支持{value}",
                "field_path": "product.attributes",
                "operator": "contains",
                "expected_value": value,
                "hardness_hint": "hard",
                "hardness_source": "explicit_core_function_rule",
                "query_spans": _exact_spans(query, value),
                "data_sources": ["instruction.attributes", "query"],
                "selection_guidance": (
                    "Query 明确表达相同功能、适用人群、使用场景或无歧义同义"
                    "要求时应选；同一含义有多个候选时只选最具体者。"
                ),
            }
        )

    required_options = reward.get("required_options_by_key")
    if isinstance(required_options, Mapping):
        for canonical_axis, requirement in required_options.items():
            if isinstance(requirement, Mapping):
                option_values = [requirement.get("value")]
                source_axis = (
                    requirement.get("source_axis") or canonical_axis
                )
            elif isinstance(requirement, list):
                option_values = requirement
                source_axis = canonical_axis
            else:
                option_values = [requirement]
                source_axis = canonical_axis
            for option_value in _clean_list(option_values):
                raw_candidates.append(
                    {
                        "constraint_type": "option",
                        "description_hint": (
                            f"{source_axis}应选择{option_value}"
                        ),
                        "field_path": (
                            f"purchase.options.{canonical_axis}"
                        ),
                        "operator": "eq",
                        "expected_value": option_value,
                        "hardness_hint": "hard",
                        "hardness_source": "explicit_option_rule",
                        "query_spans": _exact_spans(query, option_value),
                        "data_sources": [
                            "instruction.instruction_options",
                            "target_product.customization_options",
                            "query",
                        ],
                        "selection_guidance": (
                            "Query 明确要求该选项或其中组合规格时选择；如果值引入"
                            "用户未要求的品牌、型号、数量或实质规格则拒绝。纯营销"
                            "标签可忽略，但其余内容必须能唯一承载 Query 的组合要求。"
                        ),
                    }
                )
    else:
        for option_value in _clean_list(
            instruction.get("instruction_options")
        ):
            raw_candidates.append(
                {
                    "constraint_type": "option",
                    "description_hint": f"规格应包含{option_value}",
                    "field_path": "purchase.options.unresolved",
                    "operator": "contains",
                    "expected_value": option_value,
                    "hardness_hint": "hard",
                    "hardness_source": "explicit_option_rule",
                    "query_spans": _exact_spans(query, option_value),
                    "data_sources": [
                        "instruction.instruction_options",
                        "query",
                    ],
                }
            )

    raw_candidates.extend(extract_price_candidates(query))
    candidates = []
    seen = set()
    for raw in raw_candidates:
        identity = stable_hash(
            {
                "constraint_type": raw.get("constraint_type"),
                "field_path": raw.get("field_path"),
                "operator": raw.get("operator"),
                "expected_value": raw.get("expected_value"),
            }
        )
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(
            {
                "candidate_id": f"c{len(candidates) + 1:04d}",
                **raw,
            }
        )
    return {
        "schema_version": RUBRIC_CANDIDATE_VERSION,
        "extractor_version": RUBRIC_EXTRACTOR_VERSION,
        "task_id": int(task_facts["task_id"]),
        "query": query,
        "task_data_hash": str(task_facts.get("task_data_hash") or ""),
        "query_hash": str(task_facts.get("query_hash") or ""),
        "candidates": candidates,
    }


def _spans_from_quote(query: str, quote: str) -> list[dict]:
    quote = str(quote or "").strip()
    if not quote:
        return []
    return _exact_spans(query, quote)


def materialize_rubric_bundle(
    *,
    task_facts: Mapping,
    candidates: Mapping,
    curator_response: Mapping,
    curator_model: str,
    curator_prompt_version: str,
    rubric_version: str,
) -> dict:
    """Freeze selected candidates while copying all semantic fields from code."""

    if task_facts.get("schema_version") != TASK_FACTS_VERSION:
        raise ContractValidationError("unsupported task facts schema")
    if candidates.get("schema_version") != RUBRIC_CANDIDATE_VERSION:
        raise ContractValidationError("unsupported Rubric candidate schema")
    if int(candidates.get("task_id")) != int(task_facts.get("task_id")):
        raise ContractValidationError(
            "candidate task_id does not match task facts"
        )
    if str(candidates.get("query")) != str(task_facts.get("query")):
        raise ContractValidationError("candidate Query does not match task facts")
    if str(candidates.get("task_data_hash")) != str(
        task_facts.get("task_data_hash")
    ):
        raise ContractValidationError(
            "candidate task_data_hash does not match task facts"
        )
    if str(candidates.get("query_hash")) != str(
        task_facts.get("query_hash")
    ):
        raise ContractValidationError(
            "candidate query_hash does not match task facts"
        )
    candidate_rows = candidates.get("candidates")
    if not isinstance(candidate_rows, list):
        raise ContractValidationError("candidates.candidates must be a list")
    by_id = {
        str(item.get("candidate_id")): item
        for item in candidate_rows
        if isinstance(item, Mapping)
    }
    response = validate_curator_response(
        curator_response,
        candidate_ids=by_id,
    )
    query = str(task_facts.get("query") or "")
    rubrics = []
    for selected in response["selected_constraints"]:
        candidate = by_id[selected["candidate_id"]]
        hardness = selected["hardness"]
        quote_spans = _spans_from_quote(query, selected.get("query_quote", ""))
        rubrics.append(
            {
                "rubric_id": f"r{len(rubrics) + 1:04d}",
                "candidate_id": selected["candidate_id"],
                "constraint_type": candidate["constraint_type"],
                "description": selected["description"].strip(),
                "hardness": hardness,
                "hardness_source": candidate["hardness_source"],
                # Query spans are code-owned evidence. A curator paraphrase
                # falls back to the extractor's already validated spans.
                "query_spans": (
                    quote_spans
                    or deepcopy(candidate.get("query_spans") or [])
                ),
                "field_path": candidate["field_path"],
                "operator": candidate["operator"],
                "expected_value": deepcopy(candidate["expected_value"]),
                "data_sources": deepcopy(candidate["data_sources"]),
                "selection_reason": selected["selection_reason"].strip(),
            }
        )
    bundle = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_version": str(rubric_version),
        "task_id": int(task_facts["task_id"]),
        "query": query,
        "generation": {
            "extractor_version": str(candidates["extractor_version"]),
            "curator_model": str(curator_model),
            "curator_prompt_version": str(curator_prompt_version),
            "task_data_hash": str(task_facts["task_data_hash"]),
            "query_hash": str(task_facts["query_hash"]),
        },
        "rubrics": rubrics,
    }
    return validate_rubric_bundle(bundle)
