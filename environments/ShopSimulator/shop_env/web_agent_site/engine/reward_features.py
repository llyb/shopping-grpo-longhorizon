"""Compile deterministic task features and the frozen Reward v4 contract.

The legacy feature keys remain in the returned mapping because older callers
and reports consume them.  New environment goals request ``include_contract``
and receive the v4 contract in addition to those keys.
"""

from __future__ import annotations

import re
import unicodedata
import hashlib
import json

from web_agent_site.engine.comparators import (
    load_brand_aliases,
    normalize_text,
)


REWARD_FEATURE_VERSION = "shopping-reward-features-v1"
OPTION_AXIS_VERSION = "option-axis-v1"
CONTRACT_VERSION = "shopping-requirement-contract-v1"
PREFERENCE_RULE_VERSION = "shopping-preference-rules-v1"
SEVERITY_VERSION = "shopping-violation-severity-v1"
_AXIS_ALIASES = {
    "color": {"颜色", "颜色分类"},
    "size": {"尺码", "鞋码"},
    "dimensions": {"尺寸", "大小"},
    "net_content": {"净含量", "总净含量"},
    "flavor": {"口味", "食品口味"},
    "specification": {"规格", "规格描述", "规格类型"},
    "bundle": {"套餐", "套餐类型", "组合套餐"},
    "capacity": {"容量", "规格容量"},
}
_MODEL_TOKEN = re.compile(
    r"(?<![a-z0-9])(?=[a-z0-9._+-]{2,24}(?![a-z0-9]))"
    r"(?=[a-z0-9._+-]*\d)[a-z0-9._+-]+",
    flags=re.IGNORECASE,
)
_SHOP_SUFFIXES = (
    "官方旗舰店",
    "旗舰店",
    "专卖店",
    "专营店",
    "企业店",
    "店",
)
_UNSUPPORTED_FIRST_PASS_PATTERNS = (
    re.compile(r"(?:缺货|无货|没货).{0,8}(?:才|再)"),
    re.compile(r"(?:不行|没有|买不到).{0,8}(?:才|再)"),
)


def _requires_manual_review(instruction: str) -> bool:
    """Reject conditional substitutions and unreviewed numeric tolerances."""
    return any(pattern.search(str(instruction or "")) for pattern in _UNSUPPORTED_FIRST_PASS_PATTERNS)


def normalize_option_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("/", "|")
    return re.sub(r"\s+", "", text)


def canonicalize_option_axis(value: object) -> str:
    normalized = normalize_option_text(value)
    for canonical, aliases in _AXIS_ALIASES.items():
        if normalized in {
            normalize_option_text(alias) for alias in aliases
        }:
            return canonical
    return normalized


def _clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _target_option_axes(target_product: dict) -> dict[str, list[str]]:
    axes = {}
    for raw_axis, entries in (
        target_product.get("customization_options") or {}
    ).items():
        values = []
        for entry in entries or []:
            if (
                isinstance(entry, dict)
                and normalize_option_text(entry.get("value"))
            ):
                values.append(str(entry["value"]))
        axes[str(raw_axis)] = values
    return axes


def _resolve_required_options(
    option_values: list[str],
    target_product: dict,
) -> tuple[dict, list[dict]]:
    axes = _target_option_axes(target_product)
    resolved = {}
    unresolved = []
    for required_value in option_values:
        normalized_required = normalize_option_text(required_value)
        matches = [
            raw_axis
            for raw_axis, values in axes.items()
            if normalized_required
            in {normalize_option_text(value) for value in values}
        ]
        if len(matches) != 1:
            unresolved.append(
                {
                    "value": required_value,
                    "reason": (
                        "axis_not_found"
                        if not matches
                        else "axis_ambiguous"
                    ),
                    "axes": matches,
                }
            )
            continue
        raw_axis = matches[0]
        canonical_axis = canonicalize_option_axis(raw_axis)
        if canonical_axis in resolved:
            unresolved.append(
                {
                    "value": required_value,
                    "reason": "canonical_axis_collision",
                    "axes": [raw_axis],
                }
            )
            continue
        resolved[canonical_axis] = {
            "value": required_value,
            "source_axis": raw_axis,
            "source": "instruction.instruction_options",
        }
    return resolved, unresolved


def _explicit_brand(instruction: str, target_product: dict) -> list[str]:
    instruction_text = normalize_text(instruction)
    target_text = normalize_text(
        " ".join(
            str(value)
            for value in (
                target_product.get("title"),
                target_product.get("shop_name"),
            )
            if value
        )
    )
    aliases = load_brand_aliases()
    matches = {
        canonical
        for alias, canonical in aliases.items()
        if len(alias) >= 2
        and alias in instruction_text
        and alias in target_text
    }
    shop_name = normalize_text(target_product.get("shop_name"))
    for suffix in _SHOP_SUFFIXES:
        normalized_suffix = normalize_text(suffix)
        if shop_name.endswith(normalized_suffix):
            shop_name = shop_name[: -len(normalized_suffix)]
            break
    title = normalize_text(target_product.get("title"))
    for length in range(min(len(shop_name), 12), 1, -1):
        prefix = shop_name[:length]
        if prefix in instruction_text and prefix in title:
            matches.add(prefix)
            break
    return sorted(matches)


def _explicit_models(instruction: str, target_product: dict) -> list[str]:
    instruction_tokens = {
        token.casefold() for token in _MODEL_TOKEN.findall(instruction)
    }
    target_text = " ".join(
        str(value)
        for value in (
            target_product.get("title"),
            target_product.get("full_description"),
        )
        if value
    )
    target_tokens = {
        token.casefold() for token in _MODEL_TOKEN.findall(target_text)
    }
    return sorted(instruction_tokens.intersection(target_tokens))


def _contract_hash(contract: dict) -> str:
    payload = json.loads(json.dumps(contract, ensure_ascii=False))
    payload.setdefault("audit", {}).pop("contract_hash", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compile_requirement_contract(
    instruction_record: object,
    target_product: object,
    *,
    status: str = "approved",
) -> dict:
    """Compile existing reviewed annotations into a v4 requirement contract.

    Existing ShopSimulator annotations describe hard requirements (attributes
    and selected options).  We do not promote hidden target fields to user
    requirements.  A small, deterministic phrase recognizer adds a preference
    only when the instruction explicitly uses a preference marker; ambiguous
    wording remains a hard requirement or is left out for human review.
    """
    instruction = instruction_record if isinstance(instruction_record, dict) else {}
    product = target_product if isinstance(target_product, dict) else {}
    text = str(instruction.get("instruction") or "")
    category = product.get("category")
    must = [{
        "id": "category",
        "field": "category",
        "operator": "is_a",
        "value": category,
        "source_text": category,
        "severity": {"rule_id": "wrong_category_v1", "value": 1.0},
    }]
    attributes = _clean_list(instruction.get("attributes"))
    for index, value in enumerate(attributes):
        must.append({
            "id": f"core_function_{index}",
            "field": "core_function",
            "operator": "eq",
            "value": value,
            "source_text": value,
            "severity": {"rule_id": "missing_core_function_v1", "value": 0.9},
        })
    required_options, unresolved_options = _resolve_required_options(
        _clean_list(instruction.get("instruction_options")), product
    )
    effective_status = (
        "needs_review"
        if unresolved_options or _requires_manual_review(text)
        else status
    )
    option_values = _clean_list(instruction.get("instruction_options"))
    for axis, requirement in required_options.items():
        must.append({
            "id": f"option_{axis}",
            "field": "option",
            "axis": axis,
            "operator": "eq",
            "value": requirement.get("value"),
            "source_text": requirement.get("value"),
            "severity": (
                {"rule_id": "wrong_color_v1", "value": 0.3}
                if axis == "color"
                else {"rule_id": f"wrong_{axis}_v1", "value": 0.8}
                if axis in {"capacity", "size", "dimensions"}
                else {"rule_id": "wrong_option_v1", "value": 0.3}
            ),
        })
    for index, item in enumerate(unresolved_options):
        must.append({
            "id": f"option_unresolved_{index}",
            "field": "option",
            "axis": "",
            "operator": "eq",
            "value": item.get("value"),
            "source_text": item.get("value"),
            "severity": {"rule_id": "wrong_option_v1", "value": 0.3},
            "unresolved": True,
        })
    instruction_text = text
    expected_brand = _explicit_brand(instruction_text, product)
    expected_model = _explicit_models(instruction_text, product)
    for index, value in enumerate(expected_brand):
        must.append({
            "id": f"brand_{index}", "field": "brand", "operator": "eq", "value": value,
            "source_text": value, "severity": {"rule_id": "wrong_brand_v1", "value": 0.6},
        })
    for index, value in enumerate(expected_model):
        must.append({
            "id": f"model_{index}", "field": "model", "operator": "eq", "value": value,
            "source_text": value, "severity": {"rule_id": "wrong_model_v1", "value": 0.8},
        })

    upper = None
    # Keep the existing parser as the only budget parser; no target-price
    # fallback is allowed.  Import lazily to avoid a circular dependency.
    try:
        from web_agent_site.engine.constraints import explicit_budget_from_instruction

        upper = explicit_budget_from_instruction(text)
    except Exception:
        upper = None
    if upper is not None:
        must.append({
            "id": "budget", "field": "final_variant_price", "operator": "le", "value": upper,
            "unit": "CNY", "source_text": text, "severity": {"rule_id": "budget_excess_v1"},
        })
    else:
        # Final price must still be resolvable even when the user did not set
        # a budget; the requirement has no upper bound in that case.
        must.append({
            "id": "final_price", "field": "final_variant_price", "operator": "known", "value": None,
            "unit": "CNY", "source_text": None, "severity": {"rule_id": "invalid_price_v1", "value": 1.0},
        })
    must.append({
        "id": "legal_sku", "field": "legal_sku", "operator": "eq", "value": True,
        "source_text": None, "severity": {"rule_id": "invalid_sku_v1", "value": 1.0},
    })

    preferences = []
    markers = ("优先", "首选", "最好", "更喜欢")
    if any(marker in text for marker in markers):
        for axis, requirement in required_options.items():
            value = str(requirement.get("value") or "")
            if value and value in text:
                alternatives = [item for item in option_values if item != value]
                if any(token in text for token in ("也可以", "也接受", "都行", "其次")) and alternatives:
                    for item in must:
                        if item.get("id") == f"option_{axis}":
                            item["operator"] = "in"
                            item["value"] = [value, *alternatives]
                    preferences.append({
                        "id": f"preferred_{axis}",
                        "field": axis,
                        "function": "categorical_preferred_v1",
                        "preferred": [value],
                        "allowed": alternatives,
                        "allowed_requirement_id": f"option_{axis}",
                        "source_text": text,
                    })
    price_objective = {
        "active": bool(upper is not None and any(token in text for token in ("尽量便宜", "省钱", "越便宜", "便宜点"))),
    }
    if price_objective["active"]:
        price_objective["reference_budget"] = upper
    contract = {
        "contract_version": CONTRACT_VERSION,
        "status": effective_status,
        "task_mode": "single_product_purchase",
        "must": must,
        "preferences": preferences,
        "price_objective": price_objective,
        "audit": {
            "rule_version": PREFERENCE_RULE_VERSION,
            "severity_version": SEVERITY_VERSION,
            "review_status": effective_status,
        },
    }
    contract["audit"]["contract_hash"] = _contract_hash(contract)
    return contract


def compile_reward_features(
    instruction_record: object,
    target_product: object,
    *,
    include_contract: bool = False,
) -> dict:
    """Build fixed scoring inputs from existing task annotations and Gold metadata."""
    instruction = (
        instruction_record if isinstance(instruction_record, dict) else {}
    )
    product = target_product if isinstance(target_product, dict) else {}
    instruction_text = str(instruction.get("instruction") or "")
    option_values = _clean_list(instruction.get("instruction_options"))
    required_options, unresolved_options = _resolve_required_options(
        option_values,
        product,
    )
    features = {
        "reward_feature_version": REWARD_FEATURE_VERSION,
        "category": product.get("category"),
        "expected_brand": _explicit_brand(instruction_text, product),
        "expected_model": _explicit_models(instruction_text, product),
        "expected_core_functions": _clean_list(
            instruction.get("attributes")
        ),
        "required_options_by_key": required_options,
        "unresolved_option_requirements": unresolved_options,
        "option_axis_version": OPTION_AXIS_VERSION,
        "feature_sources": {
            "category": "task.target_product.category",
            "brand": "instruction_explicit_alias",
            "model": "instruction_target_token_intersection",
            "core_functions": "instruction.attributes",
            "options": "instruction.instruction_options",
        },
    }
    if include_contract:
        features["reward_contract"] = compile_requirement_contract(
            instruction_record, target_product
        )
    return features
