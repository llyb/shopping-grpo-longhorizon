"""Deterministic ShopSimulator Reward v4 implementation.

The v4 scorer is deliberately independent from an actor's natural-language
claims.  It consumes a frozen requirement contract, the final order facts and
an optional evidence ledger produced from observations shown to the actor.
The module keeps the input surface small so the v3 adapter can continue to
serve old callers while new environment goals opt in with ``reward_contract``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

from web_agent_site.engine.comparators import (
    COMPARATOR_VERSION,
    FAIL,
    PASS,
    UNVERIFIABLE,
    compare_category,
    compare_brand,
    compare_core_functions,
    compare_model,
    comparison,
    load_brand_aliases,
    normalize_text,
)
from web_agent_site.engine.reward_features import (
    REWARD_FEATURE_VERSION,
    canonicalize_option_axis,
    normalize_option_text,
)
from web_agent_site.engine.variant_price import (
    VARIANT_PRICE_VERSION,
    option_axes,
    resolve_variant_price,
)


REWARD_VERSION = "shopsimulator-reward-v4"
CONTRACT_VERSION = "shopping-requirement-contract-v1"
PREFERENCE_RULE_VERSION = "shopping-preference-rules-v1"
SEVERITY_VERSION = "shopping-violation-severity-v1"

DEFAULT_REWARDS = {
    "purchase_completion": 0.80,
    "preference_weight": 0.10,
    "price_weight": 0.05,
    "gold_weight": 0.05,
    "preferred_alternative_score": 0.50,
    "wrong_base": -0.60,
    "wrong_severity_weight": 0.40,
    "unverified_purchase": -0.60,
    "stop_with_acceptable_candidate": -0.40,
    "stop_base": -0.30,
    "stop_quality_weight": 0.20,
    "max_steps": -0.40,
    "repeat_loop": -0.50,
    "unfinished": -0.40,
    "reward_unverifiable": 0.0,
}

SEVERITY_RULES = {
    "category": ("wrong_category_v1", 1.0),
    "brand": ("wrong_brand_v1", 0.60),
    "model": ("wrong_model_v1", 0.80),
    "capacity": ("wrong_capacity_v1", 0.80),
    "size": ("wrong_size_v1", 0.80),
    "option": ("wrong_option_v1", 0.30),
    "color": ("wrong_color_v1", 0.30),
    "core_function": ("missing_core_function_v1", 0.90),
    "final_variant_price": ("budget_excess_v1", 1.0),
    "legal_sku": ("invalid_sku_v1", 1.0),
}

_OPTION_FIELDS = {"option", "color", "size", "capacity", "storage", "sku_option"}
_NUMERIC_OPERATORS = {"eq", "neq", "ge", "gt", "le", "lt", "between"}
_PREFERENCE_FUNCTIONS = {
    "categorical_preferred_v1",
    "ordered_choices_v1",
    "numeric_increasing_saturated_v1",
    "numeric_decreasing_saturated_v1",
    "boolean_preference_v1",
}
_UNIT_FACTORS = {
    # dimension, value in the frozen base unit
    "b": ("storage_bytes", 1.0),
    "kb": ("storage_bytes", 1024.0),
    "mb": ("storage_bytes", 1024.0**2),
    "gb": ("storage_bytes", 1024.0**3),
    "tb": ("storage_bytes", 1024.0**4),
    "字节": ("storage_bytes", 1.0),
    "ml": ("volume_ml", 1.0),
    "毫升": ("volume_ml", 1.0),
    "l": ("volume_ml", 1000.0),
    "升": ("volume_ml", 1000.0),
    "g": ("mass_g", 1.0),
    "克": ("mass_g", 1.0),
    "kg": ("mass_g", 1000.0),
    "千克": ("mass_g", 1000.0),
    "公斤": ("mass_g", 1000.0),
    "mm": ("length_mm", 1.0),
    "毫米": ("length_mm", 1.0),
    "cm": ("length_mm", 10.0),
    "厘米": ("length_mm", 10.0),
    "m": ("length_mm", 1000.0),
    "米": ("length_mm", 1000.0),
    "cny_fen": ("currency_fen", 1.0),
    "fen": ("currency_fen", 1.0),
    "分": ("currency_fen", 1.0),
    "cny": ("currency_fen", 100.0),
    "rmb": ("currency_fen", 100.0),
    "yuan": ("currency_fen", 100.0),
    "元": ("currency_fen", 100.0),
}


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit_definition(unit: Any) -> tuple[str, float] | None:
    normalized = normalize_text(unit, remove_space=False).replace(" ", "")
    return _UNIT_FACTORS.get(normalized)


def _normalized_number(value: Any, unit_hint: Any = None) -> tuple[str, float] | None:
    """Parse a finite numeric value into a deterministic base-unit pair."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = _finite_number(value)
        definition = _unit_definition(unit_hint) if unit_hint else ("scalar", 1.0)
        return None if number is None or definition is None else (definition[0], number * definition[1])
    text = normalize_text(value, remove_space=False)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = _finite_number(match.group(0))
    if number is None:
        return None
    suffix = text[match.end():].strip().replace(" ", "")
    definition = _unit_definition(suffix) if suffix else None
    if definition is None and unit_hint:
        definition = _unit_definition(unit_hint)
    if definition is None:
        # Unitless strings stay scalar; an explicit but unsupported unit is
        # not silently discarded.
        definition = ("scalar", 1.0) if not suffix else None
    return None if definition is None else (definition[0], number * definition[1])


def _boolean_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = normalize_text(value)
    if normalized in {"true", "yes", "1", "是", "支持", "有"}:
        return True
    if normalized in {"false", "no", "0", "否", "不支持", "无", "没有"}:
        return False
    return None


def _numeric_comparison(actual: Any, requirement: Mapping[str, Any], *, source_field: str) -> dict:
    operator = str(requirement.get("operator") or "eq")
    unit = requirement.get("unit")
    actual_value = _normalized_number(actual, unit)
    if operator == "between":
        raw_range = requirement.get("value")
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            lower_raw, upper_raw = raw_range
        else:
            lower_raw = requirement.get("lower")
            upper_raw = requirement.get("upper")
        lower = _normalized_number(lower_raw, unit)
        upper = _normalized_number(upper_raw, unit)
        required = [lower_raw, upper_raw]
        expected_values = [lower, upper]
    else:
        required = requirement.get("value")
        expected_values = [_normalized_number(required, unit)]
    if actual_value is None or any(item is None for item in expected_values):
        return comparison(
            UNVERIFIABLE,
            comparator="numeric_unit_registry_v1",
            required=required,
            actual=actual,
            source_field=source_field,
            evidence={"operator": operator, "unit": unit},
        )
    dimensions = {actual_value[0], *(item[0] for item in expected_values if item)}
    if len(dimensions) != 1 or operator not in _NUMERIC_OPERATORS:
        return comparison(
            UNVERIFIABLE,
            comparator="numeric_unit_registry_v1",
            required=required,
            actual=actual,
            source_field=source_field,
            evidence={"operator": operator, "unit": unit, "dimensions": sorted(dimensions)},
        )
    value = actual_value[1]
    target = expected_values[0][1]
    tolerance = _finite_number(requirement.get("tolerance", 0.0))
    if tolerance is None or tolerance < 0:
        return comparison(
            UNVERIFIABLE,
            comparator="numeric_unit_registry_v1",
            required=required,
            actual=actual,
            source_field=source_field,
            evidence={"operator": operator, "invalid_tolerance": requirement.get("tolerance")},
        )
    if operator == "eq":
        passed = abs(value - target) <= tolerance
    elif operator == "neq":
        passed = abs(value - target) > tolerance
    elif operator == "ge":
        passed = value + tolerance >= target
    elif operator == "gt":
        passed = value > target + tolerance
    elif operator == "le":
        passed = value - tolerance <= target
    elif operator == "lt":
        passed = value < target - tolerance
    else:
        passed = target - tolerance <= value <= expected_values[1][1] + tolerance
    return comparison(
        PASS if passed else FAIL,
        comparator="numeric_unit_registry_v1",
        required=required,
        actual=actual,
        source_field=source_field,
        evidence={"operator": operator, "unit": unit, "normalized_actual": value},
    )


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _contract_hash(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    raw_audit = payload.get("audit")
    audit = dict(raw_audit) if isinstance(raw_audit, Mapping) else {}
    audit.pop("contract_hash", None)
    payload["audit"] = audit
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_v4_goal(goal: Mapping[str, Any] | None) -> bool:
    if not isinstance(goal, Mapping):
        return False
    # The presence of an explicit contract opts into v4 even when the
    # contract is malformed; this guarantees fail-closed scoring instead of
    # silently routing a damaged v4 goal through the legacy v3 adapter.
    return "reward_contract" in goal or "contract" in goal


def _contract_from_goal(goal: Mapping[str, Any]) -> Any:
    if "reward_contract" in goal:
        return goal.get("reward_contract")
    if "contract" in goal:
        return goal.get("contract")
    return _legacy_contract(goal)


def _actual_option(selected_options: Any, axis: str) -> Any:
    if not isinstance(selected_options, Mapping):
        return None
    wanted = canonicalize_option_axis(axis)
    for raw_axis, value in selected_options.items():
        if canonicalize_option_axis(raw_axis) == wanted:
            return value
    return None


def _product_option_value(product: Mapping[str, Any], axis: str, selected_options: Any) -> Any:
    selected = _actual_option(selected_options, axis)
    if selected is not None:
        return selected
    # A missing explicit selection is only acceptable when the catalog exposes
    # one unambiguous default value for that axis.
    options = product.get("customization_options") or product.get("options") or {}
    for raw_axis, values in options.items():
        if canonicalize_option_axis(raw_axis) != canonicalize_option_axis(axis):
            continue
        if not isinstance(values, list):
            return None
        normalized = []
        for item in values:
            value = item.get("value") if isinstance(item, Mapping) else item
            if value is not None and normalize_option_text(value) not in normalized:
                normalized.append(normalize_option_text(value))
        return values[0].get("value") if len(normalized) == 1 and values else None
    return None


def _brand_result(requirement: Mapping[str, Any], product: Mapping[str, Any]) -> dict:
    operator = str(requirement.get("operator") or "eq")
    raw = requirement.get("value")
    values = raw if isinstance(raw, list) else [raw]
    actual = product.get("brand") or product.get("shop_name")
    actual_normalized = normalize_text(actual)
    aliases = load_brand_aliases()
    results = []
    for value in values:
        item = compare_brand(value, dict(product))
        # The task compiler may freeze an explicit Chinese brand token found
        # in both the instruction and the catalog title/shop name (for example
        # 久光 in 久光制药海外旗舰店).  Reproduce that same deterministic token
        # rule at scoring time while keeping one-character substrings invalid.
        token = normalize_text(value)
        canonical = aliases.get(token, token)
        catalog_aliases = {
            aliases.get(alias, alias)
            for alias in aliases
            if len(alias) >= 2 and alias in actual_normalized
        }
        token_match = len(token) >= 2 and token in actual_normalized
        alias_match = canonical in catalog_aliases
        if item.get("status") == FAIL and (token_match or alias_match):
            item = comparison(
                PASS,
                comparator="brand_explicit_token_v1",
                required=value,
                actual=actual,
                source_field="brand|shop_name",
                evidence={"token_match": token_match, "alias_match": alias_match},
            )
        results.append(item)
    statuses = [item.get("status") for item in results]
    if operator in {"eq", "in"}:
        status = PASS if PASS in statuses else (UNVERIFIABLE if UNVERIFIABLE in statuses else FAIL)
    elif operator in {"neq", "not_in"}:
        status = FAIL if PASS in statuses else (UNVERIFIABLE if UNVERIFIABLE in statuses else PASS)
    else:
        status = UNVERIFIABLE
    return comparison(
        status,
        comparator="brand_alias_registry_v1",
        required=raw,
        actual=actual,
        source_field="brand|shop_name",
        evidence={"operator": operator, "comparisons": results, "alias_version": "brand-aliases-v1"},
    )


def _enum_comparison(actual: Any, requirement: Mapping[str, Any], *, source_field: str) -> dict:
    operator = str(requirement.get("operator") or "eq")
    expected = requirement.get("value")
    values = expected if isinstance(expected, list) else [expected]
    if actual is None or not values:
        status = UNVERIFIABLE
    else:
        actual_normalized = normalize_option_text(actual)
        candidates = {normalize_option_text(value) for value in values}
        if operator in {"eq", "in"}:
            status = PASS if actual_normalized in candidates else FAIL
        elif operator in {"neq", "not_in"}:
            status = PASS if actual_normalized not in candidates else FAIL
        elif operator == "known":
            status = PASS
        else:
            status = UNVERIFIABLE
    return comparison(
        status,
        comparator="normalized_enum_registry_v1",
        required=expected,
        actual=actual,
        source_field=source_field,
        evidence={"operator": operator},
    )


def _option_result(requirement: Mapping[str, Any], product: Mapping[str, Any], selected: Any) -> dict:
    axis = str(requirement.get("axis") or requirement.get("field") or "")
    actual = _product_option_value(product, axis, selected)
    if actual is None:
        return comparison(
            UNVERIFIABLE,
            comparator="selected_option_v1",
            required=requirement.get("value"),
            actual=None,
            source_field=axis,
            evidence={"axis": axis},
        )
    operator = str(requirement.get("operator") or "eq")
    if requirement.get("unit") is not None or operator in {"ge", "gt", "le", "lt", "between"}:
        result = _numeric_comparison(actual, requirement, source_field=axis)
    else:
        result = _enum_comparison(actual, requirement, source_field=axis)
    result = dict(result)
    result["evidence"] = {**dict(result.get("evidence") or {}), "axis": axis}
    return result


def _price_result(price_resolution: Any, requirement: Mapping[str, Any] | None) -> dict:
    if not isinstance(price_resolution, Mapping) or price_resolution.get("status") != PASS:
        return comparison(
            UNVERIFIABLE,
            comparator="variant_price_v4",
            required=(requirement or {}).get("value"),
            actual=(price_resolution or {}).get("price") if isinstance(price_resolution, Mapping) else None,
            source_field="final_variant_price",
            evidence=price_resolution or {},
        )
    actual = _finite_number(price_resolution.get("price"))
    requirement = requirement or {}
    if actual is None or actual < 0:
        result = comparison(
            UNVERIFIABLE,
            comparator="variant_price_v4",
            required=requirement.get("value"),
            actual=actual,
            source_field="final_variant_price",
            evidence=price_resolution,
        )
    elif str(requirement.get("operator") or "known") == "known" or requirement.get("value") is None:
        result = comparison(
            PASS,
            comparator="variant_price_v4",
            required=requirement.get("value"),
            actual=actual,
            source_field="final_variant_price",
            evidence=price_resolution,
        )
    else:
        # The simulator stores catalog prices in yuan, while a frozen
        # contract may express the same value in fen.  Convert only the
        # observed side for an explicit fen comparison; keep the public
        # ``actual`` value in the catalog's major unit for audit/evidence
        # matching.
        compare_actual = actual
        unit_definition = _unit_definition(requirement.get("unit"))
        if unit_definition == ("currency_fen", 1.0):
            compare_actual = actual * 100.0
        result = _numeric_comparison(
            compare_actual,
            requirement,
            source_field="final_variant_price",
        )
        result = dict(result)
        result["actual"] = actual
    result = dict(result)
    result["evidence"] = {"price_resolution": dict(price_resolution), "comparison": result.get("evidence")}
    return result


def _product_field_value(product: Mapping[str, Any], field: str, selected: Any) -> Any:
    if field in _OPTION_FIELDS:
        return _product_option_value(product, field, selected)
    if field in product:
        return product.get(field)
    wanted = normalize_text(field)
    for key, value in product.items():
        if normalize_text(key) == wanted:
            return value
    attributes = product.get("attribute") or product.get("Attributes") or {}
    if isinstance(attributes, Mapping):
        for key, value in attributes.items():
            if normalize_text(key) == wanted:
                return value
    return None


def _boolean_result(requirement: Mapping[str, Any], product: Mapping[str, Any], selected: Any) -> dict:
    field = str(requirement.get("field") or "")
    operator = str(requirement.get("operator") or "true")
    expected = operator != "false" if operator in {"true", "false"} else bool(requirement.get("value", True))
    actual = _product_field_value(product, field, selected)
    actual_boolean = _boolean_value(actual)
    if actual_boolean is not None:
        status = PASS if actual_boolean == expected else FAIL
        evidence = {"operator": operator}
        actual = actual_boolean
    else:
        feature = requirement.get("value") if field in {"core_function", "core_functions", "attribute", "attributes"} else field
        feature_result = compare_core_functions([feature], dict(product))
        details = feature_result.get("evidence") or {}
        positive = feature_result.get("status") == PASS
        explicitly_negative = bool(details.get("negated"))
        if expected:
            status = PASS if positive else (FAIL if explicitly_negative else UNVERIFIABLE)
        else:
            status = PASS if explicitly_negative else (FAIL if positive else UNVERIFIABLE)
        actual = positive if positive or explicitly_negative else None
        evidence = {"operator": operator, "feature_comparison": feature_result}
    return comparison(
        status,
        comparator="boolean_feature_registry_v1",
        required=expected,
        actual=actual,
        source_field=field,
        evidence=evidence,
    )


def _legal_sku_result(product: Mapping[str, Any], selected: Any, price_resolution: Any) -> dict:
    indexed = option_axes(dict(product))
    axes = indexed.get("axes") or {}
    collisions = indexed.get("collisions") or {}
    invalid = []
    unknown = []
    selected_axes = {
        canonicalize_option_axis(axis): value
        for axis, value in dict(selected or {}).items()
    } if isinstance(selected, Mapping) else {}
    for axis, definition in axes.items():
        values = definition.get("values") or {}
        selection = selected_axes.get(axis)
        if selection is None:
            if len(values) != 1:
                unknown.append({"axis": axis, "reason": "selection_missing"})
            continue
        if normalize_option_text(selection) not in values:
            invalid.append({"axis": axis, "value": selection})
    unknown_axes = sorted(set(selected_axes) - set(axes))
    if unknown_axes:
        invalid.extend({"axis": axis, "reason": "unknown_axis"} for axis in unknown_axes)
    price_known = isinstance(price_resolution, Mapping) and price_resolution.get("status") == PASS
    if invalid:
        status = FAIL
    elif collisions or unknown or not price_known:
        status = UNVERIFIABLE
    else:
        status = PASS
    return comparison(
        status,
        comparator="legal_variant_v1",
        required=True,
        actual=status == PASS,
        source_field="variant_price|customization_options",
        evidence={
            "axis_collisions": collisions,
            "invalid_selections": invalid,
            "unknown_selections": unknown,
            "price_resolution": price_resolution or {},
        },
    )


def _composite_result(
    requirement: Mapping[str, Any],
    product: Mapping[str, Any],
    selected: Any,
    price_resolution: Any,
) -> dict:
    operator = str(requirement.get("operator") or requirement.get("field") or "")
    clauses = requirement.get("clauses") or requirement.get("requirements") or requirement.get("value") or []
    if not isinstance(clauses, list) or not clauses or not all(isinstance(item, Mapping) for item in clauses):
        return comparison(
            UNVERIFIABLE,
            comparator="composite_requirement_registry_v1",
            required=clauses,
            actual=None,
            source_field=operator,
            evidence={"operator": operator, "invalid_clauses": True},
        )
    children = [
        _requirement_result(item, product, selected, price_resolution)
        for item in clauses
    ]
    statuses = [item.get("status") for item in children]
    if operator == "all_of":
        status = FAIL if FAIL in statuses else (UNVERIFIABLE if UNVERIFIABLE in statuses else PASS)
    elif operator == "any_of":
        status = PASS if PASS in statuses else (UNVERIFIABLE if UNVERIFIABLE in statuses else FAIL)
    else:
        status = UNVERIFIABLE
    result = comparison(
        status,
        comparator="composite_requirement_registry_v1",
        required=clauses,
        actual=None,
        source_field=operator,
        evidence={"operator": operator, "children": children},
    )
    result["children"] = children
    return result


def _requirement_result(requirement: Mapping[str, Any], product: Mapping[str, Any], selected: Any, price_resolution: Any) -> dict:
    field = str(requirement.get("field") or requirement.get("type") or "")
    operator = str(requirement.get("operator") or "eq")
    if field in {"all_of", "any_of"} and requirement.get("operator") is None:
        operator = field
    value = requirement.get("value")
    if operator in {"all_of", "any_of"} or field in {"all_of", "any_of"}:
        result = _composite_result(requirement, product, selected, price_resolution)
    elif field == "category":
        result = compare_category(value, product)
    elif field == "brand":
        result = _brand_result(requirement, product)
    elif field == "model":
        values = value if isinstance(value, list) else [value]
        model_results = [compare_model([item], dict(product)) for item in values]
        statuses = [item.get("status") for item in model_results]
        if operator in {"eq", "in"}:
            status = PASS if PASS in statuses else (UNVERIFIABLE if UNVERIFIABLE in statuses else FAIL)
        elif operator in {"neq", "not_in"}:
            status = FAIL if PASS in statuses else (UNVERIFIABLE if UNVERIFIABLE in statuses else PASS)
        else:
            status = UNVERIFIABLE
        result = comparison(
            status,
            comparator="model_token_boundary_registry_v1",
            required=value,
            actual=product.get("model") or product.get("title") or product.get("Title"),
            source_field="model|title",
            evidence={"operator": operator, "comparisons": model_results},
        )
    elif field in {"core_function", "core_functions", "attribute", "attributes"}:
        if operator in {"true", "false"} or isinstance(value, bool):
            result = _boolean_result(requirement, product, selected)
        else:
            values = value if isinstance(value, list) else [value]
            result = compare_core_functions(values, product)
    elif field in _OPTION_FIELDS:
        result = _option_result(requirement, product, selected)
    elif field in {"final_variant_price", "budget", "price"}:
        result = _price_result(price_resolution, requirement)
    elif field == "legal_sku":
        result = _legal_sku_result(product, selected, price_resolution)
    elif operator in {"true", "false"} or isinstance(value, bool):
        result = _boolean_result(requirement, product, selected)
    else:
        actual = _product_field_value(product, field, selected)
        if requirement.get("unit") is not None or operator in {"ge", "gt", "le", "lt", "between"}:
            result = _numeric_comparison(actual, requirement, source_field=field)
        elif operator in {"eq", "neq", "in", "not_in", "known"}:
            result = _enum_comparison(actual, requirement, source_field=field)
        else:
            result = comparison(
                UNVERIFIABLE,
                comparator="unsupported_requirement_v1",
                required=value,
                actual=actual,
                source_field=field,
                evidence={"operator": operator},
            )
    result = dict(result)
    result.update(
        {
            "requirement_id": str(requirement.get("id") or field),
            "field": field,
            "axis": requirement.get("axis") or (field if field in _OPTION_FIELDS else None),
            "operator": operator,
            "severity": dict(requirement.get("severity") or {}),
            "source_text": requirement.get("source_text"),
        }
    )
    return result


def _facts_from_evidence(evidence: Any) -> list[Mapping[str, Any]]:
    if not isinstance(evidence, Mapping):
        return []
    raw = evidence.get("facts") or evidence.get("evidence_refs") or []
    return [item for item in raw if isinstance(item, Mapping)]


def _evidence_covers(result: Mapping[str, Any], product: Mapping[str, Any], selected: Any, evidence: Any) -> bool:
    # Pure scorer callers historically did not have an evidence ledger.  Their
    # contract is still deterministic; runtime callers pass facts explicitly.
    if evidence is None:
        return True
    children = [item for item in result.get("children") or [] if isinstance(item, Mapping)]
    if children:
        operator = str(result.get("operator") or result.get("field") or "")
        coverage = [_evidence_covers(item, product, selected, evidence) for item in children]
        return any(coverage) if operator == "any_of" and result.get("status") == PASS else all(coverage)
    facts = _facts_from_evidence(evidence)
    asin = str(product.get("asin"))
    field = str(result.get("field") or "")
    axis = canonicalize_option_axis(str(result.get("axis") or field))
    expected_value = result.get("actual")
    # Core-function comparators expose a boolean or the complete attribute
    # list as ``actual``.  The actor evidence only needs to contain the
    # required feature(s), which are retained in the nested comparator.
    if field in {"core_function", "core_functions", "attribute", "attributes"}:
        nested = (result.get("evidence") or {}).get("feature_comparison")
        if isinstance(nested, Mapping):
            expected_value = nested.get("required", expected_value)
        elif result.get("required") not in (None, True, False):
            expected_value = result.get("required")
    selected_fingerprint = json.dumps(dict(selected or {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for fact in facts:
        fact_field = str(fact.get("field") or "")
        field_matches = fact_field in {field, "*", "final_order"}
        if field in _OPTION_FIELDS and fact_field == "option":
            fact_axis = canonicalize_option_axis(str(fact.get("axis") or ""))
            field_matches = fact_axis == axis
        if str(fact.get("asin", asin)) != asin or not field_matches:
            continue
        if fact.get("status") in {"unknown", "unverifiable"}:
            continue
        fact_fingerprint = fact.get("option_fingerprint")
        if fact_fingerprint and fact_fingerprint != selected_fingerprint:
            continue
        fact_value = fact.get("value")
        fact_number = _finite_number(fact_value)
        actual_number = _finite_number(expected_value)
        numeric_match = (
            fact_number is not None
            and actual_number is not None
            and math.isclose(fact_number, actual_number, rel_tol=0.0, abs_tol=1e-9)
        )
        if fact_value is None or numeric_match:
            return True
        if isinstance(expected_value, (list, tuple, set)):
            expected_values = [normalize_option_text(item) for item in expected_value if normalize_option_text(item)]
            fact_values = [normalize_option_text(item) for item in (fact_value if isinstance(fact_value, (list, tuple, set)) else [fact_value]) if normalize_option_text(item)]
            if expected_values and field in {"core_function", "core_functions", "attribute", "attributes"}:
                joined = " ".join(fact_values)
                if all(item in joined for item in expected_values):
                    return True
            elif any(item in fact_values for item in expected_values):
                return True
        elif normalize_option_text(fact_value) == normalize_option_text(expected_value):
            return True
    return False


def _contract_invalid_reason(contract: Any) -> str | None:
    """Validate the frozen, data-only contract before scoring it."""
    if not isinstance(contract, Mapping):
        return "contract_not_object"
    if contract.get("contract_version") != CONTRACT_VERSION:
        return "contract_version_invalid"
    if contract.get("status") not in {"approved", "legacy_approved"}:
        return "contract_status_not_approved"
    audit = contract.get("audit")
    if audit is not None and not isinstance(audit, Mapping):
        return "contract_audit_invalid"
    must = contract.get("must")
    if not isinstance(must, list) or not must:
        return "contract_must_invalid"
    ids = set()
    for item in must:
        if not isinstance(item, Mapping):
            return "contract_requirement_invalid"
        requirement_id = str(item.get("id") or "")
        if not requirement_id or requirement_id in ids:
            return "contract_requirement_id_invalid"
        ids.add(requirement_id)
        severity = item.get("severity")
        if severity is not None:
            if not isinstance(severity, Mapping):
                return "contract_severity_invalid"
            configured_value = severity.get("value")
            if configured_value is not None:
                numeric = _finite_number(configured_value)
                if numeric is None or not 0.0 <= numeric <= 1.0:
                    return "contract_severity_invalid"
    preferences = contract.get("preferences", [])
    if not isinstance(preferences, list):
        return "contract_preferences_invalid"
    for preference in preferences:
        if not isinstance(preference, Mapping):
            return "contract_preference_invalid"
        if str(preference.get("function") or "") not in _PREFERENCE_FUNCTIONS:
            return "unsupported_preference_function"
    price_objective = contract.get("price_objective", {})
    if not isinstance(price_objective, Mapping):
        return "contract_price_objective_invalid"
    if price_objective.get("active"):
        budget = _finite_number(price_objective.get("reference_budget"))
        if budget is None or budget <= 0:
            return "price_objective_reference_budget_invalid"
    return None


def _preference_score(preference: Mapping[str, Any], product: Mapping[str, Any], selected: Any, evidence: Any) -> dict:
    function = str(preference.get("function") or "")
    field = str(preference.get("field") or "")
    value = _product_option_value(product, field, selected) if field in _OPTION_FIELDS else _product_field_value(product, field, selected)
    if value is None and field in {"brand", "model"}:
        value = product.get(field) or product.get("title")
    if value is None:
        return {"id": preference.get("id"), "function": function, "score": 0.0, "status": "unknown", "preferred_met": False, "unknown_reason": "fact_missing"}
    evidence_probe = {
        "field": field,
        "axis": preference.get("axis") or (field if field in _OPTION_FIELDS else None),
        "actual": value,
    }
    if evidence is not None and not _evidence_covers(evidence_probe, product, selected, evidence):
        return {
            "id": preference.get("id"),
            "function": function,
            "score": 0.0,
            "status": "evidence_missing",
            "preferred_met": False,
            "unknown_reason": "actor_evidence_missing",
        }
    score = 0.0
    preferred_met = False
    if function == "categorical_preferred_v1":
        preferred = preference.get("preferred") or []
        allowed = preference.get("allowed") or preference.get("acceptable") or []
        actual = normalize_option_text(value)
        if any(actual == normalize_option_text(item) for item in preferred):
            score, preferred_met = 1.0, True
        elif any(actual == normalize_option_text(item) for item in allowed):
            score = float(preference.get("alternative_score", 0.5))
        else:
            score = 0.0
    elif function == "ordered_choices_v1":
        choices = preference.get("choices") or preference.get("preferred") or []
        actual = normalize_option_text(value)
        ranks = [index for index, item in enumerate(choices) if (normalize_option_text(item) == actual or (isinstance(item, list) and actual in {normalize_option_text(v) for v in item}))]
        if ranks:
            k = len(choices)
            score = 1.0 if k == 1 else 1.0 - 0.5 * ranks[0] / (k - 1)
            preferred_met = ranks[0] == 0
    elif function == "numeric_increasing_saturated_v1":
        unit = preference.get("unit")
        number_pair = _normalized_number(value, unit)
        lower_pair = _normalized_number(preference.get("minimum"), unit)
        target_pair = _normalized_number(preference.get("target"), unit)
        if not number_pair or not lower_pair or not target_pair or len({number_pair[0], lower_pair[0], target_pair[0]}) != 1 or target_pair[1] <= lower_pair[1]:
            return {"id": preference.get("id"), "function": function, "score": 0.0, "status": "unknown", "preferred_met": False, "unknown_reason": "invalid_numeric_anchor"}
        number, lower, target = number_pair[1], lower_pair[1], target_pair[1]
        score = _clip((number - lower) / (target - lower))
        preferred_met = score >= 1.0
    elif function == "numeric_decreasing_saturated_v1":
        unit = preference.get("unit")
        number_pair = _normalized_number(value, unit)
        target_pair = _normalized_number(preference.get("target"), unit)
        upper_pair = _normalized_number(preference.get("maximum"), unit)
        if not number_pair or not target_pair or not upper_pair or len({number_pair[0], target_pair[0], upper_pair[0]}) != 1 or upper_pair[1] <= target_pair[1]:
            return {"id": preference.get("id"), "function": function, "score": 0.0, "status": "unknown", "preferred_met": False, "unknown_reason": "invalid_numeric_anchor"}
        number, target, upper = number_pair[1], target_pair[1], upper_pair[1]
        score = _clip((upper - number) / (upper - target))
        preferred_met = score >= 1.0
    elif function == "boolean_preference_v1":
        expected = _boolean_value(preference.get("preferred", True))
        actual = _boolean_value(value)
        if expected is None or actual is None:
            return {"id": preference.get("id"), "function": function, "score": 0.0, "status": "unknown", "preferred_met": False, "unknown_reason": "invalid_boolean_fact"}
        score, preferred_met = (1.0, True) if actual == expected else (0.0, False)
    else:
        return {"id": preference.get("id"), "function": function, "score": 0.0, "status": "unknown", "preferred_met": False, "unknown_reason": "unsupported_function"}
    return {"id": preference.get("id"), "function": function, "score": _clip(score), "status": "pass", "preferred_met": preferred_met}


def _severity(requirement: Mapping[str, Any], result: Mapping[str, Any]) -> tuple[str, float]:
    configured = requirement.get("severity") or {}
    if configured.get("rule_id") and _finite_number(configured.get("value")) is not None:
        return str(configured["rule_id"]), float(configured["value"])
    field = str(result.get("field") or "option")
    rule_id, value = SEVERITY_RULES.get(field, SEVERITY_RULES["option"])
    return rule_id, value


def _waste_cost(*, evidence: Any, redundant_actions: int = 0, invalid_actions: int = 0, waste_cost: float | None = None) -> float:
    if waste_cost is not None:
        value = _finite_number(waste_cost)
        return _clip(value or 0.0, 0.0, 0.05)
    if isinstance(evidence, Mapping):
        redundant_actions = int(evidence.get("redundant_actions", redundant_actions) or 0)
        invalid_actions = int(evidence.get("invalid_actions", invalid_actions) or 0)
    return min(0.05, 0.01 * max(0, redundant_actions) + 0.01 * max(0, invalid_actions))


@dataclass(frozen=True)
class RewardResult:
    reward: float
    reward_type: str
    reward_valid: bool
    termination_reason: str
    target_asin_match: bool
    hard_gates: dict
    weighted_score: float
    evidence: dict
    contract_hash: str | None = None
    purchase_success: bool = False
    acceptable_purchase: bool = False
    strict_success: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "reward_version": REWARD_VERSION,
                "terminal_utility": float(self.reward),
                "purchase_success": bool(self.purchase_success),
                "acceptable_purchase": bool(self.acceptable_purchase),
                "strict_success": bool(self.strict_success),
                "sampling_invalid": not bool(self.reward_valid),
                "evidence_coverage": float(self.evidence.get("evidence_coverage", 0.0)),
                "preference_satisfaction": self.evidence.get("preference_satisfaction", {}),
                "price_utility": float(self.evidence.get("price_utility", 1.0)),
                "violations": self.evidence.get("violations", []),
                "violation_severity": float(self.evidence.get("violation_severity", 0.0)),
                "violation_assessment_complete": bool(self.evidence.get("violation_assessment_complete", True)),
                "waste_cost": float(self.evidence.get("waste_cost", 0.0)),
                "requirement_verdicts": self.evidence.get("requirement_verdicts", []),
                "evidence_refs": self.evidence.get("evidence_refs", []),
                "invalid_reason": self.evidence.get("invalid_reason"),
                "reward_components": self.evidence.get("reward_components", {}),
            }
        )
        return payload


def _legacy_contract(goal: Mapping[str, Any]) -> dict:
    """Create a reviewed v4 contract from the existing v3 feature fields."""
    must = [
        {"id": "category", "field": "category", "operator": "is_a", "value": goal.get("category"), "severity": {"rule_id": "wrong_category_v1", "value": 1.0}},
    ]
    for field, values, severity in (
        ("core_function", goal.get("expected_core_functions") or [], {"rule_id": "missing_core_function_v1", "value": 0.9}),
        ("option", goal.get("required_options_by_key") or {}, {"rule_id": "wrong_option_v1", "value": 0.3}),
    ):
        if field == "option":
            for axis, item in values.items():
                must.append({"id": f"option_{axis}", "field": "option", "axis": axis, "operator": "eq", "value": item.get("value") if isinstance(item, Mapping) else item, "severity": severity})
        else:
            for index, item in enumerate(values):
                must.append({"id": f"{field}_{index}", "field": field, "operator": "eq", "value": item, "severity": severity})
    for index, value in enumerate(goal.get("expected_brand") or []):
        must.append({"id": f"brand_{index}", "field": "brand", "operator": "eq", "value": value, "severity": {"rule_id": "wrong_brand_v1", "value": 0.6}})
    for index, value in enumerate(goal.get("expected_model") or []):
        must.append({"id": f"model_{index}", "field": "model", "operator": "eq", "value": value, "severity": {"rule_id": "wrong_model_v1", "value": 0.8}})
    upper = goal.get("price_upper")
    if upper is not None:
        must.append({"id": "budget", "field": "final_variant_price", "operator": "le", "value": upper, "severity": {"rule_id": "budget_excess_v1"}})
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "approved",
        "task_mode": "single_product_purchase",
        "must": must,
        "preferences": [],
        "price_objective": {"active": False},
        "audit": {"rule_version": PREFERENCE_RULE_VERSION, "severity_version": SEVERITY_VERSION},
    }


def score_purchase(
    product: Mapping[str, Any],
    goal: Mapping[str, Any],
    *,
    selected_options: Any,
    price_resolution: Mapping[str, Any] | None = None,
    price: Any = None,
    evidence: Mapping[str, Any] | None = None,
    rewards: Mapping[str, Any] | None = None,
    redundant_actions: int = 0,
    invalid_actions: int = 0,
) -> RewardResult:
    if not isinstance(goal, Mapping):
        return RewardResult(0.0, "reward_unverifiable", False, "reward_unverifiable", False, {}, 0.0, {"invalid_reason": "goal_not_object"}, None)
    contract = _contract_from_goal(goal)
    invalid_contract_reason = _contract_invalid_reason(contract)
    if invalid_contract_reason:
        return RewardResult(0.0, "reward_unverifiable", False, "reward_unverifiable", False, {}, 0.0, {"invalid_reason": invalid_contract_reason}, None)
    audit = contract.get("audit") or {}
    computed_contract_hash = _contract_hash(contract)
    recorded_contract_hash = audit.get("contract_hash") if isinstance(audit, Mapping) else None
    if recorded_contract_hash is not None and str(recorded_contract_hash) != computed_contract_hash:
        return RewardResult(
            0.0,
            "reward_unverifiable",
            False,
            "reward_unverifiable",
            False,
            {},
            0.0,
            {
                "invalid_reason": "contract_hash_mismatch",
                "expected_contract_hash": computed_contract_hash,
                "recorded_contract_hash": str(recorded_contract_hash),
            },
            str(recorded_contract_hash),
        )
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if price_resolution is None:
        if price is not None:
            numeric = _finite_number(price)
            price_resolution = {"status": PASS if numeric is not None and numeric >= 0 else UNVERIFIABLE, "price": numeric, "version": VARIANT_PRICE_VERSION, "method": "explicit_price"}
        else:
            price_resolution = resolve_variant_price(dict(product), selected_options)
    requirement_verdicts = []
    violations = []
    fact_unknown = False
    evidence_missing = False
    for requirement in contract.get("must") or []:
        if not isinstance(requirement, Mapping):
            fact_unknown = True
            continue
        result = _requirement_result(requirement, product, selected_options, price_resolution)
        covered = _evidence_covers(result, product, selected_options, evidence)
        result["evidence_covered"] = bool(covered)
        requirement_verdicts.append(result)
        if result["status"] == FAIL:
            rule_id, severity = _severity(requirement, result)
            if result.get("field") in {"final_variant_price", "budget", "price"}:
                actual = _finite_number(result.get("actual"))
                upper = _finite_number(requirement.get("value"))
                unit_definition = _unit_definition(requirement.get("unit"))
                if unit_definition == ("currency_fen", 1.0) and upper is not None:
                    upper /= 100.0
                if actual is not None and upper and actual > upper:
                    severity = 0.25 + 0.75 * min(1.0, (actual - upper) / (0.20 * upper))
            violations.append({"requirement_id": result["requirement_id"], "field": result.get("field"), "rule_id": rule_id, "severity": severity, "actual": result.get("actual"), "required": result.get("required")})
        elif result["status"] == UNVERIFIABLE:
            fact_unknown = True
        elif not covered:
            evidence_missing = True
    hard_failures = [item for item in requirement_verdicts if item.get("status") == FAIL]
    severity = max((float(item["severity"]) for item in violations), default=0.0)
    waste = _waste_cost(evidence=evidence, redundant_actions=redundant_actions, invalid_actions=invalid_actions)
    asin_match = str(product.get("asin")) == str(goal.get("asin"))
    preference_items = [
        _preference_score(item, product, selected_options, evidence)
        for item in contract.get("preferences") or []
        if isinstance(item, Mapping)
    ]
    pref_score = min((float(item.get("score", 0.0)) for item in preference_items), default=1.0)
    all_preferred = all(bool(item.get("preferred_met")) for item in preference_items)
    price_objective = contract.get("price_objective") or {}
    budget = _finite_number(price_objective.get("reference_budget"))
    actual_price = _finite_number(price_resolution.get("price")) if isinstance(price_resolution, Mapping) else None
    if price_objective.get("active"):
        price_utility = 0.0 if budget is None or budget <= 0 or actual_price is None else _clip((budget - actual_price) / budget)
    else:
        price_utility = 1.0
    evidence_refs = _facts_from_evidence(evidence)
    evidence_coverage = (sum(1 for item in requirement_verdicts if item.get("evidence_covered")) / len(requirement_verdicts)) if requirement_verdicts else (0.0 if evidence is not None else 1.0)
    critical_missing = evidence is not None and evidence_missing
    if hard_failures:
        reward_type = "wrong_purchase"
        reward_valid = True
        reward = max(-1.0, min(1.0, float(values.get("wrong_base", -0.60)) - float(values.get("wrong_severity_weight", 0.40)) * severity - waste))
        invalid_reason = None
    elif fact_unknown or critical_missing or not requirement_verdicts:
        reward_type = "reward_unverifiable" if fact_unknown or not requirement_verdicts else "unverified_purchase"
        reward_valid = reward_type != "reward_unverifiable"
        reward = float(values.get("reward_unverifiable", 0.0) if not reward_valid else values.get("unverified_purchase", -0.60) - waste)
        invalid_reason = "required_fact_or_evidence_missing" if not reward_valid else None
    else:
        gold = asin_match and all_preferred
        if gold:
            reward_type = "gold_purchase"
        elif all_preferred:
            reward_type = "valid_alternative_purchase"
        else:
            reward_type = "acceptable_compromise_purchase"
        reward_valid = True
        gold_indicator = 1.0 if reward_type == "gold_purchase" else 0.0
        reward = float(values.get("purchase_completion", 0.80) + values.get("preference_weight", 0.10) * pref_score + values.get("price_weight", 0.05) * price_utility + values.get("gold_weight", 0.05) * gold_indicator - waste)
        reward = max(-1.0, min(1.0, reward))
        invalid_reason = None
    if reward_type in {"gold_purchase", "valid_alternative_purchase", "acceptable_compromise_purchase"}:
        components = {
            "completion": float(values.get("purchase_completion", 0.80)),
            "preference": float(values.get("preference_weight", 0.10)) * pref_score,
            "price": float(values.get("price_weight", 0.05)) * price_utility,
            "gold": float(values.get("gold_weight", 0.05)) if reward_type == "gold_purchase" else 0.0,
            "waste": -waste,
        }
    elif reward_type == "wrong_purchase":
        components = {
            "wrong_base": float(values.get("wrong_base", -0.60)),
            "severity": -float(values.get("wrong_severity_weight", 0.40)) * severity,
            "waste": -waste,
        }
    elif reward_type == "unverified_purchase":
        components = {
            "unverified": float(values.get("unverified_purchase", -0.60)),
            "waste": -waste,
        }
    else:
        components = {"placeholder": 0.0}
    return RewardResult(
        reward=reward,
        reward_type=reward_type,
        reward_valid=reward_valid,
        termination_reason=reward_type,
        target_asin_match=asin_match,
        hard_gates={item["requirement_id"]: item for item in requirement_verdicts},
        weighted_score=float(pref_score),
        evidence={
            "contract_version": contract.get("contract_version"),
            "contract_hash": (contract.get("audit") or {}).get("contract_hash") or _contract_hash(contract),
            "requirement_verdicts": requirement_verdicts,
            "preference_satisfaction": {"aggregate": pref_score, "items": preference_items},
            "price_utility": price_utility,
            "price_resolution": dict(price_resolution),
            "evidence_coverage": evidence_coverage,
            "evidence_refs": evidence_refs,
            "violations": violations,
            "violation_severity": severity,
            "violation_assessment_complete": not fact_unknown,
            "waste_cost": waste,
            "reward_components": components,
            "invalid_reason": invalid_reason,
            "reward_feature_version": REWARD_FEATURE_VERSION,
            "comparator_version": COMPARATOR_VERSION,
        },
        contract_hash=(contract.get("audit") or {}).get("contract_hash") or _contract_hash(contract),
        purchase_success=reward_type in {"gold_purchase", "valid_alternative_purchase"},
        acceptable_purchase=reward_type in {"gold_purchase", "valid_alternative_purchase", "acceptable_compromise_purchase"},
        strict_success=reward_type == "gold_purchase" and reward_valid,
    )


def score_abstain(*, effective_result_sets: int, candidate_quality: float = 0.0, known_acceptable_candidates: int = 0, waste_cost: float = 0.0, rewards: Mapping[str, Any] | None = None) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    quality = _clip(candidate_quality)
    if known_acceptable_candidates:
        reward_type, base = "stop_with_acceptable_candidate", float(values["stop_with_acceptable_candidate"])
    elif quality > 0:
        reward_type, base = "graceful_stop", float(values["stop_base"]) + float(values["stop_quality_weight"]) * quality
    else:
        reward_type, base = "early_abstain", float(values["stop_base"])
    waste = min(0.05, max(0.0, float(waste_cost)))
    return RewardResult(base - waste, reward_type, True, reward_type, False, {}, 0.0, {"candidate_quality": quality, "effective_result_sets": int(effective_result_sets), "waste_cost": waste, "reward_components": {"base": base, "waste": -waste}}, acceptable_purchase=False)


def fixed_termination(reason: str, rewards: Mapping[str, Any] | None = None) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if reason not in {"repeat_loop", "max_steps"}:
        raise ValueError(f"unsupported fixed termination reason: {reason}")
    return RewardResult(float(values[reason]), reason, True, reason, False, {}, 0.0, {"reward_components": {"base": float(values[reason])}})


__all__ = ["REWARD_VERSION", "CONTRACT_VERSION", "DEFAULT_REWARDS", "RewardResult", "is_v4_goal", "score_purchase", "score_abstain", "fixed_termination"]
