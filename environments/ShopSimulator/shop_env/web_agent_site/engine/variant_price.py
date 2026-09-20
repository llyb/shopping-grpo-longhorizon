"""Conservative option matching and variant-price resolution for ShopSimulator."""

from __future__ import annotations

import math

from web_agent_site.engine.comparators import (
    FAIL,
    PASS,
    UNVERIFIABLE,
    comparison,
)
from web_agent_site.engine.reward_features import (
    canonicalize_option_axis,
    normalize_option_text,
)


VARIANT_PRICE_VERSION = "variant-price-v1"


def _finite_price(value):
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price >= 0 else None


def option_axes(product: dict) -> dict[str, dict]:
    """Return canonical axes while preserving ambiguity as explicit evidence."""
    axes = {}
    collisions = {}
    raw = product.get("customization_options") or {}
    if not isinstance(raw, dict):
        raw = {}
    for raw_axis, entries in raw.items():
        canonical = canonicalize_option_axis(raw_axis)
        values = {}
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            normalized = normalize_option_text(entry.get("value"))
            if normalized:
                values[normalized] = {
                    "value": str(entry.get("value")),
                    "price": _finite_price(entry.get("price")),
                }
        record = {
            "canonical_axis": canonical,
            "source_axis": str(raw_axis),
            "values": values,
        }
        if canonical in axes:
            collisions.setdefault(canonical, [axes[canonical]["source_axis"]])
            collisions[canonical].append(str(raw_axis))
        else:
            axes[canonical] = record
    for canonical in collisions:
        axes.pop(canonical, None)
    return {"axes": axes, "collisions": collisions}


def _canonical_selected(selected_options: object) -> tuple[dict, list[dict]]:
    if not isinstance(selected_options, dict):
        return {}, [{"code": "selected_options_not_object"}]
    selected = {}
    errors = []
    for raw_axis, raw_value in selected_options.items():
        canonical = canonicalize_option_axis(raw_axis)
        if canonical in selected:
            errors.append(
                {
                    "code": "selected_axis_collision",
                    "axis": canonical,
                }
            )
            continue
        selected[canonical] = {
            "source_axis": str(raw_axis),
            "value": str(raw_value),
        }
    return selected, errors


def compare_required_options(
    product: dict,
    required_options_by_key: object,
    selected_options: object,
) -> dict:
    if not isinstance(required_options_by_key, dict):
        return comparison(
            UNVERIFIABLE,
            comparator="option_key_value_v1",
            required=required_options_by_key,
            actual=selected_options,
            source_field="customization_options",
        )
    indexed = option_axes(product)
    selected, errors = _canonical_selected(selected_options)
    if indexed["collisions"] or errors:
        return comparison(
            UNVERIFIABLE,
            comparator="option_key_value_v1",
            required=required_options_by_key,
            actual=selected_options,
            source_field="customization_options",
            evidence={
                "axis_collisions": indexed["collisions"],
                "selection_errors": errors,
            },
        )
    missing_axes = []
    unavailable_values = []
    wrong_values = []
    matched = []
    for canonical_axis, requirement in required_options_by_key.items():
        required_value = (
            requirement.get("value")
            if isinstance(requirement, dict)
            else requirement
        )
        product_axis = indexed["axes"].get(canonical_axis)
        if product_axis is None:
            missing_axes.append(canonical_axis)
            continue
        normalized_required = normalize_option_text(required_value)
        if normalized_required not in product_axis["values"]:
            unavailable_values.append(
                {"axis": canonical_axis, "value": required_value}
            )
            continue
        selection = selected.get(canonical_axis)
        if selection is None:
            wrong_values.append(
                {
                    "axis": canonical_axis,
                    "required": required_value,
                    "selected": None,
                }
            )
            continue
        if normalize_option_text(selection["value"]) != normalized_required:
            wrong_values.append(
                {
                    "axis": canonical_axis,
                    "required": required_value,
                    "selected": selection["value"],
                }
            )
            continue
        matched.append(
            {
                "axis": canonical_axis,
                "value": required_value,
                "source_axis": product_axis["source_axis"],
            }
        )
    if missing_axes:
        status = UNVERIFIABLE
    elif unavailable_values or wrong_values:
        status = FAIL
    else:
        status = PASS
    return comparison(
        status,
        comparator="option_key_value_v1",
        required=required_options_by_key,
        actual=selected_options,
        source_field="customization_options",
        evidence={
            "matched": matched,
            "missing_axes": missing_axes,
            "unavailable_values": unavailable_values,
            "wrong_values": wrong_values,
            "axis_version": "option-axis-v1",
        },
    )


def matching_candidate_options(
    product: dict,
    required_options_by_key: object,
) -> tuple[dict, dict]:
    """Select only values explicitly required by the task contract."""
    if not isinstance(required_options_by_key, dict):
        return {}, {"status": UNVERIFIABLE, "reason": "missing_contract"}
    indexed = option_axes(product)
    if indexed["collisions"]:
        return {}, {
            "status": UNVERIFIABLE,
            "reason": "candidate_axis_collision",
            "collisions": indexed["collisions"],
        }
    selected = {}
    for canonical_axis, requirement in required_options_by_key.items():
        required_value = (
            requirement.get("value")
            if isinstance(requirement, dict)
            else requirement
        )
        axis = indexed["axes"].get(canonical_axis)
        if axis is None:
            return {}, {
                "status": UNVERIFIABLE,
                "reason": "candidate_axis_missing",
                "axis": canonical_axis,
            }
        normalized = normalize_option_text(required_value)
        entry = axis["values"].get(normalized)
        if entry is None:
            return {}, {
                "status": FAIL,
                "reason": "candidate_required_value_missing",
                "axis": canonical_axis,
                "value": required_value,
            }
        selected[axis["source_axis"]] = entry["value"]
    return selected, {"status": PASS}


def candidate_options_for_evaluation(
    product: dict,
    required_options_by_key: object,
) -> tuple[dict, dict]:
    """Choose a deterministic purchasable variant without using Agent actions."""
    selected, resolution = matching_candidate_options(
        product,
        required_options_by_key,
    )
    if resolution["status"] != PASS:
        return selected, resolution
    indexed = option_axes(product)
    if indexed["collisions"]:
        return selected, {
            "status": UNVERIFIABLE,
            "reason": "candidate_axis_collision",
            "collisions": indexed["collisions"],
        }
    selected_canonical, errors = _canonical_selected(selected)
    if errors:
        return selected, {
            "status": UNVERIFIABLE,
            "reason": "candidate_selection_invalid",
            "errors": errors,
        }
    filled = dict(selected)
    inferred = []
    for canonical_axis, axis in indexed["axes"].items():
        if canonical_axis in selected_canonical:
            continue
        entries = list(axis["values"].values())
        if not entries:
            return selected, {
                "status": UNVERIFIABLE,
                "reason": "candidate_axis_has_no_values",
                "axis": canonical_axis,
            }
        chosen = min(
            entries,
            key=lambda entry: (
                entry["price"] is None,
                entry["price"] if entry["price"] is not None else math.inf,
                normalize_option_text(entry["value"]),
            ),
        )
        filled[axis["source_axis"]] = chosen["value"]
        inferred.append(
            {
                "axis": canonical_axis,
                "source_axis": axis["source_axis"],
                "value": chosen["value"],
                "method": "lowest_listed_price_then_lexical",
            }
        )
    return filled, {
        "status": PASS,
        "inferred_options": inferred,
    }


def _combination_price(product: dict, selected: dict) -> float | None:
    variants = product.get("variant_combinations")
    if not isinstance(variants, list):
        return None
    normalized_selected = {
        canonicalize_option_axis(key): normalize_option_text(value)
        for key, value in selected.items()
    }
    for variant in variants:
        if not isinstance(variant, dict) or not isinstance(
            variant.get("options"), dict
        ):
            continue
        normalized_variant = {
            canonicalize_option_axis(key): normalize_option_text(value)
            for key, value in variant["options"].items()
        }
        if normalized_variant == normalized_selected:
            return _finite_price(variant.get("price"))
    return None


def resolve_variant_price(product: dict, selected_options: object) -> dict:
    selected, errors = _canonical_selected(selected_options)
    indexed = option_axes(product)
    if errors or indexed["collisions"]:
        return {
            "status": UNVERIFIABLE,
            "price": None,
            "version": VARIANT_PRICE_VERSION,
            "method": "invalid_option_axes",
            "evidence": {
                "selection_errors": errors,
                "axis_collisions": indexed["collisions"],
            },
        }
    raw_selected = {
        value["source_axis"]: value["value"] for value in selected.values()
    }
    combination = _combination_price(product, raw_selected)
    if combination is not None:
        return {
            "status": PASS,
            "price": combination,
            "version": VARIANT_PRICE_VERSION,
            "method": "full_variant_combination",
            "evidence": {},
        }

    effective_axes = []
    all_prices = []
    for canonical_axis, axis in indexed["axes"].items():
        prices = {
            entry["price"]
            for entry in axis["values"].values()
            if entry["price"] is not None
        }
        all_prices.extend(prices)
        if len(prices) > 1:
            effective_axes.append(canonical_axis)
    if len(effective_axes) > 1:
        return {
            "status": UNVERIFIABLE,
            "price": None,
            "version": VARIANT_PRICE_VERSION,
            "method": "multiple_effective_price_axes",
            "evidence": {"effective_price_axes": effective_axes},
        }
    if len(effective_axes) == 1:
        price_axis = effective_axes[0]
        selection = selected.get(price_axis)
        axis = indexed["axes"][price_axis]
        if selection is None:
            return {
                "status": UNVERIFIABLE,
                "price": None,
                "version": VARIANT_PRICE_VERSION,
                "method": "effective_price_axis_unselected",
                "evidence": {"effective_price_axis": price_axis},
            }
        entry = axis["values"].get(normalize_option_text(selection["value"]))
        price = entry["price"] if entry else None
        if price is None:
            return {
                "status": UNVERIFIABLE,
                "price": None,
                "version": VARIANT_PRICE_VERSION,
                "method": "selected_variant_price_missing",
                "evidence": {"effective_price_axis": price_axis},
            }
        return {
            "status": PASS,
            "price": price,
            "version": VARIANT_PRICE_VERSION,
            "method": "unique_effective_price_axis",
            "evidence": {"effective_price_axis": price_axis},
        }

    unique_prices = {price for price in all_prices if price is not None}
    if len(unique_prices) == 1:
        return {
            "status": PASS,
            "price": next(iter(unique_prices)),
            "version": VARIANT_PRICE_VERSION,
            "method": "constant_option_price",
            "evidence": {},
        }
    pricing = [
        price
        for price in (_finite_price(value) for value in product.get("pricing") or [])
        if price is not None
    ]
    if not indexed["axes"] and len(set(pricing)) == 1:
        return {
            "status": PASS,
            "price": pricing[0],
            "version": VARIANT_PRICE_VERSION,
            "method": "single_base_price",
            "evidence": {},
        }
    return {
        "status": UNVERIFIABLE,
        "price": None,
        "version": VARIANT_PRICE_VERSION,
        "method": "price_not_deterministic",
        "evidence": {"candidate_prices": sorted(unique_prices)},
    }
