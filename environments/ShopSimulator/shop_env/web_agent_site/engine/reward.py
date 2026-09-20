"""Reward entrypoints with v3 input compatibility and v4 as the environment path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from web_agent_site.engine.comparators import (
    COMPARATOR_VERSION,
    FAIL,
    PASS,
    UNVERIFIABLE,
    compare_category,
    compare_core_functions,
    compare_model,
    comparison,
    load_brand_aliases,
    normalize_text,
)
from web_agent_site.engine.reward_features import (
    REWARD_FEATURE_VERSION,
    normalize_option_text,
)
from web_agent_site.engine.variant_price import (
    VARIANT_PRICE_VERSION,
    candidate_options_for_evaluation,
    compare_required_options,
    resolve_variant_price,
)
from web_agent_site.engine.reward_v4 import (
    REWARD_VERSION as REWARD_V4_VERSION,
    is_v4_goal,
    score_abstain as _score_abstain_v4,
    score_purchase as _score_purchase_v4,
    fixed_termination as _fixed_termination_v4,
)


REWARD_V3_VERSION = "shopsimulator-reward-v3"
# The environment config imports this symbol, so v4 is the single active
# runtime contract.  Legacy callers are dispatched by the absence of a v4
# requirement contract and still receive Reward v3 diagnostics.
REWARD_VERSION = REWARD_V4_VERSION
DIMENSION_WEIGHTS = {
    "brand": 0.35,
    "model": 0.25,
    "core_functions": 0.25,
    "key_options": 0.15,
}
DEFAULT_REWARDS = {
    "gold_purchase": 1.0,
    "valid_alternative_purchase": 0.55,
    "partial_purchase_base": -0.30,
    "partial_purchase_scale": 0.55,
    "partial_purchase_cap": 0.25,
    "graceful_stop": -0.15,
    "early_abstain": -0.35,
    "max_steps": -0.50,
    "repeat_loop": -0.65,
    "wrong_purchase": -0.85,
    "reward_unverifiable": 0.0,
}
KNOWN_ACCEPTABLE_MATCH_THRESHOLD = 0.70
KNOWN_ACCEPTABLE_COVERAGE_THRESHOLD = 0.75


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

    def to_dict(self):
        payload = asdict(self)
        preference_scoring = self.evidence.get("preference_scoring") or {}
        dimensions = preference_scoring.get("dimensions") or {}
        payload.update(
            {
                "reward_version": REWARD_V3_VERSION,
                "terminal_utility": self.reward,
                "purchase_success": self.reward_type
                in {"gold_purchase", "valid_alternative_purchase"},
                "sampling_invalid": not self.reward_valid,
                "evidence_coverage": float(
                    preference_scoring.get("evidence_coverage", 0.0)
                ),
                "dimension_scores": {
                    name: float(result.get("score", 0.0))
                    for name, result in dimensions.items()
                },
            }
        )
        return payload


def _price_gate(price_resolution: object, goal: dict) -> dict:
    upper = goal.get("price_upper")
    if upper is None:
        return comparison(
            PASS,
            comparator="budget_not_declared_v1",
            required=None,
            actual=(
                price_resolution.get("price")
                if isinstance(price_resolution, dict)
                else None
            ),
            source_field="instruction",
            evidence=price_resolution,
        )
    if (
        not isinstance(price_resolution, dict)
        or price_resolution.get("status") != PASS
    ):
        return comparison(
            UNVERIFIABLE,
            comparator="variant_price_budget_v1",
            required=upper,
            actual=(
                price_resolution.get("price")
                if isinstance(price_resolution, dict)
                else None
            ),
            source_field="variant_price",
            evidence=price_resolution,
        )
    try:
        actual = float(price_resolution["price"])
        limit = float(upper)
    except (KeyError, TypeError, ValueError):
        status = UNVERIFIABLE
        actual = price_resolution.get("price")
        limit = upper
    else:
        if (
            not math.isfinite(actual)
            or not math.isfinite(limit)
            or actual < 0
            or limit <= 0
        ):
            status = UNVERIFIABLE
        else:
            status = PASS if actual <= limit else FAIL
    return comparison(
        status,
        comparator="variant_price_budget_v1",
        required=limit,
        actual=actual,
        source_field="variant_price",
        evidence=price_resolution,
    )


def _brand_gate(expected: str, product: dict) -> dict:
    aliases = load_brand_aliases()
    canonical = aliases.get(normalize_text(expected), normalize_text(expected))
    candidate_brand_text = normalize_text(
        " ".join(
            str(value)
            for value in (
                product.get("brand"),
                product.get("shop_name"),
            )
            if value
        )
    )
    candidate_aliases = {
        candidate_canonical
        for alias, candidate_canonical in aliases.items()
        if len(alias) >= 2 and alias in candidate_brand_text
    }
    matched = (
        canonical in candidate_aliases
        or normalize_text(expected) in candidate_brand_text
    )
    return comparison(
        PASS if matched else FAIL,
        comparator="explicit_brand_alias_text_v1",
        required=expected,
        actual=candidate_brand_text,
        source_field="brand|shop_name",
        evidence={"alias_match": matched},
    )


def _score_requirements(requirements, evaluator) -> dict:
    results = [evaluator(requirement) for requirement in requirements]
    total = len(results)
    passed = sum(result["status"] == PASS for result in results)
    verifiable = sum(
        result["status"] != UNVERIFIABLE for result in results
    )
    return {
        "active": total > 0,
        "score": passed / total if total else 0.0,
        "coverage": verifiable / total if total else 0.0,
        "required_count": total,
        "passed_count": passed,
        "verifiable_count": verifiable,
        "results": results,
    }


def _option_dimension(
    product: dict,
    goal: dict,
    selected_options: object,
) -> dict:
    required = goal.get("required_options_by_key") or {}
    unresolved = goal.get("unresolved_option_requirements") or []
    results = [
        compare_required_options(
            product,
            {axis: requirement},
            selected_options,
        )
        for axis, requirement in required.items()
    ]
    selected_values = {
        normalize_option_text(value)
        for value in (
            selected_options.values()
            if isinstance(selected_options, dict)
            else ()
        )
    }
    results.extend(
        comparison(
            (
                PASS
                if normalize_option_text(item.get("value"))
                in selected_values
                else FAIL
            ),
            comparator="exact_option_value_fallback_v1",
            required=item.get("value"),
            actual=list(
                selected_options.values()
                if isinstance(selected_options, dict)
                else ()
            ),
            source_field="instruction_options",
            evidence=item,
        )
        for item in unresolved
    )
    total = len(results)
    passed = sum(result["status"] == PASS for result in results)
    verifiable = sum(
        result["status"] != UNVERIFIABLE for result in results
    )
    return {
        "active": total > 0,
        "score": passed / total if total else 0.0,
        "coverage": verifiable / total if total else 0.0,
        "required_count": total,
        "passed_count": passed,
        "verifiable_count": verifiable,
        "results": results,
    }


def _preference_dimensions(
    product: dict,
    goal: dict,
    selected_options: object,
) -> dict:
    dimensions = {
        "brand": _score_requirements(
            goal.get("expected_brand") or [],
            lambda value: _brand_gate(value, product),
        ),
        "model": _score_requirements(
            goal.get("expected_model") or [],
            lambda value: compare_model(value, product),
        ),
        "core_functions": _score_requirements(
            goal.get("expected_core_functions") or [],
            lambda value: compare_core_functions([value], product),
        ),
        "key_options": _option_dimension(
            product,
            goal,
            selected_options,
        ),
    }
    active_weight = sum(
        DIMENSION_WEIGHTS[name]
        for name, result in dimensions.items()
        if result["active"]
    )
    if active_weight:
        match_score = sum(
            DIMENSION_WEIGHTS[name] * result["score"]
            for name, result in dimensions.items()
            if result["active"]
        ) / active_weight
        evidence_coverage = sum(
            DIMENSION_WEIGHTS[name] * result["coverage"]
            for name, result in dimensions.items()
            if result["active"]
        ) / active_weight
    else:
        match_score = 1.0
        evidence_coverage = 1.0
    all_satisfied = (
        math.isclose(match_score, 1.0, abs_tol=1.0e-8)
        and math.isclose(evidence_coverage, 1.0, abs_tol=1.0e-8)
    )
    return {
        "weights": DIMENSION_WEIGHTS,
        "active_weight": active_weight,
        "match_score": match_score,
        "evidence_coverage": evidence_coverage,
        "all_satisfied": all_satisfied,
        "dimensions": dimensions,
    }


def _evaluate(
    product: dict,
    goal: dict,
    *,
    selected_options: object,
    price_resolution: object,
) -> tuple[dict, dict]:
    hard_gates = {
        "category": compare_category(goal.get("category"), product),
        "budget": _price_gate(price_resolution, goal),
    }
    preferences = _preference_dimensions(
        product,
        goal,
        selected_options,
    )
    return hard_gates, preferences


def _explicit_price_resolution(price: object) -> dict:
    try:
        value = float(price)
    except (TypeError, ValueError):
        value = None
    return {
        "status": (
            PASS
            if value is not None
            and math.isfinite(value)
            and value >= 0
            else UNVERIFIABLE
        ),
        "price": value,
        "version": VARIANT_PRICE_VERSION,
        "method": "explicit_test_price",
        "evidence": {},
    }


def _evaluate_purchase_v3(
    product: dict,
    goal: dict,
    *,
    selected_options: object,
    price_resolution: dict | None = None,
    price: object = None,
    rewards: dict[str, float] | None = None,
) -> RewardResult:
    """Score a purchase using two hard gates and continuous soft matching."""
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if price_resolution is None:
        price_resolution = (
            _explicit_price_resolution(price)
            if price is not None
            else resolve_variant_price(product, selected_options)
        )
    hard_gates, preferences = _evaluate(
        product,
        goal,
        selected_options=selected_options,
        price_resolution=price_resolution,
    )
    asin_match = str(product.get("asin")) == str(goal.get("asin"))
    hard_unverifiable = any(
        gate["status"] == UNVERIFIABLE for gate in hard_gates.values()
    )
    hard_failed = any(
        gate["status"] == FAIL for gate in hard_gates.values()
    )
    if hard_unverifiable:
        reward_type = "reward_unverifiable"
        reward_valid = False
        reward = values[reward_type]
    elif hard_failed:
        reward_type = "wrong_purchase"
        reward_valid = True
        reward = values[reward_type]
    elif preferences["all_satisfied"]:
        reward_type = (
            "gold_purchase" if asin_match else "valid_alternative_purchase"
        )
        reward_valid = True
        reward = values[reward_type]
    else:
        reward_type = "partial_alternative_purchase"
        reward_valid = True
        reward = min(
            values["partial_purchase_cap"],
            values["partial_purchase_base"]
            + values["partial_purchase_scale"]
            * preferences["match_score"],
        )
    return RewardResult(
        reward=float(reward),
        reward_type=reward_type,
        reward_valid=reward_valid,
        termination_reason=reward_type,
        target_asin_match=asin_match,
        hard_gates=hard_gates,
        weighted_score=float(preferences["match_score"]),
        evidence={
            "reward_feature_version": goal.get("reward_feature_version"),
            "expected_reward_feature_version": REWARD_FEATURE_VERSION,
            "comparator_version": COMPARATOR_VERSION,
            "variant_price_version": VARIANT_PRICE_VERSION,
            "price_resolution": price_resolution,
            "preference_scoring": preferences,
        },
    )


def _evaluate_candidate_eligibility_v3(product: dict, goal: dict) -> dict:
    """Determine whether an opened candidate is sufficiently acceptable to block abstention."""
    selected, option_resolution = candidate_options_for_evaluation(
        product,
        goal.get("required_options_by_key"),
    )
    price_resolution = resolve_variant_price(product, selected)
    hard_gates, preferences = _evaluate(
        product,
        goal,
        selected_options=selected,
        price_resolution=price_resolution,
    )
    hard_pass = all(
        gate["status"] == PASS for gate in hard_gates.values()
    )
    known_acceptable = (
        hard_pass
        and preferences["match_score"]
        >= KNOWN_ACCEPTABLE_MATCH_THRESHOLD
        and preferences["evidence_coverage"]
        >= KNOWN_ACCEPTABLE_COVERAGE_THRESHOLD
    )
    return {
        "status": PASS if known_acceptable else FAIL,
        "known_acceptable": known_acceptable,
        # Compatibility field for the Environment v2.1 session state.
        "known_valid": known_acceptable,
        "selected_options": selected,
        "option_resolution": option_resolution,
        "price_resolution": price_resolution,
        "hard_gates": hard_gates,
        "match_score": preferences["match_score"],
        "evidence_coverage": preferences["evidence_coverage"],
    }


def _evaluate_abstain_v3(
    *,
    effective_result_sets: int,
    opened_candidates: int,
    known_acceptable_candidates: int | None = None,
    known_valid_candidates: int | None = None,
    rewards: dict[str, float] | None = None,
) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    known_count = int(
        known_acceptable_candidates
        if known_acceptable_candidates is not None
        else known_valid_candidates or 0
    )
    eligible = (
        int(effective_result_sets) >= 2
        and int(opened_candidates) >= 2
        and known_count == 0
    )
    reward_type = "graceful_stop" if eligible else "early_abstain"
    return RewardResult(
        reward=float(values[reward_type]),
        reward_type=reward_type,
        reward_valid=True,
        termination_reason=reward_type,
        target_asin_match=False,
        hard_gates={},
        weighted_score=0.0,
        evidence={
            "eligible": eligible,
            "effective_result_sets": int(effective_result_sets),
            "opened_candidates": int(opened_candidates),
            "known_acceptable_candidates": known_count,
        },
    )


def _fixed_termination_v3(
    reason: str,
    rewards: dict[str, float] | None = None,
) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if reason not in {"repeat_loop", "max_steps"}:
        raise ValueError(f"unsupported fixed termination reason: {reason}")
    return RewardResult(
        reward=float(values[reason]),
        reward_type=reason,
        reward_valid=True,
        termination_reason=reason,
        target_asin_match=False,
        hard_gates={},
        weighted_score=0.0,
        evidence={},
    )


def evaluate_purchase(
    product: dict,
    goal: dict,
    *,
    selected_options: object,
    price_resolution: dict | None = None,
    price: object = None,
    rewards: dict | None = None,
    evidence: dict | None = None,
    redundant_actions: int = 0,
    invalid_actions: int = 0,
):
    """Score a terminal purchase under v4 or the explicit legacy path."""
    configured_version = (rewards or {}).get("version")
    if is_v4_goal(goal) or configured_version == REWARD_V4_VERSION:
        return _score_purchase_v4(
            product,
            goal,
            selected_options=selected_options,
            price_resolution=price_resolution,
            price=price,
            evidence=evidence,
            rewards=rewards,
            redundant_actions=redundant_actions,
            invalid_actions=invalid_actions,
        )
    return _evaluate_purchase_v3(
        product,
        goal,
        selected_options=selected_options,
        price_resolution=price_resolution,
        price=price,
        rewards=rewards,
    )


def evaluate_candidate_eligibility(
    product: dict,
    goal: dict,
    evidence: dict | None = None,
    *,
    selected_options: dict | None = None,
    price_resolution: dict | None = None,
) -> dict:
    """Return candidate acceptability while preserving the v3 result keys."""
    if not is_v4_goal(goal):
        return _evaluate_candidate_eligibility_v3(product, goal)
    required_options = goal.get("required_options_by_key")
    if not isinstance(required_options, dict):
        # Hand-authored/legacy callers may keep option requirements only in
        # the frozen contract.  Derive a deterministic choice for exact or
        # finite-set requirements; ambiguous numeric choices stay
        # unverifiable rather than being guessed.
        required_options = {}
        contract = goal.get("reward_contract") or goal.get("contract") or {}
        for item in contract.get("must") or []:
            if not isinstance(item, dict) or item.get("field") not in {"option", "color", "size", "capacity", "storage", "sku_option"}:
                continue
            axis = item.get("axis") or item.get("field")
            value = item.get("value")
            if item.get("operator") in {"eq", "in"} and axis and value not in (None, []):
                required_options[str(axis)] = {"value": value[0] if isinstance(value, list) else value}
    if selected_options is None:
        selected, option_resolution = candidate_options_for_evaluation(product, required_options)
    else:
        selected = dict(selected_options)
        option_resolution = {"status": PASS, "method": "actor_selected_options"}
    if price_resolution is None:
        price_resolution = resolve_variant_price(product, selected)
    result = _score_purchase_v4(
        product,
        goal,
        selected_options=selected,
        price_resolution=price_resolution,
        evidence=evidence,
    )
    # A candidate only becomes known after the actor has received enough
    # evidence for the same SKU.  Backend facts alone must never make a stop
    # action look justified.
    known_acceptable = bool(
        evidence is not None
        and result.acceptable_purchase
        and result.reward_valid
        and float(result.evidence.get("evidence_coverage", 0.0)) >= 1.0
    )
    return {
        "status": PASS if known_acceptable else FAIL,
        "known_acceptable": known_acceptable,
        "known_valid": known_acceptable,
        "selected_options": selected,
        "option_resolution": option_resolution,
        "price_resolution": price_resolution,
        "hard_gates": result.hard_gates,
        "match_score": result.weighted_score,
        "evidence_coverage": float(result.evidence.get("evidence_coverage", 0.0)),
    }


def evaluate_abstain(
    *,
    effective_result_sets: int,
    opened_candidates: int,
    known_acceptable_candidates: int | None = None,
    known_valid_candidates: int | None = None,
    candidate_quality: float | None = None,
    waste_cost: float = 0.0,
    rewards: dict | None = None,
):
    configured_version = (rewards or {}).get("version")
    if configured_version == REWARD_V4_VERSION:
        known_count = int(
            known_acceptable_candidates
            if known_acceptable_candidates is not None
            else known_valid_candidates or 0
        )
        quality = (
            float(candidate_quality)
            if candidate_quality is not None
            else (1.0 if int(opened_candidates) > 0 else 0.0)
        )
        return _score_abstain_v4(
            effective_result_sets=effective_result_sets,
            candidate_quality=quality,
            known_acceptable_candidates=known_count,
            waste_cost=waste_cost,
            rewards=rewards,
        )
    return _evaluate_abstain_v3(
        effective_result_sets=effective_result_sets,
        opened_candidates=opened_candidates,
        known_acceptable_candidates=known_acceptable_candidates,
        known_valid_candidates=known_valid_candidates,
        rewards=rewards,
    )


def fixed_termination(reason: str, rewards: dict | None = None):
    if (rewards or {}).get("version") == REWARD_V4_VERSION:
        return _fixed_termination_v4(reason, rewards=rewards)
    return _fixed_termination_v3(reason, rewards=rewards)
