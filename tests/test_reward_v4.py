"""Rule-level verification for the approved Reward v4 design."""

from __future__ import annotations

import unittest

from web_agent_site.engine.reward_v4 import is_v4_goal, score_abstain, score_purchase
from web_agent_site.engine.reward import evaluate_candidate_eligibility, evaluate_purchase
from web_agent_site.engine.evidence import (
    candidate_quality,
    new_ledger,
    public_ledger,
    record_observation,
)
from web_agent_site.engine.reward_features import compile_requirement_contract


def product(asin="gold", color="白色", capacity="256GB", price=5000):
    return {
        "asin": asin,
        "title": "A型号 256GB 手机",
        "brand": "示例品牌",
        "category": "数码›手机›智能手机",
        "attribute": ["无线充电"],
        "pricing": [price],
        "customization_options": {
            "颜色分类": [{"value": color, "price": price}],
            "容量": [{"value": capacity, "price": price}],
        },
    }


def contract(*, color_values=("白色", "黑色"), budget=5000, preference=True):
    preferences = []
    if preference:
        preferences.append(
            {
                "id": "color_preferred",
                "field": "color",
                "function": "categorical_preferred_v1",
                "preferred": ["白色"],
                "allowed": ["黑色"],
            }
        )
    return {
        "contract_version": "shopping-requirement-contract-v1",
        "status": "approved",
        "task_mode": "single_product_purchase",
        "must": [
            {
                "id": "category",
                "field": "category",
                "operator": "is_a",
                "value": "数码›手机›智能手机",
                "severity": {"rule_id": "wrong_category_v1", "value": 1.0},
            },
            {
                "id": "color",
                "field": "option",
                "axis": "color",
                "operator": "in",
                "value": list(color_values),
                "severity": {"rule_id": "wrong_color_v1", "value": 0.3},
            },
            {
                "id": "capacity",
                "field": "option",
                "axis": "capacity",
                "operator": "eq",
                "value": "256GB",
                "severity": {"rule_id": "wrong_capacity_v1", "value": 0.8},
            },
            {
                "id": "budget",
                "field": "final_variant_price",
                "operator": "le",
                "value": budget,
                "severity": {"rule_id": "budget_excess_v1"},
            },
        ],
        "preferences": preferences,
        "price_objective": {"active": False},
        "audit": {"review_status": "approved"},
    }


def goal(**kwargs):
    return {"asin": "gold", "reward_contract": contract(**kwargs)}


def select(color="白色", capacity="256GB"):
    return {"颜色分类": color, "容量": capacity}


class RewardV4Test(unittest.TestCase):
    def test_gold_alternative_and_authorized_compromise(self):
        gold = score_purchase(product(), goal(), selected_options=select(), price=5000)
        alternative = score_purchase(
            product("alternative"), goal(), selected_options=select(), price=5000
        )
        compromise = score_purchase(
            product(color="黑色"), goal(), selected_options=select("黑色"), price=5000
        )
        self.assertEqual((gold.reward_type, gold.reward), ("gold_purchase", 1.0))
        self.assertEqual(alternative.reward_type, "valid_alternative_purchase")
        self.assertAlmostEqual(alternative.reward, 0.95)
        self.assertEqual(compromise.reward_type, "acceptable_compromise_purchase")
        self.assertAlmostEqual(compromise.reward, 0.9)
        self.assertFalse(compromise.purchase_success)
        self.assertTrue(compromise.acceptable_purchase)

    def test_wrong_color_and_capacity_use_frozen_severity(self):
        color_goal = goal(color_values=("白色",))
        wrong_color = score_purchase(
            product(color="红色"), color_goal, selected_options=select("红色"), price=5000
        )
        wrong_capacity = score_purchase(
            product(capacity="128GB"), goal(), selected_options=select(capacity="128GB"), price=5000
        )
        self.assertEqual(wrong_color.reward_type, "wrong_purchase")
        self.assertAlmostEqual(wrong_color.reward, -0.72)
        self.assertAlmostEqual(wrong_capacity.reward, -0.92)

    def test_budget_boundary_and_one_percent_excess(self):
        boundary = score_purchase(product(), goal(), selected_options=select(), price=5000)
        excess = score_purchase(product(price=5050), goal(), selected_options=select(), price=5050)
        self.assertEqual(boundary.reward_type, "gold_purchase")
        self.assertEqual(excess.reward_type, "wrong_purchase")
        self.assertAlmostEqual(excess.reward, -0.715)

    def test_order_fact_unknown_and_missing_actor_evidence_are_distinct(self):
        unknown = score_purchase(
            product(),
            goal(),
            selected_options=select(),
            price_resolution={"status": "unverifiable", "price": None},
        )
        missing_evidence = score_purchase(
            product(), goal(), selected_options=select(), price=5000, evidence={"facts": []}
        )
        self.assertEqual((unknown.reward_type, unknown.reward_valid, unknown.reward), ("reward_unverifiable", False, 0.0))
        self.assertEqual((missing_evidence.reward_type, missing_evidence.reward_valid, missing_evidence.reward), ("unverified_purchase", True, -0.6))

    def test_known_hard_failure_wins_over_another_unknown_fact(self):
        result = score_purchase(
            product(color="红色"),
            goal(color_values=("白色",)),
            selected_options=select("红色"),
            price_resolution={"status": "unverifiable", "price": None},
        )
        self.assertEqual(result.reward_type, "wrong_purchase")
        self.assertTrue(result.reward_valid)

    def test_no_preference_has_unit_satisfaction_and_deterministic_result(self):
        task_goal = goal(preference=False)
        first = score_purchase(product(), task_goal, selected_options=select(), price=5000)
        second = score_purchase(product(), task_goal, selected_options=select(), price=5000)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.evidence["preference_satisfaction"]["aggregate"], 1.0)

    def test_contract_integrity_and_preference_registry_are_fail_closed(self):
        tampered = contract(preference=False)
        tampered["audit"]["contract_hash"] = "not-the-frozen-hash"
        result = score_purchase(product(), {"asin": "gold", "reward_contract": tampered}, selected_options=select(), price=5000)
        self.assertEqual((result.reward_type, result.reward_valid), ("reward_unverifiable", False))
        unsupported = contract(preference=False)
        unsupported["preferences"] = [{"id": "custom", "field": "color", "function": "unreviewed_v9", "preferred": ["白色"]}]
        result = score_purchase(product(), {"asin": "gold", "reward_contract": unsupported}, selected_options=select(), price=5000)
        self.assertEqual((result.reward_type, result.reward_valid), ("reward_unverifiable", False))

        malformed = contract(preference=False)
        malformed["audit"] = "not-an-object"
        result = score_purchase(product(), {"asin": "gold", "reward_contract": malformed}, selected_options=select(), price=5000)
        self.assertEqual(result.evidence["invalid_reason"], "contract_audit_invalid")

        invalid_price_objective = contract(preference=False)
        invalid_price_objective["price_objective"] = {"active": True}
        result = score_purchase(product(), {"asin": "gold", "reward_contract": invalid_price_objective}, selected_options=select(), price=5000)
        self.assertEqual(result.evidence["invalid_reason"], "price_objective_reference_budget_invalid")

        explicit_empty_goal = {"asin": "gold", "reward_contract": {}}
        explicit_empty = score_purchase(product(), explicit_empty_goal, selected_options=select(), price=5000)
        self.assertEqual(explicit_empty.evidence["invalid_reason"], "contract_version_invalid")
        self.assertTrue(is_v4_goal({"reward_contract": {}}))
        dispatched = evaluate_purchase(product(), explicit_empty_goal, selected_options=select(), price=5000)
        self.assertEqual(dispatched.reward_type, "reward_unverifiable")

    def test_stop_quality_and_known_candidate(self):
        graceful = score_abstain(
            effective_result_sets=2, candidate_quality=1.0, known_acceptable_candidates=0
        )
        blocked = score_abstain(
            effective_result_sets=2, candidate_quality=1.0, known_acceptable_candidates=1
        )
        self.assertEqual(graceful.reward_type, "graceful_stop")
        self.assertAlmostEqual(graceful.reward, -0.1)
        self.assertEqual(
            (blocked.reward_type, blocked.reward),
            ("stop_with_acceptable_candidate", -0.4),
        )

    def test_candidate_quality_does_not_mix_variant_fingerprints(self):
        task_contract = contract(preference=False)
        facts = [
            {"asin": "gold", "field": "category", "status": "known", "value": "数码›手机›智能手机"},
            {"asin": "gold", "field": "option", "axis": "color", "status": "known", "value": "白色", "option_fingerprint": "variant-a"},
            {"asin": "gold", "field": "option", "axis": "capacity", "status": "known", "value": "256GB", "option_fingerprint": "variant-b"},
            {"asin": "gold", "field": "legal_sku", "status": "known", "value": None, "option_fingerprint": "variant-a"},
            {"asin": "gold", "field": "final_variant_price", "status": "known", "value": 5000, "option_fingerprint": "variant-a"},
        ]
        quality = candidate_quality({"facts": facts}, task_contract)
        self.assertLess(quality, 1.0)

    def test_candidate_quality_counts_composite_branches(self):
        task_contract = contract(preference=False)
        task_contract["must"].append(
            {
                "id": "feature_choice",
                "field": "any_of",
                "value": [
                    {
                        "id": "wireless",
                        "field": "core_function",
                        "operator": "true",
                        "value": "无线充电",
                    },
                    {
                        "id": "waterproof",
                        "field": "core_function",
                        "operator": "true",
                        "value": "防水",
                    },
                ],
            }
        )
        facts = [
            {"asin": "gold", "field": "category", "status": "known", "value": "数码›手机›智能手机"},
            {"asin": "gold", "field": "option", "axis": "color", "status": "known", "value": "白色"},
            {"asin": "gold", "field": "option", "axis": "capacity", "status": "known", "value": "256GB"},
            {"asin": "gold", "field": "core_function", "status": "known", "value": "无线充电"},
            {"asin": "gold", "field": "legal_sku", "status": "known", "value": "gold"},
            {"asin": "gold", "field": "final_variant_price", "status": "known", "value": 5000},
        ]
        self.assertAlmostEqual(candidate_quality({"facts": facts}, task_contract), 1.0)

    def test_numeric_units_and_bounds_are_normalized(self):
        task_contract = contract(preference=False)
        task_contract["must"][2] = {
            "id": "capacity",
            "field": "capacity",
            "operator": "ge",
            "value": 512,
            "unit": "GB",
            "severity": {"rule_id": "wrong_capacity_v1", "value": 0.8},
        }
        task_goal = {"asin": "gold", "reward_contract": task_contract}
        result = score_purchase(
            product(capacity="1TB"),
            task_goal,
            selected_options=select(capacity="1TB"),
            price=5000,
        )
        self.assertEqual(result.reward_type, "gold_purchase")

    def test_preference_missing_evidence_is_zero_without_invalidating_reward(self):
        task_contract = contract(preference=False)
        task_contract["preferences"] = [
            {
                "id": "brand_preferred",
                "field": "brand",
                "function": "categorical_preferred_v1",
                "preferred": ["示例品牌"],
            }
        ]
        facts = [
            {"asin": "gold", "field": "category", "status": "known", "value": None},
            {"asin": "gold", "field": "option", "axis": "capacity", "status": "known", "value": "256GB"},
            {"asin": "gold", "field": "final_variant_price", "status": "known", "value": 5000},
            {"asin": "gold", "field": "option", "axis": "color", "status": "known", "value": "白色"},
        ]
        result = score_purchase(
            product(),
            {"asin": "gold", "reward_contract": task_contract},
            selected_options=select(),
            price=5000,
            evidence={"facts": facts},
        )
        item = result.evidence["preference_satisfaction"]["items"][0]
        self.assertEqual(item["status"], "evidence_missing")
        self.assertEqual(item["score"], 0.0)
        self.assertTrue(result.reward_valid)
        self.assertEqual(result.reward_type, "acceptable_compromise_purchase")

    def test_composite_any_of_and_boolean_negation_are_tristate(self):
        task_contract = contract(preference=False)
        task_contract["must"].append(
            {
                "id": "feature_choice",
                "field": "any_of",
                "value": [
                    {"id": "wireless", "field": "core_function", "operator": "true", "value": "无线充电"},
                    {"id": "waterproof", "field": "core_function", "operator": "true", "value": "防水"},
                ],
                "severity": {"rule_id": "missing_core_function_v1", "value": 0.9},
            }
        )
        result = score_purchase(
            product(),
            {"asin": "gold", "reward_contract": task_contract},
            selected_options=select(),
            price=5000,
        )
        self.assertEqual(result.reward_type, "gold_purchase")

    def test_compiler_freezes_spec_severity_and_screens_unsupported_conditionals(self):
        target = product()
        approved = compile_requirement_contract(
            {"instruction": "购买256GB手机", "attributes": ["无线充电"], "instruction_options": ["256GB"]},
            target,
        )
        capacity = next(item for item in approved["must"] if item.get("axis") == "capacity")
        self.assertEqual(capacity["severity"]["value"], 0.8)
        review = compile_requirement_contract(
            {"instruction": "A缺货才买B", "attributes": ["无线充电"], "instruction_options": []},
            target,
        )
        self.assertEqual(review["status"], "needs_review")

    def test_buy_now_action_is_public_legal_sku_evidence(self):
        ledger = new_ledger()
        record_observation(
            ledger,
            {
                "page_type": "product_detail",
                "actions": ["back to search", "buy now"],
                "product": {"asin": "gold", "category": "手机"},
                "selected_options": {},
                "available_options": {},
                "selected_price": 5000,
            },
            step=1,
        )
        legal = [item for item in ledger["facts"] if item["field"] == "legal_sku"]
        self.assertEqual(legal[0]["source"], "actions.buy_now")

    def test_candidate_requires_rendered_same_sku_evidence(self):
        target = product()
        self.assertFalse(evaluate_candidate_eligibility(target, goal())["known_valid"])
        ledger = new_ledger()
        record_observation(
            ledger,
            {
                "page_type": "product_detail",
                "actions": ["buy now"],
                "product": {
                    "asin": "gold",
                    "title": target["title"],
                    "brand": target["brand"],
                    "category": target["category"],
                    "key_attributes": target["attribute"],
                },
                "selected_options": select(),
                "available_options": {"颜色分类": ["白色"], "容量": ["256GB"]},
                "selected_price": 5000,
            },
            step=1,
        )
        facts = ledger["facts"]
        self.assertEqual(
            next(item["value"] for item in facts if item["field"] == "core_function"),
            ["无线充电"],
        )
        result = evaluate_candidate_eligibility(
            target,
            goal(),
            public_ledger(ledger),
        )
        self.assertTrue(result["known_valid"])
        self.assertEqual(result["evidence_coverage"], 1.0)

    def test_candidate_eligibility_uses_actor_selected_acceptable_variant(self):
        target = product(color="黑色")
        selected = select("黑色")
        ledger = new_ledger()
        record_observation(
            ledger,
            {
                "page_type": "product_detail",
                "actions": ["buy now"],
                "product": {
                    "asin": "gold",
                    "title": target["title"],
                    "brand": target["brand"],
                    "category": target["category"],
                    "key_attributes": target["attribute"],
                },
                "selected_options": selected,
                "available_options": {"颜色分类": ["黑色"], "容量": ["256GB"]},
                "selected_price": 5000,
            },
            step=1,
        )
        result = evaluate_candidate_eligibility(
            target,
            goal(),
            public_ledger(ledger),
            selected_options=selected,
            price_resolution={"status": "pass", "price": 5000},
        )
        self.assertTrue(result["known_valid"])

    def test_information_subpage_features_are_public_core_function_evidence(self):
        ledger = new_ledger()
        record_observation(
            ledger,
            {
                "page_type": "information_subpage",
                "subpage": "features",
                "product": {"asin": "gold", "key_attributes": []},
                "selected_options": {},
                "content": "支持无线充电和防水",
            },
            step=2,
        )
        self.assertTrue(
            any(
                item["field"] == "core_function"
                and item["source"] == "information_subpage.features"
                for item in ledger["facts"]
            )
        )

    def test_legal_sku_rejects_unknown_selected_axis_even_with_explicit_price(self):
        task_contract = contract(preference=False)
        task_contract["must"].append(
            {
                "id": "legal_sku",
                "field": "legal_sku",
                "operator": "eq",
                "value": True,
                "severity": {"rule_id": "invalid_sku_v1", "value": 1.0},
            }
        )
        selected = select()
        selected["不存在的规格轴"] = "值"
        result = score_purchase(
            product(),
            {"asin": "gold", "reward_contract": task_contract},
            selected_options=selected,
            price=5000,
        )
        self.assertEqual(result.reward_type, "wrong_purchase")
        self.assertEqual(result.reward, -1.0)


if __name__ == "__main__":
    unittest.main()
