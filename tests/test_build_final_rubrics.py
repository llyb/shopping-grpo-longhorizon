from __future__ import annotations

import unittest

from scripts.build_final_rubrics import _normalize_base_url, _strict_curator_response
from shopping_grpo.evaluation.contracts import ContractValidationError


class FinalRubricBuilderTests(unittest.TestCase):
    def test_normalizes_full_chat_completions_url_without_double_suffix(self):
        self.assertEqual(
            _normalize_base_url("http://127.0.0.1:3010/v1/chat/completions"),
            "http://127.0.0.1:3010/v1",
        )
        self.assertEqual(
            _normalize_base_url("http://127.0.0.1:3010/v1/"),
            "http://127.0.0.1:3010/v1",
        )

    def test_curator_gate_requires_a_contiguous_query_quote(self):
        with self.assertRaisesRegex(ContractValidationError, "not a contiguous"):
            _strict_curator_response(
                {
                    "selected_constraints": [
                        {
                            "candidate_id": "c0001",
                            "description": "白色商品",
                            "hardness": "hard",
                            "query_quote": "白色商品",
                            "selection_reason": "query",
                        }
                    ]
                },
                "想买白色手机",
            )

    def test_curator_gate_rejects_empty_selection(self):
        with self.assertRaisesRegex(ContractValidationError, "at least one"):
            _strict_curator_response({"selected_constraints": []}, "买手机")

    def test_curator_gate_accepts_exact_quote(self):
        response = _strict_curator_response(
            {
                "selected_constraints": [
                    {
                        "candidate_id": "c0001",
                        "description": "商品品类为手机",
                        "hardness": "hard",
                        "query_quote": "手机",
                        "selection_reason": "Query 明确提出商品类型",
                    }
                ]
            },
            "想买一台手机",
        )
        self.assertEqual(response["selected_constraints"][0]["query_quote"], "手机")


if __name__ == "__main__":
    unittest.main()
