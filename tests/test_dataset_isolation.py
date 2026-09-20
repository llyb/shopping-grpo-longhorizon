from __future__ import annotations

import unittest

from scripts.dataset_isolation import isolation_report, normalize_requirement


class DatasetIsolationTest(unittest.TestCase):
    def test_normalization_finds_same_requirement_with_different_spacing(self):
        self.assertEqual(
            normalize_requirement("买 白色 手机！"),
            normalize_requirement("买白色手机"),
        )

    def test_cross_split_content_overlap_is_rejected(self):
        fingerprint = "same"
        catalog = [
            {"task_id": 1, "requirement_fingerprint": fingerprint},
            {"task_id": 2, "requirement_fingerprint": fingerprint},
        ]
        report = isolation_report({"sft": [1], "evaluation": [2]}, catalog)
        self.assertFalse(report["valid"])
        self.assertIn("sft__evaluation", report["requirement_content_overlaps"])


if __name__ == "__main__":
    unittest.main()
