import json
from pathlib import Path
import unittest

from web_agent_site.engine.config import (
    load_config,
    validate_config,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "environment.json"


class EnvironmentV21ConfigTest(unittest.TestCase):
    def test_repository_config_matches_reward_contract(self):
        config = load_config(CONFIG)
        self.assertEqual(
            config["environment_version"],
            "shopsimulator-environment-v2.1",
        )
        self.assertEqual(config["reward"]["version"], "shopsimulator-reward-v4")
        self.assertEqual(config["reward"]["purchase_completion"], 0.80)
        self.assertEqual(config["reward"]["wrong_base"], -0.60)
        self.assertEqual(config["reward"]["wrong_severity_weight"], 0.40)
        self.assertEqual(
            config["reward_feature_version"],
            "shopping-reward-features-v1",
        )
        self.assertEqual(
            config["termination"]["version"],
            "shopping-termination-v3",
        )

    def test_reward_drift_is_rejected(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["reward"]["wrong_base"] = -0.4
        with self.assertRaisesRegex(ValueError, "reward values"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
