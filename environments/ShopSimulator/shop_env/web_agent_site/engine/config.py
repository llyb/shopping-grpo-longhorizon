"""Load and validate the frozen ShopSimulator Environment v2.1 contract."""

from __future__ import annotations

import json
from pathlib import Path

from web_agent_site.engine.comparators import (
    BRAND_ALIAS_VERSION,
    COMPARATOR_VERSION,
)
from web_agent_site.engine.reward_features import (
    REWARD_FEATURE_VERSION,
    OPTION_AXIS_VERSION,
)
from web_agent_site.engine.observation import OBSERVATION_VERSION
from web_agent_site.engine.reward import REWARD_VERSION
from web_agent_site.engine.reward_v4 import DEFAULT_REWARDS
from web_agent_site.engine.search import DEFAULT_FIELD_WEIGHTS, SEARCH_VERSION
from web_agent_site.engine.termination import TERMINATION_VERSION
from web_agent_site.engine.variant_price import VARIANT_PRICE_VERSION


ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"
TOOL_VERSION = "shopping-tools-v2"
SEARCH_TOP_K = 150
SEARCH_PAGE_SIZE = 20
_POSITIVE_TERMINATION_FIELDS = (
    "exact_repeat_limit",
    "no_progress_limit",
    "max_steps",
    "min_new_asins_per_result_set",
    "product_open_progress_budget",
    "subpage_progress_budget",
    "result_set_progress_budget",
)


def load_config(path):
    config_path = Path(path).resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load Environment v2.1 config {config_path}: {exc}"
        ) from exc
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Environment v2.1 config must be an object")
    if config.get("environment_version") != ENVIRONMENT_VERSION:
        raise ValueError("Environment v2.1 config has the wrong environment_version")
    search = config.get("search")
    if not isinstance(search, dict) or search.get("version") != SEARCH_VERSION:
        raise ValueError("Environment v2.1 config has the wrong search version")
    if int(search.get("top_k", 0)) != SEARCH_TOP_K:
        raise ValueError(f"Environment v2.1 search top_k must equal {SEARCH_TOP_K}")
    if int(search.get("page_size", 0)) != SEARCH_PAGE_SIZE:
        raise ValueError(
            f"Environment v2.1 search page_size must equal {SEARCH_PAGE_SIZE}"
        )
    if search.get("field_weights") != DEFAULT_FIELD_WEIGHTS:
        raise ValueError(
            "Environment v2.1 search field weights differ from the index contract"
        )
    reward = config.get("reward")
    if not isinstance(reward, dict) or reward.get("version") != REWARD_VERSION:
        raise ValueError("Environment v2.1 config has the wrong reward version")
    if set(reward) != {"version", *DEFAULT_REWARDS}:
        raise ValueError(
            "Environment v2.1 reward values have missing or unsupported keys"
        )
    reward_values = {
        key: float(reward.get(key)) for key in DEFAULT_REWARDS if key in reward
    }
    if reward_values != DEFAULT_REWARDS:
        raise ValueError(
            "Environment v2.1 reward values differ from the runtime contract"
        )
    termination = config.get("termination")
    if (
        not isinstance(termination, dict)
        or termination.get("version") != TERMINATION_VERSION
    ):
        raise ValueError("Environment v2.1 config has the wrong termination version")
    for name in _POSITIVE_TERMINATION_FIELDS:
        if int(termination.get(name, 0)) <= 0:
            raise ValueError(f"Environment v2.1 termination.{name} must be positive")
    expected_versions = {
        "reward_feature_version": REWARD_FEATURE_VERSION,
        "option_axis_version": OPTION_AXIS_VERSION,
        "variant_price_version": VARIANT_PRICE_VERSION,
        "comparator_version": COMPARATOR_VERSION,
        "brand_alias_version": BRAND_ALIAS_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "tool_version": TOOL_VERSION,
    }
    for name, expected in expected_versions.items():
        if config.get(name) != expected:
            raise ValueError(
                f"Environment v2.1 config has the wrong {name}: "
                f"expected {expected!r}"
            )
    return config
