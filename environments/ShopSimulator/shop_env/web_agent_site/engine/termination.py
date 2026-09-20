"""Bounded evidence-progress tracking for Environment v2.1."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from web_agent_site.engine.product_id import is_product_id
from web_agent_site.engine.reward_features import (
    canonicalize_option_axis,
    normalize_option_text,
)
from web_agent_site.engine.search import normalize_query


TERMINATION_VERSION = "shopping-termination-v3"
PAGINATION_ACTIONS = {"next >", "< prev"}


def canonical_action(action_name: str, action_argument: object) -> str:
    argument = (
        normalize_query(action_argument)
        if action_name == "search"
        else str(action_argument or "").strip().casefold()
    )
    return json.dumps(
        {"action": str(action_name).strip().casefold(), "argument": argument},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _result_set_fingerprint(visible_asins) -> str:
    payload = "\0".join(str(asin) for asin in visible_asins)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class EvidenceProgressTracker:
    max_steps: int = 35
    exact_repeat_limit: int = 2
    no_progress_limit: int = 4
    min_new_asins_per_result_set: int = 3
    product_open_progress_budget: int = 5
    subpage_progress_budget: int = 6
    result_set_progress_budget: int = 3
    steps: int = 0
    last_signature: str | None = None
    consecutive_repeats: int = 0
    no_progress_steps: int = 0
    redundant_actions: int = 0
    seen_asins: set[str] = field(default_factory=set)
    opened_asins: set[str] = field(default_factory=set)
    runtime_result_fingerprints: set[str] = field(default_factory=set)
    runtime_subpage_evidence: set[str] = field(default_factory=set)
    result_set_evidence: set[str] = field(default_factory=set)
    product_evidence: set[str] = field(default_factory=set)
    subpage_evidence: set[str] = field(default_factory=set)
    option_evidence: set[str] = field(default_factory=set)
    constraint_evidence: set[str] = field(default_factory=set)

    @property
    def effective_result_sets(self) -> int:
        return len(self.result_set_evidence)

    def record(
        self,
        action_name: str,
        action_argument: object,
        visible_asins=(),
        *,
        page_type: str | None = None,
        selected_options: object = None,
        constraint_evidence=(),
    ):
        self.steps += 1
        signature = canonical_action(action_name, action_argument)
        repeated_signature = signature == self.last_signature
        if repeated_signature:
            self.consecutive_repeats += 1
        else:
            self.consecutive_repeats = 0
        self.last_signature = signature

        normalized_name = str(action_name or "").strip().casefold()
        normalized_argument = str(action_argument or "").strip().casefold()
        visible_ordered = tuple(str(asin) for asin in visible_asins)
        visible = set(visible_ordered)
        new_asins = visible - self.seen_asins
        self.seen_asins.update(visible)
        runtime_progress_added = []
        credited_evidence_added = []

        discovery_action = (
            normalized_name == "search"
            or (
                normalized_name == "click"
                and normalized_argument in PAGINATION_ACTIONS
            )
        )
        if discovery_action and visible_ordered:
            fingerprint = _result_set_fingerprint(visible_ordered)
            new_runtime_result = fingerprint not in self.runtime_result_fingerprints
            self.runtime_result_fingerprints.add(fingerprint)
            if (
                new_runtime_result
                and len(new_asins) >= self.min_new_asins_per_result_set
            ):
                key = f"result_set:{fingerprint}"
                runtime_progress_added.append(key)
                if len(self.result_set_evidence) < self.result_set_progress_budget:
                    self.result_set_evidence.add(fingerprint)
                    credited_evidence_added.append(key)

        candidate_opened = (
            normalized_name == "click"
            and is_product_id(normalized_argument)
        )
        if candidate_opened:
            asin = normalized_argument
            new_runtime_product = asin not in self.opened_asins
            self.opened_asins.add(asin)
            key = f"product:{asin}"
            if new_runtime_product:
                runtime_progress_added.append(key)
                if len(self.product_evidence) < self.product_open_progress_budget:
                    self.product_evidence.add(key)
                    credited_evidence_added.append(key)

        if (
            normalized_name == "click"
            and page_type == "information_subpage"
            and visible_ordered
        ):
            asin = visible_ordered[0]
            key = f"subpage:{asin}:{normalized_argument}"
            if key not in self.runtime_subpage_evidence:
                self.runtime_subpage_evidence.add(key)
                runtime_progress_added.append(key)
                if len(self.subpage_evidence) < self.subpage_progress_budget:
                    self.subpage_evidence.add(key)
                    credited_evidence_added.append(key)

        if isinstance(selected_options, dict) and visible_ordered:
            asin = visible_ordered[0]
            for raw_axis, raw_value in selected_options.items():
                key = (
                    f"option:{asin}:{canonicalize_option_axis(raw_axis)}:"
                    f"{normalize_option_text(raw_value)}"
                )
                if key not in self.option_evidence:
                    self.option_evidence.add(key)
                    runtime_progress_added.append(key)
                    credited_evidence_added.append(key)

        for raw_evidence in constraint_evidence or ():
            key = f"constraint:{str(raw_evidence)}"
            if key not in self.constraint_evidence:
                self.constraint_evidence.add(key)
                runtime_progress_added.append(key)
                credited_evidence_added.append(key)

        if runtime_progress_added:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
            if repeated_signature:
                self.redundant_actions += 1

        reason = None
        if self.consecutive_repeats >= self.exact_repeat_limit:
            reason = "repeat_loop"
        elif self.no_progress_steps >= self.no_progress_limit:
            reason = "repeat_loop"
        elif self.steps >= self.max_steps:
            reason = "max_steps"
        return {
            "termination_version": TERMINATION_VERSION,
            "termination_reason": reason,
            "step_count": self.steps,
            "action_signature": signature,
            "consecutive_repeats": self.consecutive_repeats,
            "no_progress_steps": self.no_progress_steps,
            "redundant_actions": self.redundant_actions,
            # Compatibility field: evidence that counts toward bounded abstention
            # qualification and metrics, not the liveness decision.
            "evidence_added": credited_evidence_added,
            "credited_evidence_added": credited_evidence_added,
            "runtime_progress_added": runtime_progress_added,
            "new_asin_count": len(new_asins),
            "seen_asin_count": len(self.seen_asins),
            "opened_candidate_count": len(self.opened_asins),
            "effective_result_sets": self.effective_result_sets,
            "runtime_evidence_counts": {
                "result_set": len(self.runtime_result_fingerprints),
                "product": len(self.opened_asins),
                "subpage": len(self.runtime_subpage_evidence),
                "option": len(self.option_evidence),
                "constraint": len(self.constraint_evidence),
            },
            "evidence_counts": {
                "result_set": len(self.result_set_evidence),
                "product": len(self.product_evidence),
                "subpage": len(self.subpage_evidence),
                "option": len(self.option_evidence),
                "constraint": len(self.constraint_evidence),
            },
        }
