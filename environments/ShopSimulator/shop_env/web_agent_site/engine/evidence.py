"""Actor-visible evidence ledger used by Reward v4.

Only structured observation fields that are returned to the actor are stored.
Backend-only product fields and the private goal/contract never enter this
ledger.  Price and option facts carry an exact option fingerprint so selecting
a different variant invalidates earlier price evidence.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from web_agent_site.engine.reward_features import canonicalize_option_axis


EVIDENCE_LEDGER_VERSION = "shopping-evidence-ledger-v3"


def option_fingerprint(selected_options: Any) -> str:
    return json.dumps(
        dict(selected_options or {}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def new_ledger() -> dict:
    return {"version": EVIDENCE_LEDGER_VERSION, "facts": [], "_keys": set()}


def _append(ledger: dict, fact: dict) -> None:
    key = json.dumps(fact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    keys = ledger.setdefault("_keys", set())
    if key not in keys:
        keys.add(key)
        ledger.setdefault("facts", []).append(fact)


def record_observation(ledger: dict, observation: Mapping[str, Any], *, step: int) -> None:
    """Record public facts from one structured observation."""
    page_type = str(observation.get("page_type") or "")
    if page_type not in {"product_detail", "information_subpage"}:
        return
    product = observation.get("product") or {}
    asin = str(product.get("asin") or "")
    if not asin:
        return
    selected = observation.get("selected_options") or {}
    fingerprint = option_fingerprint(selected)
    common = {"asin": asin, "step": int(step), "page_type": page_type, "status": "known"}
    for field, source in (
        ("category", "product.category"),
        ("brand", "product.brand"),
        ("model", "product.title"),
        ("core_function", "product.key_attributes"),
    ):
        value = {
            "category": product.get("category"),
            "brand": product.get("brand"),
            "model": product.get("title"),
            "core_function": product.get("key_attributes"),
        }[field]
        if value not in (None, "", []):
            # Keep the value that was actually rendered.  A ``None`` sentinel
            # would turn a visible field into a wildcard in the scorer and
            # could therefore claim evidence for a requirement that was not
            # present in the observation (especially core functions).
            _append(ledger, {**common, "field": field, "value": value, "source": source})
    if page_type == "information_subpage":
        subpage = str(observation.get("subpage") or "").casefold()
        # Description, Features and Attributes are actor-visible sources for
        # core functions.  Reviews are intentionally excluded by the public
        # protocol and cannot establish an official product capability.
        if subpage in {"description", "features", "attributes"}:
            content = observation.get("content")
            if content not in (None, "", []):
                _append(
                    ledger,
                    {
                        **common,
                        "field": "core_function",
                        "value": content,
                        "source": f"information_subpage.{subpage}",
                    },
                )
    available = observation.get("available_options") or {}
    for raw_axis, value in selected.items():
        canonical_axis = canonicalize_option_axis(raw_axis)
        _append(
            ledger,
            {
                **common,
                "field": "option",
                "axis": canonical_axis,
                "value": value,
                "source": "selected_options",
                "option_fingerprint": fingerprint,
            },
        )
    selected_axes = {canonicalize_option_axis(axis) for axis in selected}
    for raw_axis, values in available.items():
        canonical_axis = canonicalize_option_axis(raw_axis)
        if canonical_axis in selected_axes or not isinstance(values, list):
            continue
        unique = [value for value in dict.fromkeys(str(item) for item in values) if value]
        if len(unique) == 1:
            _append(
                ledger,
                {
                    **common,
                    "field": "option",
                    "axis": canonical_axis,
                    "value": unique[0],
                    "source": "available_options.single_default",
                    "option_fingerprint": fingerprint,
                },
            )
    actions = {str(item).casefold() for item in observation.get("actions") or []}
    if available or "buy now" in actions:
        _append(
            ledger,
            {
                **common,
                "field": "legal_sku",
                "value": None,
                "source": (
                    "available_options"
                    if available
                    else "actions.buy_now"
                ),
                "option_fingerprint": fingerprint,
            },
        )
    if observation.get("selected_price") is not None:
        _append(
            ledger,
            {
                **common,
                "field": "final_variant_price",
                "value": observation.get("selected_price"),
                "source": "selected_price",
                "option_fingerprint": fingerprint,
            },
        )


def public_ledger(ledger: Mapping[str, Any] | None, *, redundant_actions: int = 0, invalid_actions: int = 0) -> dict:
    value = ledger or {}
    return {
        "version": value.get("version", EVIDENCE_LEDGER_VERSION),
        "facts": [dict(item) for item in value.get("facts") or [] if isinstance(item, Mapping)],
        "redundant_actions": max(0, int(redundant_actions)),
        "invalid_actions": max(0, int(invalid_actions)),
    }


def candidate_quality(ledger: Mapping[str, Any] | None, contract: Mapping[str, Any] | None) -> float:
    """Return maximum same-candidate coverage over the contract obligations.

    Composite requirements are represented as branch alternatives: ``all_of``
    contributes every child, while ``any_of`` contributes one obligation that
    is covered when any complete branch is observed.  This mirrors the Reward
    v4 tri-state scorer and avoids treating ``any_of`` itself as an unknown
    evidence field.
    """
    facts = [item for item in (ledger or {}).get("facts") or [] if isinstance(item, Mapping)]
    requirements = [item for item in (contract or {}).get("must") or [] if isinstance(item, Mapping)]
    def leaf(item: Mapping[str, Any]) -> tuple[str, str, str] | None:
        field = str(item.get("field") or item.get("type") or "")
        if not field or field in {"all_of", "any_of"}:
            return None
        if field in {"budget", "price"}:
            field = "final_variant_price"
        if field in {"color", "size", "capacity", "storage", "sku_option"}:
            field = "option"
        if field in {"core_functions", "attribute", "attributes"}:
            field = "core_function"
        axis = canonicalize_option_axis(str(item.get("axis") or field)) if field == "option" else ""
        return (str(item.get("id") or f"{field}:{axis}"), field, axis)

    def branches(item: Mapping[str, Any]) -> list[list[tuple[str, str, str]]]:
        operator = str(item.get("operator") or item.get("field") or "")
        if operator not in {"all_of", "any_of"}:
            value = leaf(item)
            return [[value]] if value is not None else []
        clauses = item.get("clauses") or item.get("requirements") or item.get("value") or []
        clauses = [child for child in clauses if isinstance(child, Mapping)]
        if not clauses:
            return []
        child_branches = [branches(child) for child in clauses]
        if operator == "any_of":
            return [branch for options in child_branches for branch in options]
        combined = [[]]
        for options in child_branches:
            combined = [prefix + branch for prefix in combined for branch in options]
        return combined

    # Each top-level requirement is one quality unit.  A unit with several
    # branches is covered if one branch's leaves are all present.
    obligations: list[tuple[str, list[list[tuple[str, str, str]]]]] = []
    seen_semantics: set[tuple[str, str]] = set()
    for index, item in enumerate(requirements):
        alternatives = branches(item)
        normalized: list[list[tuple[str, str, str]]] = []
        for branch in alternatives:
            unique: list[tuple[str, str, str]] = []
            for value in branch:
                semantic = (value[1], value[2])
                if semantic not in {(entry[1], entry[2]) for entry in unique}:
                    unique.append(value)
                seen_semantics.add(semantic)
            if unique:
                normalized.append(unique)
        if normalized:
            obligations.append((str(item.get("id") or f"requirement:{index}"), normalized))
    for key, field, axis in (
        ("__legal_sku__", "legal_sku", ""),
        ("__final_variant_price__", "final_variant_price", ""),
    ):
        if (field, axis) not in seen_semantics:
            obligations.append((key, [[(key, field, axis)]]))
            seen_semantics.add((field, axis))
    if not obligations:
        return 0.0
    # Variant-dependent facts must stay attached to the same observed SKU.
    # Static product facts (category/brand/model/core function) may be reused
    # across fingerprints, but combining an option or price from two variants
    # would fabricate a candidate that the actor never actually verified.
    variant_keys_by_asin: dict[str, set[str]] = {}
    for fact in facts:
        asin = str(fact.get("asin") or "")
        fingerprint = fact.get("option_fingerprint")
        if asin and fact.get("status") == "known" and fingerprint:
            variant_keys_by_asin.setdefault(asin, set()).add(str(fingerprint))

    by_candidate: dict[tuple[str, str], set[str]] = {}
    variant_fields = {"option", "legal_sku", "final_variant_price"}
    for fact in facts:
        asin = str(fact.get("asin") or "")
        field = str(fact.get("field") or "")
        if not asin or fact.get("status") != "known":
            continue
        fingerprint = fact.get("option_fingerprint")
        if fingerprint:
            candidate_keys = [(asin, str(fingerprint))]
        else:
            known_fingerprints = variant_keys_by_asin.get(asin)
            # A variant-dependent fact without a fingerprint is not enough to
            # score any explicitly observed variant.  If no variant has been
            # observed, keep the legacy un-fingerprinted candidate key.
            if field in variant_fields and known_fingerprints:
                candidate_keys = []
            else:
                candidate_keys = [
                    (asin, value) for value in sorted(known_fingerprints or {""})
                ]
        for key, alternatives in obligations:
            matching = {
                leaf_key
                for branch in alternatives
                for leaf_key, leaf_field, leaf_axis in branch
                if field == leaf_field
                and (leaf_field != "option" or canonicalize_option_axis(str(fact.get("axis") or "")) == leaf_axis)
            }
            if not matching:
                continue
            for candidate_key in candidate_keys:
                by_candidate.setdefault(candidate_key, set()).update(matching)
    total = len(obligations)
    # Collapse leaf matches back to top-level units; a composite unit counts
    # only when one complete alternative branch is covered.
    scores = []
    for covered in by_candidate.values():
        units = 0
        for key, alternatives in obligations:
            if any(all(leaf_key in covered for leaf_key, _, _ in branch) for branch in alternatives):
                units += 1
        scores.append(units / total)
    return max(scores, default=0.0)


__all__ = ["EVIDENCE_LEDGER_VERSION", "new_ledger", "record_observation", "public_ledger", "candidate_quality", "option_fingerprint"]
