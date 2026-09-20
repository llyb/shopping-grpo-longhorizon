"""Shared helpers for task-disjoint SFT, GRPO, development and evaluation data."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata


_AXIS_ALIASES = {
    "color": {"颜色", "颜色分类"},
    "size": {"尺码", "鞋码"},
    "dimensions": {"尺寸", "大小"},
    "net_content": {"净含量", "总净含量"},
    "flavor": {"口味", "食品口味"},
    "specification": {"规格", "规格描述", "规格类型"},
    "bundle": {"套餐", "套餐类型", "组合套餐"},
    "capacity": {"容量", "规格容量"},
}
_UNSUPPORTED_FIRST_PASS_PATTERNS = (
    re.compile(r"(?:缺货|无货|没货).{0,8}(?:才|再)"),
    re.compile(r"(?:不行|没有|买不到).{0,8}(?:才|再)"),
)


def normalize_requirement(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", value)


def _normalize_option(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("/", "|")
    return re.sub(r"\s+", "", value)


def _canonical_axis(value: object) -> str:
    normalized = _normalize_option(value)
    for canonical, aliases in _AXIS_ALIASES.items():
        if normalized in {_normalize_option(alias) for alias in aliases}:
            return canonical
    return normalized


def _contract_admissible(product: dict, instruction: dict) -> bool:
    instruction_text = str(instruction.get("instruction") or "").strip()
    if not instruction_text or any(
        pattern.search(instruction_text)
        for pattern in _UNSUPPORTED_FIRST_PASS_PATTERNS
    ):
        return False
    axes = {}
    for raw_axis, entries in (product.get("customization_options") or {}).items():
        axes[str(raw_axis)] = {
            _normalize_option(item.get("value") if isinstance(item, dict) else item)
            for item in entries or []
        }
    resolved_axes = set()
    for value in instruction.get("instruction_options") or []:
        normalized = _normalize_option(value)
        matches = [axis for axis, values in axes.items() if normalized in values]
        if len(matches) != 1:
            return False
        canonical = _canonical_axis(matches[0])
        if canonical in resolved_axes:
            return False
        resolved_axes.add(canonical)
    return True


def load_task_catalog(product_corpus: str | Path) -> list[dict]:
    """Reproduce ``get_goals`` ordering without loading the environment stack."""
    path = Path(product_corpus)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        products = json.load(stream)
    catalog = []
    for product in products:
        for instruction in product.get("instructions") or []:
            if not instruction.get("attributes"):
                continue
            text = instruction.get("instruction") or ""
            catalog.append(
                {
                    "task_id": len(catalog),
                    "instruction": text,
                    "requirement_fingerprint": hashlib.sha256(
                        normalize_requirement(text).encode("utf-8")
                    ).hexdigest(),
                    "contract_admissible": _contract_admissible(product, instruction),
                }
            )
    return catalog


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def task_id(row: dict) -> int:
    value = row.get("task_id")
    if value is None:
        value = (row.get("extra_info") or {}).get("task_id")
    if value is None:
        raise ValueError("dataset row is missing task_id")
    return int(value)


def task_ids(path: str | Path) -> list[int]:
    return [task_id(row) for row in read_jsonl(path)]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: str | Path, rows) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def isolation_report(named_ids: dict[str, list[int]], catalog: list[dict]) -> dict:
    by_id = {int(row["task_id"]): row for row in catalog}
    unknown = {
        name: sorted(set(ids) - set(by_id))
        for name, ids in named_ids.items()
        if set(ids) - set(by_id)
    }
    duplicate_within = {
        name: len(ids) - len(set(ids))
        for name, ids in named_ids.items()
        if len(ids) != len(set(ids))
    }
    id_overlaps = {}
    content_overlaps = {}
    names = list(named_ids)
    for index, left in enumerate(names):
        left_ids = set(named_ids[left])
        left_fingerprints = {
            by_id[value]["requirement_fingerprint"]
            for value in left_ids
            if value in by_id
        }
        for right in names[index + 1 :]:
            pair = f"{left}__{right}"
            right_ids = set(named_ids[right])
            common_ids = sorted(left_ids & right_ids)
            if common_ids:
                id_overlaps[pair] = common_ids[:20]
            right_fingerprints = {
                by_id[value]["requirement_fingerprint"]
                for value in right_ids
                if value in by_id
            }
            common_content = sorted(left_fingerprints & right_fingerprints)
            if common_content:
                content_overlaps[pair] = common_content[:20]
    return {
        "schema_version": "shopping-dataset-isolation-v1",
        "counts": {name: len(ids) for name, ids in named_ids.items()},
        "unknown_task_ids": unknown,
        "duplicate_task_ids_within_split": duplicate_within,
        "task_id_overlaps": id_overlaps,
        "requirement_content_overlaps": content_overlaps,
        "valid": not any((unknown, duplicate_within, id_overlaps, content_overlaps)),
    }


def require_isolated(named_ids: dict[str, list[int]], catalog: list[dict]) -> dict:
    report = isolation_report(named_ids, catalog)
    if not report["valid"]:
        raise ValueError(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report
