#!/usr/bin/env python3
"""Build and freeze Final-200 Rubric bundles.

The script implements the offline construction stage only:

    private TaskFacts -> deterministic candidate superset -> DeepSeek V4 Flash
    selection -> code-owned schema/materialization gate -> frozen Rubric JSONL

It never changes ``data/evaluation/tasks.jsonl`` and never sends the private
gold product or TaskFacts to the Actor.  The Flash request contains only the
public Query and the code-generated candidate records.  API credentials are
read from the environment and are never written to output metadata.
代码先从商品和任务中提取所有可验证属性，生成候选约束；
然后让 DeepSeek 根据用户 Query 判断哪些约束是真正用户要求的；
最后代码检查引用合法性并冻结成 Rubric。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from shopping_grpo.evaluation.contracts import (
    ContractValidationError,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.model_client import (
    DEFAULT_FLASH_MODEL,
    OpenAIJSONClient,
)
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_PROMPT_VERSION,
    build_rubric_curator_messages,
)
from shopping_grpo.evaluation.rubric import (
    RUBRIC_CANDIDATE_VERSION,
    RUBRIC_EXTRACTOR_VERSION,
    TASK_FACTS_VERSION,
    build_task_facts,
    extract_rubric_candidates,
    materialize_rubric_bundle,
)
from web_agent_site.engine.reward_features import compile_reward_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "data/evaluation/tasks.jsonl"
DEFAULT_CORPUS = ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"


class RubricBuildError(RuntimeError):
    """Raised when a Final-200 task cannot pass the construction gate."""


def _normalize_base_url(value: str) -> str:
    value = str(value or "").strip().rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")].rstrip("/")
    if not value:
        raise ValueError("OPENAI_BASE_URL or --base-url is required")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RubricBuildError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_products(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        products = json.load(stream)
    if not isinstance(products, list):
        raise RubricBuildError("product corpus must contain a JSON array")
    return products


def _load_tasks(path: Path) -> list[int]:
    rows = _read_jsonl(path)
    task_ids = []
    for row in rows:
        value = row.get("task_id")
        if value is None:
            value = (row.get("extra_info") or {}).get("task_id")
        if isinstance(value, bool) or value is None:
            raise RubricBuildError(f"{path} contains a row without integer task_id")
        task_ids.append(int(value))
    if len(task_ids) != len(set(task_ids)):
        raise RubricBuildError("Final-200 task manifest contains duplicate task IDs")
    return task_ids


def _task_records(products: list[dict]) -> dict[int, tuple[dict, dict]]:
    """Reproduce ShopSimulator's goal ordering for private TaskFacts."""

    records: dict[int, tuple[dict, dict]] = {}
    task_id = 0
    for product in products:
        for instruction in product.get("instructions") or []:
            if not instruction.get("attributes"):
                continue
            records[task_id] = (product, instruction)
            task_id += 1
    return records


def _make_task_facts(task_id: int, product: Mapping, instruction: Mapping) -> dict:
    query = str(instruction.get("instruction") or "").strip()
    if not query:
        raise RubricBuildError(f"task {task_id} has an empty Query")
    reward_goal = {
        "asin": product.get("asin"),
        "category": product.get("category"),
    }
    reward_goal.update(compile_reward_features(instruction, product, include_contract=True))
    return build_task_facts(
        task_id=task_id,
        query=query,
        target_product=product,
        instruction_record=instruction,
        reward_goal=reward_goal,
    )


def _strict_curator_response(response: Mapping, query: str) -> dict:
    """Apply gates that are stricter than generic JSON schema validation."""

    selected = response.get("selected_constraints")
    if not isinstance(selected, list) or not selected:
        raise ContractValidationError("curator must select at least one constraint")
    for index, item in enumerate(selected):
        if not isinstance(item, Mapping):
            raise ContractValidationError(f"selected_constraints[{index}] must be an object")
        quote = str(item.get("query_quote") or "").strip()
        if not quote:
            raise ContractValidationError(
                f"selected_constraints[{index}].query_quote is required"
            )
        if quote not in query:
            raise ContractValidationError(
                f"selected_constraints[{index}].query_quote is not a contiguous Query span"
            )
    return deepcopy(dict(response))


def _task_rows(
    *,
    task_ids: list[int],
    records: Mapping[int, tuple[dict, dict]],
) -> tuple[list[dict], list[dict]]:
    facts_rows = []
    candidate_rows = []
    for task_id in task_ids:
        if task_id not in records:
            raise RubricBuildError(f"task {task_id} is outside ShopSimulator goal ordering")
        product, instruction = records[task_id]
        facts = _make_task_facts(task_id, product, instruction)
        candidates = extract_rubric_candidates(facts)
        if not candidates.get("candidates"):
            raise RubricBuildError(f"task {task_id} produced no Rubric candidates")
        facts_rows.append(facts)
        candidate_rows.append(candidates)
    return facts_rows, candidate_rows


def _index_by_task(rows: list[dict]) -> dict[int, dict]:
    indexed = {}
    for row in rows:
        if "task_id" in row:
            indexed[int(row["task_id"])] = row
    return indexed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--product-corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_FLASH_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--rubric-version", default="rubric-final200-v1")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--build-candidates-only", action="store_true")
    parser.add_argument("--require-approved", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_tokens < 1 or args.retries < 0 or args.schema_retries < 0:
        raise SystemExit("token and retry values are invalid")
    task_ids = _load_tasks(args.tasks)
    if args.expected_count < 1 or len(task_ids) != args.expected_count:
        raise SystemExit(
            f"task manifest contains {len(task_ids)} rows; expected {args.expected_count}"
        )
    products = _load_products(args.product_corpus)
    records = _task_records(products)
    facts_rows, candidate_rows = _task_rows(task_ids=task_ids, records=records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "task_facts": args.output_dir / "task_facts.jsonl",
        "candidates": args.output_dir / "rubric_candidates.jsonl",
        "responses": args.output_dir / "curator_responses.jsonl",
        "rubrics": args.output_dir / "rubrics.jsonl",
        "review": args.output_dir / "review_queue.jsonl",
    }
    _write_jsonl(paths["task_facts"], facts_rows)
    _write_jsonl(paths["candidates"], candidate_rows)

    task_id_set = set(task_ids)
    existing_responses = {
        task_id: row
        for task_id, row in _index_by_task(_read_jsonl(paths["responses"])).items()
        if task_id in task_id_set
    }
    existing_rubrics = {
        task_id: row
        for task_id, row in _index_by_task(_read_jsonl(paths["rubrics"])).items()
        if task_id in task_id_set
    }
    response_rows = list(existing_responses.values())
    rubric_rows = []
    review_rows = []
    client = None
    if not args.build_candidates_only:
        if not args.api_key:
            raise SystemExit("--api-key or OPENAI_API_KEY is required")
        client = OpenAIJSONClient(
            model=args.model,
            base_url=_normalize_base_url(args.base_url),
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            response_format_json=True,
            thinking=False,
        )

    for facts, candidates in zip(facts_rows, candidate_rows):
        task_id = int(facts["task_id"])
        if task_id in existing_rubrics:
            rubric = validate_rubric_bundle(
                existing_rubrics[task_id], expected_task_id=task_id
            )
            rubric_rows.append(rubric)
            if any(item.get("hardness") == "needs_review" for item in rubric.get("rubrics", [])):
                review_rows.append(rubric)
            continue
        if args.build_candidates_only:
            continue
        messages = build_rubric_curator_messages(
            task_id=task_id,
            query=facts["query"],
            candidates=candidates["candidates"],
        )
        last_error = None
        response_payload = None
        for attempt in range(args.schema_retries + 1):
            try:
                response_payload = client.complete_json(messages)
                response = _strict_curator_response(
                    response_payload["result"], facts["query"]
                )
                rubric = materialize_rubric_bundle(
                    task_facts=facts,
                    candidates=candidates,
                    curator_response=response,
                    curator_model=args.model,
                    curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
                    rubric_version=args.rubric_version,
                )
                break
            except (ContractValidationError, KeyError, TypeError, ValueError) as exc:
                last_error = str(exc)
                if attempt >= args.schema_retries:
                    raise RubricBuildError(
                        f"task {task_id} failed Rubric gate after {attempt + 1} attempts: {last_error}"
                    ) from exc
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "上一次 JSON 未通过代码门禁。只修正 Schema/Query 原文引用问题，"
                            "不得新增候选、修改底层字段或期望值；重新输出唯一 JSON 对象。"
                        ),
                    }
                ]
        response_record = {
            "task_id": task_id,
            "task_data_hash": facts["task_data_hash"],
            "result": response,
            "metadata": response_payload.get("metadata", {}),
        }
        response_rows.append(response_record)
        rubric_rows.append(rubric)
        if any(item.get("hardness") == "needs_review" for item in rubric.get("rubrics", [])):
            review_rows.append(rubric)

    if not args.build_candidates_only:
        _write_jsonl(paths["responses"], sorted(response_rows, key=lambda row: int(row["task_id"])))
        _write_jsonl(paths["rubrics"], sorted(rubric_rows, key=lambda row: int(row["task_id"])))
        _write_jsonl(paths["review"], sorted(review_rows, key=lambda row: int(row["task_id"])))
        if len(rubric_rows) != len(task_ids):
            raise RubricBuildError(
                f"frozen Rubric count {len(rubric_rows)} does not equal task count {len(task_ids)}"
            )
        if args.require_approved and review_rows:
            raise RubricBuildError(
                f"{len(review_rows)} Rubric bundles still contain needs_review constraints"
            )

    hardness = Counter(
        item["hardness"]
        for rubric in rubric_rows
        for item in rubric.get("rubrics", [])
    )
    metadata = {
        "schema_version": "shopping-final200-rubric-build-v1",
        "task_manifest": str(args.tasks),
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "product_corpus": str(args.product_corpus),
        "task_facts_version": TASK_FACTS_VERSION,
        "candidate_version": RUBRIC_CANDIDATE_VERSION,
        "extractor_version": RUBRIC_EXTRACTOR_VERSION,
        "curator_model": args.model,
        "curator_prompt_version": RUBRIC_CURATOR_PROMPT_VERSION,
        "rubric_version": args.rubric_version,
        "build_candidates_only": bool(args.build_candidates_only),
        "rubric_count": len(rubric_rows),
        "constraint_count": sum(hardness.values()),
        "hardness_counts": dict(sorted(hardness.items())),
        "needs_review_bundle_count": len(review_rows),
        "files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
            if path.exists()
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
