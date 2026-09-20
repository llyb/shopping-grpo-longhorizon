"""Turn raw Teacher rollouts into reproducible SFT JSONL artifacts.

The collector records everything needed for auditing in ``raw.jsonl``. This
module is the deterministic second half of the pipeline: it accepts only strict
Reward v4 gold purchases, removes private reasoning and terminal Reward text,
excludes held-out tasks, and creates task-disjoint train/validation files.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

from shopping_grpo.environment.actions import (
    RUNTIME_GUARD_FIELD,
    action_reject_reason,
)
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action
from shopping_grpo.training.sft.dataset import split_rows_by_task


COLLECTION_SCHEMA_VERSION = "shopping-sft-collection-v2"
ALLOWED_MESSAGE_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}
ALLOWED_TOOL_CALL_KEYS = {"id", "type", "function"}
ALLOWED_FUNCTION_KEYS = {"name", "arguments"}


def acceptance_reasons(trajectory: dict) -> tuple[bool, list[str]]:
    """Return whether one rollout is safe to train on and every rejection reason."""

    reasons = []
    steps = trajectory.get("steps") or []
    terminal = trajectory.get("terminal_result") or {}
    reward = terminal.get("reward_detail") or {}

    if trajectory.get("error"):
        reasons.append("has_error")
    if trajectory.get("release_error"):
        reasons.append("release_error")
    if trajectory.get("infrastructure_invalid") is True:
        reasons.append("infrastructure_invalid")
    if trajectory.get("status") != "done":
        reasons.append("status_not_done")
    if trajectory.get("done") is not True:
        reasons.append("trajectory_not_done")
    if terminal.get("done") is not True or terminal.get("over") is not True:
        reasons.append("environment_not_done")
    if not any(
        step.get("tool_name") == "buy_now"
        or step.get("env_action") == "click[Buy Now]"
        for step in steps
    ):
        reasons.append("missing_buy")

    if reward.get("reward_version") != "shopsimulator-reward-v4":
        reasons.append("reward_v4_required")
    if reward.get("reward_type") != "gold_purchase":
        reasons.append("reward_v4_not_gold_purchase")
    if reward.get("reward_valid") is not True:
        reasons.append("reward_v4_invalid")
    if reward.get("sampling_invalid") is not False:
        reasons.append("sampling_invalid")
    if reward.get("purchase_success") is not True:
        reasons.append("purchase_not_successful")
    if reward.get("termination_reason") != "gold_purchase":
        reasons.append("termination_not_gold_purchase")
    if reward.get("strict_success") is not True:
        reasons.append("strict_success_required")
    if float(reward.get("evidence_coverage", 0.0)) < 1.0:
        reasons.append("incomplete_key_evidence")

    for index, message in enumerate(trajectory.get("messages") or []):
        if (
            message.get("role") == "assistant"
            and len(message.get("tool_calls") or []) > 1
        ):
            reasons.append(f"message_{index}.multiple_tool_calls")

    for index, step in enumerate(steps):
        reason = _tool_step_reject_reason(trajectory, step)
        if reason:
            reasons.append(f"step_{index}.{reason}")

    return not reasons, reasons


def build_sft_row(trajectory: dict) -> dict:
    """Build one action-only training row without audit-only Teacher content."""

    terminal_tool_call_id = _terminal_tool_call_id(trajectory)
    blocked_call_ids = {
        (blocked.get("tool_call") or {}).get("id")
        for blocked in trajectory.get("blocked_tool_calls") or []
    }
    blocked_call_ids.discard(None)
    return {
        "trajectory_id": trajectory.get("trajectory_id"),
        "task_id": int(trajectory["task_id"]),
        "messages": _training_messages(
            trajectory.get("messages") or [],
            blocked_call_ids,
            terminal_tool_call_id,
        ),
        "tools": deepcopy(SHOP_TOOL_SCHEMAS),
    }


def build_collection_artifacts(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    held_out_task_ids: Iterable[int] = (),
    validation_ratio: float = 0.1,
    seed: int = 42,
    collection_config: dict | None = None,
) -> dict:
    """Rebuild all derived files from raw rollouts, which remain the source of truth."""

    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    held_out = {int(task_id) for task_id in held_out_task_ids}
    accepted_trajectories = []
    rejected_rows = []
    sft_rows = []
    accepted_task_ids = set()
    reject_reasons = Counter()
    total = 0
    held_out_excluded = 0
    duplicate_tasks_excluded = 0
    quality_rejected = 0

    for trajectory in read_jsonl(raw_path):
        total += 1
        accepted, reasons = acceptance_reasons(trajectory)
        task_id = int(trajectory["task_id"])
        if accepted and task_id in held_out:
            accepted = False
            reasons = ["held_out_task"]
            held_out_excluded += 1
        elif accepted and task_id in accepted_task_ids:
            accepted = False
            reasons = ["duplicate_task"]
            duplicate_tasks_excluded += 1
        elif not accepted:
            quality_rejected += 1

        if accepted:
            accepted_task_ids.add(task_id)
            accepted_trajectories.append(trajectory)
            sft_rows.append(build_sft_row(trajectory))
            continue

        reject_reasons.update(reasons)
        rejected_rows.append(
            {
                "trajectory_id": trajectory.get("trajectory_id"),
                "task_id": task_id,
                "status": trajectory.get("status"),
                "reject_reasons": reasons,
            }
        )

    train_rows, validation_rows = split_rows_by_task(
        sft_rows,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "accepted": output_dir / "accepted.jsonl",
        "rejected": output_dir / "rejected.jsonl",
        "sft": output_dir / "sft.jsonl",
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "stats": output_dir / "reject_stats.json",
        "metadata": output_dir / "metadata.json",
    }
    write_jsonl(paths["accepted"], accepted_trajectories)
    write_jsonl(paths["rejected"], rejected_rows)
    write_jsonl(paths["sft"], sft_rows)
    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["validation"], validation_rows)

    summary = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "total": total,
        "accepted": len(sft_rows),
        "rejected": quality_rejected,
        "held_out_excluded": held_out_excluded,
        "duplicate_tasks_excluded": duplicate_tasks_excluded,
        "train": len(train_rows),
        "validation": len(validation_rows),
        "reject_reasons": dict(sorted(reject_reasons.items())),
    }
    paths["stats"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if collection_config is None and paths["metadata"].exists():
        previous_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        collection_config = previous_metadata.get("collection_config") or {}
    metadata = {
        **summary,
        "environment": "shopsimulator-environment-v2.1",
        "reward": "shopsimulator-reward-v4",
        "validation_ratio": float(validation_ratio),
        "split_seed": int(seed),
        "collection_config": deepcopy(collection_config or {}),
        "files": {
            name: {
                "path": str(path),
                "rows": _line_count(path),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
            if name not in {"stats", "metadata"}
        },
        "raw": {
            "path": str(raw_path),
            "rows": total,
            "sha256": _sha256(raw_path),
        },
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def task_ids_from_jsonl(path: str | Path | None) -> set[int]:
    """Read task IDs from a benchmark/task JSONL; a missing optional file means none."""

    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        return set()
    return {int(row["task_id"]) for row in read_jsonl(path)}


def read_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tool_step_reject_reason(trajectory: dict, step: dict) -> str | None:
    tool_call = step.get("tool_call") or {}
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw_arguments = function.get("arguments")
    if not isinstance(name, str) or not name:
        return "missing_tool_name"
    try:
        arguments = json.loads(raw_arguments or "{}")
    except (TypeError, json.JSONDecodeError):
        return "invalid_arguments_json"
    if not isinstance(arguments, dict):
        return "arguments_not_object"
    if name != step.get("tool_name"):
        return "tool_name_mismatch"
    try:
        expected_action = tool_call_to_action(name, arguments)
    except (KeyError, TypeError, ValueError):
        return "unknown_or_invalid_tool"
    if expected_action != step.get("env_action"):
        return "env_action_mismatch"
    return action_reject_reason(
        name,
        arguments,
        _previous_observation(trajectory, tool_call.get("id")),
    )


def _previous_observation(trajectory: dict, tool_call_id: str | None) -> str:
    if not tool_call_id:
        return ""
    messages = trajectory.get("messages") or []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if not any(
            call.get("id") == tool_call_id
            for call in message.get("tool_calls") or []
        ):
            continue
        for previous in reversed(messages[:index]):
            if previous.get(RUNTIME_GUARD_FIELD) is True:
                continue
            if previous.get("role") == "tool":
                content = previous.get("content")
                return content if isinstance(content, str) else ""
        return ""
    return ""


def _terminal_tool_call_id(trajectory: dict) -> str | None:
    if trajectory.get("done") is not True:
        return None
    terminal_steps = [step for step in trajectory.get("steps") or [] if step.get("done")]
    if not terminal_steps:
        return None
    return (terminal_steps[-1].get("tool_call") or {}).get("id")


def _training_messages(messages, blocked_call_ids, terminal_tool_call_id):
    clean_messages = []
    for message in messages:
        if _is_blocked_training_message(message, blocked_call_ids):
            continue
        clean_messages.append(_sanitize_message(message, terminal_tool_call_id))
    return clean_messages


def _is_blocked_training_message(message, blocked_call_ids):
    if message.get(RUNTIME_GUARD_FIELD) is True:
        return True
    if message.get("role") == "tool":
        return message.get("tool_call_id") in blocked_call_ids
    if message.get("role") == "assistant":
        return any(
            call.get("id") in blocked_call_ids
            for call in message.get("tool_calls") or []
        )
    return False


def _sanitize_message(message, terminal_tool_call_id):
    clean = {key: deepcopy(message[key]) for key in ALLOWED_MESSAGE_KEYS if key in message}
    if clean.get("role") == "tool" and clean.get("tool_call_id") == terminal_tool_call_id:
        clean["content"] = "购买已完成。"
    if "tool_calls" in clean:
        clean["tool_calls"] = [_sanitize_tool_call(call) for call in clean["tool_calls"]]
    return clean


def _sanitize_tool_call(tool_call):
    clean = {
        key: deepcopy(tool_call[key])
        for key in ALLOWED_TOOL_CALL_KEYS
        if key in tool_call
    }
    if "function" in clean:
        clean["function"] = {
            key: clean["function"][key]
            for key in ALLOWED_FUNCTION_KEYS
            if key in clean["function"]
        }
    return clean


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
