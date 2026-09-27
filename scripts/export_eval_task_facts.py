#!/usr/bin/env python3
"""Export private evaluation facts using the ShopSimulator runtime itself."""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "environments/ShopSimulator/shop_env"))

def main():
    parser = argparse.ArgumentParser(description="导出 Final-200 私有 TaskFacts")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from shopping_grpo.evaluation.artifacts import index_jsonl, write_jsonl_atomic
    from shopping_grpo.evaluation.rollout import load_tasks
    from shopping_grpo.evaluation.task_facts import task_facts_from_environment
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals
    from web_agent_site.utils import DEFAULT_FILE_PATH, DEBUG_PROD_SIZE
    task_ids = [row["task_id"] for row in load_tasks(args.benchmark)]
    if args.output.exists():
        cached = index_jsonl(args.output, key="task_id", allowed_keys=set(task_ids))
        if set(cached) != set(task_ids):
            raise ValueError("cached TaskFacts do not cover the benchmark")
        return
    products, product_dict, prices, _ = load_products(
        filepath=DEFAULT_FILE_PATH, num_products=DEBUG_PROD_SIZE
    )
    goals = get_goals(products, prices)
    # Match web_agent_site.app's fixed-task goal order without changing global RNG.
    random.Random(223).shuffle(goals)
    facts = task_facts_from_environment(
        task_ids=task_ids, goals=goals, product_item_dict=product_dict
    )
    write_jsonl_atomic(args.output, facts)


if __name__ == "__main__":
    main()
