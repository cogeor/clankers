"""Command-line interface for clankers_synthetic."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for dataset generation."""
    parser = argparse.ArgumentParser(
        prog="clankers_synthetic",
        description="Generate synthetic trajectory datasets using LLM-proposed skill plans.",
    )
    parser.add_argument("--scene", required=True, help="Path to scene spec JSON file")
    parser.add_argument("--task", required=True, help="Path to task spec JSON file")
    parser.add_argument("--out", required=True, help="Output directory for dataset")
    parser.add_argument("--n-plans", type=int, default=10, help="Number of plans to generate")
    parser.add_argument("--model", default="gpt-5", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-refine-iters", type=int, default=3, help="Max refinement iterations"
    )
    parser.add_argument(
        "--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)"
    )

    args = parser.parse_args(argv)

    # Load specs -- import here to keep top-level imports lightweight
    from clankers_synthetic.specs import SceneSpec, TaskSpec

    with open(args.scene) as f:
        scene = SceneSpec(**json.load(f))

    with open(args.task) as f:
        task = TaskSpec(**json.load(f))

    # Import pipeline here to avoid circular imports at module level
    from clankers_synthetic.pipeline import generate_dataset

    manifest = generate_dataset(
        scene=scene,
        task=task,
        output_dir=args.out,
        n_plans=args.n_plans,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_refine_iters=args.max_refine_iters,
        openai_api_key=args.api_key,
    )

    print(f"Dataset generated: {manifest.n_episodes} episodes in {manifest.output_dir}")
    return 0
