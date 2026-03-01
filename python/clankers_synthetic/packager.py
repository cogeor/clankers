"""Dataset packager: writes validated traces to JSON episodes + metadata + provenance."""
from __future__ import annotations

import json
import os
import random
from typing import Any

from clankers_synthetic.specs import (
    DatasetManifest,
    ExecutionTrace,
    ValidationReport,
)


class DatasetPackager:
    """Convert validated execution traces into a dataset directory.

    Output structure::

        dataset/
        +-- metadata.json           # Schema version, provenance, stats
        +-- episodes/
        |   +-- ep_000000.json      # Episode data (JSON)
        |   +-- ep_000001.json
        |   +-- ...
        +-- splits.json             # train/val/test split indices
        +-- provenance/
            +-- plans/              # CanonicalPlan JSON per episode
            +-- validation/         # ValidationReport per episode
            +-- prompts/            # LLM prompts/responses (optional)
    """

    SCHEMA_VERSION = "1.0.0"

    def package(
        self,
        traces: list[tuple[ExecutionTrace, ValidationReport]],
        output_dir: str,
        scene_spec_hash: str = "",
        task_spec_hash: str = "",
        prompt_template_version: str = "1.0.0",
        llm_model: str = "gpt-5",
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        plans: list[dict] | None = None,
        prompts: list[dict] | None = None,
    ) -> DatasetManifest:
        """Write dataset to output_dir.

        Args:
            traces: List of (ExecutionTrace, ValidationReport) tuples.
            output_dir: Directory to write to (created if needed).
            scene_spec_hash: SHA-256 hash of SceneSpec JSON.
            task_spec_hash: SHA-256 hash of TaskSpec JSON.
            prompt_template_version: Version of prompt template used.
            llm_model: LLM model used for generation.
            split_ratios: (train, val, test) ratios, must sum to 1.0.
            seed: Random seed for deterministic splits.
            plans: Optional list of CanonicalPlan dicts (one per trace).
            prompts: Optional list of prompt/response dicts (one per trace).

        Returns:
            DatasetManifest with paths, stats, split sizes.

        Raises:
            ValueError: If traces is empty.
        """
        n = len(traces)
        if n == 0:
            raise ValueError("No traces to package")

        # Create directory structure
        episodes_dir = os.path.join(output_dir, "episodes")
        provenance_dir = os.path.join(output_dir, "provenance")
        plans_dir = os.path.join(provenance_dir, "plans")
        validation_dir = os.path.join(provenance_dir, "validation")
        prompts_dir = os.path.join(provenance_dir, "prompts")

        for d in [episodes_dir, plans_dir, validation_dir, prompts_dir]:
            os.makedirs(d, exist_ok=True)

        # Write episodes and collect stats
        rewards = []
        lengths = []
        max_forces = []

        for i, (trace, report) in enumerate(traces):
            # Write episode data as JSON
            ep_path = os.path.join(episodes_dir, f"ep_{i:06d}.json")
            ep_data = {
                "plan_id": trace.plan_id,
                "steps": [
                    {
                        "obs": step.obs,
                        "action": step.action,
                        "next_obs": step.next_obs,
                        "reward": step.reward,
                        "terminated": step.terminated,
                        "truncated": step.truncated,
                    }
                    for step in trace.steps
                ],
                "total_reward": trace.total_reward,
                "terminated": trace.terminated,
                "truncated": trace.truncated,
            }
            with open(ep_path, "w") as f:
                json.dump(ep_data, f)

            # Write validation report
            val_path = os.path.join(validation_dir, f"ep_{i:06d}.json")
            with open(val_path, "w") as f:
                json.dump(report.dict(), f, indent=2, default=str)

            # Write plan provenance
            if plans and i < len(plans):
                plan_path = os.path.join(plans_dir, f"ep_{i:06d}.json")
                with open(plan_path, "w") as f:
                    json.dump(plans[i], f, indent=2, default=str)

            # Write prompt provenance
            if prompts and i < len(prompts):
                prompt_path = os.path.join(prompts_dir, f"ep_{i:06d}.json")
                with open(prompt_path, "w") as f:
                    json.dump(prompts[i], f, indent=2, default=str)

            # Collect stats
            rewards.append(trace.total_reward)
            lengths.append(len(trace.steps))
            max_forces.append(report.metrics.max_contact_force)

        # Generate deterministic splits
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)

        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        splits = {
            "train": sorted(indices[:n_train]),
            "val": sorted(indices[n_train : n_train + n_val]),
            "test": sorted(indices[n_train + n_val :]),
        }

        splits_path = os.path.join(output_dir, "splits.json")
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=2)

        # Compute stats
        mean_reward = sum(rewards) / n
        std_reward = (
            (sum((r - mean_reward) ** 2 for r in rewards) / n) ** 0.5
            if n > 1
            else 0.0
        )

        stats = {
            "mean_reward": round(mean_reward, 4),
            "std_reward": round(std_reward, 4),
            "mean_episode_length": round(sum(lengths) / n, 1),
            "mean_contact_force": round(sum(max_forces) / n, 2),
            "max_contact_force": round(max(max_forces), 2),
        }

        # Write metadata
        split_sizes = {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }

        metadata = {
            "schema_version": self.SCHEMA_VERSION,
            "generator": "clankers-synthetic",
            "n_episodes": n,
            "scene_spec_hash": scene_spec_hash,
            "task_spec_hash": task_spec_hash,
            "prompt_template_version": prompt_template_version,
            "llm_model": llm_model,
            "stats": stats,
            "split_sizes": split_sizes,
        }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return DatasetManifest(
            output_dir=output_dir,
            n_episodes=n,
            n_original=n,
            n_augmented=0,
            schema_version=self.SCHEMA_VERSION,
            scene_spec_hash=scene_spec_hash,
            task_spec_hash=task_spec_hash,
            prompt_template_version=prompt_template_version,
            llm_model=llm_model,
            stats=stats,
            split_sizes=split_sizes,
        )
