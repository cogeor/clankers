"""Run Cosmos-Transfer2.5 inference on prepared input.

Invokes the Cosmos inference script as a subprocess for clean isolation.
Requires the cosmos-transfer2.5 repository to be cloned locally.

Usage:
    python -m clankers.cosmos infer --spec output/cosmos/spec.json
    python -m clankers.cosmos infer --spec spec.json --cosmos-repo ~/cosmos-transfer2.5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from clankers.cosmos import DEFAULT_MODEL_DIR


def _find_cosmos_repo() -> Path | None:
    """Locate the cosmos-transfer2.5 repository."""
    # 1. Environment variable
    env_path = os.environ.get("COSMOS_REPO_PATH")
    if env_path:
        p = Path(env_path)
        if (p / "examples" / "inference.py").exists():
            return p

    # 2. Common locations
    candidates = [
        Path.home() / "cosmos-transfer2.5",
        Path.home() / "src" / "cosmos-transfer2.5",
        DEFAULT_MODEL_DIR / "cosmos-transfer2.5",
        Path.cwd() / "cosmos-transfer2.5",
    ]
    for c in candidates:
        if (c / "examples" / "inference.py").exists():
            return c

    return None


def infer(
    spec_path: Path,
    cosmos_repo: Path | None = None,
    model_dir: Path | None = None,
    distilled: bool = False,
    num_gpus: int = 1,
) -> Path:
    """Run Cosmos-Transfer2.5 inference.

    Parameters
    ----------
    spec_path : Path
        Path to spec.json (produced by prepare.py).
    cosmos_repo : Path, optional
        Path to cloned cosmos-transfer2.5 repository.
        Auto-detected from COSMOS_REPO_PATH env or common locations.
    model_dir : Path, optional
        Override model directory (for custom checkpoint locations).
    distilled : bool
        Use distilled (4-step) model instead of full model.
    num_gpus : int
        Number of GPUs for distributed inference (default: 1).

    Returns
    -------
    Path
        Path to the output directory containing generated video.
    """
    # Resolve cosmos repo
    if cosmos_repo is None:
        cosmos_repo = _find_cosmos_repo()
    if cosmos_repo is None:
        print(
            "Error: Cannot find cosmos-transfer2.5 repository.\n"
            "Either:\n"
            "  1. Set COSMOS_REPO_PATH environment variable\n"
            "  2. Clone to ~/cosmos-transfer2.5:\n"
            "     git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5\n"
            "  3. Pass --cosmos-repo /path/to/cosmos-transfer2.5",
            file=sys.stderr,
        )
        sys.exit(1)

    inference_script = cosmos_repo / "examples" / "inference.py"
    if not inference_script.exists():
        print(f"Error: {inference_script} not found", file=sys.stderr)
        sys.exit(1)

    # Load spec to check frame count
    spec = json.loads(spec_path.read_text())
    output_dir = Path(spec.get("output_dir", spec_path.parent / "output"))

    # Warn about distilled + long video
    num_steps = spec.get("num_steps", 35)
    if distilled and num_steps > 4:
        print(f"Warning: distilled model uses 4 steps, spec says {num_steps}. Overriding to 4.")
        spec["num_steps"] = 4
        spec_path.write_text(json.dumps(spec, indent=2))

    print(f"Cosmos inference:")
    print(f"  Spec: {spec_path}")
    print(f"  Repo: {cosmos_repo}")
    print(f"  Output: {output_dir}")
    print(f"  Model: {'distilled' if distilled else 'full'}")
    print(f"  GPUs: {num_gpus}")
    print()

    # Build command
    if num_gpus > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--master_port=12341",
            str(inference_script),
            "--params_file", str(spec_path),
        ]
    else:
        cmd = [
            sys.executable,
            str(inference_script),
            "--params_file", str(spec_path),
        ]

    if distilled:
        cmd.extend(["--model", "edge/distilled"])

    # Set environment
    env = os.environ.copy()
    if model_dir:
        env["HF_HOME"] = str(model_dir)

    # Run with real-time output
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(cosmos_repo),
    )

    if result.returncode != 0:
        print(f"\nCosmos inference failed (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nInference complete. Output: {output_dir}")
    return output_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Cosmos-Transfer2.5 inference")
    parser.add_argument("--spec", type=Path, required=True, help="Path to spec.json")
    parser.add_argument(
        "--cosmos-repo", type=Path, default=None,
        help="Path to cosmos-transfer2.5 repo (or set COSMOS_REPO_PATH)",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Override HF_HOME for model location",
    )
    parser.add_argument(
        "--distilled", action="store_true",
        help="Use distilled (4-step) model",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs (default: 1)",
    )
    args = parser.parse_args()

    infer(
        spec_path=args.spec,
        cosmos_repo=args.cosmos_repo,
        model_dir=args.model_dir,
        distilled=args.distilled,
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
