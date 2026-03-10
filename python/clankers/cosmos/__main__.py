"""Entry point for python -m clankers.cosmos.

Dispatches to subcommands: prepare, infer, run.
"""

from __future__ import annotations

import argparse
import sys


def _run_full_pipeline(remaining: list[str]) -> None:
    """Run prepare + infer as a single command."""
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Full Cosmos pipeline: prepare + infer")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with PNG frames")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--model-id", type=str, default=None, help="HuggingFace model ID")
    parser.add_argument("--control", type=str, default="edge", help="ControlNet variant")
    args = parser.parse_args(remaining)

    from clankers.cosmos.infer import infer
    from clankers.cosmos.prepare import prepare

    print("=== Step 1: Prepare ===\n")
    spec_path = prepare(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    print("\n=== Step 2: Infer ===\n")
    kwargs: dict[str, object] = {
        "spec_path": spec_path,
        "control_type": args.control,
    }
    if args.model_id:
        kwargs["model_id"] = args.model_id
    infer(**kwargs)


def main() -> None:
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="python -m clankers.cosmos",
        description="Cosmos-Transfer2.5 sim-to-real pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    # prepare
    sub.add_parser("prepare", help="Convert PNG frames to Cosmos input (MP4 + spec)")

    # infer
    sub.add_parser("infer", help="Run Cosmos inference on prepared input")

    # run (full pipeline)
    sub.add_parser("run", help="Full pipeline: prepare + infer")

    args, remaining = parser.parse_known_args()

    if args.command == "prepare":
        from clankers.cosmos.prepare import main as prepare_main

        sys.argv = [sys.argv[0], *remaining]
        prepare_main()
    elif args.command == "infer":
        from clankers.cosmos.infer import main as infer_main

        sys.argv = [sys.argv[0], *remaining]
        infer_main()
    elif args.command == "run":
        _run_full_pipeline(remaining)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
