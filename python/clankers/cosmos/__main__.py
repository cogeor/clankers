"""Entry point for python -m clankers.cosmos.

Dispatches to subcommands: download, prepare, infer, run.
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
    parser.add_argument("--cosmos-repo", type=Path, default=None, help="Cosmos repo path")
    parser.add_argument("--distilled", action="store_true", help="Use distilled model")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    args = parser.parse_args(remaining)

    from clankers.cosmos.infer import infer
    from clankers.cosmos.prepare import prepare

    print("=== Step 1: Prepare ===\n")
    num_steps = 4 if args.distilled else 35
    spec_path = prepare(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_steps=num_steps,
    )

    print("\n=== Step 2: Infer ===\n")
    infer(
        spec_path=spec_path,
        cosmos_repo=args.cosmos_repo,
        distilled=args.distilled,
        num_gpus=args.num_gpus,
    )


def main() -> None:
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="python -m clankers.cosmos",
        description="Cosmos-Transfer2.5 sim-to-real pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    # download
    sub.add_parser("download", help="Download Cosmos-Transfer2.5-2B model")

    # prepare
    sub.add_parser("prepare", help="Convert PNG frames to Cosmos input (MP4 + spec)")

    # infer
    sub.add_parser("infer", help="Run Cosmos inference on prepared input")

    # run (full pipeline)
    sub.add_parser("run", help="Full pipeline: prepare + infer")

    args, remaining = parser.parse_known_args()

    if args.command == "download":
        from clankers.cosmos.download import main as download_main

        sys.argv = [sys.argv[0], *remaining]
        download_main()
    elif args.command == "prepare":
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
