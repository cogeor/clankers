"""Download Cosmos-Transfer2.5-2B model from HuggingFace.

Usage:
    python -m clankers.cosmos.download
    python -m clankers.cosmos.download --model-dir /path/to/models
    python -m clankers.cosmos.download --variant distilled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clankers.cosmos import COSMOS_HF_REPO, DEFAULT_MODEL_DIR


def download_model(
    model_dir: Path = DEFAULT_MODEL_DIR,
    variant: str = "general",
) -> Path:
    """Download Cosmos-Transfer2.5-2B from HuggingFace.

    Parameters
    ----------
    model_dir : Path
        Local directory for model storage. Created if needed.
    variant : str
        Model variant: "general" (full, supports autoregressive) or
        "distilled" (4-step, faster, 93 frames only).

    Returns
    -------
    Path
        Path to the downloaded model directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Error: huggingface_hub is required.\n"
            "Install with: pip install 'clankers[cosmos]'",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir.mkdir(parents=True, exist_ok=True)

    # Map variant to HF subdirectory pattern
    if variant == "distilled":
        allow = ["distilled/general/*", "LICENSE", "README.md"]
    elif variant == "general":
        allow = ["general/*", "LICENSE", "README.md"]
    else:
        print(f"Unknown variant '{variant}'. Use 'general' or 'distilled'.", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {COSMOS_HF_REPO} ({variant}) to {model_dir}")
    print("This may take a while on first download (~10-20 GB)...")

    local_dir = snapshot_download(
        repo_id=COSMOS_HF_REPO,
        local_dir=str(model_dir / "Cosmos-Transfer2.5-2B"),
        allow_patterns=allow,
    )

    print(f"Download complete: {local_dir}")
    return Path(local_dir)


def main() -> None:
    """CLI entry point for model download."""
    parser = argparse.ArgumentParser(
        description="Download Cosmos-Transfer2.5-2B model",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Model storage directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--variant",
        choices=["general", "distilled"],
        default="general",
        help="Model variant (default: general)",
    )
    args = parser.parse_args()
    download_model(model_dir=args.model_dir, variant=args.variant)


if __name__ == "__main__":
    main()
