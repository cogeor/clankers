#!/usr/bin/env python3
"""Train a vision-based behavioral cloning policy.

Architecture: image + joint positions → CNN + position encoder → velocity.

Data: MCAP episodes recorded by ``clankers-record`` with camera images
and joint states.

Usage::

    # From a directory of MCAP recordings:
    py -3.12 python/examples/train_vision_bc.py \\
        --data-dir output/arm_episodes/ \\
        --joint-dim 6 --epochs 100 --output vision_bc.onnx

    # From a single MCAP file:
    py -3.12 python/examples/train_vision_bc.py \\
        --mcap output/episode_001.mcap \\
        --joint-dim 6 --output vision_bc.onnx
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Imports from clankers
# ---------------------------------------------------------------------------

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from clankers.vision_dataset import VisionPositionDataset
from clankers.vision_model import VisionPolicyNet

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    model: VisionPolicyNet,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    lr: float,
) -> VisionPolicyNet:
    """Train the model with MSE loss on velocity targets."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        n_train = 0
        for images, positions, velocities in train_loader:
            optimizer.zero_grad()
            pred = model(images, positions)
            loss = criterion(pred, velocities)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
            n_train += len(images)
        train_loss /= max(n_train, 1)

        # -- Validate --
        val_loss_str = ""
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for images, positions, velocities in val_loader:
                    pred = model(images, positions)
                    loss = criterion(pred, velocities)
                    val_loss += loss.item() * len(images)
                    n_val += len(images)
            val_loss /= max(n_val, 1)
            val_loss_str = f"  val_loss={val_loss:.6f}"

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch:>4}/{epochs}  train_loss={train_loss:.6f}{val_loss_str}")

    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(
    model: VisionPolicyNet,
    image_shape: tuple[int, int, int],
    joint_dim: int,
    path: str,
    control_dt: float = 0.02,
) -> None:
    """Export model to ONNX with two named inputs and metadata."""
    model.eval()

    c, h, w = image_shape
    dummy_image = torch.randn(1, c, h, w)
    dummy_positions = torch.randn(1, joint_dim)

    torch.onnx.export(
        model,
        (dummy_image, dummy_positions),
        path,
        input_names=["image", "joint_positions"],
        output_names=["velocity"],
        dynamic_axes={
            "image": {0: "batch"},
            "joint_positions": {0: "batch"},
            "velocity": {0: "batch"},
        },
        opset_version=17,
    )

    # Embed metadata for Rust-side loading
    try:
        import json

        import onnx

        onnx_model = onnx.load(path)

        metadata = {
            "clanker_policy_type": "vision",
            "clanker_image_channels": str(c),
            "clanker_image_height": str(h),
            "clanker_image_width": str(w),
            "clanker_joint_dim": str(joint_dim),
            "clanker_velocity_dim": str(model.velocity_dim),
            "clanker_prediction_mode": "velocity",
            "clanker_control_dt": str(control_dt),
            "action_transform": "none",
            "action_scale": json.dumps([1.0] * model.velocity_dim),
            "action_offset": json.dumps([0.0] * model.velocity_dim),
        }

        for key, value in metadata.items():
            entry = onnx_model.metadata_props.add()
            entry.key = key
            entry.value = value

        onnx.save(onnx_model, path)
        print(f"ONNX metadata embedded: image=({c},{h},{w}), joints={joint_dim}")
    except ImportError:
        print("  (onnx package not installed — metadata not embedded)")

    print(f"Exported ONNX model to {path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_onnx(path: str, image_shape: tuple[int, int, int], joint_dim: int) -> None:
    """Validate exported ONNX model can run inference."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(path)
        c, h, w = image_shape
        dummy_img = np.random.randn(1, c, h, w).astype(np.float32)
        dummy_pos = np.random.randn(1, joint_dim).astype(np.float32)

        inputs = {
            session.get_inputs()[0].name: dummy_img,
            session.get_inputs()[1].name: dummy_pos,
        }
        outputs = session.run(None, inputs)
        print(f"ONNX validation passed: output shape = {outputs[0].shape}")
    except ImportError:
        print("  (onnxruntime not installed — skipping validation)")
    except Exception as e:
        print(f"  ONNX validation failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a vision-based behavioral cloning policy (image + positions → velocity)",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="Directory of MCAP episode files")
    src.add_argument("--mcap", help="Single MCAP episode file")

    p.add_argument("--joint-dim", type=int, default=6, help="Number of joints (default: 6)")
    p.add_argument(
        "--control-dt", type=float, default=0.02, help="Control timestep (default: 0.02)"
    )
    p.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--hidden", type=int, default=128, help="Hidden layer size (default: 128)")
    p.add_argument("--cnn-features", type=int, default=128, help="CNN feature dim (default: 128)")
    p.add_argument("--val-split", type=float, default=0.1, help="Validation split (default: 0.1)")
    p.add_argument(
        "--output", default="vision_bc.onnx", help="Output ONNX path (default: vision_bc.onnx)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load dataset
    print("Loading dataset...")
    if args.data_dir:
        dataset = VisionPositionDataset.from_mcap_dir(
            args.data_dir,
            joint_dim=args.joint_dim,
            control_dt=args.control_dt,
        )
    else:
        dataset = VisionPositionDataset.from_mcap_file(
            args.mcap,
            joint_dim=args.joint_dim,
            control_dt=args.control_dt,
        )

    print(f"  {len(dataset)} samples, image={dataset.image_shape}, joints={dataset.joint_dim}")

    # 2. Train/val split
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # 3. Build model
    c, _h, _w = dataset.image_shape
    model = VisionPolicyNet(
        image_channels=c,
        joint_dim=dataset.joint_dim,
        velocity_dim=dataset.joint_dim,
        cnn_features=args.cnn_features,
        pos_features=64,
        hidden=args.hidden,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters")

    # 4. Train
    print("Training...")
    model = train(model, train_loader, val_loader, args.epochs, args.lr)

    # 5. Save PyTorch weights
    pt_path = args.output.replace(".onnx", ".pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Saved PyTorch weights to {pt_path}")

    # 6. Export ONNX
    export_onnx(model, dataset.image_shape, dataset.joint_dim, args.output, args.control_dt)

    # 7. Validate
    validate_onnx(args.output, dataset.image_shape, dataset.joint_dim)


if __name__ == "__main__":
    main()
