"""Model-agnostic behavioral cloning trainer using JointEncoder + TrajectoryDataset.

Trains a neural network to predict next joint positions from current joint
positions (or to predict actions or velocities from positions).  The model
can be any ``nn.Module`` with matching input/output dimensions.

When a ``--scene`` spec is provided, the encoder discovers all joints
(including gripper prismatic joints), so the model learns to control
all joints end-to-end without any gripper hack.

Usage::

    # Train 8-joint velocity model (recommended — all joints incl. gripper)
    py -3.12 python/examples/train_joint_bc.py \\
        --trace output/arm_pick_dataset/dry_run_trace.json \\
        --scene python/clankers_synthetic/scenes/arm_pick_cube.json \\
        --mode velocity --control-dt 0.02

    # Train from a single trace file (auto-detect joints from obs dim)
    py -3.12 python/examples/train_joint_bc.py \\
        --trace output/arm_pick_dataset/dry_run_trace.json

    # Train from a packaged dataset directory
    py -3.12 python/examples/train_joint_bc.py \\
        --dataset-dir output/arm_pick_dataset/

    # Train action predictor (input=pos, output=action)
    py -3.12 python/examples/train_joint_bc.py \\
        --trace output/arm_pick_dataset/dry_run_trace.json \\
        --mode action

    # Train velocity predictor (6-DOF arm only)
    py -3.12 python/examples/train_joint_bc.py \\
        --trace output/arm_pick_dataset/dry_run_trace.json \\
        --mode velocity --control-dt 0.02
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "python")

from clankers.trajectory_dataset import TrajectoryDataset

# ---------------------------------------------------------------------------
# Default MLP model
# ---------------------------------------------------------------------------


class JointMLP(nn.Module):
    """Simple MLP for joint-position behavioral cloning."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    dataset: TrajectoryDataset,
    model: nn.Module,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_split: float = 0.1,
) -> nn.Module:
    """Train a model on the dataset with MSE loss."""
    n = len(dataset)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Training: {n_train} train, {n_val} val, {epochs} epochs")
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_pos, batch_target in train_loader:
            pred = model(batch_pos)
            loss = criterion(pred, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_pos)
            train_count += len(batch_pos)
        train_loss /= max(train_count, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_pos, batch_target in val_loader:
                pred = model(batch_pos)
                val_loss += criterion(pred, batch_target).item() * len(batch_pos)
                val_count += len(batch_pos)
        val_loss /= max(val_count, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

    print(f"Final val_loss: {val_loss:.6f}")
    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(
    model: nn.Module,
    dataset: TrajectoryDataset,
    path: str,
    mode: str = "position",
    control_dt: float = 0.02,
) -> None:
    """Export model to ONNX with joint encoder metadata."""
    model.eval()
    dummy = torch.randn(1, dataset.input_dim)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["joint_positions"],
        output_names=["predicted_targets"],
        dynamic_axes={
            "joint_positions": {0: "batch"},
            "predicted_targets": {0: "batch"},
        },
    )

    # Embed encoder metadata in ONNX model
    try:
        import onnx

        onnx_model = onnx.load(path)
        meta = onnx_model.metadata_props.add()
        meta.key = "clanker_joint_encoder"
        meta.value = dataset.encoder.to_json()
        meta = onnx_model.metadata_props.add()
        meta.key = "clanker_input_dim"
        meta.value = str(dataset.input_dim)
        meta = onnx_model.metadata_props.add()
        meta.key = "clanker_target_dim"
        meta.value = str(dataset.target_dim)
        meta = onnx_model.metadata_props.add()
        meta.key = "clanker_prediction_mode"
        meta.value = mode
        meta = onnx_model.metadata_props.add()
        meta.key = "clanker_control_dt"
        meta.value = str(control_dt)
        onnx.save(onnx_model, path)
        print(
            f"ONNX metadata embedded: {dataset.encoder.dof} joints, "
            f"input_dim={dataset.input_dim}, target_dim={dataset.target_dim}, "
            f"mode={mode}"
        )
    except ImportError:
        print("  (onnx package not installed — metadata not embedded)")

    print(f"Exported ONNX model to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a behavioral cloning model from trajectory data"
    )
    parser.add_argument("--trace", type=str, help="Path to a single trace JSON file")
    parser.add_argument("--dataset-dir", type=str, help="Path to packaged dataset dir")
    parser.add_argument("--scene", type=str, help="Scene spec JSON for joint names")
    parser.add_argument(
        "--joint-names",
        type=str,
        nargs="+",
        help="Explicit joint names (space-separated)",
    )
    parser.add_argument(
        "--mode",
        choices=["position", "action", "velocity"],
        default="position",
        help="Prediction mode: position (next pos), action (raw action), velocity (rad/s)",
    )
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="(deprecated, use --mode action) Train to predict actions",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.02,
        help="Control timestep for velocity mode (default: 0.02s)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--output", type=str, default="joint_bc.onnx", help="ONNX output path")
    args = parser.parse_args()

    if not args.trace and not args.dataset_dir:
        parser.error("Provide --trace or --dataset-dir")

    # Resolve mode (backward compat for --include-actions)
    mode = args.mode
    if args.include_actions:
        mode = "action"

    include_actions = mode == "action"
    include_velocities = mode == "velocity"

    # Load dataset
    joint_names = args.joint_names
    scene_spec = args.scene

    if args.trace:
        print(f"Loading trace: {args.trace} (mode={mode})")
        dataset = TrajectoryDataset.from_trace_file(
            args.trace,
            joint_names=joint_names,
            scene_spec=scene_spec,
            include_actions=include_actions,
            include_velocities=include_velocities,
            control_dt=args.control_dt,
        )
    else:
        print(f"Loading dataset: {args.dataset_dir} (mode={mode})")
        dataset = TrajectoryDataset.from_dataset_dir(
            args.dataset_dir,
            joint_names=joint_names,
            scene_spec=scene_spec,
            include_actions=include_actions,
            include_velocities=include_velocities,
            control_dt=args.control_dt,
        )

    print(
        f"Dataset: {len(dataset)} samples, input_dim={dataset.input_dim}, "
        f"target_dim={dataset.target_dim}"
    )
    print(f"Encoder: {dataset.encoder}")

    # Build model
    model = JointMLP(
        input_dim=dataset.input_dim,
        output_dim=dataset.target_dim,
        hidden=args.hidden,
    )

    # Train
    model = train(
        dataset,
        model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # Export ONNX
    export_onnx(model, dataset, args.output, mode=mode, control_dt=args.control_dt)

    # Save PyTorch checkpoint alongside
    pt_path = args.output.replace(".onnx", ".pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Saved PyTorch weights to {pt_path}")

    # Save encoder metadata alongside
    enc_path = args.output.replace(".onnx", "_encoder.json")
    dataset.encoder.save(enc_path)
    print(f"Saved encoder metadata to {enc_path}")


if __name__ == "__main__":
    main()
