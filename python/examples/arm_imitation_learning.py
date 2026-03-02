"""Imitation learning for the 6-DOF arm: behavioral cloning (offline & online).

Offline mode: loads expert demonstrations from MCAP files recorded by arm_bench.
Online mode: connects to the arm_gym server, runs a scripted expert to collect
data, then trains a BC policy.

Usage:
    # Offline (from arm_bench MCAP recordings)
    py -3.12 python/examples/arm_imitation_learning.py --offline recordings/

    # Online (from arm_gym server)
    py -3.12 python/examples/arm_imitation_learning.py --online --port 9879

    # Train + export ONNX
    py -3.12 python/examples/arm_imitation_learning.py --offline recordings/ --output arm_bc.onnx

    # Evaluate learned policy online
    py -3.12 python/examples/arm_imitation_learning.py --online --port 9879 --eval arm_bc.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "python")


# ---------------------------------------------------------------------------
# BC Policy
# ---------------------------------------------------------------------------


class BCPolicy(nn.Module):
    """Simple MLP behavioral cloning policy: obs (12) -> action (6)."""

    def __init__(self, obs_dim: int = 12, act_dim: int = 6, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Offline data loading
# ---------------------------------------------------------------------------


def load_offline_data(directory: str) -> tuple[np.ndarray, np.ndarray]:
    """Load MCAP recordings and return (observations, actions) arrays.

    Observations are 12-dim: 6 joint positions + 6 joint velocities.
    Actions are 6-dim: joint command targets.
    """
    from clankers.mcap_loader import McapEpisodeLoader

    mcap_files = sorted(Path(directory).glob("*.mcap"))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found in {directory}")

    all_obs = []
    all_actions = []

    for path in mcap_files:
        loader = McapEpisodeLoader(str(path))
        data = loader.load()

        positions = data.get("joint_positions")
        velocities = data.get("joint_velocities")
        actions = data.get("actions")

        if positions is None or velocities is None or actions is None:
            print(f"  skipping {path.name}: missing data channels")
            continue

        # Truncate to 6 joints if needed (e.g., 8-DOF with gripper)
        pos = positions[:, :6] if positions.shape[1] > 6 else positions
        vel = velocities[:, :6] if velocities.shape[1] > 6 else velocities
        act = actions[:, :6] if actions.shape[1] > 6 else actions

        # Observations: concat positions + velocities -> 12-dim
        obs = np.concatenate([pos, vel], axis=1)

        t = min(len(obs), len(act))
        all_obs.append(obs[:t])
        all_actions.append(act[:t])

        print(f"  loaded {path.name}: {t} steps")

    if not all_obs:
        raise ValueError("No valid episodes found")

    return np.concatenate(all_obs), np.concatenate(all_actions)


# ---------------------------------------------------------------------------
# Online data collection
# ---------------------------------------------------------------------------

# Scripted expert: cycle through predefined joint angle targets
EXPERT_TARGETS = [
    [0.0, -0.5, 1.0, 0.0, -0.5, 0.0],
    [0.5, -0.3, 0.8, 0.2, -0.3, 0.1],
    [-0.5, -0.7, 1.2, -0.2, -0.6, -0.1],
    [0.3, -0.4, 0.6, 0.1, -0.4, 0.2],
    [0.0, -0.5, 1.0, 0.0, -0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]


def collect_online_data(
    host: str, port: int, n_episodes: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Connect to arm_gym and collect expert demonstrations."""
    from clankers.env import ClankerEnv

    env = ClankerEnv(host=host, port=port)
    env.connect()
    print(f"  connected to {host}:{port}")

    all_obs = []
    all_actions = []

    for ep in range(n_episodes):
        obs, _info = env.reset()
        ep_obs = []
        ep_actions = []

        steps_per_target = 50
        for step in range(300):
            # Scripted expert: cycle through targets
            target_idx = (step // steps_per_target) % len(EXPERT_TARGETS)
            action = np.array(EXPERT_TARGETS[target_idx], dtype=np.float32)

            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())

            obs, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

        all_obs.append(np.array(ep_obs))
        all_actions.append(np.array(ep_actions))
        print(f"  episode {ep + 1}/{n_episodes}: {len(ep_obs)} steps")

    env.close()
    return np.concatenate(all_obs), np.concatenate(all_actions)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_bc(
    obs: np.ndarray,
    actions: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.1,
) -> BCPolicy:
    """Train a BC policy with MSE loss."""
    n = len(obs)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    # Shuffle
    indices = np.random.permutation(n)
    obs = obs[indices]
    actions = actions[indices]

    obs_train = torch.tensor(obs[:n_train], dtype=torch.float32)
    act_train = torch.tensor(actions[:n_train], dtype=torch.float32)
    obs_val = torch.tensor(obs[n_train:], dtype=torch.float32)
    act_val = torch.tensor(actions[n_train:], dtype=torch.float32)

    policy = BCPolicy(obs_dim=obs.shape[1], act_dim=actions.shape[1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(obs_train, act_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"\nTraining BC: {n_train} train, {n_val} val, {epochs} epochs")

    for epoch in range(epochs):
        policy.train()
        train_loss = 0.0
        for batch_obs, batch_act in train_loader:
            pred = policy(batch_obs)
            loss = criterion(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_obs)
        train_loss /= n_train

        # Validation
        policy.eval()
        with torch.no_grad():
            val_pred = policy(obs_val)
            val_loss = criterion(val_pred, act_val).item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    return policy


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(policy: BCPolicy, path: str) -> None:
    """Export the trained policy to ONNX with clanker metadata."""
    policy.eval()
    dummy = torch.randn(1, 12)

    torch.onnx.export(
        policy,
        dummy,
        path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action": {0: "batch"},
        },
    )
    print(f"\nExported ONNX model to {path}")


# ---------------------------------------------------------------------------
# Online evaluation
# ---------------------------------------------------------------------------


def evaluate_policy_online(
    onnx_path: str, host: str, port: int, n_episodes: int = 3
) -> None:
    """Run the learned policy on arm_gym and measure tracking error."""
    from clankers.env import ClankerEnv

    # Load policy
    policy = BCPolicy()
    # Load from ONNX via torch for evaluation
    state_dict = torch.load(
        onnx_path.replace(".onnx", ".pt"), weights_only=True
    ) if os.path.exists(onnx_path.replace(".onnx", ".pt")) else None

    if state_dict is not None:
        policy.load_state_dict(state_dict)
    else:
        print("  warning: no .pt weights found, using random policy for eval")

    policy.eval()

    env = ClankerEnv(host=host, port=port)
    env.connect()

    for ep in range(n_episodes):
        obs, _info = env.reset()
        total_error = 0.0
        steps = 0

        for _step in range(300):
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = policy(obs_t).squeeze(0).numpy()

            obs, terminated, truncated, _info = env.step(action)
            # Tracking error: distance between commanded and observed positions
            total_error += float(np.sum((action - obs[:6]) ** 2))
            steps += 1

            if terminated or truncated:
                break

        avg_error = total_error / max(steps, 1)
        print(f"  eval episode {ep + 1}: {steps} steps, avg_tracking_mse={avg_error:.6f}")

    env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Arm imitation learning (behavioral cloning)")
    parser.add_argument("--offline", type=str, default=None, help="Directory of MCAP recordings")
    parser.add_argument("--online", action="store_true", help="Collect data from arm_gym server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Gym server host")
    parser.add_argument("--port", type=int, default=9879, help="Gym server port")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to collect (online)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--output", type=str, default="arm_bc.onnx", help="ONNX output path")
    parser.add_argument("--eval", type=str, default=None, help="Evaluate ONNX policy online")
    args = parser.parse_args()

    if args.eval:
        print(f"Evaluating policy: {args.eval}")
        evaluate_policy_online(args.eval, args.host, args.port)
        return

    if not args.offline and not args.online:
        parser.error("Specify --offline <dir> or --online")

    # Collect data
    if args.offline:
        print(f"Loading offline data from: {args.offline}")
        obs, actions = load_offline_data(args.offline)
    else:
        print(f"Collecting online data from {args.host}:{args.port}")
        obs, actions = collect_online_data(args.host, args.port, args.episodes)

    print(f"\nDataset: {len(obs)} samples, obs_dim={obs.shape[1]}, act_dim={actions.shape[1]}")

    # Train
    policy = train_bc(obs, actions, epochs=args.epochs)

    # Export
    export_onnx(policy, args.output)

    # Also save PyTorch weights for evaluation
    pt_path = args.output.replace(".onnx", ".pt")
    torch.save(policy.state_dict(), pt_path)
    print(f"Saved PyTorch weights to {pt_path}")


if __name__ == "__main__":
    main()
