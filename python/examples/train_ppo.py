#!/usr/bin/env python3
"""Minimal PPO training example using Stable-Baselines3.

Connects to a running Clankers simulation server and trains a PPO agent
with a configurable reward function.

Usage
-----
1. Start a Clankers server:  ``cargo run``
2. Train the agent:          ``python examples/train_ppo.py``

Requirements: ``pip install clanker-gym[sb3]``
"""

from __future__ import annotations

import argparse
import sys


def make_env(
    host: str,
    port: int,
    reward_type: str,
) -> object:
    """Create a ClankerGymnasiumEnv with the specified reward function.

    Parameters
    ----------
    host : str
        Clankers server address.
    port : int
        Clankers server port.
    reward_type : str
        One of "distance", "sparse", "composite".

    Returns
    -------
    ClankerGymnasiumEnv
        Gymnasium-compatible environment.
    """
    from clanker_gym.gymnasium_env import ClankerGymnasiumEnv
    from clanker_gym.rewards import (
        ActionPenaltyReward,
        CompositeReward,
        DistanceReward,
        SparseReward,
    )

    # Default: first 3 obs values are end-effector pos, next 3 are target pos.
    ee_indices = [0, 1, 2]
    target_indices = [3, 4, 5]

    if reward_type == "distance":
        reward_fn = DistanceReward(
            pos_a_indices=ee_indices,
            pos_b_indices=target_indices,
        )
    elif reward_type == "sparse":
        reward_fn = SparseReward(
            pos_a_indices=ee_indices,
            pos_b_indices=target_indices,
            threshold=0.05,
        )
    elif reward_type == "composite":
        reward_fn = CompositeReward()
        reward_fn.add(
            DistanceReward(
                pos_a_indices=ee_indices,
                pos_b_indices=target_indices,
            ),
            weight=1.0,
        )
        reward_fn.add(ActionPenaltyReward(scale=0.01), weight=0.1)
    else:
        msg = f"Unknown reward type: {reward_type}"
        raise ValueError(msg)

    return ClankerGymnasiumEnv(
        host=host,
        port=port,
        reward_fn=reward_fn,
    )


def train(
    host: str = "127.0.0.1",
    port: int = 9876,
    reward_type: str = "distance",
    total_timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    save_path: str | None = None,
) -> None:
    """Run PPO training.

    Parameters
    ----------
    host : str
        Server address.
    port : int
        Server port.
    reward_type : str
        Reward function type ("distance", "sparse", "composite").
    total_timesteps : int
        Total training steps.
    learning_rate : float
        PPO learning rate.
    n_steps : int
        Steps per rollout buffer collection.
    batch_size : int
        Minibatch size for PPO updates.
    n_epochs : int
        Epochs per PPO update.
    save_path : str | None
        Path to save the trained model. None skips saving.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print(
            "stable-baselines3 is required. Install with: "
            "pip install clanker-gym[sb3]",
            file=sys.stderr,
        )
        sys.exit(1)

    env = make_env(host=host, port=port, reward_type=reward_type)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        verbose=1,
    )

    print(f"Training PPO for {total_timesteps} timesteps...")
    print(f"  Server: {host}:{port}")
    print(f"  Reward: {reward_type}")
    model.learn(total_timesteps=total_timesteps)

    if save_path is not None:
        model.save(save_path)
        print(f"Model saved to {save_path}")

    env.close()  # type: ignore[union-attr]
    print("Training complete.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on a Clankers environment."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server address")
    parser.add_argument("--port", type=int, default=9876, help="Server port")
    parser.add_argument(
        "--reward",
        default="distance",
        choices=["distance", "sparse", "composite"],
        help="Reward function type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--save", default=None, help="Path to save trained model"
    )

    args = parser.parse_args()

    train(
        host=args.host,
        port=args.port,
        reward_type=args.reward,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
