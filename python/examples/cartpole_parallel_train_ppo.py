"""Train PPO on cart-pole with parallel environments via SB3 VecEnv.

Compares training with N parallel envs (ClankerSB3VecEnv) against the
single-env baseline (ClankerGymnasiumEnv).

Start the servers first:
    # Single-env server (port 9877):
    cargo run -p clankers-examples --bin cartpole_gym --release

    # Vec-env server (port 9878, N envs):
    cargo run -p clankers-examples --bin cartpole_vec_gym --release -- 8

Then run this script:
    py -3.12 python/examples/cartpole_parallel_train_ppo.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "python")

from stable_baselines3 import PPO

from clanker_gym.gymnasium_env import make_cartpole_gymnasium_env
from clanker_gym.sb3_vec_env import make_cartpole_sb3_vec_env


def evaluate_with_single_env(model, port: int = 9877, n_episodes: int = 20) -> dict:
    """Evaluate trained model using single-env server."""
    env = make_cartpole_gymnasium_env(port=port)
    episode_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        episode_lengths.append(steps)

    env.close()
    return {
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "min_length": int(np.min(episode_lengths)),
        "max_length": int(np.max(episode_lengths)),
        "survived_500": sum(1 for l in episode_lengths if l >= 500),
        "n_episodes": n_episodes,
    }


def train_single_env(total_timesteps: int, port: int = 9877) -> tuple[PPO, float]:
    """Train PPO with a single environment. Returns (model, train_time)."""
    env = make_cartpole_gymnasium_env(port=port)
    env.reset()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        device="cpu",
        policy_kwargs={"net_arch": [64, 64]},
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - t0

    env.close()
    return model, elapsed


def train_vec_env(total_timesteps: int, port: int = 9878) -> tuple[PPO, float]:
    """Train PPO with parallel vectorized environments. Returns (model, train_time)."""
    vec_env = make_cartpole_sb3_vec_env(port=port)
    vec_env.reset()

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        device="cpu",
        policy_kwargs={"net_arch": [64, 64]},
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - t0

    vec_env.close()
    return model, elapsed


def main() -> None:
    total_timesteps = 50_000

    print("=" * 60)
    print("Cart-Pole PPO: Single vs Parallel Environment Training")
    print("=" * 60)
    print(f"\nTotal timesteps: {total_timesteps}")
    print(f"PPO config: MlpPolicy(64x64), lr=3e-4, n_steps=2048, batch=64")

    # --- Single-env training ---
    print(f"\n{'─' * 60}")
    print("Training with SINGLE environment (port 9877)...")
    print(f"{'─' * 60}")
    try:
        model_single, time_single = train_single_env(total_timesteps, port=9877)
        print(f"  Training time: {time_single:.1f}s")

        print("  Evaluating...")
        eval_single = evaluate_with_single_env(model_single, port=9877)
        print(f"  Mean length:  {eval_single['mean_length']:.0f} +/- {eval_single['std_length']:.0f}")
        print(f"  Survived 500: {eval_single['survived_500']}/{eval_single['n_episodes']}")
    except Exception as e:
        print(f"  SKIPPED (server not running?): {e}")
        model_single, time_single, eval_single = None, None, None

    # --- Vec-env training ---
    print(f"\n{'─' * 60}")
    print("Training with PARALLEL environments (port 9878)...")
    print(f"{'─' * 60}")
    try:
        model_vec, time_vec = train_vec_env(total_timesteps, port=9878)
        print(f"  Training time: {time_vec:.1f}s")

        print("  Evaluating (using single-env server)...")
        eval_vec = evaluate_with_single_env(model_vec, port=9877)
        print(f"  Mean length:  {eval_vec['mean_length']:.0f} +/- {eval_vec['std_length']:.0f}")
        print(f"  Survived 500: {eval_vec['survived_500']}/{eval_vec['n_episodes']}")
    except Exception as e:
        print(f"  SKIPPED (server not running?): {e}")
        model_vec, time_vec, eval_vec = None, None, None

    # --- Comparison ---
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")

    if time_single is not None and time_vec is not None:
        speedup = time_single / time_vec
        print(f"  Single-env:   {time_single:.1f}s")
        print(f"  Parallel-env: {time_vec:.1f}s")
        print(f"  Speedup:      {speedup:.2f}x")

        if eval_single is not None and eval_vec is not None:
            print(f"\n  Single-env quality:   {eval_single['mean_length']:.0f} mean steps, "
                  f"{eval_single['survived_500']}/{eval_single['n_episodes']} survived")
            print(f"  Parallel-env quality: {eval_vec['mean_length']:.0f} mean steps, "
                  f"{eval_vec['survived_500']}/{eval_vec['n_episodes']} survived")
    else:
        print("  Could not compare (one or both servers not running)")

    print("\nDone.")


if __name__ == "__main__":
    main()
