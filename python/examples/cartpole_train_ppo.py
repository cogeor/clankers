"""Train PPO on cart-pole using Stable-Baselines3.

End-to-end RL training pipeline validation:
1. Connect to cartpole_gym server via ClankerGymnasiumEnv
2. Train PPO for N timesteps
3. Evaluate trained policy
4. Compare against PD baseline

Start the server first:
    cargo run -p clankers-examples --bin cartpole_gym

Then run this script (requires Python 3.12 with PyTorch + SB3):
    py -3.12 python/examples/cartpole_train_ppo.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "python")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from clanker_gym.gymnasium_env import make_cartpole_gymnasium_env


class ProgressCallback(BaseCallback):
    """Print training progress every N steps."""

    def __init__(self, print_freq: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self.print_freq:
            ep_rewards = self.training_env.get_attr("_last_obs")
            infos = self.locals.get("infos", [])
            # Get episode info if available
            ep_info = None
            for info in infos:
                if "episode" in info:
                    ep_info = info["episode"]
                    break

            if ep_info:
                print(
                    f"  step={self.num_timesteps:6d}: "
                    f"ep_reward={ep_info['r']:.0f}, ep_len={ep_info['l']:.0f}"
                )
            else:
                print(f"  step={self.num_timesteps:6d}")
            self._last_print = self.num_timesteps
        return True


def evaluate_policy(env, model, n_episodes: int = 50) -> dict:
    """Run n_episodes with the trained model and collect stats."""
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "max_length": np.max(episode_lengths),
        "min_length": np.min(episode_lengths),
        "survived_500": sum(1 for l in episode_lengths if l >= 500),
        "n_episodes": n_episodes,
    }


def main() -> None:
    print("=== Cart-Pole PPO Training ===\n")

    # Create environment
    print("Connecting to cart-pole gym server...")
    env = make_cartpole_gymnasium_env(port=9877)
    env.reset()  # Force connection
    print(f"  obs_space: {env.observation_space}")
    print(f"  act_space: {env.action_space}")

    # PPO hyperparameters tuned for CartPole
    total_timesteps = 50_000

    print(f"\nTraining PPO for {total_timesteps} timesteps...")
    print(f"  policy: MlpPolicy (64x64)")
    print(f"  learning_rate: 3e-4")
    print(f"  n_steps: 2048")
    print(f"  batch_size: 64")

    t0 = time.perf_counter()

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
        policy_kwargs={"net_arch": [64, 64]},
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=ProgressCallback(print_freq=4096),
    )

    train_time = time.perf_counter() - t0
    print(f"\nTraining completed in {train_time:.1f}s")

    # Evaluate
    print(f"\nEvaluating trained policy (50 episodes)...")
    eval_results = evaluate_policy(env, model, n_episodes=50)

    print(f"\n=== Evaluation Results ===")
    print(f"  Mean reward:  {eval_results['mean_reward']:.1f} +/- {eval_results['std_reward']:.1f}")
    print(f"  Mean length:  {eval_results['mean_length']:.1f} +/- {eval_results['std_length']:.1f}")
    print(f"  Max length:   {eval_results['max_length']}")
    print(f"  Min length:   {eval_results['min_length']}")
    print(f"  Survived 500: {eval_results['survived_500']}/{eval_results['n_episodes']}")

    success = eval_results["mean_length"] > 400
    if success:
        print(f"\n  SUCCESS: PPO learned to balance the pole!")
        print(f"  Mean episode length {eval_results['mean_length']:.0f} > 400 threshold")
    else:
        print(f"\n  PARTIAL: Mean length {eval_results['mean_length']:.0f} (target: >400)")
        print(f"  May need more training timesteps or hyperparameter tuning")

    # Save model
    model_path = "python/examples/cartpole_ppo_model"
    model.save(model_path)
    print(f"\n  Model saved to: {model_path}.zip")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
