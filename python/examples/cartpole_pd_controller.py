"""PD baseline controller for cart-pole balance.

Validates that the physics is correct: a properly tuned PD controller
should keep the pole balanced for the full 500 steps if the dynamics
match CartPole-v1.

Start the server first:
    cargo run -p clankers-examples --bin cartpole_gym

Then run this script:
    python python/examples/cartpole_pd_controller.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "python")
from clanker_gym.env import ClankerEnv
from clanker_gym.rewards import ConstantReward
from clanker_gym.terminations import cartpole_termination


def pd_controller(obs: np.ndarray) -> float:
    """PD controller for cart-pole balance.

    Based on linearized CartPole-v1 dynamics. Gains tuned for:
    - Cart mass: 1.0 kg
    - Pole mass: 0.1 kg
    - Pole half-length: 0.5 m
    - Force magnitude: 10 N
    - Gravity: 9.8 m/s^2

    Observation layout: [cart_pos, cart_vel, pole_angle, pole_vel]

    Returns a normalized force in [-1, 1].
    """
    cart_pos, cart_vel, pole_angle, pole_vel = obs[:4]

    # LQR-inspired gains for CartPole-v1
    # The classic LQR solution for CartPole gives gains roughly:
    #   K = [k_pos, k_vel, k_angle, k_angle_vel]
    # where angle gains dominate (keeping pole upright is primary objective)
    k_pos = 1.0
    k_vel = 1.5
    k_angle = 15.0
    k_angle_vel = 3.0

    force = (
        k_pos * cart_pos
        + k_vel * cart_vel
        + k_angle * pole_angle
        + k_angle_vel * pole_vel
    )

    # Normalize to [-1, 1] (server scales by effort limit = 10N)
    return float(np.clip(force, -1.0, 1.0))


def main() -> None:
    env = ClankerEnv(host="127.0.0.1", port=9877)

    print("=== Cart-Pole PD Controller Baseline ===\n")
    print("Connecting to cart-pole gym server...")
    env.connect()
    print(f"  obs_space: {env.observation_space}")
    print(f"  act_space: {env.action_space}")

    reward_fn = ConstantReward(value=1.0)
    termination_fn = cartpole_termination()

    num_episodes = 20
    max_steps = 500
    results = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        max_angle = 0.0
        max_pos = 0.0

        for step in range(max_steps):
            force = pd_controller(obs)
            action = np.array([force, 0.0], dtype=np.float32)
            obs, terminated, truncated, info = env.step(action)

            total_reward += reward_fn.compute(obs)
            max_angle = max(max_angle, abs(float(obs[2])))
            max_pos = max(max_pos, abs(float(obs[0])))

            # Python-side termination
            py_terminated = termination_fn.is_terminated(obs, step + 1)

            if terminated or truncated or py_terminated:
                break

        results.append({
            "episode": ep + 1,
            "steps": step + 1,
            "reward": total_reward,
            "max_angle_deg": np.degrees(max_angle),
            "max_pos": max_pos,
            "survived": step + 1 >= max_steps,
        })

        status = "BALANCED" if step + 1 >= max_steps else "FELL"
        print(
            f"  ep={ep+1:2d}: steps={step+1:4d} ({status}) "
            f"max_angle={np.degrees(max_angle):.1f}deg "
            f"max_pos={max_pos:.2f}m"
        )

    env.close()

    # Summary
    survived = sum(1 for r in results if r["survived"])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["reward"] for r in results])

    print(f"\n=== Summary ===")
    print(f"  Episodes:     {num_episodes}")
    print(f"  Survived:     {survived}/{num_episodes} ({100*survived/num_episodes:.0f}%)")
    print(f"  Avg steps:    {avg_steps:.0f}/{max_steps}")
    print(f"  Avg reward:   {avg_reward:.0f}")

    if survived >= num_episodes * 0.8:
        print(f"\n  SUCCESS: PD controller balances pole in {survived}/{num_episodes} episodes")
        print(f"  Physics dynamics are consistent with CartPole-v1")
    else:
        print(f"\n  NEEDS TUNING: Only {survived}/{num_episodes} episodes balanced")
        print(f"  Physics may differ from CartPole-v1 or gains need adjustment")


if __name__ == "__main__":
    main()
