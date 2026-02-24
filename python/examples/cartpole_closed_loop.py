"""Closed-loop cart-pole balance with proportional controller.

Validates the full Python action pipeline:
1. Connect to cartpole_gym server
2. Read observations (cart_pos, cart_vel, pole_angle, pole_vel)
3. Compute proportional control action
4. Apply action and verify state changes

Start the server first:
    cargo run -p clankers-examples --bin cartpole_gym

Then run this script:
    python python/examples/cartpole_closed_loop.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "python")
from clanker_gym.env import ClankerEnv
from clanker_gym.rewards import ConstantReward
from clanker_gym.terminations import cartpole_termination


def pd_controller(obs: np.ndarray) -> float:
    """Simple PD controller for cart-pole balance.

    Observation layout: [cart_pos, cart_vel, pole_angle, pole_vel]

    Returns a force in [-1, 1] to apply to the cart.
    """
    cart_pos, cart_vel, pole_angle, pole_vel = obs[:4]

    # PD gains tuned for CartPole-v1 parameters
    # Pole angle control (primary)
    kp_angle = 5.0
    kd_angle = 1.0
    # Cart position control (secondary, keeps cart centered)
    kp_pos = 0.5
    kd_pos = 1.0

    force = (
        kp_angle * pole_angle
        + kd_angle * pole_vel
        + kp_pos * cart_pos
        + kd_pos * cart_vel
    )

    return float(np.clip(force, -1.0, 1.0))


def main() -> None:
    env = ClankerEnv(host="127.0.0.1", port=9877)

    print("Connecting to cart-pole gym server...")
    resp = env.connect()
    print(f"  obs_space: {env.observation_space}")
    print(f"  act_space: {env.action_space}")

    reward_fn = ConstantReward(value=1.0)
    termination_fn = cartpole_termination()

    num_episodes = 5
    episode_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0

        print(f"\n--- Episode {ep + 1} ---")
        print(f"  Initial: cart_pos={obs[0]:+.4f}, pole_angle={obs[2]:+.4f}")

        while True:
            # Compute PD control action
            force = pd_controller(obs)
            action = np.array([force, 0.0], dtype=np.float32)  # [cart_force, pole_torque=0]

            # Step
            obs, terminated, truncated, info = env.step(action)
            step_count += 1

            # Compute reward in Python
            reward = reward_fn.compute(obs)
            total_reward += reward

            # Check Python-side termination
            py_terminated = termination_fn.is_terminated(obs, step_count)

            if step_count % 100 == 0:
                print(
                    f"  step={step_count:4d}: cart_pos={obs[0]:+.4f}, "
                    f"pole_angle={obs[2]:+.4f}, force={force:+.3f}"
                )

            if terminated or truncated or py_terminated:
                reason = "rust" if (terminated or truncated) else "python"
                print(
                    f"  Done at step {step_count} ({reason}): "
                    f"cart_pos={obs[0]:+.4f}, pole_angle={obs[2]:+.4f}, "
                    f"total_reward={total_reward:.0f}"
                )
                break

        episode_rewards.append(total_reward)

    env.close()

    avg_reward = np.mean(episode_rewards)
    print(f"\n=== Results ===")
    print(f"  Episodes: {num_episodes}")
    print(f"  Rewards: {[int(r) for r in episode_rewards]}")
    print(f"  Average: {avg_reward:.1f}")
    print(f"  Max steps (500) = perfect balance")

    if avg_reward > 200:
        print("  PD controller is balancing the pole!")
    else:
        print("  PD controller needs tuning (or physics params differ)")


if __name__ == "__main__":
    main()
