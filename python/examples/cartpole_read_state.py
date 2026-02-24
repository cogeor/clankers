"""Read cart-pole state from running gym server.

Validates the Rust→TCP→Python observation pipeline.
Start the server first:
    cargo run -p clankers-examples --bin cartpole_gym

Then run this script:
    python python/examples/cartpole_read_state.py
"""

from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "python")
from clanker_gym.env import ClankerEnv


def main() -> None:
    env = ClankerEnv(host="127.0.0.1", port=9877)

    print("Connecting to cart-pole gym server...")
    resp = env.connect()
    print(f"  env_info: {resp.get('env_info', {})}")
    print(f"  obs_space: {env.observation_space}")
    print(f"  act_space: {env.action_space}")

    # Reset to get initial state
    obs, info = env.reset()
    print(f"\nInitial state (after reset):")
    print(f"  cart_pos={obs[0]:.4f}, cart_vel={obs[1]:.4f}")
    print(f"  pole_angle={obs[2]:.4f}, pole_vel={obs[3]:.4f}")

    # Step with zero action for 50 steps — pole should start falling under gravity
    print("\nStepping with zero action (no force on cart)...")
    for i in range(50):
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, terminated, truncated, info = env.step(action)
        if i % 10 == 0 or terminated or truncated:
            print(
                f"  step={i:3d}: cart_pos={obs[0]:+.4f}, cart_vel={obs[1]:+.4f}, "
                f"pole_angle={obs[2]:+.4f}, pole_vel={obs[3]:+.4f}, "
                f"term={terminated}, trunc={truncated}"
            )
        if terminated or truncated:
            print("  Episode ended!")
            break

    # Reset and step with constant positive force
    obs, info = env.reset()
    print(f"\nAfter second reset: cart_pos={obs[0]:.4f}, pole_angle={obs[2]:.4f}")

    print("\nStepping with positive cart force (push right)...")
    for i in range(50):
        action = np.array([1.0, 0.0], dtype=np.float32)  # max force right
        obs, terminated, truncated, info = env.step(action)
        if i % 10 == 0 or terminated or truncated:
            print(
                f"  step={i:3d}: cart_pos={obs[0]:+.4f}, cart_vel={obs[1]:+.4f}, "
                f"pole_angle={obs[2]:+.4f}, pole_vel={obs[3]:+.4f}"
            )
        if terminated or truncated:
            print("  Episode ended!")
            break

    env.close()
    print("\nDone. Observations validated successfully.")


if __name__ == "__main__":
    main()
