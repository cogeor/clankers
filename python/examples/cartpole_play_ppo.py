"""Load a trained PPO model and run it on the cart-pole server.

Visualizes the policy in action by printing state at each step.

Start the server first:
    cargo run -p clankers-examples --bin cartpole_gym --release

Then run this script:
    py -3.12 python/examples/cartpole_play_ppo.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "python")

from stable_baselines3 import PPO

from clanker_gym.gymnasium_env import make_cartpole_gymnasium_env


def render_bar(value: float, width: int = 40, lo: float = -1.0, hi: float = 1.0) -> str:
    """Render a value as an ASCII bar centered at zero."""
    clamped = max(lo, min(hi, value))
    normalized = (clamped - lo) / (hi - lo)  # 0..1
    center = int((0.0 - lo) / (hi - lo) * width)
    pos = int(normalized * width)
    bar = list("." * width)
    bar[center] = "|"
    if pos < center:
        for i in range(pos, center):
            bar[i] = "#"
    elif pos > center:
        for i in range(center + 1, min(pos + 1, width)):
            bar[i] = "#"
    bar[pos] = "*"
    return "".join(bar)


def main() -> None:
    model_path = "python/examples/cartpole_ppo_model.zip"

    print("=== Cart-Pole PPO Playback ===\n")
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print("  Model loaded successfully")

    print("Connecting to cart-pole gym server...")
    env = make_cartpole_gymnasium_env(port=9877)

    num_episodes = 5
    slow_mode = True  # Add delay to watch the policy work

    for ep in range(num_episodes):
        obs, _info = env.reset()
        total_reward = 0.0
        max_angle = 0.0

        print(f"\n{'=' * 70}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'=' * 70}")
        print(
            f"{'step':>5}  {'cart_pos':>8}  {'cart_vel':>8}  "
            f"{'pole_deg':>8}  {'action':>7}  cart position"
        )

        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            max_angle = max(max_angle, abs(float(obs[2])))

            cart_pos = float(obs[0])
            cart_vel = float(obs[1])
            pole_deg = np.degrees(float(obs[2]))
            act_val = float(action[0])

            # Print every 10 steps or on termination
            if step % 10 == 0 or terminated or truncated:
                bar = render_bar(cart_pos, width=50, lo=-2.5, hi=2.5)
                print(
                    f"{step + 1:5d}  {cart_pos:+8.4f}  {cart_vel:+8.4f}  "
                    f"{pole_deg:+8.3f}  {act_val:+7.3f}  [{bar}]"
                )

            if slow_mode and step < 100:
                time.sleep(0.02)

            if terminated or truncated:
                break

        status = "BALANCED" if step + 1 >= 500 else "FELL"
        print(
            f"\n  Result: {status} | steps={step + 1} "
            f"| reward={total_reward:.0f} "
            f"| max_angle={np.degrees(max_angle):.2f}deg"
        )

    env.close()
    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
