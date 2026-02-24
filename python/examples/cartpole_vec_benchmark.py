"""Python VecEnv benchmark for cart-pole.

Connects to a VecGymServer and measures batch step throughput.

Start the server first:
    cargo run -p clankers-examples --bin cartpole_vec_gym -- 8

Then run this script:
    python python/examples/cartpole_vec_benchmark.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "python")
from clanker_gym.vec_env import ClankerVecEnv
from clanker_gym.rewards import ConstantReward
from clanker_gym.terminations import cartpole_termination


def main() -> None:
    host = "127.0.0.1"
    port = 9878

    env = ClankerVecEnv(host=host, port=port)

    print("Connecting to vectorized cart-pole server...")
    resp = env.connect()
    num_envs = env.num_envs
    print(f"  num_envs: {num_envs}")
    print(f"  obs_space: {env.observation_space}")
    print(f"  act_space: {env.action_space}")

    # Reset all
    t0 = time.perf_counter()
    obs, infos = env.reset()
    reset_time = time.perf_counter() - t0
    print(f"\nReset {num_envs} envs in {reset_time*1000:.1f}ms")
    print(f"  obs shape: {obs.shape}")
    print(f"  env 0 initial: cart_pos={obs[0, 0]:+.4f}, pole_angle={obs[0, 2]:+.4f}")

    # Benchmark: step all envs with constant action
    num_batches = 500
    reward_fn = ConstantReward(value=1.0)
    termination_fn = cartpole_termination()

    total_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs, dtype=int)

    print(f"\nBenchmarking {num_batches} batch steps across {num_envs} envs...")
    t0 = time.perf_counter()

    for step_idx in range(num_batches):
        # Simple proportional controller for each env
        actions = []
        for i in range(num_envs):
            if obs.shape[1] >= 4:
                # PD controller
                cart_pos = obs[i, 0]
                cart_vel = obs[i, 1]
                pole_angle = obs[i, 2]
                pole_vel = obs[i, 3]
                force = float(np.clip(
                    5.0 * pole_angle + 1.0 * pole_vel + 0.5 * cart_pos + 1.0 * cart_vel,
                    -1.0, 1.0
                ))
            else:
                force = 0.0
            actions.append([force, 0.0])

        actions_arr = np.array(actions, dtype=np.float32)
        obs, terminated, truncated, infos = env.step(actions_arr)

        for i in range(num_envs):
            total_rewards[i] += reward_fn.compute(obs[i])
            episode_lengths[i] += 1

        if step_idx % 100 == 0:
            print(
                f"  batch {step_idx:4d}: "
                f"env0 cart_pos={obs[0, 0]:+.4f}, pole_angle={obs[0, 2]:+.4f}"
            )

    elapsed = time.perf_counter() - t0
    total_steps = num_batches * num_envs
    steps_per_sec = total_steps / elapsed
    batches_per_sec = num_batches / elapsed
    ms_per_batch = elapsed * 1000.0 / num_batches

    print(f"\n=== Benchmark Results ===")
    print(f"  Environments:    {num_envs}")
    print(f"  Batches:         {num_batches}")
    print(f"  Total steps:     {total_steps}")
    print(f"  Wall time:       {elapsed:.2f}s")
    print(f"  Steps/sec:       {steps_per_sec:.0f}")
    print(f"  Batches/sec:     {batches_per_sec:.1f}")
    print(f"  ms/batch:        {ms_per_batch:.2f}")
    print(f"  Mean reward:     {np.mean(total_rewards):.0f}")
    print(f"  Mean ep length:  {np.mean(episode_lengths):.0f}")

    # Final state across envs
    print(f"\n  Final observations (first 4 envs):")
    for i in range(min(4, num_envs)):
        print(
            f"    env {i}: cart_pos={obs[i, 0]:+.4f}, "
            f"pole_angle={obs[i, 2]:+.4f}"
        )

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
