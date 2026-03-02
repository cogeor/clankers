"""Evaluation utilities for Clanker environments.

Provides ``evaluate_policy()`` for computing success rates, mean rewards,
and optional per-component reward breakdowns using ``CompositeReward``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

try:
    import gymnasium
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for clankers evaluation. "
        "Install with: pip install clankers[sb3]"
    ) from exc

from clankers.rewards import CompositeReward

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Results from policy evaluation.

    Attributes
    ----------
    mean_reward : float
        Mean episode return across all evaluation episodes.
    std_reward : float
        Standard deviation of episode returns.
    mean_length : float
        Mean episode length.
    success_rate : float
        Fraction of episodes where info["is_success"] was True at termination.
        NaN if is_success was never present in info.
    episode_rewards : list[float]
        Per-episode total rewards.
    episode_lengths : list[int]
        Per-episode lengths.
    reward_breakdown : dict[str, float]
        Mean per-component rewards (only if reward_fn is CompositeReward).
    """

    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_length: float = 0.0
    success_rate: float = float("nan")
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    reward_breakdown: dict[str, float] = field(default_factory=dict)


def evaluate_policy(
    env: gymnasium.Env,
    policy: Callable[[np.ndarray], np.ndarray | int],
    n_episodes: int = 10,
    reward_fn: CompositeReward | None = None,
) -> EvalResult:
    """Evaluate a policy on an environment.

    Parameters
    ----------
    env : gymnasium.Env
        Environment to evaluate on.
    policy : callable
        Function mapping observation -> action. Can be a simple function
        or ``model.predict`` from SB3 (wrap to extract action only).
    n_episodes : int
        Number of episodes to run (default: 10).
    reward_fn : CompositeReward | None
        If provided, compute per-component reward breakdown.

    Returns
    -------
    EvalResult
        Aggregated evaluation metrics.
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    successes: list[bool] = []
    has_success_info = False

    # Per-component reward accumulators
    component_sums: dict[str, float] = {}

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        prev_obs = obs

        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

            # Reward breakdown
            if reward_fn is not None:
                breakdown = reward_fn.breakdown(
                    obs=prev_obs,
                    action=action,
                    next_obs=obs,
                    info=info,
                )
                for name, value in breakdown:
                    component_sums[name] = component_sums.get(name, 0.0) + value

            prev_obs = obs

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        # Check for success
        if "is_success" in info:
            has_success_info = True
            successes.append(bool(info["is_success"]))

    # Compute aggregates
    rewards_arr = np.array(episode_rewards)
    result = EvalResult(
        mean_reward=float(np.mean(rewards_arr)),
        std_reward=float(np.std(rewards_arr)),
        mean_length=float(np.mean(episode_lengths)),
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
    )

    if has_success_info and successes:
        result.success_rate = float(np.mean(successes))

    if component_sums:
        total_steps = sum(episode_lengths)
        result.reward_breakdown = {
            k: v / max(total_steps, 1) for k, v in component_sums.items()
        }

    logger.info(
        "Eval: %d episodes, mean_reward=%.3f (+/-%.3f), mean_len=%.1f, success=%.1f%%",
        n_episodes,
        result.mean_reward,
        result.std_reward,
        result.mean_length,
        result.success_rate * 100 if not np.isnan(result.success_rate) else 0.0,
    )

    return result
