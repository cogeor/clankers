"""Tests for observation and reward wrappers."""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if gymnasium is not installed.
gymnasium = pytest.importorskip("gymnasium")

from gymnasium import spaces as gym_spaces  # noqa: E402

from clanker_gym.wrappers import (  # noqa: E402
    ClipReward,
    FrameStack,
    NormalizeObservation,
    NormalizeReward,
)


# ---------------------------------------------------------------------------
# Helpers -- minimal gymnasium env that returns deterministic observations.
# ---------------------------------------------------------------------------


class DummyEnv(gymnasium.Env):
    """Minimal gymnasium env for testing wrappers.

    Observations cycle through a fixed sequence.  No server required.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, obs_shape: tuple[int, ...] = (4,), obs_sequence: list[np.ndarray] | None = None) -> None:
        super().__init__()
        self.observation_space = gym_spaces.Box(
            low=-10.0, high=10.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = gym_spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        if obs_sequence is not None:
            self._obs_sequence = [np.asarray(o, dtype=np.float32) for o in obs_sequence]
        else:
            self._obs_sequence = [
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32),
                np.array([3.0, 6.0, 9.0, 12.0], dtype=np.float32),
            ]
        self._idx = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        self._idx = 0
        return self._obs_sequence[self._idx].copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._idx = (self._idx + 1) % len(self._obs_sequence)
        obs = self._obs_sequence[self._idx].copy()
        return obs, 0.0, False, False, {}


# ---------------------------------------------------------------------------
# NormalizeObservation tests
# ---------------------------------------------------------------------------


class TestNormalizeObservation:
    def test_first_observation_is_zero(self):
        """After one observation, mean == obs so normalized should be ~0."""
        env = NormalizeObservation(DummyEnv())
        obs, _ = env.reset()
        # After one sample, var is 0, so normalised = 0 / sqrt(0+eps) = 0
        np.testing.assert_allclose(obs, 0.0, atol=1e-4)

    def test_stats_update_correctly(self):
        """Verify Welford stats track mean/var of observed data."""
        raw_env = DummyEnv()
        env = NormalizeObservation(raw_env)
        env.reset()

        # Collect raw observations for manual stats.
        raw_obs = [raw_env._obs_sequence[0]]
        for _ in range(2):
            obs, _, _, _, _ = env.step(np.array([0.0]))
            # Advance raw idx to match.
        raw_obs.append(raw_env._obs_sequence[1])
        raw_obs.append(raw_env._obs_sequence[2])

        raw_arr = np.stack(raw_obs[:env._count], axis=0)
        expected_mean = raw_arr.mean(axis=0)
        # Population variance (ddof=0), matching Welford's m2/count.
        expected_var = raw_arr.var(axis=0, ddof=0)

        np.testing.assert_allclose(env._mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(env._var, expected_var, atol=1e-6)

    def test_normalization_formula(self):
        """Check the normalize formula manually after a few steps."""
        seq = [
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([20.0, 40.0], dtype=np.float32),
        ]
        raw_env = DummyEnv(obs_shape=(2,), obs_sequence=seq)
        env = NormalizeObservation(raw_env, epsilon=1e-8, clip=10.0)
        env.reset()  # processes seq[0]
        obs, _, _, _, _ = env.step(np.array([0.0]))  # processes seq[1]

        # After 2 obs: mean=[15,30], var=[25,100]
        expected_mean = np.array([15.0, 30.0])
        expected_var = np.array([25.0, 100.0])
        np.testing.assert_allclose(env._mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(env._var, expected_var, atol=1e-6)

        # Normalized = (20 - 15) / sqrt(25 + 1e-8) = 5/5 = 1.0
        #              (40 - 30) / sqrt(100 + 1e-8) = 10/10 = 1.0
        np.testing.assert_allclose(obs, [1.0, 1.0], atol=1e-4)

    def test_clipping(self):
        """Ensure normalized observations are clipped to [-clip, clip]."""
        # Extreme jump: first obs = 0, second obs = 1e6.
        seq = [
            np.array([0.0], dtype=np.float32),
            np.array([1e6], dtype=np.float32),
        ]
        raw_env = DummyEnv(obs_shape=(1,), obs_sequence=seq)
        env = NormalizeObservation(raw_env, clip=5.0)
        env.reset()
        obs, _, _, _, _ = env.step(np.array([0.0]))
        assert obs.item() <= 5.0
        assert obs.item() >= -5.0

    def test_output_dtype_is_float32(self):
        """Wrapper output should always be float32."""
        env = NormalizeObservation(DummyEnv())
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        obs2, _, _, _, _ = env.step(np.array([0.0]))
        assert obs2.dtype == np.float32

    def test_custom_epsilon(self):
        """Custom epsilon propagates correctly."""
        env = NormalizeObservation(DummyEnv(), epsilon=1e-4)
        assert env.epsilon == 1e-4

    def test_observation_space_unchanged(self):
        """NormalizeObservation should not change the observation space."""
        raw_env = DummyEnv()
        env = NormalizeObservation(raw_env)
        assert env.observation_space.shape == raw_env.observation_space.shape


# ---------------------------------------------------------------------------
# FrameStack tests
# ---------------------------------------------------------------------------


class TestFrameStack:
    def test_observation_space_shape(self):
        """Observation space gains a leading n_frames dimension."""
        raw_env = DummyEnv(obs_shape=(4,))
        env = FrameStack(raw_env, n_frames=3)
        assert env.observation_space.shape == (3, 4)

    def test_reset_fills_all_frames(self):
        """On reset, all frames should be copies of the initial observation."""
        raw_env = DummyEnv()
        env = FrameStack(raw_env, n_frames=3)
        obs, _ = env.reset()
        assert obs.shape == (3, 4)
        # All frames should be identical to the reset obs.
        for i in range(3):
            np.testing.assert_array_equal(obs[i], obs[0])

    def test_step_shifts_frames(self):
        """After a step, the newest frame should be the latest obs."""
        seq = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 3.0, 0.0], dtype=np.float32),
        ]
        raw_env = DummyEnv(obs_sequence=seq)
        env = FrameStack(raw_env, n_frames=3)
        obs_reset, _ = env.reset()  # All frames = seq[0]

        obs1, _, _, _, _ = env.step(np.array([0.0]))  # newest = seq[1]
        # Frames: [seq[0], seq[0], seq[1]]
        np.testing.assert_array_equal(obs1[0], seq[0])
        np.testing.assert_array_equal(obs1[1], seq[0])
        np.testing.assert_array_equal(obs1[2], seq[1])

        obs2, _, _, _, _ = env.step(np.array([0.0]))  # newest = seq[2]
        # Frames: [seq[0], seq[1], seq[2]]
        np.testing.assert_array_equal(obs2[0], seq[0])
        np.testing.assert_array_equal(obs2[1], seq[1])
        np.testing.assert_array_equal(obs2[2], seq[2])

    def test_n_frames_1_is_identity(self):
        """n_frames=1 should just add a leading dim of size 1."""
        raw_env = DummyEnv()
        env = FrameStack(raw_env, n_frames=1)
        obs, _ = env.reset()
        assert obs.shape == (1, 4)
        expected = raw_env._obs_sequence[0]
        np.testing.assert_array_equal(obs[0], expected)

    def test_space_bounds_replicated(self):
        """Low and high bounds should be replicated across frames."""
        raw_env = DummyEnv(obs_shape=(2,))
        env = FrameStack(raw_env, n_frames=3)
        assert env.observation_space.low.shape == (3, 2)
        assert env.observation_space.high.shape == (3, 2)
        for i in range(3):
            np.testing.assert_array_equal(
                env.observation_space.low[i], raw_env.observation_space.low
            )
            np.testing.assert_array_equal(
                env.observation_space.high[i], raw_env.observation_space.high
            )

    def test_dtype_preserved(self):
        """The dtype of the stacked obs should match the base env."""
        raw_env = DummyEnv()
        env = FrameStack(raw_env, n_frames=2)
        obs, _ = env.reset()
        assert obs.dtype == raw_env.observation_space.dtype

    def test_2d_observation(self):
        """FrameStack works with 2D observations (e.g., small images)."""
        seq = [
            np.ones((3, 3), dtype=np.float32) * i for i in range(4)
        ]
        raw_env = DummyEnv(obs_shape=(3, 3), obs_sequence=seq)
        env = FrameStack(raw_env, n_frames=2)
        assert env.observation_space.shape == (2, 3, 3)
        obs, _ = env.reset()
        assert obs.shape == (2, 3, 3)


# ---------------------------------------------------------------------------
# Composition tests
# ---------------------------------------------------------------------------


class TestWrapperComposition:
    def test_normalize_then_framestack(self):
        """Composing NormalizeObservation -> FrameStack should work."""
        raw_env = DummyEnv()
        env = FrameStack(NormalizeObservation(raw_env), n_frames=2)
        obs, _ = env.reset()
        assert obs.shape == (2, 4)
        obs2, _, _, _, _ = env.step(np.array([0.0]))
        assert obs2.shape == (2, 4)

    def test_framestack_then_normalize(self):
        """Composing FrameStack -> NormalizeObservation should work."""
        raw_env = DummyEnv()
        env = NormalizeObservation(FrameStack(raw_env, n_frames=3))
        obs, _ = env.reset()
        assert obs.shape == (3, 4)
        obs2, _, _, _, _ = env.step(np.array([0.0]))
        assert obs2.shape == (3, 4)


# ---------------------------------------------------------------------------
# Helpers -- dummy env that returns configurable rewards.
# ---------------------------------------------------------------------------


class RewardDummyEnv(gymnasium.Env):
    """Minimal gymnasium env that returns configurable rewards per step.

    The ``rewards`` list is cycled through on each ``step()`` call.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, rewards: list[float] | None = None) -> None:
        super().__init__()
        self.observation_space = gym_spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym_spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self._rewards = rewards if rewards is not None else [1.0]
        self._step_idx = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        self._step_idx = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = self._rewards[self._step_idx % len(self._rewards)]
        self._step_idx += 1
        return np.zeros(4, dtype=np.float32), reward, False, False, {}


# ---------------------------------------------------------------------------
# NormalizeReward tests
# ---------------------------------------------------------------------------


class TestNormalizeReward:
    def test_normalize_reward_basic(self):
        """Normalized rewards should be finite and non-zero after a few steps."""
        env = NormalizeReward(RewardDummyEnv(rewards=[1.0, 2.0, 3.0]))
        env.reset()
        rewards = []
        for _ in range(6):
            _, r, _, _, _ = env.step(np.array([0.0, 0.0]))
            rewards.append(r)
        # All rewards should be finite.
        assert all(np.isfinite(r) for r in rewards)
        # At least one non-zero reward after the first step.
        assert any(r != 0.0 for r in rewards)

    def test_normalize_reward_reset_clears_return(self):
        """The discounted return should reset to zero on reset()."""
        env = NormalizeReward(RewardDummyEnv(rewards=[5.0]))
        env.reset()
        for _ in range(5):
            env.step(np.array([0.0, 0.0]))
        assert env._return != 0.0
        env.reset()
        assert env._return == 0.0

    def test_normalize_reward_clip(self):
        """Normalized rewards must be within [-clip, clip]."""
        env = NormalizeReward(RewardDummyEnv(rewards=[1000.0]), clip=3.0)
        env.reset()
        for _ in range(10):
            _, r, _, _, _ = env.step(np.array([0.0, 0.0]))
            assert -3.0 <= r <= 3.0

    def test_normalize_reward_parameters(self):
        """Constructor parameters should be stored correctly."""
        env = NormalizeReward(
            RewardDummyEnv(), gamma=0.95, epsilon=1e-4, clip=5.0
        )
        assert env.gamma == 0.95
        assert env.epsilon == 1e-4
        assert env.clip == 5.0


# ---------------------------------------------------------------------------
# ClipReward tests
# ---------------------------------------------------------------------------


class TestClipReward:
    def test_clip_reward_basic(self):
        """Values within range pass through unchanged."""
        env = ClipReward(RewardDummyEnv(rewards=[5.0]), min_reward=-10.0, max_reward=10.0)
        env.reset()
        _, r, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert r == 5.0

    def test_clip_reward_clips_high(self):
        """Values above max_reward are clipped."""
        env = ClipReward(RewardDummyEnv(rewards=[100.0]), min_reward=-10.0, max_reward=10.0)
        env.reset()
        _, r, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert r == 10.0

    def test_clip_reward_clips_low(self):
        """Values below min_reward are clipped."""
        env = ClipReward(RewardDummyEnv(rewards=[-100.0]), min_reward=-10.0, max_reward=10.0)
        env.reset()
        _, r, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert r == -10.0

    def test_clip_reward_custom_bounds(self):
        """Custom bounds work correctly."""
        env = ClipReward(RewardDummyEnv(rewards=[50.0]), min_reward=-1.0, max_reward=1.0)
        env.reset()
        _, r, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert r == 1.0

    def test_clip_reward_negative_within_range(self):
        """Negative values within range pass through."""
        env = ClipReward(RewardDummyEnv(rewards=[-5.0]), min_reward=-10.0, max_reward=10.0)
        env.reset()
        _, r, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert r == -5.0
