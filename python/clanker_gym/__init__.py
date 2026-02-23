"""Clanker Gym: Python client for Clankers simulation training."""

from clanker_gym.spaces import Box, Discrete
from clanker_gym.client import GymClient
from clanker_gym.env import ClankerEnv
from clanker_gym.vec_env import ClankerVecEnv

__all__ = [
    "Box",
    "Discrete",
    "GymClient",
    "ClankerEnv",
    "ClankerVecEnv",
]

__version__ = "0.1.0"
