"""Reward base typed to the motion command."""
from active_adaptation.envs.mdp.rewards.base import Reward
from ..command import MotionTracking
TrackReward = Reward[MotionTracking]
