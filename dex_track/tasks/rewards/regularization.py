"""Action and joint-limit regularization."""
import torch
from .common import TrackReward


class wrist_action_regularization(TrackReward, namespace='dex_track'):
    """Raw wrist action magnitude, averaged over its 12 dimensions."""
    def _compute(self):
        return self.env.input_managers['wrist'].action_buf.square().mean(-1, keepdim=True)



class finger_action_rate(TrackReward, namespace='dex_track'):
    """Raw finger action delta, averaged over 24 dimensions; no dt division."""
    def _compute(self):
        return self.env.input_managers['finger'].action_delta.square().mean(-1, keepdim=True)



class finger_joint_pos_limits(TrackReward, namespace='dex_track'):
    """Mimic-lite linear actual-angle soft-limit penalty, summed over fingers."""
    def __init__(self, weight=10., soft_factor=.9, enabled=True):
        super().__init__(weight, enabled)
        if not 0 < soft_factor <= 1:
            raise ValueError('soft_factor must be in (0, 1]')
        self.soft_factor = soft_factor

    def _initialize(self, env):
        super()._initialize(env)
        self.finger = env.input_managers['finger']
        limits = self.finger.asset.data.joint_pos_limits[:, self.finger.joint_ids]
        midpoint = limits.mean(-1)
        half = (limits[..., 1]-limits[..., 0])*.5*self.soft_factor
        self.lower, self.upper = midpoint-half, midpoint+half

    def _compute(self):
        q = self.finger.asset.data.joint_pos[:, self.finger.joint_ids]
        return -((self.lower-q).clamp_min(0)+(q-self.upper).clamp_min(0)).sum(-1, keepdim=True)



class survival(TrackReward, namespace='dex_track'):
    def _compute(self):
        return torch.ones(self.num_envs, 1, device=self.device)
