"""Independent reward terms; computing a reward never mutates the command."""
import torch
from active_adaptation.envs.mdp.rewards.base import Reward as BaseReward
from ..command import MotionTracking
from ..wrist_math import log, multiply, inverse

TrackReward = BaseReward[MotionTracking]


class _exponential_tracking(TrackReward):
    def __init__(self, weight, sigma, enabled=True):
        super().__init__(weight, enabled)
        if sigma <= 0:
            raise ValueError('Reward sigma must be positive')
        self.sigma = sigma


class object_tracking(_exponential_tracking, namespace='dex_track'):
    def _compute(self):
        return self.command_manager.scene_objects.reward(self.sigma).unsqueeze(-1)


class tip_object_distance(_exponential_tracking, namespace='dex_track'):
    def _compute(self):
        c = self.command_manager
        reference_distance = c.ref_mano_tip_object_distance
        values = []
        for si, side in enumerate(c.hand_sides):
            sdf, _ = c.sdf_query(c.robot_tip_pos_w[:, si], side)
            values.append(torch.exp(-(sdf - reference_distance[:, si]).clamp_min(0) / self.sigma).mean(-1))
        return torch.stack(values, -1).mean(-1, keepdim=True)


class wrist_position_tracking(_exponential_tracking, namespace='dex_track'):
    def _compute(self):
        c = self.command_manager
        error = (c.ref_mano_wrist_pos_w - c.robot_wrist_pos_w).norm(dim=-1)
        return torch.exp(-error / self.sigma).mean(-1, keepdim=True)


class wrist_orientation_tracking(_exponential_tracking, namespace='dex_track'):
    def _compute(self):
        c = self.command_manager
        error = log(multiply(c.ref_mano_wrist_quat_w, inverse(c.robot_wrist_quat_w))).norm(dim=-1)
        return torch.exp(-error / self.sigma).mean(-1, keepdim=True)


class fingertip_tracking(_exponential_tracking, namespace='dex_track'):
    def _compute(self):
        c = self.command_manager
        error = (c.ref_mano_tip_pos_w - c.robot_tip_pos_w).norm(dim=-1)
        return torch.exp(-error / self.sigma).mean(-1).mean(-1, keepdim=True)


class action_regularization(TrackReward, namespace='dex_track'):
    def _compute(self):
        wrist = self.env.input_managers['wrist'].action_buf
        finger_delta = self.env.input_managers['finger'].action_delta
        return torch.cat((wrist, finger_delta), -1).square().mean(-1, keepdim=True)
