"""Absolute finger position targets about a named nominal pose."""
import torch
from active_adaptation.envs.mdp.actions.base import Action


class FingerNominalPosition(Action, namespace='dex_track'):
    """Unclipped absolute targets: named nominal + scale_rad * sampled action."""
    def __init__(self, nominal, action_scale=1.0):
        super().__init__()
        self.action_dim = 24
        self.nominal_config = dict(nominal)
        self.action_scale = float(action_scale)
        if not 0 < self.action_scale < float('inf'):
            raise ValueError('action_scale must be finite and positive')

    def _initialize(self, env):
        super()._initialize(env)
        from dex_track.assets import description
        self.names = list(description()['joint_names']['finger'])
        self.joint_ids, _ = self.asset.find_joints(self.names, preserve_order=True)
        self.action_buf = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.joint_target = self.asset.data.joint_pos[:, self.joint_ids].clone()
        self.action_delta = torch.zeros_like(self.action_buf)
        self.has_previous_action = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if set(self.nominal_config) != set(self.names):
            raise ValueError('Nominal must specify all 24 finger joints by name')
        self.nominal = torch.tensor([self.nominal_config[n] for n in self.names], device=self.device)
        if not torch.isfinite(self.nominal).all():
            raise ValueError('Nominal must be finite')

    def reset(self, env_ids, tensordict):
        self.action_buf[env_ids] = 0
        self.action_delta[env_ids] = 0
        self.has_previous_action[env_ids] = False
        self.joint_target[env_ids] = self.asset.data.joint_pos[env_ids][:, self.joint_ids]

    def process_action(self, action):
        self.action_delta.copy_(torch.where(self.has_previous_action[:, None], action-self.action_buf, 0.))
        self.action_buf.copy_(action)
        self.has_previous_action[:] = True
        self.joint_target[:] = self.nominal + self.action_scale * action

    def apply_action(self, substep):
        self.asset.set_joint_position_target(self.joint_target, joint_ids=self.joint_ids)
