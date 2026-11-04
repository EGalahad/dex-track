"""Integrated wrist twist targets and Cartesian PD wrench."""
import torch
from active_adaptation.envs.mdp.actions.base import Action
from dex_track.tasks.wrist_math import integrate


class WristVelocityWrench(Action, namespace="dex_track"):
    def __init__(self, linear_speed=.5, angular_speed=2., kp_pos=350., kd_pos=25.,
                 kp_rot=40., kd_rot=3., max_force=100., max_torque=10.,
                 desired_velocity_feedforward=True):
        super().__init__()
        self.linear_speed, self.angular_speed = linear_speed, angular_speed
        self.gains = (kp_pos, kd_pos, kp_rot, kd_rot, max_force, max_torque)
        self.desired_velocity_feedforward = bool(desired_velocity_feedforward)
        self.action_dim = 12

    def _initialize(self, env):
        super()._initialize(env)
        from dex_track.tasks.tensor_kernels import select_kernel, wrist_wrench
        self._wrist_wrench = select_kernel(env, wrist_wrench)
        from dex_track.assets import body_name
        if not getattr(self.asset, 'is_free_hand_pair', False):
            raise ValueError('WristVelocityWrench requires the current free-hand pair')
        self.names = [f'{side}_wrist_{kind}{axis}' for kind in ('v', 'omega_')
                      for side in ('right', 'left') for axis in ('x', 'y', 'z')]
        self.body_ids, _ = self.asset.find_bodies(
            [body_name(n, env.backend) for n in ['R_forearm_rot_y_link', 'L_forearm_rot_y_link']], preserve_order=True)
        self.action_buf = torch.zeros(self.num_envs, 12, device=self.device)
        self.target_p = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.target_q = torch.zeros(self.num_envs, 2, 4, device=self.device)
        self.target_q[..., 0] = 1
        self.v = torch.zeros_like(self.target_p)
        self.omega = torch.zeros_like(self.v)
        self.zero_velocity = torch.zeros_like(self.v)

    def state(self):
        return self.asset.direct_wrist_state()

    def reset(self, env_ids, tensordict):
        self.action_buf[env_ids] = 0
        self.v[env_ids], self.omega[env_ids] = 0, 0
        # Command owns reference-to-URDF conversion; controller only sees wrist roots.
        reference_p, reference_q = self.env.command_manager.reference_control_wrist_pose_w(env_ids)
        self.target_p[env_ids] = reference_p
        self.target_q[env_ids] = reference_q
        zero = torch.zeros(len(env_ids), 2, 3, device=self.device)
        self._write_wrench(zero, zero, env_ids)

    def process_action(self, action):
        self.action_buf.copy_(action)
        processed = action.clamp(-1, 1)
        self.v[:] = processed[:, :6].reshape(-1, 2, 3) * self.linear_speed
        self.omega[:] = processed[:, 6:12].reshape(-1, 2, 3) * self.angular_speed
        self.target_p[:], self.target_q[:] = integrate(
            self.target_p, self.target_q, self.v, self.omega, self.env.step_dt)

    def _write_wrench(self, force, torque, env_ids=None):
        if self.env.backend == 'mjlab':
            self.asset.write_external_wrench_to_sim(force, torque, body_ids=self.body_ids, env_ids=env_ids)
        else:
            self.asset.set_external_force_and_torque(force, torque, body_ids=self.body_ids,
                                                    env_ids=env_ids, is_global=True)

    def apply_action(self, substep):
        p, q, v, omega = self.state()
        force, torque = self._wrist_wrench(p, q, v, omega,
            self.asset.direct_wrist_com(), self.target_p, self.target_q,
            self.v if self.desired_velocity_feedforward else self.zero_velocity,
            self.omega if self.desired_velocity_feedforward else self.zero_velocity, self.gains)
        self._write_wrench(force, torque)
