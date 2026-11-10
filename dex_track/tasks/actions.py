"""Floating-wrist wrench control and filtered finger position / integrated velocity targets."""
import torch
from active_adaptation.envs.mdp.actions.base import Action
from .transforms import integrate


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
        from dex_track.tasks.transforms import select_kernel, wrist_wrench
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


class FingerFilteredPosition(Action, namespace='dex_track'):
    """Absolute nominal-position offsets with a second-order low-pass filter.

    Policy action still means an absolute offset about the named nominal pose
    (``target = nominal + action_scale * a``). Each physics substep, a fixed
    second-order low-pass filter produces the PD position/velocity targets::

        v_dot = omega^2 * (target - q_bar) - 2*zeta*omega * v_bar
        v_bar <- v_bar + dt * v_dot
        q_bar <- q_bar + dt * v_bar

    ``omega`` (rad/s) and ``zeta`` are constants, not policy outputs. Use
    ``zeta >= 1`` to avoid resonant peaking. Filter state resets to the current
    finger joints with ``v_bar = 0`` on episode start.
    """

    def __init__(self, nominal, action_scale=0.5, omega=20.0, zeta=1.0):
        super().__init__()
        self.action_dim = 24
        self.nominal_config = dict(nominal)
        self.action_scale = float(action_scale)
        if not 0 < self.action_scale < float('inf'):
            raise ValueError('action_scale must be finite and positive')
        self.omega = float(omega)
        self.zeta = float(zeta)
        if not 0 < self.omega < float('inf'):
            raise ValueError('omega must be finite and positive')
        if not 0 < self.zeta < float('inf'):
            raise ValueError('zeta must be finite and positive')

    def _initialize(self, env):
        super()._initialize(env)
        from dex_track.assets import description
        self.names = list(description()['joint_names']['finger'])
        self.joint_ids, _ = self.asset.find_joints(self.names, preserve_order=True)
        self.action_buf = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.joint_target = self.asset.data.joint_pos[:, self.joint_ids].clone()
        self.q_bar = self.joint_target.clone()
        self.v_bar = torch.zeros_like(self.q_bar)
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
        self.q_bar[env_ids] = self.joint_target[env_ids]
        self.v_bar[env_ids] = 0

    def process_action(self, action):
        self.action_delta.copy_(torch.where(self.has_previous_action[:, None], action-self.action_buf, 0.))
        self.action_buf.copy_(action)
        self.has_previous_action[:] = True
        self.joint_target[:] = self.nominal + self.action_scale * action

    def apply_action(self, substep):
        target = self.joint_target
        v_dot = (
            (self.omega ** 2) * (target - self.q_bar)
            - 2.0 * self.zeta * self.omega * self.v_bar
        )
        dt = float(self.env.physics_dt)
        self.v_bar.add_(dt * v_dot)
        self.q_bar.add_(dt * self.v_bar)
        self.asset.set_joint_position_target(self.q_bar, joint_ids=self.joint_ids)
        self.asset.set_joint_velocity_target(self.v_bar, joint_ids=self.joint_ids)


class FingerVelocity(Action, namespace='dex_track'):
    """Integrate bounded joint velocities once per policy step; PD velocity target is zero.

    ``action_scale`` is rad/s, so ``q_target += clamp(action, -1, 1) * scale * dt``.
    Reset starts the target at the actual reset joint positions, including noise.
    The integrated position target is not clipped to joint limits.
    """
    def __init__(self, action_scale=2.0):
        super().__init__()
        self.action_dim = 24
        self.action_scale = float(action_scale)
        if not 0 < self.action_scale < float('inf'):
            raise ValueError('action_scale must be finite and positive')

    def _initialize(self, env):
        super()._initialize(env)
        from dex_track.assets import description
        self.names = list(description()['joint_names']['finger'])
        self.joint_ids, _ = self.asset.find_joints(self.names, preserve_order=True)
        self.action_buf = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.action_delta = torch.zeros_like(self.action_buf)
        self.has_previous_action = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.joint_target = self.asset.data.joint_pos[:, self.joint_ids].clone()
        self.zero_velocity = torch.zeros_like(self.joint_target)

    def reset(self, env_ids, tensordict):
        self.action_buf[env_ids] = 0
        self.action_delta[env_ids] = 0
        self.has_previous_action[env_ids] = False
        self.joint_target[env_ids] = self.asset.data.joint_pos[env_ids][:, self.joint_ids]

    def process_action(self, action):
        self.action_delta.copy_(torch.where(self.has_previous_action[:, None], action-self.action_buf, 0.))
        self.action_buf.copy_(action)
        self.has_previous_action[:] = True
        self.joint_target.add_(action.clamp(-1, 1) * self.action_scale * self.env.step_dt)

    def apply_action(self, substep):
        self.asset.set_joint_position_target(self.joint_target, joint_ids=self.joint_ids)
        self.asset.set_joint_velocity_target(self.zero_velocity, joint_ids=self.joint_ids)
