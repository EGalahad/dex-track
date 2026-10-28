"""AA action term: integrated wrist twist targets and finger position PD."""
import torch
from active_adaptation.envs.mdp.actions.base import Action
from dex_track.tasks.wrist_math import integrate, inverse, multiply, rotate, log, zxy_joint_angles


class WristVelocityWrench(Action, namespace="dex_track"):
    def __init__(self, linear_speed=.5, angular_speed=2., kp_pos=350., kd_pos=25.,
                 kp_rot=40., kd_rot=3., max_force=100., max_torque=10.):
        super().__init__()
        self.linear_speed, self.angular_speed = linear_speed, angular_speed
        self.gains = (kp_pos, kd_pos, kp_rot, kd_rot, max_force, max_torque)
        self.action_dim = 12

    def _initialize(self, env):
        super()._initialize(env)
        from dex_track.tasks.tensor_kernels import select_kernel, wrist_wrench
        self._wrist_wrench = select_kernel(env, wrist_wrench)
        from dex_track.assets import description, robot_model, body_name
        cfg = description()
        joint_names = sum((cfg['joint_names'][k] for k in ('wrist_trans', 'wrist_rot')), [])
        self.joint_ids, _ = self.asset.find_joints(joint_names, preserve_order=True)
        self.names = [f'{side}_wrist_{kind}{axis}' for kind in ('v', 'omega_')
                      for side in ('right', 'left') for axis in ('x', 'y', 'z')]
        self.body_ids, _ = self.asset.find_bodies(
            [body_name(n, env.backend) for n in ['R_forearm_rot_y_link', 'L_forearm_rot_y_link']], preserve_order=True)
        model = robot_model()
        palm_ids = [model.site(cfg['site_names']['palm'][side]).id for side in ('right', 'left')]
        # These static sites define the MANO-compatible wrist frame on XHand.
        expected_bodies = [model.body(f'{side}_forearm_rot_y_link').id for side in ('R', 'L')]
        if model.site_bodyid[palm_ids].tolist() != expected_bodies:
            raise ValueError('Wrist reference sites must be attached to the controlled links')
        self.palm_p = torch.tensor(model.site_pos[palm_ids], device=self.device, dtype=torch.float32)
        self.palm_q = torch.tensor(model.site_quat[palm_ids], device=self.device, dtype=torch.float32)
        roots = [model.body(f'{side}_forearm_pos_x_link').id for side in ('R', 'L')]
        self.base_p = torch.tensor(model.body_pos[roots], device=self.device, dtype=torch.float32)
        self.base_q = torch.tensor(model.body_quat[roots], device=self.device, dtype=torch.float32)
        self.action_buf = torch.zeros(self.num_envs, 12, device=self.device)
        self.target_p = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.target_q = torch.zeros(self.num_envs, 2, 4, device=self.device)
        self.target_q[..., 0] = 1
        self.v = torch.zeros_like(self.target_p)
        self.omega = torch.zeros_like(self.v)
        self.joint_target = torch.zeros(self.num_envs, 12, device=self.device)

    def state(self):
        d = self.asset.data
        return (d.body_link_pos_w[:, self.body_ids], d.body_link_quat_w[:, self.body_ids],
                d.body_link_lin_vel_w[:, self.body_ids], d.body_link_ang_vel_w[:, self.body_ids])

    def reset(self, env_ids, tensordict):
        self.action_buf[env_ids] = 0
        self.v[env_ids], self.omega[env_ids] = 0, 0
        reference_p, reference_q = self.env.command_manager.reference_wrist_pose_w(env_ids)
        # T_world_link = T_world_MANO * inverse(T_link_palm). No sim pose read.
        target_q = multiply(reference_q, inverse(self.palm_q).expand(len(env_ids), -1, -1))
        self.target_q[env_ids] = target_q
        self.target_p[env_ids] = reference_p - rotate(target_q, self.palm_p.expand(len(env_ids), -1, -1))
        self._update_joint_target(env_ids)
        zero = torch.zeros(len(env_ids), 2, 3, device=self.device)
        self._write_wrench(zero, zero, env_ids)

    def process_action(self, action):
        self.action_buf.copy_(action)
        processed = action.clamp(-1, 1)
        self.v[:] = processed[:, :6].reshape(-1, 2, 3) * self.linear_speed
        self.omega[:] = processed[:, 6:12].reshape(-1, 2, 3) * self.angular_speed
        self.target_p[:], self.target_q[:] = integrate(
            self.target_p, self.target_q, self.v, self.omega, self.env.step_dt)
        self._update_joint_target()

    def _update_joint_target(self, env_ids=slice(None)):
        target_p, target_q = self.target_p[env_ids], self.target_q[env_ids]
        base_q = self.base_q.expand(target_p.shape[0], -1, -1)
        local_p = rotate(inverse(base_q), target_p-self.env.scene.env_origins[env_ids, None]-self.base_p)
        local_q = multiply(inverse(base_q), target_q)
        near = self.joint_target[env_ids, 6:12].reshape(-1, 2, 3)
        self.joint_target[env_ids, :6] = local_p.flatten(1)
        self.joint_target[env_ids, 6:12] = zxy_joint_angles(local_q, near).flatten(1)

    def _write_wrench(self, force, torque, env_ids=None):
        if self.env.backend == 'mjlab':
            self.asset.write_external_wrench_to_sim(force, torque, body_ids=self.body_ids, env_ids=env_ids)
        else:
            self.asset.set_external_force_and_torque(force, torque, body_ids=self.body_ids,
                                                    env_ids=env_ids, is_global=True)

    def apply_action(self, substep):
        p, q, v, omega = self.state()
        force, torque = self._wrist_wrench(p, q, v, omega,
            self.asset.data.body_com_pos_w[:, self.body_ids], self.target_p, self.target_q,
            self.v, self.omega, self.gains)
        self._write_wrench(force, torque)
        self.asset.set_joint_position_target(self.joint_target, joint_ids=self.joint_ids)

    def controller_state(self):
        p, q, _, _ = self.state()
        return torch.cat((self.target_p-p, log(multiply(self.target_q, inverse(q))),
                          self.v, self.omega), -1).flatten(1)


class FingerPosition(Action, namespace="dex_track"):
    """Absolute finger targets in joint-limit coordinates; independent of wrists."""
    def __init__(self):
        super().__init__()
        self.action_dim = 24

    def _initialize(self, env):
        super()._initialize(env)
        from dex_track.assets import description
        self.names = list(description()['joint_names']['finger'])
        self.joint_ids, _ = self.asset.find_joints(self.names, preserve_order=True)
        self.limits = self.asset.data.soft_joint_pos_limits[:, self.joint_ids]
        self.action_buf = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.joint_target = self.asset.data.joint_pos[:, self.joint_ids].clone()
        self.action_delta = torch.zeros_like(self.action_buf)
        self.has_previous_action = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids, tensordict):
        self.action_buf[env_ids] = 0
        self.action_delta[env_ids] = 0
        self.has_previous_action[env_ids] = False
        self.joint_target[env_ids] = self.asset.data.joint_pos[env_ids][:, self.joint_ids]

    def process_action(self, action):
        self.action_delta.copy_(torch.where(self.has_previous_action[:, None], action-self.action_buf, 0.))
        self.action_buf.copy_(action)
        self.has_previous_action[:] = True
        processed = action.clamp(-1, 1)
        low, high = self.limits.unbind(-1)
        self.joint_target[:] = low + (processed + 1) * .5 * (high-low)

    def apply_action(self, substep):
        self.asset.set_joint_position_target(self.joint_target, joint_ids=self.joint_ids)
