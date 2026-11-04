"""Robot and controller state; no scene or reference geometry."""
import torch
from .common import TrackObservation
from active_adaptation.envs.utils import find_joints
from ..reference.math import matrix_from_quat, quat_apply_inverse


class _proprioception_history(TrackObservation):
    """Newest-first policy-frame history, filled from reset state per env."""
    def __init__(self, history_steps=None):
        super().__init__()
        steps = None if history_steps is None else tuple(history_steps)
        if steps is not None and (not steps or any(type(x) is not int or x < 0 for x in steps)
                                  or len(set(steps)) != len(steps)):
            raise ValueError('history_steps must be unique nonnegative integer offsets')
        self.history_steps = steps
        self.history_buffer = None

    def _ensure_history(self):
        if self.history_buffer is None:
            value = self._current()
            self.capacity = max(self.history_steps) + 1
            self.history_offsets = torch.tensor(self.history_steps, device=self.device)
            self.history_head = 0
            self.history_buffer = value[:, None].expand(-1, self.capacity, -1).clone()
            self.pending_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.has_pending_reset = False

    def _finish_reset(self):
        # Observation reset callbacks can precede command/action reset callbacks.
        # Delay state sampling until compute/update, after the entire reset.
        if self.has_pending_reset:
            current = self._current()
            self.history_buffer[:] = torch.where(
                self.pending_reset[:, None, None], current[:, None], self.history_buffer)
            self.pending_reset.zero_()
            self.has_pending_reset = False

    def update(self):
        if self.history_steps is not None:
            self._ensure_history()
            self._finish_reset()
            self.history_head = (self.history_head - 1) % self.capacity
            self.history_buffer[:, self.history_head] = self._current()

    def reset(self, env_ids, tensordict=None):
        if self.history_steps is not None:
            self._ensure_history()
            self.pending_reset[env_ids] = True
            self.has_pending_reset = True

    def compute(self):
        if self.history_steps is None:
            return self._current()
        self._ensure_history()
        self._finish_reset()
        indices = (self.history_offsets + self.history_head) % self.capacity
        return self.history_buffer[:, indices].flatten(1)


class wrist_projected_gravity(_proprioception_history, namespace='dex_track'):
    def _current(self):
        c = self.command_manager
        gravity = torch.zeros_like(c.robot_wrist_pos_w)
        gravity[..., 2] = -1.
        return quat_apply_inverse(c.robot_wrist_quat_w, gravity).flatten(1)


class wrist_velocity(_proprioception_history, namespace='dex_track'):
    def __init__(self, local_frame=False, history_steps=None):
        super().__init__(history_steps=history_steps)
        self.local_frame = bool(local_frame)

    def _current(self):
        c = self.command_manager
        v, w = c.robot_wrist_lin_vel_w, c.robot_wrist_ang_vel_w
        if self.local_frame:
            v = quat_apply_inverse(c.robot_wrist_quat_w, v)
            w = quat_apply_inverse(c.robot_wrist_quat_w, w)
        return torch.cat((v, w), -1).flatten(1)


class finger_joint_pos(_proprioception_history, namespace='dex_track'):
    """Finger-only qpos; wrist pose and twist have dedicated observations."""
    def _initialize(self, env):
        super()._initialize(env)
        self.asset = env.scene['robot']
        self.ids, _ = find_joints(self.asset, '(right|left)_hand_.*')
        if len(self.ids) != 24:
            raise ValueError('Expected 24 finger joints')

    def _current(self):
        return self.asset.data.joint_pos[:, self.ids]


class finger_joint_vel(finger_joint_pos, namespace='dex_track'):
    def _current(self):
        return self.asset.data.joint_vel[:,self.ids]


class previous_action(_proprioception_history, namespace='dex_track'):
    def _current(self):
        return torch.cat([self.env.input_managers[n].action_buf for n in ('wrist','finger')],-1)


class wrist_controller_state(TrackObservation, namespace='dex_track'):
    """Current PD target URDF wrist frame in the current URDF wrist frame.

    Each hand contributes local translation and the first two rotation columns.
    No palm translation or rotation is composed into this observation.
    """
    def compute(self):
        action = self.env.input_managers['wrist']
        from ..wrist_math import inverse, multiply, rotate
        current_p, current_q, _, _ = action.state()
        delta = rotate(inverse(current_q), action.target_p-current_p)
        rotation = matrix_from_quat(multiply(inverse(current_q), action.target_q))
        return torch.cat((delta, rotation[..., :2].flatten(-2)), -1).flatten(1)
