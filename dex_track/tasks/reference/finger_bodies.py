"""Full finger link poses relative to each hand's wrist link (not palm site)."""
import numpy as np
import torch
from dex_track.assets import description, robot_model, body_name
from .hand import ALLLINK_BODY_NAMES_BY_SIDE
from ..wrist_math import inverse, multiply, rotate, log

SIDES = ('right', 'left')

def local_poses(position, quaternion):
    """Input [..., 2, 13, xyz/wxyz], wrist first, then 12 finger links."""
    wrist_inv = inverse(quaternion[..., :1, :])
    p = rotate(wrist_inv.expand_as(quaternion[..., 1:, :]), position[..., 1:, :]-position[..., :1, :])
    q = multiply(wrist_inv.expand_as(quaternion[..., 1:, :]), quaternion[..., 1:, :])
    return p, q

def fk_local(joint_positions, joint_names, device, relative_to_wrist=True):
    """FK after time resampling, preserving the exact joint interpolation used at reset."""
    import mujoco
    model = robot_model()
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[model.joint(n).id] for n in joint_names]
    ids = np.array([[model.body(n).id for n in ALLLINK_BODY_NAMES_BY_SIDE[s]] for s in SIDES])
    rows = joint_positions.detach().cpu().numpy()
    positions, quaternions = [], []
    for row in rows:
        data.qpos[:] = model.qpos0
        data.qpos[addresses] = row
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[ids].copy())
        quaternions.append(data.xquat[ids].copy())
    p = torch.as_tensor(np.asarray(positions), device=device, dtype=torch.float32)
    q = torch.as_tensor(np.asarray(quaternions), device=device, dtype=torch.float32)
    return local_poses(p, q) if relative_to_wrist else (p[:, :, 1:], q[:, :, 1:])

def nominal_local(nominal, device):
    names = description()['joint_names']['finger']
    if set(nominal) != set(names):
        raise ValueError('Nominal must contain exactly the 24 named finger joints')
    model = robot_model()
    values = np.array([nominal[n] for n in names])
    limits = np.array([model.jnt_range[model.joint(n).id] for n in names])
    if not np.isfinite(values).all() or ((values < limits[:, 0]) | (values > limits[:, 1])).any():
        raise ValueError('Nominal finger joints must be finite and within limits')
    return fk_local(torch.as_tensor(values[None]), names, device)

class FingerBodyReference:
    def __init__(self, command):
        self.command = command
        self.body_names = [ALLLINK_BODY_NAMES_BY_SIDE[s] for s in SIDES]
        self.body_ids = [[command.robot.body_names.index(body_name(n, command.env.backend)) for n in row] for row in self.body_names]
        self.retarget = None
        self.retarget_world = None
        self.nominals = {}

    def prepare(self, reference, nominal=None):
        if reference == 'retarget':
            if self.retarget is None:
                self.retarget = fk_local(self.command.dataset.reset.joint_pos, self.command.robot.joint_names, self.command.device)
            return None
        if reference != 'nominal':
            raise ValueError(reference)
        key = tuple(sorted(dict(nominal or {}).items()))
        if key not in self.nominals:
            self.nominals[key] = nominal_local(dict(key), self.command.device)
        return key

    def actual(self):
        state = self.command.finger_body_state()
        return state.position_local, state.quaternion_local

    def world_trajectory(self):
        """Same global retarget pose used by future finger-error observations."""
        if self.retarget_world is None:
            self.prepare('retarget')
            c = self.command
            if hasattr(c, 'free_root_states'):
                p, q = self.retarget
                root = c.free_root_states
                root_q = root[..., None, 3:7].expand_as(q)
                self.retarget_world = (root[..., None, :3] + rotate(root_q, p),
                                       multiply(root_q, q))
            else:
                self.retarget_world = fk_local(c.dataset.reset.joint_pos,
                    c.robot.joint_names, c.device, relative_to_wrist=False)
        return self.retarget_world

    def target(self, reference, key=None):
        if reference == 'retarget':
            indices = self.command.frame_indices
            return tuple(v[indices] for v in self.retarget)
        return self.nominals[key]

    def errors(self, reference, key=None):
        return self.command.finger_body_tracking_errors(reference, key)


def finger_body_reference(command):
    if not hasattr(command, '_finger_body_reference'):
        command._finger_body_reference = FingerBodyReference(command)
    return command._finger_body_reference
