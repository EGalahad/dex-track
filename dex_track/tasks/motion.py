"""Typed motion containers, following any4hdmi's TensorClass convention.

Frames use the dataset's world axes, with an origin at zero. The command adds
AA env origins to positions. Retargeted robot states live in a reset-only buffer.
"""
from dataclasses import dataclass
import math

import torch
from tensordict import TensorClass
from any4hdmi.dataset.interpolation import interpolate_qpos_qvel_packed_torch
from dex_track.data import dataset_fps
from .reference.math import matrix_from_quat, quat_from_matrix


class MotionData(TensorClass):
    mano_wrist_pos_w: torch.Tensor
    mano_wrist_rotmat_w: torch.Tensor
    mano_joint_pos_w: torch.Tensor
    object_pos_w: torch.Tensor
    object_rotmat_w: torch.Tensor


class ResetData(TensorClass):
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class ObjectState(TensorClass):
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor


@dataclass
class MotionDataset:
    frames: MotionData
    reset: ResetData
    objects: ObjectState
    starts: torch.Tensor
    lengths: torch.Tensor
    object_indices: torch.Tensor
    mano_joint_names: dict[str, list[str]]
    hand_sides: tuple[str, ...]
    fps: float
    source_fps: float

    def frame_indices(self, motion_ids, steps):
        return self.starts[motion_ids] + steps.clamp(min=0).minimum(self.lengths[motion_ids] - 1)

    def get_slice(self, motion_ids, steps, offsets):
        indices = self.frame_indices(motion_ids[:, None], steps[:, None] + offsets[None])
        return self.frames[indices]


def _resample_containers(containers, lengths, source_fps, target_fps):
    """One shared per-clip timeline for every reference and reset tensor.

    Reuse the same any4hdmi interpolation as mimic-lite: vectors/scalar joints
    and velocities use linear interpolation; orientations use shortest-arc SLERP.
    """
    if source_fps == target_fps:
        return containers, lengths
    columns = [[], []]  # configuration/position fields, velocity fields
    widths = [0, 0]
    schema = []
    quaternion_slices = []
    for container_index, container in enumerate(containers):
        for name, tensor in container.to_tensordict().items():
            is_rotation = name.endswith('rotmat_w')
            value = quat_from_matrix(tensor) if is_rotation else tensor
            group = int(name.endswith('vel_w') or name == 'joint_vel')
            start = widths[group]
            flat = value.flatten(1)
            widths[group] += flat.shape[1]
            columns[group].append(flat)
            if is_rotation or name.endswith('quat_w'):
                quaternion_slices.extend((i, i + 4) for i in range(start, widths[group], 4))
            schema.append((container_index, name, group, start, widths[group], value.shape[1:], is_rotation))
    qpos, qvel = [torch.cat(parts, -1) for parts in columns]
    qpos, qvel, output_lengths = interpolate_qpos_qvel_packed_torch(
        list(qpos.split(lengths)), list(qvel.split(lengths)),
        source_fps=source_fps, target_fps=target_fps,
        quaternion_slices=tuple(quaternion_slices))
    outputs = [{} for _ in containers]
    for index, name, group, start, end, shape, is_rotation in schema:
        value = (qpos, qvel)[group][:, start:end].reshape(-1, *shape)
        outputs[index][name] = matrix_from_quat(value) if is_rotation else value
    total = sum(output_lengths)
    return [type(c)(**values, batch_size=[total], device=qpos.device)
            for c, values in zip(containers, outputs)], output_lengths


def load_motion_dataset(raw, *, scene_state, step_dt, joint_order, device):
    """Read meta.fps and resample each complete clip to the policy rate."""
    source_fps = dataset_fps(raw)
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError(f'Invalid policy step_dt={step_dt}')
    target_fps = 1. / step_dt
    sides = ('right', 'left')
    lengths = raw['motion_num_frames'].to(device=device, dtype=torch.long)
    starts = raw['length_starts'].to(device=device, dtype=torch.long)
    if lengths.ndim != 1 or not len(lengths) or (lengths < 2).any():
        raise ValueError('Expected nonempty clips with at least two frames')
    if not torch.equal(starts, lengths.cumsum(0) - lengths):
        raise ValueError('Non-contiguous motion starts')
    total = int(lengths.sum())
    def stack(prefix, suffix):
        return torch.stack([raw[f'{prefix}_{s}_{suffix}'] for s in sides], 1).to(device=device, dtype=torch.float32)
    frames = MotionData(
        mano_wrist_pos_w=stack('mano', 'wrist_pos'),
        mano_wrist_rotmat_w=stack('mano', 'wrist_rot'),
        mano_joint_pos_w=stack('mano', 'joints'),
        object_pos_w=stack('obj', 'pos'),
        object_rotmat_w=stack('obj', 'rotmat'),
        batch_size=[total], device=device,
    )
    reset = ResetData(
        joint_pos=raw['joint_pos'][:, joint_order].to(device=device, dtype=torch.float32),
        joint_vel=raw['joint_vel'][:, joint_order].to(device=device, dtype=torch.float32),
        batch_size=[total], device=device,
    )
    objects = ObjectState(**{f'{k}_w': v.to(device=device, dtype=torch.float32)
                             for k, v in scene_state.items()}, batch_size=[total], device=device)
    (frames, reset, objects), new_lengths = _resample_containers(
        [frames, reset, objects], lengths.tolist(), source_fps, target_fps)
    lengths = torch.tensor(new_lengths, dtype=torch.long, device=device)
    starts = lengths.cumsum(0) - lengths
    return MotionDataset(frames, reset, objects, starts, lengths,
        torch.stack([raw[f'motion_object_slot_{s}'] for s in sides], 1).to(device=device, dtype=torch.long),
        {s: list(raw[f'mano_{s}_joint_names']) for s in sides}, sides, target_fps, source_fps)
