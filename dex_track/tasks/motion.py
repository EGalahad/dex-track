"""Aligned MANO, current free-root robot, and complete object trajectories."""
from dataclasses import dataclass
import math

import torch
from tensordict import TensorClass
from .resampling import resample_packed
from dex_track.data import dataset_fps
from .transforms import matrix_from_quat, quat_from_matrix, pose_velocities


class ManoData(TensorClass):
    mano_wrist_pos_w: torch.Tensor
    mano_wrist_rotmat_w: torch.Tensor
    mano_joint_pos_w: torch.Tensor


class RobotPose(TensorClass):
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_pos: torch.Tensor


class ObjectPose(TensorClass):
    pos_w: torch.Tensor
    quat_w: torch.Tensor


class RobotData(TensorClass):
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor


class ObjectState(TensorClass):
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor


@dataclass
class MotionDataset:
    mano: ManoData
    robot: RobotData
    objects: ObjectState
    starts: torch.Tensor
    lengths: torch.Tensor
    hand_slots: torch.Tensor
    mano_joint_names: dict[str, list[str]]
    hand_sides: tuple[str, ...]
    fps: float
    source_fps: float

    def frame_indices(self, motion_ids, steps):
        return self.starts[motion_ids] + steps.clamp(min=0).minimum(self.lengths[motion_ids] - 1)

    def get_slice(self, motion_ids, steps, offsets):
        indices = self.frame_indices(motion_ids[:, None], steps[:, None] + offsets[None])
        return self.mano[indices]

    def hand_object_trajectory(self):
        """Select each segment's associated objects from its complete scene."""
        slots = torch.repeat_interleave(self.hand_slots, self.lengths, dim=0)
        frames = torch.arange(len(slots), device=slots.device)[:, None]
        return self.objects.pos_w[frames, slots], matrix_from_quat(self.objects.quat_w[frames, slots])


def _resample_containers(containers, lengths, source_fps, target_fps):
    """One shared per-clip timeline for every reference and reset tensor.

    Reuse the same any4hdmi interpolation as mimic-lite: vectors/scalar joints
    and velocities use linear interpolation; orientations use shortest-arc SLERP.
    """
    if source_fps == target_fps:
        return containers, lengths
    columns = []  # Pose fields only. No stored velocity is interpolated.
    width = 0
    schema = []
    quaternion_slices = []
    for container_index, container in enumerate(containers):
        for name, tensor in container.to_tensordict().items():
            is_rotation = name.endswith('rotmat_w')
            value = quat_from_matrix(tensor) if is_rotation else tensor
            if 'vel' in name:
                raise ValueError('Resampling accepts pose-only containers')
            start = width
            flat = value.flatten(1)
            width += flat.shape[1]
            columns.append(flat)
            if is_rotation or name.endswith('quat_w'):
                quaternion_slices.extend((i, i + 4) for i in range(start, width, 4))
            schema.append((container_index, name, start, width, value.shape[1:], is_rotation))
    qpos = torch.cat(columns, -1)
    # The shared interpolation API accepts an empty velocity channel.
    empty = qpos.new_empty((len(qpos),0))
    qpos, _, output_lengths = resample_packed(
        list(qpos.split(lengths)), list(empty.split(lengths)),
        source_fps=source_fps, target_fps=target_fps,
        quaternion_slices=tuple(quaternion_slices))
    outputs = [{} for _ in containers]
    for index, name, start, end, shape, is_rotation in schema:
        value = qpos[:, start:end].reshape(-1, *shape)
        outputs[index][name] = matrix_from_quat(value) if is_rotation else value
    total = sum(output_lengths)
    return [type(c)(**values, batch_size=[total], device=qpos.device)
            for c, values in zip(containers, outputs)], output_lengths


def load_motion_dataset(raw, *, scene_state, hand_slots, step_dt, joint_names, device):
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
    mano = ManoData(
        mano_wrist_pos_w=stack('mano', 'wrist_pos'),
        mano_wrist_rotmat_w=stack('mano', 'wrist_rot'),
        mano_joint_pos_w=stack('mano', 'joints'),
        batch_size=[total], device=device,
    )
    if 'qpos' not in raw or raw.get('robot', {}).get('representation') != 'free-root-qpos-wxyz-v1':
        raise ValueError('Convert this legacy dataset with data_pipeline/convert_motion_storage.py before training')
    names = [name for row in raw['robot']['finger_joint_names'] for name in row]
    order = [names.index(name) for name in joint_names]
    qpos = raw['qpos'].to(device=device, dtype=torch.float32).reshape(total, 2, 19)
    position = qpos[:, :, :3]
    joint_pos = qpos[:, :, 7:].reshape(total, 24)[:, order]
    robot = RobotPose(
        root_pos_w=position, root_quat_w=qpos[:, :, 3:7], joint_pos=joint_pos,
        batch_size=[total], device=device,
    )
    objects = ObjectPose(**{f'{k}_w': v.to(device=device, dtype=torch.float32)
                             for k, v in scene_state.items()}, batch_size=[total], device=device)
    (mano, robot, objects), new_lengths = _resample_containers(
        [mano, robot, objects], lengths.tolist(), source_fps, target_fps)
    lengths = torch.tensor(new_lengths, dtype=torch.long, device=device)
    starts = lengths.cumsum(0) - lengths
    # Derive velocities on the final policy timeline, never across segment ends.
    root_v, root_w = pose_velocities(
        robot.root_pos_w, robot.root_quat_w, starts, lengths, step_dt)
    frame = torch.arange(int(lengths.sum()), device=device)
    lower = torch.repeat_interleave(starts, lengths)
    upper = torch.repeat_interleave(starts + lengths - 1, lengths)
    previous, following = torch.maximum(frame - 1, lower), torch.minimum(frame + 1, upper)
    elapsed = ((following - previous) * step_dt).clamp_min(step_dt)
    joint_v = (robot.joint_pos[following] - robot.joint_pos[previous]) / elapsed[:, None]
    robot = RobotData(root_pos_w=robot.root_pos_w, root_quat_w=robot.root_quat_w, joint_pos=robot.joint_pos,
                      joint_vel=joint_v, root_lin_vel_w=root_v, root_ang_vel_w=root_w, batch_size=robot.batch_size, device=device)
    object_v, object_w = pose_velocities(objects.pos_w, objects.quat_w, starts, lengths, step_dt)
    objects = ObjectState(pos_w=objects.pos_w, quat_w=objects.quat_w, lin_vel_w=object_v, ang_vel_w=object_w, batch_size=objects.batch_size, device=device)
    return MotionDataset(mano, robot, objects, starts, lengths,
        hand_slots.to(device=device, dtype=torch.long),
        {s: list(raw[f'mano_{s}_joint_names']) for s in sides}, sides, target_fps, source_fps)
