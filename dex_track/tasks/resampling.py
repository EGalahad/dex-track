"""Per-clip linear/shortest-arc SLERP resampling, preserving native frame timing.

Extracted from the validated EGalahad/any4hdmi implementation; no runtime dependency
on another development checkout. Quaternion columns are explicit.
"""
from __future__ import annotations
import math
import torch

def resampled_length(
    length: int,
    *,
    source_fps: float,
    target_fps: float,
) -> int:
    if length <= 0:
        raise ValueError(f"Expected positive sequence length, got {length}")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(
            f"fps must be positive, got source_fps={source_fps}, target_fps={target_fps}"
        )
    if length == 1 or source_fps == target_fps:
        return int(length)
    duration = (length - 1) / float(source_fps)
    return max(1, int(math.floor(duration * float(target_fps) + 1e-9)) + 1)


def _packed_interp_plan_torch(
    clip_lengths: list[int],
    *,
    source_fps: float,
    target_fps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    clip_lengths_t = torch.as_tensor(clip_lengths, device=device, dtype=torch.long)
    output_lengths = [
        resampled_length(
            int(length),
            source_fps=source_fps,
            target_fps=target_fps,
        )
        for length in clip_lengths
    ]
    output_lengths_t = torch.as_tensor(output_lengths, device=device, dtype=torch.long)
    total_output = int(output_lengths_t.sum().item())

    input_starts = torch.cumsum(
        torch.cat([clip_lengths_t.new_zeros(1), clip_lengths_t[:-1]]),
        dim=0,
    )
    output_starts = torch.cumsum(
        torch.cat([output_lengths_t.new_zeros(1), output_lengths_t[:-1]]),
        dim=0,
    )

    clip_ids = torch.repeat_interleave(
        torch.arange(len(clip_lengths), device=device, dtype=torch.long),
        output_lengths_t,
    )
    output_indices = torch.arange(total_output, device=device, dtype=torch.long)
    local_output_idx = output_indices - output_starts.index_select(0, clip_ids)

    source_pos = local_output_idx.to(dtype=dtype) * (
        float(source_fps) / float(target_fps)
    )
    single_frame_mask = clip_lengths_t.index_select(0, clip_ids) <= 1
    max_right = (clip_lengths_t.index_select(0, clip_ids) - 1).clamp_min(0)

    right_local = torch.ceil(source_pos).to(dtype=torch.long)
    right_local = torch.where(
        single_frame_mask, torch.zeros_like(right_local), right_local.clamp_min(1)
    )
    right_local = torch.minimum(right_local, max_right)
    left_local = torch.where(
        single_frame_mask, torch.zeros_like(right_local), right_local - 1
    )

    alpha = torch.where(
        single_frame_mask,
        torch.zeros_like(source_pos),
        source_pos - left_local.to(dtype=dtype),
    )

    global_left = input_starts.index_select(0, clip_ids) + left_local
    global_right = input_starts.index_select(0, clip_ids) + right_local
    return output_lengths, global_left, global_right, alpha


def _packed_slerp_by_indices_torch(
    quat: torch.Tensor,
    *,
    global_left: torch.Tensor,
    global_right: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    batch_shape = quat.shape[1:-1]
    quat_dim = quat.shape[-1]
    flat = quat.reshape(quat.shape[0], -1, quat_dim)
    q0 = flat[global_left]
    q1 = flat[global_right]

    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    flip_mask = dot < 0.0
    q1 = torch.where(flip_mask, -q1, q1)
    dot = torch.where(flip_mask, -dot, dot).clamp(-1.0, 1.0)

    alpha_quat = alpha.view(-1, 1, 1)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * alpha_quat

    safe_denom = torch.where(
        sin_theta_0 > 1e-8, sin_theta_0, torch.ones_like(sin_theta_0)
    )
    s0 = torch.sin(theta_0 - theta) / safe_denom
    s1 = torch.sin(theta) / safe_denom
    slerp_out = s0 * q0 + s1 * q1

    nlerp_out = (1.0 - alpha_quat) * q0 + alpha_quat * q1
    out = torch.where(dot > 0.9995, nlerp_out, slerp_out)
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out.reshape(alpha.shape[0], *batch_shape, quat_dim)


def resample_packed(
    qpos_list: list[torch.Tensor],
    qvel_list: list[torch.Tensor],
    *,
    source_fps: float,
    target_fps: float,
    quaternion_slices: tuple[tuple[int, int], ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if len(qpos_list) != len(qvel_list):
        raise ValueError(
            f"Expected qpos/qvel list lengths to match, got {len(qpos_list)} and {len(qvel_list)}"
        )
    if not qpos_list:
        raise ValueError("Expected at least one qpos/qvel clip")

    prepared_qpos: list[torch.Tensor] = []
    prepared_qvel: list[torch.Tensor] = []
    clip_lengths: list[int] = []
    device = qpos_list[0].device

    for qpos, qvel in zip(qpos_list, qvel_list, strict=True):
        if qpos.ndim != 2:
            raise ValueError(
                f"Expected qpos to be rank 2, got shape {tuple(qpos.shape)}"
            )
        if qvel.ndim != 2:
            raise ValueError(
                f"Expected qvel to be rank 2, got shape {tuple(qvel.shape)}"
            )
        if qpos.device != device or qvel.device != device:
            raise ValueError(
                "Expected all qpos/qvel tensors in a batch to share the same device"
            )
        if qpos.shape[0] != qvel.shape[0]:
            raise ValueError(
                f"Expected qpos/qvel frame counts to match, got {tuple(qpos.shape)} and {tuple(qvel.shape)}"
            )
        prepared_qpos.append(qpos.to(dtype=torch.float32).contiguous())
        prepared_qvel.append(qvel.to(dtype=torch.float32).contiguous())
        clip_lengths.append(int(qpos.shape[0]))

    if source_fps == target_fps or all(length <= 1 for length in clip_lengths):
        return (
            torch.cat(prepared_qpos, dim=0),
            torch.cat(prepared_qvel, dim=0),
            clip_lengths,
        )

    packed_qpos = torch.cat(prepared_qpos, dim=0)
    packed_qvel = torch.cat(prepared_qvel, dim=0)
    output_lengths, global_left, global_right, alpha = _packed_interp_plan_torch(
        clip_lengths,
        source_fps=source_fps,
        target_fps=target_fps,
        device=device,
        dtype=packed_qpos.dtype,
    )

    def _packed_lerp(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        x0 = flat[global_left]
        x1 = flat[global_right]
        out = (1.0 - alpha.unsqueeze(1)) * x0 + alpha.unsqueeze(1) * x1
        return out.reshape(alpha.shape[0], *x.shape[1:]).contiguous()

    packed_qpos_interp = _packed_lerp(packed_qpos)
    if quaternion_slices is None:
        quaternion_slices = ((3, 7),) if packed_qpos.shape[1] >= 7 else ()
    for start, stop in quaternion_slices:
        if stop - start != 4 or start < 0 or stop > packed_qpos.shape[1]:
            raise ValueError(
                f"Invalid quaternion qpos slice {(start, stop)} for width {packed_qpos.shape[1]}"
            )
        packed_qpos_interp[:, start:stop] = _packed_slerp_by_indices_torch(
            packed_qpos[:, start:stop],
            global_left=global_left,
            global_right=global_right,
            alpha=alpha,
        )

    packed_qvel_interp = _packed_lerp(packed_qvel)
    return packed_qpos_interp, packed_qvel_interp, output_lengths
