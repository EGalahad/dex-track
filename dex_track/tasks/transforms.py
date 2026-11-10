"""Shared pose transforms, tracking errors, and optional compiled tensor kernels."""
import os
import torch
from active_adaptation.utils.math import (
    matrix_from_quat, quat_from_matrix,
    quat_rotate as quat_apply, quat_rotate_inverse as quat_apply_inverse,
)


def multiply(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)


def inverse(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def rotate(q, v):
    t = 2 * torch.linalg.cross(q[..., 1:], v)
    return v + q[..., :1] * t + torch.linalg.cross(q[..., 1:], t)


def exp(rotvec):
    angle = rotvec.norm(dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle / 2), .5 * torch.sinc(angle / (2 * torch.pi)) * rotvec), -1)


def log(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    n = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(n, q[..., :1].clamp(min=0))
    return q[..., 1:] * torch.where(n > 1e-7, angle / n.clamp(min=1e-7), 2.0)


def integrate(p, q, v, omega, dt):
    return p + v * dt, torch.nn.functional.normalize(multiply(exp(omega * dt), q), dim=-1)


def limit_norm(x, limit):
    return x * (limit / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1)


def pd_wrench(p, q, v, omega, target_p, target_q, desired_v, desired_omega,
              kp_pos, kd_pos, kp_rot, kd_rot, max_force, max_torque):
    force = kp_pos * (target_p - p) + kd_pos * (desired_v - v)
    torque = kp_rot * log(multiply(target_q, inverse(q))) + kd_rot * (desired_omega - omega)
    return limit_norm(force, max_force), limit_norm(torque, max_torque)


def pose_error_in_anchor(actual_p, actual_r, target_p, target_r, anchor_r):
    """Position delta and rotation correction in anchor axes (first two rows)."""
    inv = anchor_r.transpose(-1, -2)
    dp = (inv @ (target_p-actual_p)[..., None]).squeeze(-1)
    dr = inv @ target_r @ actual_r.transpose(-1, -2) @ anchor_r
    return torch.cat((dp, dr[..., :2, :].flatten(-2)), -1)


def compute_tracking_errors(actual_p, actual_q, target_p, target_q):
    """Return world position delta, local quaternion error, distance and angle."""
    delta = target_p - actual_p
    error_q = multiply(inverse(actual_q), target_q)
    return delta, error_q, delta.norm(dim=-1), log(error_q).norm(dim=-1)


def pose_velocities(position, quaternion, starts, lengths, dt):
    """Centered differences internally, one-sided at each clip boundary.

    Quaternion increments use the shortest world-frame rotation, including
    across equivalent q/-q representations. Never difference separate clips.
    """
    frame = torch.arange(position.shape[0], device=position.device)
    lower = torch.repeat_interleave(starts, lengths)
    upper = torch.repeat_interleave(starts + lengths - 1, lengths)
    previous = torch.maximum(frame - 1, lower)
    following = torch.minimum(frame + 1, upper)
    elapsed = ((following - previous) * dt).clamp_min(dt)
    elapsed = elapsed.reshape(-1, *([1] * (position.ndim - 1)))
    linear = (position[following] - position[previous]) / elapsed
    angular = log(multiply(quaternion[following], inverse(quaternion[previous]))) / elapsed
    return linear, angular


def select_kernel(env, function):
    if not env.cfg.get('tensor_compile', False):
        return function
    from active_adaptation.utils.rollout_compile import compile_rollout_function
    compiled = compile_rollout_function(function)
    if os.environ.get('DEX_VERIFY_KERNELS') != '1':
        return compiled
    # Diagnostic only: compare identical real inputs once, outside timed training.
    checked = False
    def verified(*args):
        nonlocal checked
        result = compiled(*args)
        if not checked:
            expected = function(*args)
            for actual, reference in zip(result, expected):
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
            print(f'DEX_KERNEL_PARITY_OK {function.__name__}', flush=True)
            checked = True
        return result
    return verified


def site_kinematics(p, q, v, omega, site_p, site_q):
    offset = rotate(q, site_p)
    return p + offset, multiply(q, site_q), v + torch.linalg.cross(omega, offset), omega


def wrist_wrench(p, q, v, omega, com_p, target_p, target_q, desired_v, desired_w, gains):
    force, torque = pd_wrench(p, q, v, omega, target_p, target_q, desired_v, desired_w, *gains)
    return force, torque - torch.linalg.cross(com_p - p, force)
