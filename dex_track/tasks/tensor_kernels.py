"""Optional fusion of pure tensor math; simulator reads/writes stay with callers."""
import os
import torch
from dex_track.tasks.wrist_math import rotate, multiply, pd_wrench, limit_norm


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


def object_wrench(p, q, v, omega, com_p, target_p, target_q, target_v, target_w,
                  mass, weight, gains, gravity_compensation):
    force, torque = pd_wrench(p, q, v, omega, target_p, target_q, target_v, target_w, *gains)
    torque = torque - torch.linalg.cross(com_p - p, force)
    if gravity_compensation:
        force = force.clone()
        force[:, 2:3] += mass * 9.81
    return limit_norm(force, gains[-2]) * weight, limit_norm(torque, gains[-1]) * weight
