"""World-frame SE(3) velocity integration and wrench PD (quaternions are wxyz)."""
import torch


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
