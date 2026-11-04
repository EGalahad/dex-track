"""World-frame pose derivatives on each clip's policy-rate timeline."""
import torch
from ..wrist_math import inverse, multiply, log


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
