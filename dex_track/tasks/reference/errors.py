"""Shared pose-error math, independent of observation/reward consumers."""
import torch


def pose_error_in_anchor(actual_p, actual_r, target_p, target_r, anchor_r):
    """Position delta and rotation correction in anchor axes (first two rows)."""
    inv = anchor_r.transpose(-1, -2)
    dp = (inv @ (target_p-actual_p)[..., None]).squeeze(-1)
    dr = inv @ target_r @ actual_r.transpose(-1, -2) @ anchor_r
    return torch.cat((dp, dr[..., :2, :].flatten(-2)), -1)
