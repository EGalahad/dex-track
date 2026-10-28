"""MANO reference errors shared by intent terms."""
import torch
from ..command import MotionTracking
from ..reference.math import matrix_from_quat, quat_apply_inverse


def _future_mano_traj_parts(command: MotionTracking, offsets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Future reference minus current robot, in wrist axes; each value is (B, K, S, width)."""
    fut_wt = command.ref_mano_wrist_pos_w_future(offsets)
    R_fut = command.ref_mano_wrist_rotmat_w_future(offsets)
    fut_tip = command.ref_mano_tip_pos_w_future(offsets)
    B, K, S = (fut_wt.shape[0], fut_wt.shape[1], fut_wt.shape[2])
    cur_wt = command.robot_wrist_pos_w[:, None]
    cur_wq = command.robot_wrist_quat_w[:, None].expand(B, K, S, 4)
    cur_tip = command.robot_tip_pos_w[:, None]
    R_cur = matrix_from_quat(cur_wq)
    d_wt = quat_apply_inverse(cur_wq, fut_wt - cur_wt)
    R_d = R_cur.transpose(-1, -2) @ R_fut
    d_r6 = R_d[..., :2].reshape(B, K, S, 6)
    tip_q = cur_wq[:, :, :, None, :].expand(B, K, S, 5, 4)
    d_tip = quat_apply_inverse(tip_q, fut_tip - cur_tip)
    return {'mano_wrist_pos_wrist': d_wt, 'mano_wrist_rot6d_wrist': d_r6, 'mano_tip_pos_wrist': d_tip.reshape(B, K, S, 15)}
