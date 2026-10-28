"""Human references and object spatial corrections in the current wrist anchor."""
import torch
from .common import TrackObservation
from .helpers import _future_mano_traj_parts
from ..reference.math import matrix_from_quat


def spatial_error_in_wrist(current_p,current_r,future_p,future_r,wrist_p,wrist_r):
    """W^-1 T_future T_current^-1 W; invariant to the object's local frame."""
    inverse_wrist = wrist_r.transpose(-1,-2)
    actual_p = (inverse_wrist @ (current_p-wrist_p)[...,None]).squeeze(-1)
    actual_r = inverse_wrist @ current_r
    target_p = (inverse_wrist[:,None] @ (future_p-wrist_p[:,None])[...,None]).squeeze(-1)
    target_r = inverse_wrist[:,None] @ future_r
    rotation = target_r @ actual_r[:,None].transpose(-1,-2)
    translation = target_p-(rotation @ actual_p[:,None,...,None]).squeeze(-1)
    # Match HDMI spatial observations: first two rows, flattened.
    return torch.cat((translation,rotation[...,:2,:].flatten(-2)),-1)


class _future_reference(TrackObservation):
    def __init__(self,future_steps=(1,2,3,4,5,6,7,8,9,10)):
        super().__init__()
        if not future_steps or any(int(t)<1 for t in future_steps):
            raise ValueError('future_steps must be nonempty positive frame offsets')
        self.future_steps = tuple(map(int,future_steps))

    def _initialize(self,env):
        super()._initialize(env)
        self.offsets = torch.tensor(self.future_steps,device=self.device)


class mano_wrist_error_wrist(_future_reference, namespace='dex_track'):
    def compute(self):
        parts = _future_mano_traj_parts(self.command_manager,self.offsets)
        return torch.cat((parts['mano_wrist_pos_wrist'],parts['mano_wrist_rot6d_wrist']),-1).flatten(1)


class mano_tip_error_wrist(_future_reference, namespace='dex_track'):
    """Future reference minus current robot fingertip positions, in current wrist axes."""
    def compute(self):
        return _future_mano_traj_parts(self.command_manager,self.offsets)['mano_tip_pos_wrist'].flatten(1)


class object_spatial_error_wrist(_future_reference, namespace='dex_track'):
    def compute(self):
        c = self.command_manager
        return spatial_error_in_wrist(c.object_pos_w,matrix_from_quat(c.object_quat_w),
            c.ref_object_pos_w_future(self.offsets),c.ref_object_rotmat_w_future(self.offsets),
            c.robot_wrist_pos_w,matrix_from_quat(c.robot_wrist_quat_w)).flatten(1)
