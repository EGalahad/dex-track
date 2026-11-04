"""Intent views over command-owned current/future tracking errors."""
import torch
from .common import TrackObservation
from ..reference.math import matrix_from_quat
from ..reference.errors import pose_error_in_anchor




class _future_reference(TrackObservation):
    def __init__(self,future_steps=(1,2,3,4,5,6,7,8,9,10)):
        super().__init__()
        if not future_steps or any(int(t)<1 for t in future_steps):
            raise ValueError('future_steps must be nonempty positive frame offsets')
        self.future_steps = tuple(map(int,future_steps))

    def _initialize(self,env):
        super()._initialize(env)
        self.offsets = self.command_manager.future_offsets(self.future_steps)


class mano_wrist_error_wrist(_future_reference, namespace='dex_track'):
    def __init__(self, future_steps=(1,2,3,4,5,6,7,8,9,10), rotation_rows=False):
        super().__init__(future_steps)
        self.rotation_rows = bool(rotation_rows)

    def compute(self):
        errors = self.command_manager.wrist_tracking_errors(self.offsets)
        r = errors.rotation_error_local
        rot6d = r[..., :2, :].flatten(-2) if self.rotation_rows else r[..., :2].flatten(-2)
        return torch.cat((errors.position_delta_local, rot6d), -1).flatten(1)




class mano_finger_pos_future_palm(_future_reference, namespace='dex_track'):
    """20 MANO finger xyz per hand in each corresponding FUTURE MANO palm.

    Metres, no bone/unit/hand-size normalization. Flatten order is future step,
    hand (right,left), finger (thumb,index,middle,ring,pinky), joint, xyz.
    Wrist world motion is provided separately by the wrist future observation.
    """
    def compute(self):
        return self.command_manager.topology.future_fingers(self.offsets).flatten(1)




def finger_reference_in_anchor(local_p, local_q, future_root, current_root):
    """Future retarget links in the CURRENT reference wrist-link frame.

    Mirrors mimic-lite ref_body_*_future_local (reference anchor, not robot
    feedback or a pose error). Each hand has its own full wrist SE(3) anchor.
    """
    from ..wrist_math import inverse, multiply, rotate
    anchor_inv = inverse(current_root[:, None, :, 3:7])
    relative_q = multiply(anchor_inv, future_root[..., 3:7])
    relative_p = rotate(anchor_inv, future_root[..., :3] - current_root[:, None, :, :3])
    q = relative_q[..., None, :].expand_as(local_q)
    return relative_p[..., None, :] + rotate(q, local_p), multiply(q, local_q)


class _future_finger_reference(_future_reference):
    def _initialize(self, env):
        super()._initialize(env)
        from ..reference.finger_bodies import finger_body_reference
        self.reference = finger_body_reference(self.command_manager)
        self.reference.prepare('retarget')
        if not hasattr(self.command_manager, 'free_root_states'):
            raise ValueError('Future finger references require retarget wrist root states (free-hand pair)')

    def poses(self):
        c = self.command_manager
        indices = c.dataset.frame_indices(c.motion_ids[:, None], c.motion_steps[:, None] + self.offsets)
        p, q = (value[indices] for value in self.reference.retarget)
        return finger_reference_in_anchor(p, q, c.free_root_states[indices], c.free_root_states[c.frame_indices])








class object_pose_error_wrist(_future_reference, namespace='dex_track'):
    """Direct future-minus-current position, unlike legacy spatial-transform translation."""
    def compute(self):
        return self.command_manager.object_pose_error_future(self.offsets).flatten(1)


class finger_body_error_future_local(_future_finger_reference, namespace='dex_track'):
    """Future retarget link errors in each CURRENT robot finger-link frame.

    12 links/hand; each link emits local displacement xyz and the first two
    rows of R_actual.T @ R_reference. Future offsets use the policy timestep.
    """
    def compute(self):
        return self.command_manager.finger_body_pose_error_future(self.offsets).flatten(1)
