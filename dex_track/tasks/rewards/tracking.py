"""Tracking rewards, including disabled diagnostic terms."""
import torch
from .common import TrackReward


class _exponential_tracking(TrackReward):
    def __init__(self, weight, sigma, enabled=True):
        super().__init__(weight, enabled)
        if sigma <= 0:
            raise ValueError('Reward sigma must be positive')
        self.sigma = sigma



class object_tracking(_exponential_tracking, namespace='dex_track'):
    def __init__(self, weight, sigma, enabled=True, top_k=3):
        super().__init__(weight, sigma, enabled)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 128:
            raise ValueError('Object reward top_k must be an integer in [1, 128]')
        self.top_k = top_k

    def _compute(self):
        return self.command_manager.scene_objects.reward(self.sigma, top_k=self.top_k).unsqueeze(-1)



class _object_velocity_tracking(_exponential_tracking):
    """World-frame vector L2 error; equal mean over valid scene objects."""
    def _compute(self):
        objects = self.command_manager.scene_objects
        error = (objects.sim(self.field)-objects.ref(self.field)).norm(dim=-1)
        value = torch.exp(-error/self.sigma).masked_fill(~objects.mask, 0.)
        return (value.sum(-1)/objects.mask.sum(-1).clamp_min(1)).unsqueeze(-1)



class object_linear_velocity_tracking(_object_velocity_tracking, namespace='dex_track'):
    field = 'lin_vel'



class object_angular_velocity_tracking(_object_velocity_tracking, namespace='dex_track'):
    field = 'ang_vel'



class tip_object_distance(_exponential_tracking, namespace='dex_track'):
    def _compute(self):
        c = self.command_manager
        reference_distance = c.ref_mano_tip_object_distance
        values = []
        for si, side in enumerate(c.hand_sides):
            sdf, _ = c.sdf_query(c.robot_tip_pos_w[:, si], side)
            values.append(torch.exp(-(sdf - reference_distance[:, si]).clamp_min(0) / self.sigma).mean(-1))
        return torch.stack(values, -1).mean(-1, keepdim=True)



class wrist_position_tracking(_exponential_tracking, namespace='dex_track'):
    def _compute(self):
        c = self.command_manager
        error = c.wrist_tracking_errors().position_error
        return torch.exp(-error / self.sigma).mean(-1, keepdim=True)



class wrist_orientation_tracking(_exponential_tracking, namespace='dex_track'):
    def _compute(self):
        c = self.command_manager
        error = c.wrist_tracking_errors().orientation_error
        return torch.exp(-error / self.sigma).mean(-1, keepdim=True)



class _velocity_tracking(_exponential_tracking):
    def _initialize(self, env):
        super()._initialize(env)
        self.command_manager.prepare_velocity_reference(self.kind)

    def _compute(self):
        errors = self.command_manager.velocity_tracking_errors(self.kind)
        error = getattr(errors, self.component + '_error')
        if self.kind == 'finger':
            error = error.mean(-1)  # mean link error within each hand
        return torch.exp(-error/self.sigma).mean(-1, keepdim=True)



class wrist_linear_velocity_tracking(_velocity_tracking, namespace='dex_track'):
    kind, component = 'wrist', 'linear'



class wrist_angular_velocity_tracking(_velocity_tracking, namespace='dex_track'):
    kind, component = 'wrist', 'angular'



class finger_linear_velocity_tracking(_velocity_tracking, namespace='dex_track'):
    kind, component = 'finger', 'linear'



class finger_angular_velocity_tracking(_velocity_tracking, namespace='dex_track'):
    kind, component = 'finger', 'angular'



class fingertip_tracking(_exponential_tracking, namespace='dex_track'):
    def _compute(self):
        c = self.command_manager
        error = c.mano_tip_tracking_errors().position_error
        return torch.exp(-error / self.sigma).mean(-1).mean(-1, keepdim=True)



class finger_body_tracking(_exponential_tracking, namespace='dex_track'):
    """Per-hand exp(-mean link error/sigma), then the mean of the two hands.

    Position is Euclidean distance in metres; orientation is geodesic angle
    in radians. Both refer to the actual/target hand's own wrist link frame.
    """
    def __init__(self, weight, sigma, reference, component, nominal=None, enabled=True):
        super().__init__(weight, sigma, enabled)
        if reference not in ('nominal', 'retarget') or component not in ('position', 'orientation'):
            raise ValueError((reference, component))
        self.reference, self.component = reference, component
        self.nominal = dict(nominal or {})

    def _initialize(self, env):
        super()._initialize(env)
        from ..reference.finger_bodies import finger_body_reference
        self.fingers = finger_body_reference(self.command_manager)
        self.reference_key = self.fingers.prepare(self.reference, self.nominal)

    def _compute(self):
        errors = self.fingers.errors(self.reference, self.reference_key)
        error = errors[0 if self.component == 'position' else 1]
        return torch.exp(-error.mean(-1)/self.sigma).mean(-1, keepdim=True)



class mano_bone_direction_tracking(_exponential_tracking, namespace='dex_track'):
    """Mean palm-local cosine error over 20 bones; no retarget target."""
    def _compute(self):
        error = self.command_manager.topology.bone_errors()
        return torch.exp(-error/self.sigma).mean(-1, keepdim=True)



class mano_object_laplacian_tracking(_exponential_tracking, namespace='dex_track'):
    """RMS residual in metres on the shared MANO/object reference graph."""
    def _compute(self):
        error = self.command_manager.topology.laplacian_errors()
        return torch.exp(-error/self.sigma).mean(-1, keepdim=True)
