"""Unweighted current-frame tracking diagnostics; never contribute to reward."""
from .common import TrackReward


class _metric(TrackReward):
    def __init__(self, weight=1., enabled=False, track_var=True):
        super().__init__(weight=weight, enabled=enabled, track_var=track_var)

    def object_mean(self, values):
        mask = self.command_manager.scene_objects.mask
        return (values.masked_fill(~mask, 0.).sum(-1) / mask.sum(-1).clamp_min(1)).unsqueeze(-1)


class object_point_error(_metric, namespace='dex_track'):
    """Mean across valid objects of each object's top-k scaled point error (m)."""
    def __init__(self, top_k=3, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 128:
            raise ValueError('top_k must be an integer in [1, 128]')
        self.top_k = top_k

    def _compute(self):
        return self.object_mean(self.command_manager.object_tracking_error(self.top_k))


class object_position_error(_metric, namespace='dex_track'):
    """Physical object-origin translation error (m), without point normalization."""
    def _compute(self):
        _, error, _ = self.command_manager.object_pose_tracking_errors()
        return self.object_mean(error)


class object_orientation_error(_metric, namespace='dex_track'):
    """Geodesic object orientation error (rad)."""
    def _compute(self):
        _, _, error = self.command_manager.object_pose_tracking_errors()
        return self.object_mean(error)


class wrist_position_error(_metric, namespace='dex_track'):
    def _compute(self):
        _, _, _, error, _ = self.command_manager.wrist_tracking_errors()
        return error.mean(-1, keepdim=True)


class wrist_orientation_error(_metric, namespace='dex_track'):
    def _compute(self):
        _, _, _, _, error = self.command_manager.wrist_tracking_errors()
        return error.mean(-1, keepdim=True)


class _finger_error(_metric):
    def _initialize(self, env):
        super()._initialize(env)
        from ..reference.finger_bodies import finger_body_reference
        finger_body_reference(self.command_manager).prepare('retarget')


class finger_position_error(_finger_error, namespace='dex_track'):
    """Mean over hands/12 links of wrist-relative retarget shape error (m)."""
    def _compute(self):
        return self.command_manager.finger_body_tracking_errors()[0].mean((-1,-2)).unsqueeze(-1)


class finger_orientation_error(_finger_error, namespace='dex_track'):
    """Mean wrist-relative geodesic orientation error over hands/12 links (rad)."""
    def _compute(self):
        return self.command_manager.finger_body_tracking_errors()[1].mean((-1,-2)).unsqueeze(-1)
