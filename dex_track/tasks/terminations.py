"""Independent termination terms using AA's update/reset/compute lifecycle."""
import math
import torch

from active_adaptation.envs.mdp.terminations.base import Termination

from .command import MotionTracking
from .exceed import ConsecutiveExceed


TrackTermination = Termination[MotionTracking]


class object_tracking_error(TrackTermination, namespace='dex_track'):
    """Any valid object's top-three point error exceeds threshold consecutively."""

    def __init__(self, threshold=.2, min_steps=5, enabled=True, curriculum=None):
        super().__init__(enabled=enabled)
        if not math.isfinite(threshold) or threshold <= 0 or min_steps < 1 or int(min_steps) != min_steps:
            raise ValueError('Expected positive threshold and positive integer min_steps')
        self.threshold, self.min_steps = threshold, min_steps
        self.curriculum = dict(curriculum or {})
        if self.curriculum:
            final = self.curriculum['final_threshold']
            start, end = self.curriculum['start_fraction'], self.curriculum['end_fraction']
            if not (math.isfinite(final) and 0 < final <= threshold and 0 <= start < end <= 1):
                raise ValueError('Invalid object termination curriculum')

    def _initialize(self, env):
        super()._initialize(env)
        self.exceed = ConsecutiveExceed(
            (self.num_envs, self.command_manager.scene_objects.count),
            self.device, self.threshold, self.min_steps)

    def update(self):
        c = self.command_manager
        threshold = self.threshold
        if self.curriculum and self.env.training:
            total = getattr(self.env, 'total_iters', None) or self.env.cfg.total_iters
            progress = self.env.current_iter / total
            start, end = self.curriculum['start_fraction'], self.curriculum['end_fraction']
            alpha = min(1., max(0., (progress-start)/(end-start)))
            threshold += alpha*(self.curriculum['final_threshold']-self.threshold)
        self.exceed.threshold = threshold
        self.env.extra['curriculum/object_tracking_threshold'] = threshold
        self.exceed.update(c.object_tracking_error(), c.scene_objects.mask)

    def reset(self, env_ids, tensordict=None):
        self.exceed.reset(env_ids)

    def compute(self, termination):
        # Reading/logging a termination must never advance its counter.
        return (self.exceed.steps >= self.min_steps).any(-1, keepdim=True)


class robot_velocity_limit(TrackTermination, namespace='dex_track'):
    def __init__(self, max_lin_vel=100., max_ang_vel=200., max_joint_vel=200., enabled=True):
        super().__init__(enabled=enabled)
        self.max_lin_vel, self.max_ang_vel, self.max_joint_vel = max_lin_vel, max_ang_vel, max_joint_vel

    def compute(self, termination):
        d = self.command_manager.robot.data
        fast = d.body_link_lin_vel_w.norm(dim=-1).amax(-1) > self.max_lin_vel
        fast |= d.body_link_ang_vel_w.norm(dim=-1).amax(-1) > self.max_ang_vel
        fast |= d.joint_vel.abs().amax(-1) > self.max_joint_vel
        return fast.unsqueeze(-1)


class object_velocity_limit(TrackTermination, namespace='dex_track'):
    def __init__(self, max_lin_vel=100., max_ang_vel=200., enabled=True):
        super().__init__(enabled=enabled)
        self.max_lin_vel, self.max_ang_vel = max_lin_vel, max_ang_vel

    def compute(self, termination):
        objects = self.command_manager.scene_objects
        fast = objects.sim('lin_vel').norm(dim=-1) > self.max_lin_vel
        fast |= objects.sim('ang_vel').norm(dim=-1) > self.max_ang_vel
        return (fast & objects.mask).any(-1, keepdim=True)


class motion_end(TrackTermination, namespace='dex_track'):
    def __init__(self, enabled=True):
        super().__init__(is_timeout=True, enabled=enabled)

    def compute(self, termination):
        c = self.command_manager
        return (c.motion_steps >= c.dataset.lengths[c.motion_ids] - 1).unsqueeze(-1)
