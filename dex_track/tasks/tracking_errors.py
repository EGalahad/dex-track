"""Command-owned tracking cache shared by observations, rewards and terminations.

Invalidated on simulator-state refresh and reference selection/advance. Offset
registrations are canonical tensors: no per-step device-to-host cache-key copy.
"""
from types import SimpleNamespace
import torch
from .reference.math import matrix_from_quat, quat_from_matrix, quat_apply_inverse
from .wrist_math import inverse, multiply, rotate, log
from .reference.errors import pose_error_in_anchor


class TrackingErrorsMixin:
    def prepare_velocity_reference(self, kind):
        """Immutable derivatives; shared by velocity rewards and diagnostics."""
        if not hasattr(self, '_velocity_references'):
            self._velocity_references = {}
        if kind not in self._velocity_references:
            from .reference.velocities import pose_velocities
            if kind == 'wrist':
                p = self.dataset.frames.mano_wrist_pos_w
                q = quat_from_matrix(self.dataset.frames.mano_wrist_rotmat_w)
            elif kind == 'finger':
                from .reference.finger_bodies import finger_body_reference
                p, q = finger_body_reference(self).world_trajectory()
            else:
                raise ValueError(kind)
            self._velocity_references[kind] = pose_velocities(
                p, q, self.dataset.starts, self.dataset.lengths, self.env.step_dt)
        return self._velocity_references[kind]

    def velocity_tracking_errors(self, kind):
        """World-frame MANO wrist or retarget finger-link velocity errors."""
        def compute():
            reference = self.prepare_velocity_reference(kind)
            target_v, target_w = (v[self.frame_indices] for v in reference)
            if kind == 'wrist':
                actual_v, actual_w = self.robot_palm_lin_vel_w, self.robot_palm_ang_vel_w
            else:
                from .reference.finger_bodies import finger_body_reference
                ids = [row[1:] for row in finger_body_reference(self).body_ids]
                d = self.robot.data
                actual_v = d.body_link_lin_vel_w[:, ids]
                actual_w = d.body_link_ang_vel_w[:, ids]
            return SimpleNamespace(linear_error=(actual_v-target_v).norm(dim=-1),
                                   angular_error=(actual_w-target_w).norm(dim=-1))
        return self._tracking_value(('velocity', kind), compute)

    def _invalidate_tracking(self):
        self._tracking_cache.clear()

    def _tracking_value(self, key, compute):
        if key not in self._tracking_cache:
            self._tracking_cache[key] = compute()
        return self._tracking_cache[key]

    def future_offsets(self, steps):
        key = tuple(map(int, steps))
        if key not in self._future_offsets:
            self._future_offsets[key] = torch.tensor(key, device=self.device)
        return self._future_offsets[key]

    def wrist_tracking_errors(self, offsets=None):
        def compute():
            p, q = self.robot_palm_pos_w, self.robot_palm_quat_w
            if offsets is None:
                target_p, target_q = self.ref_mano_wrist_pos_w, self.ref_mano_wrist_quat_w
            else:
                target_p = self.ref_mano_wrist_pos_w_future(offsets)
                target_q = quat_from_matrix(self.ref_mano_wrist_rotmat_w_future(offsets))
                p, q = p[:, None], q[:, None]
            delta = target_p-p
            error_q = multiply(inverse(q), target_q)
            return SimpleNamespace(
                position_delta_w=delta,
                position_delta_local=quat_apply_inverse(q.expand_as(target_q), delta),
                rotation_error_local=matrix_from_quat(error_q),
                position_error=delta.norm(dim=-1),
                orientation_error=log(error_q).norm(dim=-1))
        return self._tracking_value(('wrist', offsets), compute)

    def mano_tip_tracking_errors(self, offsets=None):
        def compute():
            actual = self.robot_tip_pos_w
            if offsets is None:
                target, q = self.ref_mano_tip_pos_w, self.robot_palm_quat_w[:, :, None]
            else:
                target = self.ref_mano_tip_pos_w_future(offsets)
                actual = actual[:, None]
                q = self.robot_palm_quat_w[:, None, :, None]
            delta = target-actual
            return SimpleNamespace(position_delta_w=delta,
                position_delta_local=quat_apply_inverse(q.expand(*delta.shape[:-1], 4), delta),
                position_error=delta.norm(dim=-1))
        return self._tracking_value(('mano_tip', offsets), compute)

    def finger_body_state(self):
        def compute():
            from .reference.finger_bodies import finger_body_reference, local_poses
            reference = finger_body_reference(self)
            d = self.robot.data
            p, q = d.body_link_pos_w[:, reference.body_ids], d.body_link_quat_w[:, reference.body_ids]
            lp, lq = local_poses(p, q)
            return SimpleNamespace(position_w=p[:, :, 1:], quaternion_w=q[:, :, 1:],
                                   position_local=lp, quaternion_local=lq)
        return self._tracking_value('finger_body_state', compute)

    def finger_body_tracking_errors(self, reference='retarget', key=None):
        """Historical reward metric: each hand's own wrist-local finger poses.

        Kept distinct from full world-reference pose errors: wrist displacement
        is already rewarded by wrist tracking and must not be added here.
        """
        def compute():
            from .reference.finger_bodies import finger_body_reference
            state = self.finger_body_state()
            p, q = finger_body_reference(self).target(reference, key)
            return ((state.position_local-p).norm(dim=-1),
                    log(multiply(state.quaternion_local, inverse(q))).norm(dim=-1))
        return self._tracking_value(('finger_body_tracking', reference, key), compute)

    def finger_body_pose_error_future(self, offsets):
        def compute():
            from .reference.finger_bodies import finger_body_reference
            reference = finger_body_reference(self)
            indices = self.dataset.frame_indices(self.motion_ids[:, None], self.motion_steps[:, None]+offsets)
            p, q = (v[indices] for v in reference.retarget)
            roots = self.free_root_states[indices]
            target_p = roots[..., None, :3]+rotate(roots[..., None, 3:7].expand_as(q), p)
            target_q = multiply(roots[..., None, 3:7].expand_as(q), q)
            state = self.finger_body_state()
            actual_p = state.position_w[:, None]-self.env.scene.env_origins[:, None, None, None]
            actual_r = matrix_from_quat(state.quaternion_w[:, None])
            return pose_error_in_anchor(actual_p, actual_r, target_p, matrix_from_quat(target_q), actual_r)
        return self._tracking_value(('finger_body_future', offsets), compute)

    def object_pose_error_future(self, offsets):
        def compute():
            return pose_error_in_anchor(self.object_pos_w[:, None], matrix_from_quat(self.object_quat_w)[:, None],
                self.ref_object_pos_w_future(offsets), self.ref_object_rotmat_w_future(offsets),
                matrix_from_quat(self.robot_wrist_quat_w)[:, None])
        return self._tracking_value(('object_pose_future', offsets), compute)


    def object_pose_tracking_errors(self):
        """All valid scene objects once, not duplicated per associated hand."""
        def compute():
            scene = self.scene_objects
            delta = scene.ref('pos')-scene.sim('pos')
            error_q = multiply(inverse(scene.sim('quat')), scene.ref('quat'))
            return SimpleNamespace(position_delta_w=delta,
                position_error=delta.norm(dim=-1), orientation_error=log(error_q).norm(dim=-1))
        return self._tracking_value('object_pose_current', compute)
