# Ported from FlashDexRetarget 6f95e13 (MIT); see LICENSE.
"""MANO reference and robot state properties for wrist and fingertip tracking."""

from __future__ import annotations

import torch
from .math import quat_from_matrix


# Canonical order of the 13 collision links used by scene observations.
def _alllink_names(prefix: str, side: str) -> list[str]:
    return [
        f"{prefix}_forearm_rot_y_link",  # palm
        f"{side}_hand_thumb_bend_link",
        f"{side}_hand_thumb_rota_link1",
        f"{side}_hand_thumb_rota_link2",
        f"{side}_hand_index_bend_link",
        f"{side}_hand_index_rota_link1",
        f"{side}_hand_index_rota_link2",
        f"{side}_hand_mid_link1",
        f"{side}_hand_mid_link2",
        f"{side}_hand_ring_link1",
        f"{side}_hand_ring_link2",
        f"{side}_hand_pinky_link1",
        f"{side}_hand_pinky_link2",
    ]


ALLLINK_BODY_NAMES_BY_SIDE = {
    "right": _alllink_names("R", "right"),
    "left": _alllink_names("L", "left"),
}


class HandPropertiesMixin:

    def _urdf_wrist_state(self):
        def compute():
            if getattr(self.robot, 'is_free_hand_pair', False):
                return self.robot.direct_wrist_state()
            d = self.robot.data
            ids = self.wrist_body_ids
            return (d.body_link_pos_w[:, ids], d.body_link_quat_w[:, ids],
                    d.body_link_lin_vel_w[:, ids], d.body_link_ang_vel_w[:, ids])
        return self._tracking_value('urdf_wrist_state', compute)

    def _semantic_palm_state(self):
        def compute():
            p, q, v, w = self._urdf_wrist_state()
            return self._site_kinematics(p, q, v, w,
                self.semantic_palm_p.expand_as(p), self.semantic_palm_q.expand_as(q))
        return self._tracking_value('semantic_palm_state', compute)

    @property
    def robot_wrist_pos_w(self):
        return self._urdf_wrist_state()[0]

    @property
    def robot_wrist_quat_w(self):
        return self._urdf_wrist_state()[1]

    @property
    def robot_wrist_lin_vel_w(self):
        return self._urdf_wrist_state()[2]

    @property
    def robot_wrist_ang_vel_w(self):
        return self._urdf_wrist_state()[3]

    @property
    def robot_palm_pos_w(self):
        return self._semantic_palm_state()[0]

    @property
    def robot_palm_quat_w(self):
        return self._semantic_palm_state()[1]

    @property
    def robot_palm_lin_vel_w(self):
        return self._semantic_palm_state()[2]

    @property
    def robot_palm_ang_vel_w(self):
        return self._semantic_palm_state()[3]

    @property
    def robot_tip_pos_w(self) -> torch.Tensor:
        """Robot fingertip site positions (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self.hand_sides:
            ids = self._tip_site_ids[side]
            parts.append(self._sites.site_pos_w[:, ids])
        return torch.stack(parts, dim=1)

    @property
    def ref_mano_wrist_pos_w(self):
        return self.current_ref_motion.mano_wrist_pos_w + self.env.scene.env_origins[:, None]

    @property
    def ref_mano_wrist_quat_w(self):
        return quat_from_matrix(self.current_ref_motion.mano_wrist_rotmat_w)

    @property
    def ref_mano_tip_pos_w(self):
        return self._mano_tip_pos_w(self.current_ref_motion)

    def _mano_tip_pos_w(self, motion):
        return torch.stack([motion.mano_joint_pos_w[:, si, self._tip_mano_ids[side]]
                            + self.env.scene.env_origins[:, None]
                            for si, side in enumerate(self.hand_sides)], 1)

    def ref_mano_tip_pos_w_future(self, offsets):
        motion = self.ref_motion_future(offsets)
        return torch.stack([motion.mano_joint_pos_w[:, :, si, self._tip_mano_ids[side]]
                            + self.env.scene.env_origins[:, None, None]
                            for si, side in enumerate(self.hand_sides)], 2)

    def ref_mano_wrist_pos_w_future(self, offsets):
        return self.ref_motion_future(offsets).mano_wrist_pos_w + self.env.scene.env_origins[:, None, None]

    def ref_mano_wrist_rotmat_w_future(self, offsets):
        return self.ref_motion_future(offsets).mano_wrist_rotmat_w

    @property
    def ref_mano_tip_object_distance(self):
        return torch.stack([self.sdf_query(self.ref_mano_tip_pos_w[:, si], side,
            object_pos_w=self.ref_object_pos_w[:, si], object_quat_w=self.ref_object_quat_w[:, si])[0]
            for si, side in enumerate(self.hand_sides)], 1)
