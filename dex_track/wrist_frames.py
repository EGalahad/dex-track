"""Explicit URDF wrist roots and MANO-compatible semantic palm transforms."""
WRIST_BODY = {'right': 'R_forearm_rot_y_link', 'left': 'L_forearm_rot_y_link'}
# Historical v2 calibration only; v3 runtime reads dataset.robot.palm_in_root.
# MJCF root bodies map to URDF right_hand_link / left_hand_link.
PALM_QUAT_ROOT = {'right': (.5, .5, .5, -.5), 'left': (.5, -.5, .5, .5)}


# Ported from FlashDexRetarget 6f95e13 (MIT); see LICENSE.
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
