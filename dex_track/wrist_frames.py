"""Explicit URDF wrist roots and MANO-compatible semantic palm transforms."""
WRIST_BODY = {'right': 'R_forearm_rot_y_link', 'left': 'L_forearm_rot_y_link'}
# MJCF root bodies map to URDF right_hand_link / left_hand_link.
PALM_QUAT_ROOT = {'right': (.5, .5, .5, -.5), 'left': (.5, -.5, .5, .5)}
