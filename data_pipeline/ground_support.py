"""Geometry helpers and release admission for reviewed scene transformations.

These helpers do not modify a dataset. Caller supplies every collision vertex
in world metres over every native frame. Physical settling remains mandatory.
"""
import json
from pathlib import Path
import numpy as np


def common_object_lift(object_vertices):
    return max(0., -min(float(np.asarray(v)[..., 2].min()) for v in object_vertices))


def additional_hand_lift(hand_vertices, object_lift, allowed_depth=.005):
    minimum = min(float(np.asarray(v)[..., 2].min()) for v in hand_vertices)
    return max(0., -minimum-object_lift-allowed_depth)


def transform_points(points, translation):
    return np.asarray(points)+np.asarray(translation)


def local_contact_from_world(nearest_world, object_position, object_rotation):
    # Row-vector convention; contact_contact_pos_full is object-local.
    return np.einsum('...i,...ij->...j', np.asarray(nearest_world)-object_position, object_rotation)


def require_publishable(review_manifest):
    manifest=json.loads(Path(review_manifest).read_text())
    if manifest.get('publish_allowed') is not True:
        raise ValueError('Dataset is still staging; publication requires an approved manifest.')
    if manifest.get('quarantine_motion_count', 0):
        raise ValueError('Quarantined motions require an explicitly approved filtered release.')
    return manifest
