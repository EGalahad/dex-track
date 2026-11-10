"""Per-motion reference topology payload, aligned to an explicit policy timeline."""
import hashlib
import numpy as np

SCHEMA = 'dex-track-topology-v2'


def raw_motion_digest(raw, index):
    start = int(raw['length_starts'][index]); length = int(raw['motion_num_frames'][index])
    h = hashlib.sha256()
    h.update(str(float(raw['meta']['fps'])).encode())
    for side in ('right', 'left'):
        h.update(str(raw[f'mano_{side}_joint_names']).encode())
        for key in (f'mano_{side}_wrist_pos', f'mano_{side}_wrist_rot', f'mano_{side}_joints'):
            value = np.ascontiguousarray(raw[key][start:start+length].cpu().numpy())
            h.update(str(value.shape).encode()); h.update(value.tobytes())
        slot = int(raw[f'motion_object_slot_{side}'][index])
        for key in ('scene_object_pos','scene_object_quat'):
            value = np.ascontiguousarray(raw[key][start:start+length,slot].cpu().numpy())
            h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def validate_topology_payload(payload, mano, object_pos, object_rot, fps, source_digest):
    if str(payload['schema']) != SCHEMA or float(payload['fps']) != float(fps):
        raise ValueError('Topology schema/policy FPS mismatch; regenerate topology.npz')
    bits, points = payload['edge_bits'], payload['object_points']
    if bits.dtype != np.uint8 or bits.shape != (len(mano), 2, 71, 9):
        raise ValueError('Expected uint8 topology [policy_frames,2,71,9]')
    if points.shape != (2, 50, 3) or not np.isfinite(points).all():
        raise ValueError('Expected finite fixed object-local samples [2,50,3]')
    if str(payload['reference_sha256']) != source_digest:
        raise ValueError('Stale topology: original MANO/object reference changed')
    if str(payload['points_sha256']) != hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest():
        raise ValueError('Topology object sample checksum mismatch')
    adjacency = np.unpackbits(bits, axis=-1, bitorder='little')
    if adjacency[..., 71:].any():
        raise ValueError('Topology padding bits must be zero')
    adjacency = adjacency[..., :71]
    if not np.array_equal(adjacency, adjacency.swapaxes(-1, -2)) or np.diagonal(adjacency, axis1=-2, axis2=-1).any():
        raise ValueError('Expected symmetric topology without self edges')
    if not adjacency.any(axis=-1).all():
        raise ValueError('Topology has isolated vertices')
