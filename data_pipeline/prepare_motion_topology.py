"""Offline topology.npz builder; does not launch physics, policy or training."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion-file', required=True)
    parser.add_argument('--fps', type=float, default=50.)
    parser.add_argument('--cache', required=True)
    args = parser.parse_args()
    import active_adaptation as aa
    aa.set_backend('mjlab')
    from dex_track.data import load_scene_catalog
    from dex_track.motion_storage import resolve_motion_path
    from dex_track.tasks.motion import load_motion_dataset
    from topology_builder import TopologyBuilder
    from dex_track.tasks.reference import topology_math as geometry
    from dex_track.assets import body_name
    from dex_track.topology_storage import SCHEMA, raw_motion_digest, validate_topology_payload
    path = resolve_motion_path(args.motion_file)
    manifest = json.loads(path.read_text())
    folders = [path.parent/row['path'] for row in manifest['motions']] if 'motions' in manifest else [path]
    catalog = load_scene_catalog(path); raw = catalog['raw']
    dataset = load_motion_dataset(raw, scene_state=catalog['state'], step_dt=1/args.fps,
        joint_order=list(range(raw['joint_pos'].shape[-1])), device='cpu')
    sides = ('right', 'left')
    names = [body_name(n, 'mjlab') for side in sides for n, _ in geometry.landmark_specs(side)]
    c = SimpleNamespace(dataset=dataset, hand_sides=sides, robot=SimpleNamespace(body_names=names),
        env=SimpleNamespace(backend='mjlab'), scene_objects=SimpleNamespace(catalog=catalog),
        palm_position_root={'right':[.008,.001,-.023], 'left':[.008,-.001,-.023]},
        device='cpu', laplacian_mode='on_the_fly', prepare_topology_dataset=True)
    topology = TopologyBuilder(c, args.cache)
    for mi, metadata in enumerate(folders):
        start, length = int(dataset.starts[mi]), int(dataset.lengths[mi])
        points = topology.object_points[mi].numpy()
        payload = dict(schema=np.asarray(SCHEMA), fps=np.asarray(dataset.fps),
            object_points=points, edge_bits=topology.edge_bits[start:start+length].numpy(),
            reference_sha256=np.asarray(raw_motion_digest(raw, mi)),
            points_sha256=np.asarray(hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()))
        validate_topology_payload(payload, topology.mano[start:start+length].numpy(),
            dataset.frames.object_pos_w[start:start+length].numpy(), dataset.frames.object_rotmat_w[start:start+length].numpy(), dataset.fps, raw_motion_digest(raw, mi))
        target = metadata.parent/'topology.npz'
        temporary = target.with_suffix('.partial.npz')
        np.savez_compressed(temporary, **payload); temporary.replace(target)
        print(json.dumps(dict(path=str(target), policy_frames=length, fps=dataset.fps, bytes=target.stat().st_size)), flush=True)
    # Invalidate cached loader results after attaching new sidecars.
    from dex_track.motion_storage import _load_motion_data_cached
    _load_motion_data_cached.cache_clear()


if __name__ == '__main__':
    main()
