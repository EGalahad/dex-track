"""Rebuild reference-object SDF caches from a portable v3 dataset (no rollout)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Dataset root, index.json, or segment metadata.json')
    parser.add_argument('--object', action='append', default=[], help='Optional object ID; repeat to select several')
    parser.add_argument('--device', default='cpu', help='Warp bake device, e.g. cpu or cuda:0')
    parser.add_argument('--resolution', type=int, default=128)
    args = parser.parse_args()
    if args.resolution < 2:
        parser.error('--resolution must be at least 2')
    entry = args.dataset.resolve()
    if entry.is_dir():
        entry /= 'index.json'
    data = json.loads(entry.read_text())
    metadata = [entry.parent / row['path'] for row in data['motions']] if 'motions' in data else [entry]
    jobs = set()
    for path in metadata:
        motion = json.loads(path.read_text())
        if motion.get('schema') != 'dex-track-motion-folders-v3':
            raise ValueError('Expected pose-only v3; convert with data_pipeline/convert_pose_only_v3.py')
        with np.load(path.parent/'objects.npz',allow_pickle=False) as arrays:
            for side in ('right','left'):
                slot = int(arrays[f'motion_object_slot_{side}'][0])
                oid = motion['motion_scene_object_ids'][0][slot]
                if motion['motion_segments'][0]['hand_objects'][side] != oid:
                    raise ValueError('Hand assignment differs from scene slot')
                scale = motion['motion_scene_object_scales'][0][slot]
                obj = (path.parent/'objects'/oid).resolve()
                if not args.object or obj.name in args.object:
                    jobs.add((str(obj/'convex'),float(scale)))
    if not jobs:
        parser.error('No assigned objects matched the selection')
    # The reference package is import-safe; do not initialize AA or a simulator.
    project = Path(__file__).resolve().parents[1]
    source = project / 'dex_track/tasks'
    sys.path.insert(0, str(source if source.is_dir() else Path(__file__).parent))
    from reference.sdf import bake_object_sdf_grid, object_sdf_extent, _cache_path
    for mesh, scale in sorted(jobs):
        extent = object_sdf_extent(mesh, scale)
        grid = bake_object_sdf_grid(mesh, scale, extent, args.resolution, 'cpu', bake_device=args.device)
        print(f'{_cache_path(mesh, extent, args.resolution, scale)} shape={tuple(grid.shape)}', flush=True)
    print(f'Prepared {len(jobs)} object/scale grids. No trajectories or meshes changed.')


if __name__ == '__main__':
    main()
