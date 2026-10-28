"""Build an auditable human-reference view for full-corpus rigid tracking.

This is a preparation stage, not training admission. Keep every source decision,
then run collision generation, reset-only retargeting and physical QA separately.
All scene poses travel inside each clip, so upstream retarget start cropping also
crops the scene; scene trajectories must never be reconstructed by frame count.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import source_root
SOURCE = source_root()
sys.path.insert(0, str(SOURCE / 'src'))
from retarget.human_demo import clips as C
from retarget.human_demo.sources.base import (
    Demo, Hand, YUP_TO_ZUP, _forward, _mano, _rotate, _to_clip, _write,
)
from retarget.human_demo.sources.oakink2 import Source as OakSource


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def scene_arrays(clip, ids, poses, scales):
    ids = sorted(set(ids))
    if not 1 <= len(ids) <= 8:
        raise ValueError(f'Complete scene has {len(ids)} objects (allowed 1..8)')
    value = np.stack([poses[o] for o in ids], axis=1).astype(np.float32)
    if len(value) != C.num_frames(clip) or not np.isfinite(value).all():
        raise ValueError('Scene timeline must match the human reference exactly')
    for side in ('right', 'left'):
        oid = C.obj_id(clip, side)
        if oid not in ids or not np.allclose(value[:, ids.index(oid)], C.obj_pose(clip, side), atol=1e-6):
            raise ValueError(f'{side}: held and scene object trajectories differ')
    clip.update(scene_object_ids=np.array(ids), scene_object_poses=value,
                scene_object_scales=np.array([scales.get(o, 1.) for o in ids], np.float32))
    return clip


def hand_from_compact(data, prefix, ix):
    q = data[f'{prefix}__pose_coeffs'][ix].astype(np.float64)
    if not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-3):
        raise ValueError('Invalid MANO quaternion')
    aa = Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(len(ix), 16, 3)
    return Hand(aa[:, 0], aa[:, 1:].reshape(len(ix), 45),
                data[f'{prefix}__betas'][ix].astype(np.float64),
                wrist=data[f'{prefix}__tsl'][ix].astype(np.float64))


def associate(source, hands, poses, annotation):
    """One fixed object per hand: MANO-link contact counts, then annotation/distance.

    Restrict association to the audited complete scene, never choose a background
    object that was omitted from the physical scene. Record the fallback explicitly.
    """
    objects, evidence = {}, {}
    for side, pre in (('right', 'rh'), ('left', 'lh')):
        model = _mano(source.mano_dir, side, tuple(sorted(source.mano.items())))
        _, verts = _forward(model, hands[side])
        link = model.lbs_weights.numpy().argmax(1)
        counts, distances = {}, {}
        for oid, pose in poses.items():
            local = (verts - pose[:, None, :3, 3]) @ pose[:, :3, :3]
            near = source.near(oid, local.reshape(-1, 3), source.cfg.contact).reshape(verts.shape[:2])
            counts[oid] = sum(int((near[:, link == k] < source.cfg.contact).any(1).sum()) for k in range(16))
            # Fallback still uses surface distance, rather than object-origin distance.
            wrist_local = np.einsum('tji,tj->ti', pose[:, :3, :3], hands[side].wrist - pose[:, :3, 3])
            distances[oid] = float(source.near(oid, wrist_local, float('inf')).mean())
        annotated = [o for o in annotation.get(f'obj_list_{pre}', []) or [] if o in poses]
        if len(annotated) > 1:
            raise ValueError(f'{side}: multiple annotated hand objects need association switching')
        best = max(counts, key=counts.get)
        method = 'MANO_16_link_contact_frame_count'
        if counts[best] == 0:
            best = annotated[0] if annotated else min(distances, key=distances.get)
            method = 'annotation_no_contact' if annotated else 'nearest_mean_wrist_surface_no_contact'
        objects[side] = best
        evidence[side] = dict(object_id=best, method=method, contact_counts=counts,
                              mean_wrist_surface_distance=distances)
    return objects, evidence


def oak_unit(job):
    compact, raw, output, mano, records = job
    torch.set_num_threads(1)
    output = Path(output)
    cfg = OmegaConf.load(SOURCE / 'config/retarget/import_human_demo.yaml')
    source_cfg = OmegaConf.load(SOURCE / 'config/retarget/source/oakink2.yaml')
    cfg.mano_dir = str(mano)
    source = OakSource(Path(raw), source_cfg, str(mano), cfg.mano_tip_vertex)
    seq = records[0]['sequence']
    compact_file = Path(compact) / f'{seq}.npz'
    with np.load(compact_file, allow_pickle=False) as archive:
        data = dict(archive)
    frame_ids = data['frame_ids']
    fps = float(data['fps'])
    if fps != 120. or np.any(np.diff(frame_ids) != 1):
        raise ValueError(f'{seq}: source timeline is not contiguous 120 Hz')
    by_object = {str(o): data['object_pose'][:, i].astype(np.float64)
                 for i, o in enumerate(data['object_ids'])}
    # Table fitting uses the whole recording, as in the official importer.
    context = {o: p for o, p in by_object.items() if source._mesh_path(o).exists()}
    rows, shared = [], {}
    for record in records:
        name = record['stage_id']
        target = output / 'human_demo' / f'{name}.npz'
        result_file = output / 'clip-records' / f'{name}.json'
        if target.is_file() and result_file.is_file():
            cached = json.loads(result_file.read_text())
            if cached['status'] == 'human_reference_ready' and digest(target) == cached.get('sha256'):
                rows.append(cached)
                continue
        row = dict(dataset='oakink2', clip=name, status='error',
                   source=str(compact_file), source_stage=record, training_admission=False)
        try:
            start, end = record['frame_interval_half_open']
            ix = np.flatnonzero((frame_ids >= start) & (frame_ids < end))
            if len(ix) < 2:
                raise ValueError('Stage has fewer than two valid source frames')
            required = sorted(set(record['required_object_ids']))
            if not 1 <= len(required) <= 8:
                raise ValueError('Required scene object count outside 1..8')
            poses = {o: by_object[o][ix] for o in required}
            hands = {s: hand_from_compact(data, pre, ix) for s, pre in (('right', 'rh'), ('left', 'lh'))}
            objects, evidence = associate(source, hands, poses, record['source_annotation'])
            demo = Demo(name, fps, hands, objects, poses, to_world=YUP_TO_ZUP,
                        scales=source.scales, context=context)
            result = _write(source, demo, cfg, output / 'human_demo', shared)
            if 'skip' in result:
                raise ValueError(result['skip'])
            clip = C.load_clip(target)
            world = {o: _to_clip(_rotate(p, YUP_TO_ZUP), 0., fps, cfg.fps) for o, p in poses.items()}
            scene_arrays(clip, required, world, source.scales)
            C.save_clip(target, clip)
            row.update(status='human_reference_ready', fps=float(cfg.fps), frames=C.num_frames(clip),
                       required_objects=required, hand_assignment=evidence,
                       raw_frame_interval=[int(frame_ids[ix[0]]), int(frame_ids[ix[-1]]) + 1],
                       objects_for_mesh=sorted(set(required) | set(context)),
                       sha256=digest(target))
        except Exception as e:
            row['error'] = repr(e)
        write_json(result_file, row)
        rows.append(row)
    return rows


def prepare_oak(args):
    source_manifest = json.loads(args.stage_manifest.read_text())
    decisions = []
    groups = defaultdict(list)
    for r in source_manifest['stages']:
        status = r['simulation_eligibility']['status']
        decisions.append(dict(clip=r['stage_id'], status=status, eligibility=r['simulation_eligibility']))
        if status == 'candidate_rigid_pose_tracking':
            groups[r['sequence']].append(r)
    write_json(args.output / 'selection.json', dict(dataset='oakink2',
        source_manifest=str(args.stage_manifest), source_sha256=digest(args.stage_manifest),
        decisions=decisions, counts=dict(Counter(r['status'] for r in decisions)),
        training_admission=False))
    jobs = [(str(args.compact), str(args.raw), str(args.output), str(args.mano), rows)
            for rows in groups.values()]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(oak_unit, job): job[-1] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rows.extend(future.result())
            except Exception as e:
                rows.extend(dict(sequence=r['sequence'], clip=r['stage_id'], status='error',
                    source_stage=r, error=repr(e), training_admission=False) for r in futures[future])
            write_json(args.output / 'progress.json', dict(units_done=i, units_total=len(jobs),
                clips_done=len(rows), counts=dict(Counter(r['status'] for r in rows))))
            print(f'OakInk2 {i}/{len(jobs)} recordings, {len(rows)} stages', flush=True)
    source_cfg = OmegaConf.load(SOURCE / 'config/retarget/source/oakink2.yaml')
    source = OakSource(args.raw, source_cfg)
    needed = sorted({oid for r in rows for oid in r.get('objects_for_mesh', [])})
    for oid in needed:
        target = args.output / 'objects' / oid / 'visual.obj'
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source.mesh(oid).export(target)
    write_json(args.output / 'manifest.json', dict(schema='dex-track-rigid-human-view-v1',
        dataset='oakink2', preparation_attempts_finished=True,
        preparation_complete=all(r['status']=='human_reference_ready' for r in rows), training_admission=False,
        source_units=len(groups), source_stages=len(decisions), clips=rows,
        counts=dict(Counter(r['status'] for r in rows)),
        pending=['collision generation', 'reset-only retarget', 'replay/physical QA', 'final full-corpus pack']))


TACO_UNSUPPORTED = {
    'cut': 'cutting/material removal', 'scrape_off': 'surface material removal',
    'smear': 'deposition/deformation', 'screw': 'thread constraint',
}
TACO_REVIEW_TOOLS = {'brush', 'roller', 'glue_gun', 'soap'}


def prepare_taco(args):
    """Reuse the complete existing import, with its tool AND target trajectories."""
    rows = []
    for path in sorted((args.prepared / 'human_demo').glob('*.npz')):
        parts = path.stem.removeprefix('taco_').split('__')
        action, tool = parts[:2]
        row = dict(dataset='taco', clip=path.stem, source=str(path), action=action,
                   tool=tool, training_admission=False)
        if action in TACO_UNSUPPORTED:
            row.update(status='exclude', reason=TACO_UNSUPPORTED[action])
        elif tool in TACO_REVIEW_TOOLS:
            row.update(status='review', reason='deformable contact or articulated tool requires asset/interaction review')
        else:
            try:
                clip = C.load_clip(path)
                poses = {C.obj_id(clip, side): C.obj_pose(clip, side) for side in ('right', 'left')}
                scales = {C.obj_id(clip, side): C.obj_scale(clip, side) for side in ('right', 'left')}
                scene_arrays(clip, poses, poses, scales)
                target = args.output / 'human_demo' / path.name
                C.save_clip(target, clip)
                for oid in poses:
                    link = args.output / 'objects' / oid
                    if not link.exists():
                        link.symlink_to((args.prepared / 'objects' / oid).resolve(), target_is_directory=True)
                row.update(status='human_reference_ready', frames=C.num_frames(clip), fps=float(clip['fps']),
                           required_objects=sorted(poses), sha256=digest(target),
                           goal='rigid tool/target pose tracking; contents and surface effects are not simulated')
            except Exception as e:
                row.update(status='error', error=repr(e))
        rows.append(row)
    write_json(args.output / 'manifest.json', dict(schema='dex-track-rigid-human-view-v1', dataset='taco',
        preparation_complete=True, training_admission=False, source_units=len(rows), clips=rows,
        counts=dict(Counter(r['status'] for r in rows)),
        pending=['review flagged tools', 'collision generation', 'reset-only retarget', 'physical QA', 'final full-corpus pack']))
    print(Counter(r['status'] for r in rows), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    sub = p.add_subparsers(dest='dataset', required=True)
    oak = sub.add_parser('oakink2')
    for arg in ('raw', 'compact', 'stage-manifest', 'mano'):
        oak.add_argument('--' + arg, type=Path, required=True)
    taco = sub.add_parser('taco')
    taco.add_argument('--prepared', type=Path, required=True)
    args = p.parse_args()
    for name in ('human_demo', 'clip-records', 'objects'):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    if args.dataset == 'oakink2':
        prepare_oak(args)
    else:
        prepare_taco(args)


if __name__ == '__main__':
    main()
