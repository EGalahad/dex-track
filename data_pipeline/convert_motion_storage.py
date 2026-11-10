"""Migrate current training folders to free-root qpos, without contact arrays.

Original source and experiment snapshots are inputs only. In-place migration
stages and verifies a new corpus before replacing its index/motion directories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import tempfile

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]


def convert_robot(raw, model_path=None):
    """Convert virtual-wrist retarget coordinates into the current free roots."""
    if 'qpos' in raw:
        if raw.get('robot', {}).get('representation') != 'free-root-qpos-wxyz-v1':
            raise ValueError('Unknown robot qpos convention')
        raw.pop('joint_vel', None)
        return raw, {'already_current': True}
    import mujoco
    model_path = Path(model_path or PROJECT/'third_party/FlashDexRetarget/assets/robot/xhand/bimanual.xml')
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    free_models = [mujoco.MjModel.from_xml_path(str(PROJECT/'assets/github_xhand'/f'{side}.xml'))
                   for side in ('right', 'left')]
    free_data = [mujoco.MjData(m) for m in free_models]
    body_pairs = [[(j, model.body(m.body(j).name).id) for j in range(1, m.nbody)] for m in free_models]
    names = [model.joint(i).name for i in range(model.njnt)]
    finger_names = [[n for n in names if n.startswith(side + '_hand_')] for side in ('right', 'left')]
    if list(map(len, finger_names)) != [12, 12]:
        raise ValueError(f'Expected two 12-joint hands: {finger_names}')
    addresses = [[int(model.jnt_qposadr[model.joint(n).id]) for n in row] for row in finger_names]
    roots = [model.body(n).id for n in ('R_forearm_rot_y_link', 'L_forearm_rot_y_link')]
    old = raw['joint_pos'].cpu().numpy()
    if old.shape[1] != model.nq or model.nq != 36:
        raise ValueError('Legacy qpos does not match the pinned retarget model')
    qpos = np.empty((len(old), 2, 19), dtype=np.float32)
    max_root_error = 0.
    max_body_error = max_rotation_error = 0.
    for i, row in enumerate(old):
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        qpos[i, :, :3] = data.xpos[roots]
        qpos[i, :, 3:7] = data.xquat[roots]
        for side, ids in enumerate(addresses):
            qpos[i, side, 7:] = row[ids]
        max_root_error = max(max_root_error, float(np.max(np.abs(qpos[i, :, :3] - data.xpos[roots]))))
        for side, (m, d, pairs) in enumerate(zip(free_models, free_data, body_pairs)):
            d.qpos[:] = qpos[i, side]
            mujoco.mj_kinematics(m, d)
            free_ids, old_ids = np.array(pairs).T
            max_body_error = max(max_body_error, float(np.max(np.abs(d.xpos[free_ids] - data.xpos[old_ids]))))
            max_rotation_error = max(max_rotation_error, float(np.max(np.abs(d.xmat[free_ids] - data.xmat[old_ids]))))
    if max_body_error > 1e-6 or max_rotation_error > 1e-6:
        raise ValueError(f'Free-hand FK differs: position={max_body_error}, rotation={max_rotation_error}')
    for side, ids in enumerate(addresses):
        np.testing.assert_array_equal(qpos[:, side, 7:], old[:, ids])
    np.testing.assert_allclose(np.linalg.norm(qpos[:, :, 3:7], axis=-1), 1., atol=1e-6)
    raw['qpos'] = torch.from_numpy(qpos.reshape(len(old), 38))
    raw['robot'] = dict(representation='free-root-qpos-wxyz-v1', hand_sides=['right', 'left'],
                        qpos_per_hand=['position_xyz', 'quaternion_wxyz', '12_finger_joints'],
                        finger_joint_names=finger_names,
                        velocity='derived_per_segment_after_policy_resampling; centered_inside_one_sided_at_ends')
    raw.pop('joint_pos')
    raw.pop('joint_vel', None)
    return raw, dict(source_qpos_width=36, qpos_width=38, frames=len(old),
                     finger_coordinates_exact=True, root_position_float32_max_abs_error_m=max_root_error,
                     all_frames_all_hand_bodies_fk_max_abs_error_m=max_body_error,
                     all_frames_all_hand_bodies_rotation_matrix_max_abs_error=max_rotation_error,
                     retarget_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())


def segment_records(raw, manifest=None):
    if 'motion_segments' in raw and manifest is None:
        for i, item in enumerate(raw['motion_segments']):
            item['scene_objects'] = raw['motion_scene_object_ids'][i]
            item['hand_objects'] = {side: Path(raw[f'{side}_object_mesh_dirs'][int(raw[f'motion_object_slot_{side}'][i])]).name
                                    for side in ('right', 'left')}
        return
    stages = {}
    if manifest:
        source = json.loads(Path(manifest).read_text())
        records = source.get('stages', source.get('motions', [])) if isinstance(source, dict) else source
        stages = {r.get('stage_id', r.get('clip')): r for r in records}
    result = []
    for i, name in enumerate(raw['motion_filename']):
        recording, separator, stage = name.rpartition('__st')
        item = dict(id=name, source_recording=recording if separator else name,
                    stage_index=int(stage) if separator and stage.isdigit() else None,
                    stored_fps=float(raw['meta']['fps']), frames=int(raw['motion_num_frames'][i]),
                    hand_objects={})
        item['scene_objects'] = raw['motion_scene_object_ids'][i]
        for side in ('right', 'left'):
            slot = int(raw[f'motion_object_slot_{side}'][i])
            item['hand_objects'][side] = Path(raw[f'{side}_object_mesh_dirs'][slot]).name
        if name in stages:
            s = stages[name]
            item['source_stage_fps'] = s['fps']
            item['source_stage_frame_interval_half_open'] = s['frame_interval_half_open']
            item['primitive'] = s.get('primitive')
            item['source_annotation'] = s.get('source_annotation')
            item['source_program_sha256'] = s.get('source_program_sha256')
        # Retargeting may crop the stage again. A stage interval alone cannot
        # establish the exact source frame corresponding to stored frame zero.
        item['retarget_crop_source_interval'] = None
        item['retarget_crop_provenance'] = raw.get('scene_selection_provenance', [{}] * len(raw['motion_filename']))[i]
        result.append(item)
    raw['motion_segments'] = result


def verify(raw, loaded):
    checked = []
    total = int(raw['motion_num_frames'].sum())
    for key, value in raw.items():
        if key.startswith(('contact_', 'tips_')) or key in ('motion_object_slot_right', 'motion_object_slot_left', 'motion_topology'):
            continue
        if torch.is_tensor(value):
            if key.startswith('scene_object_'):
                for i, length in enumerate(raw['motion_num_frames'].tolist()):
                    start = int(raw['length_starts'][i]); width = len(raw['motion_scene_object_ids'][i])
                    torch.testing.assert_close(value[start:start+length, :width], loaded[key][start:start+length, :width], rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, loaded[key], rtol=0, atol=0)
            checked.append(key)
    for side in ('right', 'left'):
        for i in range(len(raw['motion_filename'])):
            def asset(d):
                j = int(d[f'motion_object_slot_{side}'][i])
                return d[f'{side}_object_mesh_dirs'][j], d[f'{side}_object_mesh_scales'][j]
            assert asset(raw) == asset(loaded)
    for key in ('motion_segments', 'motion_scene_object_ids', 'motion_scene_object_scales', 'motion_scene_supports', 'robot'):
        if key in raw:
            assert raw[key] == loaded[key], key
    for a, b in zip(raw.get('motion_topology', []), loaded.get('motion_topology', [])):
        if a is None:
            assert b is None
        else:
            assert a.keys() == b.keys()
            for key in a: np.testing.assert_array_equal(a[key], b[key])
    assert not any(k.startswith(('contact_', 'tips_')) for k in loaded)
    assert 'joint_pos' not in loaded and 'joint_vel' not in loaded
    return dict(valid_trajectories_exact=True, topology_exact=True, assignments_exact=True,
                source_frames=total, verified_tensor_keys=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--in-place', action='store_true')
    parser.add_argument('--segment-manifest', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--distinct-objects', type=int)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.in_place == (args.destination is not None):
        parser.error('Choose --in-place or --destination')
    api = runpy.run_path(str(PROJECT/'dex_track/motion_storage.py'))
    source = api['resolve_motion_path'](args.source)
    raw = torch.load(source, map_location='cpu', weights_only=False) if source.suffix == '.pt' else api['load_motion_data'](source)
    indices = list(range(len(raw['motion_filename'])))
    if args.distinct_objects is not None:
        indices, seen = [], set()
        for i in torch.randperm(len(raw['motion_filename']), generator=torch.Generator().manual_seed(args.seed)).tolist():
            slot = int(raw['motion_object_slot_right'][i]); oid = Path(raw['right_object_mesh_dirs'][slot]).name
            if oid in seen: continue
            seen.add(oid); indices.append(i)
            if len(indices) == args.distinct_objects: break
        if len(indices) != args.distinct_objects: raise ValueError('Insufficient distinct objects')
        raw = api['subset_motion_data'](raw, indices)
    source_index_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    raw, conversion = convert_robot(raw, args.model)
    segment_records(raw, args.segment_manifest)
    raw = {k: v for k, v in raw.items() if not k.startswith(('contact_', 'tips_'))}
    root = source.parent if args.in_place else args.destination.resolve()
    if not args.in_place and root.exists(): raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{root.name}-v2-', dir=root.parent))
    try:
        index = api['write_motion_folders'](raw, stage)
        for name in ('objects', 'supports'):
            if (source.parent/name).exists(): (stage/name).symlink_to((source.parent/name).resolve(), target_is_directory=True)
        loaded = api['load_motion_data'](index)
        report = dict(schema=api['SCHEMA'], source=str(source), source_index_sha256=source_index_sha,
                      motions=len(indices), robot_conversion=conversion,
                      segments_with_source_stage_interval=sum('source_stage_frame_interval_half_open' in s for s in raw['motion_segments']),
                      exact_retarget_crop_interval_recovered=False, **verify(raw, loaded))
        assert not list((stage/'motions').rglob('contact.npz'))
        assert not list((stage/'motions').rglob('scene.npz'))
        (stage/'conversion-verification.json').write_text(json.dumps(report, indent=2)+'\n')
        if args.in_place:
            backup = root/'backup'/'schema-v1-before-20261109'
            if backup.exists(): raise FileExistsError(backup)
            backup.mkdir(parents=True)
            moved = []
            try:
                for name in ('motions', 'index.json', 'conversion-verification.json'):
                    if (root/name).exists():
                        (root/name).rename(backup/name); moved.append(name)
                for name in ('motions', 'index.json', 'conversion-verification.json'):
                    (stage/name).rename(root/name)
            except BaseException:
                for name in moved:
                    target = root/name
                    if target.is_dir(): shutil.rmtree(target)
                    elif target.exists(): target.unlink()
                    (backup/name).rename(target)
                raise
            print('Verified v2 installed; temporary rollback data:', backup, flush=True)
        else:
            stage.rename(root)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if stage.exists(): shutil.rmtree(stage)


if __name__ == '__main__':
    main()
