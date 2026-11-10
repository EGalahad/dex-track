"""Validate the packed bimanual data contract before constructing a simulator."""
from pathlib import Path
import math
import torch
import runpy

_STORAGE = runpy.run_path(str(Path(__file__).with_name("motion_storage.py")))
load_motion_data = _STORAGE["load_motion_data"]
resolve_motion_path = _STORAGE["resolve_motion_path"]


def dataset_fps(raw):
    """Packed sample rate belongs to the dataset, never to the task config."""
    meta = raw.get('meta')
    if not isinstance(meta, dict) or 'fps' not in meta:
        raise ValueError('Missing dataset meta.fps; record the source sample rate when packing the dataset')
    try:
        fps = float(meta['fps'])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid dataset meta.fps={meta['fps']!r}") from error
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'Invalid dataset meta.fps={fps}')
    return fps


def validate_training_object_limit(path, max_objects=8, excluded_motions=None):
    """Check required episode objects, not the full recording's background list."""
    path = resolve_motion_path(path)
    data = load_motion_data(path)
    names = data['motion_filename']
    ids = data.get('motion_scene_object_ids')
    if ids is None or len(ids) != len(names) or not data.get('scene_selection_provenance'):
        raise ValueError('Missing episode object selection and provenance; neither hand assignments nor whole-recording context alone establish required episode objects')
    counts = []
    for i, name in enumerate(names):
        key = Path(str(name)).stem
        if excluded_motions and key in excluded_motions:
            raise ValueError(f'{name}: excluded from physical training: {excluded_motions[key]}')
        objects = set(ids[i])
        count = len(objects)
        if not 0 < count <= max_objects:
            raise ValueError(f'{name}: {count} required episode objects exceeds training contract (1..{max_objects}, table excluded); dataset load rejected')
        counts.append(count)
    return counts


def validate_motion(path, action_dim=36, require_sim_assets=False):
    path = resolve_motion_path(path)
    data = load_motion_data(path)
    fps = dataset_fps(data)
    lengths = data["motion_num_frames"]
    starts = data["length_starts"]
    if lengths.ndim != 1 or len(lengths) == 0 or (lengths < 2).any():
        raise ValueError("Expected nonempty clips with at least two frames")
    if not torch.equal(starts, lengths.cumsum(0) - lengths):
        raise ValueError("Non-contiguous clip starts")
    frames = int(lengths.sum())
    if data.get('robot', {}).get('representation') != 'free-root-qpos-wxyz-v1':
        raise ValueError('Expected current free-root qpos; migrate the legacy dataset first')
    if data['qpos'].shape != (frames, 38):
        raise ValueError(f"qpos: expected {(frames, 38)}, got {data['qpos'].shape}")
    roots = data['qpos'].reshape(frames, 2, 19)[:, :, 3:7]
    if ((roots.norm(dim=-1) - 1).abs() > 1e-4).any():
        raise ValueError('Robot root quaternions must be normalized wxyz')
    if len(set(n for row in data['robot']['finger_joint_names'] for n in row)) != 24:
        raise ValueError('Expected 24 unique named finger joints')
    required = ['qpos']
    assets = set()
    for side in ("right", "left"):
        required += [f"mano_{side}_{suffix}" for suffix in
                     ("wrist_pos", "wrist_rot", "joints")]
        slots = data[f"motion_object_slot_{side}"]
        if slots.shape != lengths.shape:
            raise ValueError(f"Invalid {side} scene slot shape")
        for i, slot in enumerate(slots.tolist()):
            ids = data['motion_scene_object_ids'][i]
            if slot < 0 or slot >= len(ids):
                raise ValueError(f"Invalid {side} scene slot")
            if data['motion_segments'][i]['hand_objects'][side] != ids[slot]:
                raise ValueError(f"{side}: scene slot disagrees with hand_objects")
    for ids in data['motion_scene_object_ids']:
        for oid in ids:
            obj = path.parent/'objects'/oid
            if not (obj/'visual.obj').is_file(): raise FileNotFoundError(obj/'visual.obj')
            if not any((obj/'convex').glob('*.obj')): raise FileNotFoundError(obj/'convex')
            if require_sim_assets:
                if not (obj/'.isaac_usd/object.usd').is_file(): raise FileNotFoundError(obj/'.isaac_usd/object.usd')
                if not any((obj/'.sdf_cache').glob('*')): raise FileNotFoundError(obj/'.sdf_cache')
            assets.add(str(obj.resolve()))
    for side in ('right','left'):
        palm = data['robot']['palm_in_root'][side]
        p, q = torch.tensor(palm['position_xyz']), torch.tensor(palm['quaternion_wxyz'])
        if p.shape != (3,) or q.shape != (4,) or not torch.isfinite(p).all() or not torch.isfinite(q).all() or abs(float(q.norm())-1)>1e-5:
            raise ValueError(f'Invalid dataset palm_in_root.{side}')
    expected_shapes = {'qpos': (frames,38)}
    for side in ('right','left'):
        expected_shapes.update({f'mano_{side}_wrist_pos':(frames,3), f'mano_{side}_wrist_rot':(frames,3,3), f'mano_{side}_joints':(frames,20,3)})
    for key in required:
        value = data[key]
        if tuple(value.shape) != expected_shapes[key]:
            raise ValueError(f'{key}: expected {expected_shapes[key]}, got {tuple(value.shape)}')
        if value.shape[0] != frames or not torch.isfinite(value).all():
            raise ValueError(f"Invalid frame count or non-finite values: {key}")
    return {"motion_file": str(path), "clips": len(lengths), "frames": frames, "fps": fps,
            "action_dim": action_dim, "objects": sorted(assets),
            "sim_assets_checked": require_sim_assets}


def load_scene_catalog(path):
    """Canonical packed scene; slot count is the dataset maximum, never a constant eight."""
    path = resolve_motion_path(path)
    raw = load_motion_data(path)
    dataset_fps(raw)
    counts = validate_training_object_limit(path)
    slots = max(counts)
    ids = raw['motion_scene_object_ids']
    if any(len(x) != len(set(x)) for x in ids):
        raise ValueError('Scene object IDs must be unique within each clip')
    lengths = raw['motion_num_frames'].long()
    starts = raw['length_starts'].long()
    total = int(lengths.sum())
    mask = torch.arange(slots)[None, :] < torch.tensor(counts)[:, None]
    hand_slots = torch.empty(len(ids), 2, dtype=torch.long)
    directories, scales = [], []
    scene_scales = raw['motion_scene_object_scales']
    if scene_scales is not None and len(scene_scales) != len(ids):
        raise ValueError('Scene scale metadata must have one row per clip')
    for i, objects in enumerate(ids):
        dirs = [f'objects/{oid}' for oid in objects]
        size = list(scene_scales[i]) if scene_scales is not None else [1.] * len(objects)
        if len(size) != len(objects) or any(not math.isfinite(float(s)) or float(s) <= 0 for s in size):
            raise ValueError(f'{raw["motion_filename"][i]}: invalid scene object scales')
        for si, side in enumerate(('right', 'left')):
            index = int(raw[f'motion_object_slot_{side}'][i])
            if not 0 <= index < len(objects):
                raise ValueError(f'{side}: invalid scene slot {index}')
            if raw['motion_segments'][i]['hand_objects'][side] != objects[index]:
                raise ValueError(f'{side}: scene slot and hand_objects disagree')
            hand_slots[i, si] = index
        directories.append(dirs + [None] * (slots-len(dirs)))
        scales.append(size + [1.] * (slots-len(size)))
    state = {}
    for name, width in [('pos',3),('quat',4)]:
        value = raw[f'scene_object_{name}'].float()
        if value.shape != (total,slots,width) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid scene_object_{name}: expected {(total,slots,width)} finite values')
        state[name] = value
    frame_mask = torch.repeat_interleave(mask,lengths,dim=0)
    if ((state['quat'].norm(dim=-1)[frame_mask]-1).abs()>1e-3).any():
        raise ValueError('Scene quaternions must be normalized wxyz')
    # Mesh layouts are fixed per environment; reset samples compatible clips only.
    # train.py calls this preflight before AA chooses a backend. Importing the
    # dex_track package here would register MDP modules and cache AA's terrain
    # module with no backend selected, leaving the later terrain registry empty.
    import runpy
    support_contract = runpy.run_path(str(Path(__file__).with_name('supports.py')))
    SCHEMA = support_contract['SCHEMA']
    support_layout_key = support_contract['support_layout_key']
    validate_supports = support_contract['validate_supports']
    supports = raw.get('motion_scene_supports', [[] for _ in ids])
    if len(supports) != len(ids):
        raise ValueError('motion_scene_supports must have one list per motion (empty when absent)')
    if 'motion_scene_supports' in raw and raw.get('meta', {}).get('scene_support_schema') != SCHEMA:
        raise ValueError('Missing or unsupported meta.scene_support_schema')
    for row in supports:
        validate_supports(row)
    layouts, layout_supports, group_keys, groups, clip_group = [], [], [], [], []
    for i in range(len(ids)):
        objects = tuple(zip(directories[i],scales[i]))
        key = (objects, support_layout_key(supports[i]))
        if key not in group_keys:
            group_keys.append(key); layouts.append(objects)
            layout_supports.append(supports[i]); groups.append([])
        g = group_keys.index(key); groups[g].append(i); clip_group.append(g)
    return dict(path=path, raw=raw, slots=slots, mask=mask, hand_slots=hand_slots,
                directories=directories, scales=scales, state=state, layouts=layouts,
                groups=groups, clip_group=clip_group, supports=supports,
                layout_supports=layout_supports)


def configure_scene_slots(cfg, allow_play_subset=False):
    """Use AA's normal scene/MDP factories, sized from the selected dataset."""
    from omegaconf import OmegaConf
    catalog = load_scene_catalog(cfg.task.motion_file)
    groups = len(catalog['groups'])
    if cfg.task.num_envs < groups and allow_play_subset:
        selected_path = _STORAGE['select_play_subset'](catalog, int(cfg.task.num_envs), int(cfg.get('seed', 0)))
        OmegaConf.set_struct(cfg, False)
        cfg.task.motion_file = str(selected_path)
        cfg.task.command.motion_file = str(selected_path)
        catalog = load_scene_catalog(selected_path)
        groups = len(catalog['groups'])
    if cfg.task.num_envs < groups:
        raise ValueError(f'Dataset has {groups} distinct object layouts; num_envs={cfg.task.num_envs} cannot represent each. Select a smaller dataset or increase num_envs.')
    OmegaConf.set_struct(cfg, False)
    cfg.task.motion_file = str(catalog['path'])
    cfg.task.command.motion_file = str(catalog['path'])
    cfg.task.objects = {f'object_{i}': {'_target_':'dex_track.scene_object',
        'motion_file':str(catalog['path']), 'slot':i} for i in range(catalog['slots'])}
    collision_variant = cfg.task.robot.get('collision_variant', 'simplified')
    for entry in cfg.task.objects.values():
        entry['collision_variant'] = collision_variant
    if cfg.task.robot.name == 'dex_track.xhand_free':
        cfg.task.objects['left_hand'] = {'_target_':'dex_track.xhand_free', 'side':'left',
            'finger_damping':cfg.task.robot.get('finger_damping',2.5), 'collision_variant':collision_variant}
        for key in ('finger_stiffness', 'finger_armature', 'finger_passive_damping'):
            if cfg.task.robot.get(key) is not None:
                cfg.task.objects['left_hand'][key] = cfg.task.robot[key]
        if cfg.task.robot.get('palm_position_root') is not None:
            cfg.task.objects['left_hand']['palm_position_root'] = cfg.task.robot.palm_position_root
    if any(catalog['layout_supports']):
        cfg.task.objects['support'] = {'_target_':'dex_track.scene_support',
                                     'motion_file':str(catalog['path'])}
    print(f"Scene allocation: {catalog['slots']} object slots, {groups} mesh layouts, {len(catalog['mask'])} clips (table excluded)",flush=True)
    return catalog['slots']
