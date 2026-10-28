"""Validate the packed bimanual data contract before constructing a simulator."""
from pathlib import Path
import math
import torch


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
    path = Path(path).resolve()
    data = torch.load(path, map_location='cpu', weights_only=False)
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


def validate_motion(path, action_dim=36, require_sim_assets=True):
    path = Path(path).resolve()
    data = torch.load(path, map_location="cpu", weights_only=False)
    fps = dataset_fps(data)
    lengths = data["motion_num_frames"]
    starts = data["length_starts"]
    if lengths.ndim != 1 or len(lengths) == 0 or (lengths < 2).any():
        raise ValueError("Expected nonempty clips with at least two frames")
    if not torch.equal(starts, lengths.cumsum(0) - lengths):
        raise ValueError("Non-contiguous clip starts")
    frames = int(lengths.sum())
    for key in ("joint_pos", "joint_vel"):
        if data[key].shape != (frames, action_dim):
            raise ValueError(f"{key}: expected {(frames, action_dim)}, got {data[key].shape}")
    required = ["joint_pos", "joint_vel"]
    assets = set()
    for side in ("right", "left"):
        required += [f"mano_{side}_{suffix}" for suffix in
                     ("wrist_pos", "wrist_rot", "joints")]
        required += [f"obj_{side}_{suffix}" for suffix in ("pos", "rotmat", "vel", "angvel")]
        directories = data[f"{side}_object_mesh_dirs"]
        slots = data[f"motion_object_slot_{side}"]
        if slots.shape != lengths.shape or (slots < 0).any() or (slots >= len(directories)).any():
            raise ValueError(f"Invalid {side} object slot mapping")
        for directory in directories:
            obj = path.parent / directory
            if not (obj / "visual.obj").is_file():
                raise FileNotFoundError(obj / "visual.obj")
            if require_sim_assets:
                if not (obj / ".isaac_usd/object.usd").is_file():
                    raise FileNotFoundError(obj / ".isaac_usd/object.usd")
                if not any((obj / ".sdf_cache").glob("*")):
                    raise FileNotFoundError(obj / ".sdf_cache")
            assets.add(str(obj.resolve()))
    for key in required:
        value = data[key]
        if value.shape[0] != frames or not torch.isfinite(value).all():
            raise ValueError(f"Invalid frame count or non-finite values: {key}")
    return {"motion_file": str(path), "clips": len(lengths), "frames": frames, "fps": fps,
            "action_dim": action_dim, "objects": sorted(assets),
            "sim_assets_checked": require_sim_assets}


def load_scene_catalog(path):
    """Canonical packed scene; slot count is the dataset maximum, never a constant eight."""
    from scipy.spatial.transform import Rotation
    path = Path(path).resolve()
    raw = torch.load(path, map_location='cpu', weights_only=False)
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
    scene_scales = raw.get('motion_scene_object_scales')
    if scene_scales is not None and len(scene_scales) != len(ids):
        raise ValueError('Scene scale metadata must have one row per clip')
    for i, objects in enumerate(ids):
        dirs = [f'objects/{oid}' for oid in objects]
        size = list(scene_scales[i]) if scene_scales is not None else [1.] * len(objects)
        if len(size) != len(objects) or any(not math.isfinite(float(s)) or float(s) <= 0 for s in size):
            raise ValueError(f'{raw["motion_filename"][i]}: invalid scene object scales')
        for si, side in enumerate(('right', 'left')):
            k = int(raw[f'motion_object_slot_{side}'][i])
            directory = raw[f'{side}_object_mesh_dirs'][k]
            oid = Path(directory).name
            if oid not in objects:
                raise ValueError(f'{raw["motion_filename"][i]}: hand object {oid} absent from required scene')
            index = objects.index(oid)
            hand_slots[i, si] = index
            dirs[index] = directory
            held_scale = float(raw[f'{side}_object_mesh_scales'][k])
            if scene_scales is not None and not math.isclose(float(size[index]), held_scale, rel_tol=1e-6):
                raise ValueError(f'{raw["motion_filename"][i]}: scene/hand scale mismatch for {oid}')
            size[index] = held_scale
        directories.append(dirs + [None] * (slots-len(dirs)))
        scales.append(size + [1.] * (slots-len(size)))
    fields = ('pos', 'quat', 'lin_vel', 'ang_vel')
    state = {}
    if 'scene_object_pos' in raw:
        for name, width in zip(fields, (3, 4, 3, 3)):
            value = raw[f'scene_object_{name}'].float()
            if value.shape != (total, slots, width) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid scene_object_{name}: expected {(total,slots,width)} finite values')
            state[name] = value
    else:
        # Lossless compatibility for old two-hand packs, never infer a missing receiver pose.
        state = {k: torch.zeros(total, slots, w) for k,w in zip(fields,(3,4,3,3))}
        state['quat'][...,0] = 1
        for i, objects in enumerate(ids):
            if set(hand_slots[i].tolist()) != set(range(len(objects))):
                raise ValueError(f'{raw["motion_filename"][i]}: required objects lack scene trajectories; repack all scene_object_* fields')
            sl = slice(int(starts[i]), int(starts[i]+lengths[i]))
            for si, side in enumerate(('right','left')):
                slot = int(hand_slots[i,si])
                state['pos'][sl,slot] = raw[f'obj_{side}_pos'][sl]
                q = Rotation.from_matrix(raw[f'obj_{side}_rotmat'][sl].numpy()).as_quat()[:,[3,0,1,2]]
                state['quat'][sl,slot] = torch.from_numpy(q).float()
                state['lin_vel'][sl,slot] = raw[f'obj_{side}_vel'][sl]
                state['ang_vel'][sl,slot] = raw[f'obj_{side}_angvel'][sl]
    frame_mask = torch.repeat_interleave(mask,lengths,dim=0)
    if ((state['quat'].norm(dim=-1)[frame_mask]-1).abs()>1e-3).any():
        raise ValueError('Scene quaternions must be normalized wxyz')
    # Mesh layouts are fixed per environment; reset samples compatible clips only.
    layouts, groups, clip_group = [], [], []
    for i in range(len(ids)):
        key = tuple(zip(directories[i],scales[i]))
        if key not in layouts:
            layouts.append(key); groups.append([])
        g = layouts.index(key); groups[g].append(i); clip_group.append(g)
    return dict(path=path, raw=raw, slots=slots, mask=mask, hand_slots=hand_slots,
                directories=directories, scales=scales, state=state, layouts=layouts,
                groups=groups, clip_group=clip_group)


def configure_scene_slots(cfg):
    """Use AA's normal scene/MDP factories, sized from the selected dataset."""
    from omegaconf import OmegaConf
    catalog = load_scene_catalog(cfg.task.motion_file)
    groups = len(catalog['groups'])
    if cfg.task.num_envs < groups:
        raise ValueError(f'Dataset has {groups} distinct object layouts; num_envs={cfg.task.num_envs} cannot represent each. Select a smaller dataset or increase num_envs.')
    OmegaConf.set_struct(cfg, False)
    cfg.task.objects = {f'object_{i}': {'_target_':'dex_track.scene_object',
        'motion_file':str(catalog['path']), 'slot':i} for i in range(catalog['slots'])}
    print(f"Scene allocation: {catalog['slots']} object slots, {groups} mesh layouts, {len(catalog['mask'])} clips (table excluded)",flush=True)
    return catalog['slots']
