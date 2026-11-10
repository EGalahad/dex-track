"""Portable per-motion NPZ folders with JSON metadata."""
from pathlib import Path
import copy
import hashlib
import json
from functools import lru_cache
import numpy as np
import torch

SCHEMA = 'dex-track-motion-folders-v3'
LEGACY_SCHEMA = 'dex-track-motion-folders-v2'
GLOBAL_KEYS = {'meta', 'mano_right_joint_names', 'mano_left_joint_names', 'robot', '_dataset'}


def resolve_motion_path(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        return path / ('index.json' if (path/'index.json').is_file() else 'metadata.json')
    if not path.exists() and path.name == 'motion.pt' and (path.parent/'index.json').is_file():
        return path.parent/'index.json'
    return path


def subset_motion_data(raw, indices):
    indices = [int(i) for i in indices]
    n = len(raw['motion_filename']); total = int(raw['motion_num_frames'].sum())
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= n:
        raise ValueError('Expected nonempty unique in-range motion indices')
    slices = [slice(int(raw['length_starts'][i]), int(raw['length_starts'][i]+raw['motion_num_frames'][i])) for i in indices]
    out = {}
    for key, value in raw.items():
        if key in GLOBAL_KEYS:
            out[key] = copy.deepcopy(value)
        elif torch.is_tensor(value) and value.ndim and value.shape[0] == total:
            out[key] = torch.cat([value[s] for s in slices]).clone()
        elif torch.is_tensor(value) and value.ndim and value.shape[0] == n:
            out[key] = value[indices].clone()
        elif isinstance(value, list) and len(value) == n:
            out[key] = [copy.deepcopy(value[i]) for i in indices]
        else:
            out[key] = copy.deepcopy(value)
    out['length_starts'] = out['motion_num_frames'].cumsum(0)-out['motion_num_frames']
    slots = max(map(len, out['motion_scene_object_ids']))
    for key in ('scene_object_pos','scene_object_quat'):
        if key in out: out[key] = out[key][:, :slots].clone()
    return out


def _group(key):
    if key == 'qpos': return 'robot'
    if key.startswith('mano_'): return 'mano'
    return 'objects'


def write_motion_folders(raw, destination):
    """Lossless numeric arrays, no pickle; shared assets stay relative to index.json."""
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    if 'qpos' not in raw or raw.get('robot', {}).get('representation') != 'free-root-qpos-wxyz-v1':
        raise ValueError('Convert legacy robot coordinates before writing a v3 dataset')
    if (destination/'index.json').exists(): raise FileExistsError(destination/'index.json')
    if not raw.get('robot', {}).get('palm_in_root'):
        raise ValueError('v3 requires dataset robot.palm_in_root calibration')
    if any('vel' in k or k.startswith('obj_') for k, v in raw.items() if torch.is_tensor(v)):
        raise ValueError('v3 stores poses only; use data_pipeline/convert_pose_only_v3.py')
    contract = copy.deepcopy(raw.get('_dataset', {}))
    contract.update(schema=SCHEMA, revision='2026-11-10-pose-only-v3', robot=copy.deepcopy(raw['robot']),
                    motion_metadata_defaults={k:copy.deepcopy(raw[k]) for k in ('meta','mano_right_joint_names','mano_left_joint_names')})
    (destination/'dataset.json').write_text(json.dumps(contract,indent=2))
    rows = []
    for i, name in enumerate(raw['motion_filename']):
        clip = subset_motion_data(raw, [i]); folder = destination/'motions'/f'{i:04d}-{Path(str(name)).stem}'
        folder.mkdir(parents=True, exist_ok=False); arrays = {}
        def encode(value, key):
            if torch.is_tensor(value):
                group = _group(key); arrays.setdefault(group,{})[key] = value.detach().cpu().numpy()
                return {'__array__': key, 'file': group+'.npz', 'kind': 'torch'}
            if isinstance(value, dict): return {k:encode(v,key+'__'+k) for k,v in value.items()}
            if isinstance(value, (tuple,list)): return [encode(v,key+'__'+str(j)) for j,v in enumerate(value)]
            if isinstance(value, np.generic): return value.item()
            return value
        for asset_dir in ('objects','supports','collision_simplified'):
            (folder/asset_dir).symlink_to(Path('../..')/asset_dir,target_is_directory=True)
        metadata = {k:encode(v,k) for k,v in clip.items()
                    if k not in GLOBAL_KEYS and k not in ('motion_topology', 'joint_pos', 'joint_vel')
                    and not k.startswith(('contact_', 'tips_'))}
        metadata['schema'] = SCHEMA
        metadata['dataset_file'] = '../../dataset.json'
        metadata['meta'] = {}
        if clip.get('motion_topology') and clip['motion_topology'][0] is not None:
            np.savez_compressed(folder/'topology.npz', **clip['motion_topology'][0])
        for group, contents in arrays.items(): np.savez_compressed(folder/(group+'.npz'), **contents)
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
        rows.append(dict(path=str((folder/'metadata.json').relative_to(destination)),name=str(name),
                         frames=int(clip['motion_num_frames'][0]),object_ids=clip['motion_scene_object_ids'][0]))
    manifest = dict(schema=SCHEMA, motions=rows, fps=float(raw['meta']['fps']))
    (destination/'index.json').write_text(json.dumps(manifest,indent=2))
    contract.update(segments=len(rows), frames=sum(r['frames'] for r in rows),
                    index_sha256=hashlib.sha256((destination/'index.json').read_bytes()).hexdigest())
    (destination/'dataset.json').write_text(json.dumps(contract,indent=2))
    return destination/'index.json'


def load_motion_data(path):
    path = resolve_motion_path(path)
    topology_versions = ()
    if path.suffix == '.json':
        manifest = json.loads(path.read_text())
        metadata = [path.parent/row['path'] for row in manifest['motions']] if 'motions' in manifest else [path]
        dependencies = []
        for m in metadata:
            md = json.loads(m.read_text())
            dependencies.extend((m, m.parent/'topology.npz', m.parent/md.get('dataset_file','../../dataset.json')))
        topology_versions = tuple((str(p), p.stat().st_mtime_ns if p.exists() else None) for p in dependencies)
    return _load_motion_data_cached(str(path), path.stat().st_mtime_ns, topology_versions)


@lru_cache(maxsize=2)
def _load_motion_data_cached(filename, mtime_ns, topology_versions=()):
    path = Path(filename)
    if path.suffix != '.json':
        raise ValueError('Training accepts NPZ motion folders only; convert the original PT explicitly.')
    manifest = json.loads(path.read_text())
    if 'motion_filename' in manifest and 'motion_num_frames' in manifest:
        manifest = dict(schema=manifest.get('schema', LEGACY_SCHEMA),motions=[dict(path=path.name)])
    if manifest.get('schema') != SCHEMA:
        raise ValueError(f'{path}: expected pose-only v3; convert with data_pipeline/convert_pose_only_v3.py --source V2 --destination V3 --palm-calibration calibration.json --receipt receipt.json')
    clips=[]
    for row in manifest['motions']:
        metadata_path = path.parent/row['path']; cache={}
        def decode(value):
            if isinstance(value,dict) and '__array__' in value:
                filename=value['file']
                if filename not in cache:
                    with np.load(metadata_path.parent/filename,allow_pickle=False) as f:
                        if any('vel' in k or k.startswith('obj_') for k in f.files):
                            raise ValueError(f'{metadata_path}: forbidden velocity/redundant arrays in {filename}')
                        cache[filename]={k:f[k] for k in f.files}
                return torch.from_numpy(cache[filename][value['__array__']].copy())
            if isinstance(value,dict): return {k:decode(v) for k,v in value.items()}
            if isinstance(value,list): return [decode(v) for v in value]
            return value
        # Historical packs contain contact annotations for offline audits. The
        # training task does not consume them; never open their archive here.
        metadata = {k: v for k, v in json.loads(metadata_path.read_text()).items()
                    if not k.startswith(('contact_', 'tips_')) and k != 'schema'}
        md = json.loads(metadata_path.read_text())
        if md.get('schema') != SCHEMA:
            raise ValueError(f'{metadata_path}: legacy segment; convert with data_pipeline/convert_pose_only_v3.py')
        contract = json.loads((metadata_path.parent/md.get('dataset_file','../../dataset.json')).read_text())
        if contract.get('schema') != SCHEMA or not contract.get('robot',{}).get('palm_in_root'):
            raise ValueError('Missing v3 dataset contract or robot.palm_in_root')
        defaults = copy.deepcopy(contract.get('motion_metadata_defaults',{}))
        defaults['meta'] = dict(defaults.get('meta',{}), **metadata.get('meta',{}))
        metadata = dict(defaults, **{k:v for k,v in metadata.items() if k not in ('meta','dataset_file')}, meta=defaults['meta'])
        metadata['robot'] = copy.deepcopy(contract['robot'])
        clip = decode(metadata)
        clip['_dataset'] = contract
        if any(('vel' in k or k.startswith('obj_')) and torch.is_tensor(v) for k,v in clip.items()):
            raise ValueError(f'{metadata_path}: forbidden redundant/velocity arrays in v3')
        topology_path = metadata_path.parent/'topology.npz'
        if topology_path.exists():
            with np.load(topology_path, allow_pickle=False) as f:
                clip['motion_topology'] = [{k: f[k].copy() for k in f.files}]
        else:
            clip['motion_topology'] = [None]
        clips.append(clip)
    if not clips: raise ValueError('Empty motion dataset')
    for clip in clips[1:]:
        for key in GLOBAL_KEYS:
            if clip.get(key) != clips[0].get(key):
                raise ValueError(f'Cannot combine incompatible dataset metadata: {key}')
    out={}; scene_width=max(len(c['motion_scene_object_ids'][0]) for c in clips)
    for key in clips[0]:
        values=[c[key] for c in clips]
        if key in GLOBAL_KEYS: out[key]=copy.deepcopy(values[0])
        elif torch.is_tensor(values[0]):
            if key.startswith('scene_object_') and values[0].ndim>=3:
                padded=[]
                for value in values:
                    padding=torch.zeros((len(value),scene_width-value.shape[1],*value.shape[2:]),dtype=value.dtype)
                    if key == 'scene_object_quat': padding[...,0] = 1.
                    padded.append(torch.cat((value,padding),dim=1))
                values=padded
            out[key]=torch.cat(values) if values[0].ndim else values[0].clone()
        elif isinstance(values[0],list): out[key]=sum(values,[])
        else: out[key]=copy.deepcopy(values[0])
    out['length_starts']=out['motion_num_frames'].cumsum(0)-out['motion_num_frames']
    return out


def select_play_subset(catalog, num_envs, seed):
    """Playback only: one randomly selected motion per sampled layout."""
    import tempfile, os, shutil
    if num_envs < 1: raise ValueError('num_envs must be positive')
    generator=torch.Generator().manual_seed(seed)
    selected_groups=torch.randperm(len(catalog['groups']),generator=generator)[:num_envs].tolist()
    indices=[]
    for group in selected_groups:
        choices=catalog['groups'][group]
        indices.append(choices[int(torch.randint(len(choices),(1,),generator=generator))])
    source=catalog['path']; signature=hashlib.sha256((str(source)+str(source.stat().st_mtime_ns)+str(indices)).encode()).hexdigest()[:20]
    cache=source.parent/'.play-selections';cache.mkdir(exist_ok=True);destination=cache/signature
    if not (destination/'index.json').exists():
        temporary=Path(tempfile.mkdtemp(prefix=signature+'-',dir=cache))
        try:
            write_motion_folders(subset_motion_data(catalog['raw'],indices),temporary)
            for name in ('objects','supports','collision_simplified'):
                if (source.parent/name).exists():(temporary/name).symlink_to(source.parent/name,target_is_directory=True)
            selection=dict(source=str(source),seed=seed,num_envs=num_envs,original_indices=indices,
                           names=[catalog['raw']['motion_filename'][i] for i in indices],layout_indices=selected_groups)
            (temporary/'selection.json').write_text(json.dumps(selection,indent=2))
            try: temporary.rename(destination)
            except FileExistsError: shutil.rmtree(temporary)
        except BaseException:
            shutil.rmtree(temporary,ignore_errors=True);raise
    print(f'Playback subset: sampled {len(indices)} motions/layouts with seed={seed}; manifest {destination/"selection.json"}',flush=True)
    return destination/'index.json'


def save_motion_dataset(raw, path):
    """Canonical writer: stage first, then back up the old data and commit."""
    import datetime, shutil, tempfile
    path=Path(path).resolve();root=path if path.suffix=='' else path.parent
    root.mkdir(parents=True,exist_ok=True)
    backup=root/'backup'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    temporary=Path(tempfile.mkdtemp(prefix='.storage-stage-',dir=root))
    moved=[]
    try:
        write_motion_folders(raw,temporary)
        for name in ('index.json','motions','dataset.json'):
            if (root/name).exists():
                backup.mkdir(parents=True,exist_ok=True);shutil.move(str(root/name),str(backup/name));moved.append(name)
        (temporary/'motions').rename(root/'motions')
        (temporary/'index.json').replace(root/'index.json')
        (temporary/'dataset.json').replace(root/'dataset.json')
    except BaseException:
        for name in moved:
            if (root/name).is_dir():shutil.rmtree(root/name)
            elif (root/name).exists():(root/name).unlink()
            shutil.move(str(backup/name),str(root/name))
        raise
    finally:shutil.rmtree(temporary,ignore_errors=True)
    if path.suffix=='.pt' and path.exists():
        backup.mkdir(parents=True,exist_ok=True);shutil.move(str(path),str(backup/path.name))
    return root/'index.json'
