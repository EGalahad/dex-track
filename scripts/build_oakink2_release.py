"""Build a portable, checksummed tracking dataset on its data server.

Only explicit reference/mesh/program/training inputs are included. Never include
RGB recordings, SMPL-X arrays, MANO model weights, checkpoints or credentials.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compact', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--audit-v2', type=Path, required=True)
    p.add_argument('--training', action='append', default=[], metavar='NAME=PATH')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    reference = out/'reference'; reference.mkdir(exist_ok=True)
    mappings = [(str(a.compact.resolve()), 'reference'),
                (str((a.compact/'program').resolve()), 'reference/program'),
                (str((a.compact/'object_repair').resolve()), 'reference/object_repair'),
                (str(a.audit_v2.resolve()), 'annotations/audit-v2')]

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k,v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str):
            for old, new in sorted(mappings, key=lambda x: -len(x[0])):
                value = value.replace(old, new)
            if value.startswith(('/data/elijah/', '/home/elijah/')):
                value = 'preparation-source/' + '/'.join(Path(value).parts[-2:])
        return value

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(portable(value), ensure_ascii=False, indent=2)+'\n')

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(source) != sha(target): raise ValueError(f'Existing release file differs: {target}')
            return
        try: os.link(source.resolve(), target)
        except OSError: shutil.copy2(source, target)

    refs = sorted(a.compact.glob('*.npz'))
    if len(refs) != 627: raise ValueError(f'Expected 627 recordings, got {len(refs)}')
    for source in refs: copy(source, reference/source.name)
    write(reference/'manifest.json', json.loads((a.compact/'manifest.json').read_text()))
    for name in ('object_repair','program'):
        target = reference/f'{name}.tar.gz'
        if not target.exists():
            with tarfile.open(target.with_suffix('.partial'), 'w:gz', dereference=True) as tar:
                tar.add((a.compact/name).resolve(), arcname=name)
            target.with_suffix('.partial').replace(target)
    base = json.loads(a.baseline.read_text())
    additions = json.loads((a.audit_v2/'new-preparation-candidates.json').read_text())
    by_id = {r['stage_id']: r for r in base['stages']}
    for row in additions['stages']: by_id[row['stage_id']] = row
    rows = list(by_id.values())
    if len(rows) != 2840: raise ValueError(f'Expected 2840 stages, got {len(rows)}')
    counts = dict(Counter(r['simulation_eligibility']['status'] for r in rows))
    annotation = dict(schema='oakink2-stage-objects-release-v1', source_recordings=627,
        source_stages=2840, simulation_eligibility_counts=counts,
        training_admission=False, human_verified_complete=False,
        source_hashes={'baseline':sha(a.baseline), 'audit_increment':sha(a.audit_v2/'new-preparation-candidates.json')},
        stages=rows)
    write(out/'annotations/stage_objects.json', annotation)
    write(out/'annotations/rigid_candidates.json', {**{k:v for k,v in annotation.items() if k!='stages'},
        'stages':[r for r in rows if r['simulation_eligibility']['status']=='candidate_rigid_pose_tracking']})
    write(out/'annotations/audit_v2_summary.json', json.loads((a.audit_v2/'summary.json').read_text()))
    packages=[]
    import numpy as np
    import torch
    for entry in a.training:
        name, directory = entry.split('=',1)
        if not name.replace('_','').replace('-','').isalnum(): raise ValueError(name)
        directory=Path(directory).resolve(); target=out/'training'/name
        raw=torch.load(directory/'motion.pt',map_location='cpu',weights_only=False)
        fps = raw.get('meta', {}).get('fps')
        if fps is None:
            # Legacy packs may omit metadata; recover it from the exact source
            # clips, never from a task default or an assumed dataset frame rate.
            rates = []
            for clip in raw['motion_filename']:
                with np.load(directory/'retargeted'/f'{clip}.npz', allow_pickle=False) as source:
                    rates.append(float(source['fps']))
            if not rates or len(set(rates)) != 1:
                raise ValueError(f'{name}: inconsistent source clip frame rates')
            fps = rates[0]
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'{name}: invalid frame rate {fps}')
        raw = portable(raw)
        raw.setdefault('meta', {})['fps'] = fps
        required=raw['motion_scene_object_ids']
        if max(map(len,required))>8: raise ValueError(f'{name}: over 8 objects')
        target.mkdir(parents=True, exist_ok=True)
        torch.save(raw, target/'motion.partial.pt')
        (target/'motion.partial.pt').replace(target/'motion.pt')
        for oid in sorted({x for scene in required for x in scene}):
            asset=directory/'objects'/oid
            if not (asset/'visual.obj').is_file(): raise FileNotFoundError(asset)
            for source in sorted(asset.rglob('*')):
                if source.is_file() and source.suffix.lower() in {'.obj','.mtl','.ply','.usd','.usda','.usdc','.npy','.npz','.pt','.json','.png','.jpg'}:
                    copy(source,target/'objects'/oid/source.relative_to(asset))
        packages.append(dict(name=name, clips=raw['motion_filename'], frames=int(raw['motion_num_frames'].sum()), fps=fps))
    write(out/'release.json', dict(schema='dex-track-oakink2-v1', source='kelvin34501/OakInk-v2',
        code='https://github.com/EGalahad/dex-track', reference_recordings=len(refs),
        stage_count=len(rows), selection_counts=counts, training_packages=packages,
        full_corpus_training_ready=False,
        pending='Full-corpus retarget, complete-scene collision/support QA and final training admission'))
    files=[f for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SHA256SUMS']
    (out/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(out)}\n' for f in files))
    print(json.dumps({'files':len(files),'bytes':sum(f.stat().st_size for f in files),
                      'reference_recordings':len(refs),'stages':len(rows),'selection_counts':counts,
                      'training_packages':packages},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
