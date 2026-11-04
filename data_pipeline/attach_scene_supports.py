"""Create an immutable corpus view with optional per-motion fixed supports.

The plan must explicitly cover every motion. This copies existing NPZ members
without recomputing retargets or trajectories, and adds scene_supports_json.
It does not establish physical training admission.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dex_track.supports import (CLIP_KEY, SCHEMA, encode_clip_supports,
                                read_clip_supports, validate_supports, write_support_asset)
from dex_track.data import load_scene_catalog


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    source=a.source.resolve();out=a.output.resolve()
    if out.exists(): raise FileExistsError(out)
    plan=json.loads(a.plan.read_text())
    source_sha=sha(source/'motion.pt')
    if source_sha != plan['source_motion_sha256']:
        raise ValueError('Support plan was prepared for a different motion dataset')
    before=torch.load(source/'motion.pt',map_location='cpu',weights_only=False)
    names=list(before['motion_filename'])
    rows={r['motion']:r for r in plan['motions']}
    if len(rows)!=len(plan['motions']) or set(rows)!=set(names):
        raise ValueError('Plan must cover every motion exactly once')
    for name in names:
        row=rows[name]
        validate_supports(row['supports'])
        path=source/'retargeted'/f'{Path(name).stem}.npz'
        if sha(path)!=row['source_npz_sha256']:
            raise ValueError(f'{name}: source NPZ changed after plan generation')
        with zipfile.ZipFile(path) as z:
            if f'{CLIP_KEY}.npy' in z.namelist():
                raise ValueError(f'{name}: source already has supports; explicit revision needed')
    out.mkdir(parents=True);(out/'retargeted').mkdir()
    for name in names:
        row=rows[name];filename=f'{Path(name).stem}.npz'
        src=source/'retargeted'/filename;dst=out/'retargeted'/filename
        shutil.copyfile(src,dst)
        payload=np.array(encode_clip_supports(row['supports']))
        stream=io.BytesIO();np.lib.format.write_array(stream,payload,allow_pickle=False)
        with zipfile.ZipFile(dst,'a',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(f'{CLIP_KEY}.npy',stream.getvalue())
        # Verify existing compressed payloads byte-for-byte, not just numerical
        # equality after a second serialization. All original arrays are intact.
        with zipfile.ZipFile(src) as old, zipfile.ZipFile(dst) as new, src.open('rb') as fs, dst.open('rb') as fd:
            for member in old.infolist():
                other=new.getinfo(member.filename)
                assert (member.header_offset,member.compress_size,member.CRC)==(other.header_offset,other.compress_size,other.CRC)
                fs.seek(member.header_offset);fd.seek(other.header_offset)
                # ZIP64 local extra fields can differ from central-directory
                # extras. Read local lengths so the entire payload is compared.
                header=fs.read(30)
                name_len,extra_len=struct.unpack('<HH',header[26:30])
                fs.seek(member.header_offset)
                length=30+name_len+extra_len+member.compress_size
                assert fs.read(length)==fd.read(length), (name,member.filename)
        with np.load(dst,allow_pickle=False) as z:
            assert read_clip_supports(z)==row['supports']
        write_support_asset(out,row['supports'])
    for child in source.iterdir():
        if child.name not in {'motion.pt','retargeted','manifest.json','supports'}:
            (out/child.name).symlink_to(child,target_is_directory=child.is_dir())
    raw=dict(before);raw['meta']=copy.deepcopy(before['meta'])
    raw['meta']['scene_support_schema']=SCHEMA
    raw['motion_scene_supports']=[rows[name]['supports'] for name in names]
    raw['motion_scene_support_provenance']=[{k:v for k,v in rows[name].items() if k not in {'motion','supports'}} for name in names]
    torch.save(raw,out/'motion.pt')
    after=torch.load(out/'motion.pt',map_location='cpu',weights_only=False)
    tensors=0
    for key,value in before.items():
        if isinstance(value,torch.Tensor):
            assert torch.equal(value,after[key]),key
            assert torch.isfinite(after[key]).all(),key
            tensors+=1
        elif key!='meta': assert value==after[key],key
    for key,value in before['meta'].items(): assert value==after['meta'][key],key
    cat=load_scene_catalog(out/'motion.pt')
    motion_sha=sha(out/'motion.pt')
    report=dict(schema=SCHEMA,source_dataset=str(source),dataset=str(out),
        source_motion_sha256=source_sha,motion_sha256=motion_sha,
        clips=len(names),frames=int(after['motion_num_frames'].sum()),layouts=len(cat['groups']),
        slots=cat['slots'],unchanged_finite_tensors=tensors,
        clips_with_support=sum(bool(r) for r in raw['motion_scene_supports']),
        support_instances=sum(map(len,raw['motion_scene_supports'])),
        original_npz_members_byte_identical=True,all_clips_explicit_optional_support=True,
        packed_and_clip_supports_equal=True,retarget_recomputed=False,training_admission=False)
    manifest=json.loads((source/'manifest.json').read_text())
    manifest.update(parent_dataset=str(source),parent_motion_sha256=source_sha,
        motion_sha256=motion_sha,layouts=len(cat['groups']),training_admission=False,
        scene_supports={'schema':SCHEMA,'plan':'support-plan.json','verification':'support-verification.json',
                        'kind':'inferred simulation platforms; candidate physics validation'})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (out/'support-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (out/'support-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sha(source/'motion.pt')==source_sha
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
