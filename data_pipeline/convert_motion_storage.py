"""Convert a legacy pack to per-motion folders, optionally selecting distinct held objects."""
import argparse, json, hashlib, os, runpy, shutil
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--distinct-objects',type=int);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'dex_track/motion_storage.py'))
    raw=(torch.load(a.source,map_location='cpu',weights_only=False) if a.source.suffix=='.pt' else api['load_motion_data'](a.source));n=len(raw['motion_filename']);indices=list(range(n));representatives=[]
    if a.distinct_objects is not None:
        if a.distinct_objects<1:raise ValueError('distinct-objects must be positive')
        indices=[];seen=set();g=torch.Generator().manual_seed(a.seed)
        for i in torch.randperm(n,generator=g).tolist():
            slot=int(raw['motion_object_slot_right'][i]);oid=Path(raw['right_object_mesh_dirs'][slot]).name
            if oid in seen:continue
            seen.add(oid);indices.append(i);representatives.append(oid)
            if len(indices)==a.distinct_objects:break
        if len(indices)!=a.distinct_objects:raise ValueError(f'Only {len(indices)} distinct right-hand objects available')
    chosen=api['subset_motion_data'](raw,indices);dest=a.destination.resolve();index=api['write_motion_folders'](chosen,dest)
    source=api['resolve_motion_path'](a.source);objects=sorted({oid for row in chosen['motion_scene_object_ids'] for oid in row})
    def copy_asset(src,dst):
        try:os.link(src,dst)
        except OSError:shutil.copy2(src,dst)
        return str(dst)
    for oid in objects:
        shutil.copytree(source.parent/'objects'/oid,dest/'objects'/oid,copy_function=copy_asset)
    if (source.parent/'supports').is_dir():shutil.copytree(source.parent/'supports',dest/'supports',copy_function=copy_asset)
    backup=dest/'backup';backup.mkdir()
    if source.suffix=='.pt':(backup/'legacy-motion.pt').symlink_to(source.resolve())
    # Reload using only NPZ+JSON, verifying every selected valid trajectory value.
    loaded=api['load_motion_data'](index);total=int(chosen['motion_num_frames'].sum());verified=[]
    for key,value in chosen.items():
        if key in ['motion_object_slot_right','motion_object_slot_left']:continue
        if torch.is_tensor(value) and key not in ['motion_object_slot_right','motion_object_slot_left']:
            if key.startswith('scene_object_'):
                for i,length in enumerate(chosen['motion_num_frames']):
                    start=int(chosen['length_starts'][i]);slots=len(chosen['motion_scene_object_ids'][i]);assert torch.equal(value[start:start+length,:slots],loaded[key][start:start+length,:slots]),key
            else:assert torch.equal(value,loaded[key]),key
            verified.append(key)
        elif key not in api['GLOBAL_KEYS']:assert value==loaded[key],key
    for side in ('right','left'):
        for i in range(len(indices)):
            def entry(d):
                j=int(d[f'motion_object_slot_{side}'][i]);return d[f'{side}_object_mesh_dirs'][j],d[f'{side}_object_mesh_scales'][j]
            assert entry(chosen)==entry(loaded)
    report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),schema=api['SCHEMA'],motions=len(indices),frames=total,seed=a.seed,source_indices=indices,representative_object_ids=representatives,scene_object_ids=objects,verified_tensor_keys=verified,valid_trajectories_exact=True,metadata_and_supports_exact=True,index=str(index))
    (dest/'conversion-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
