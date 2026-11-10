"""Enrich a packed motion with all required rigid-object trajectories.

Context JSON maps clip stems to entries with object_id, poses_npz, poses_key,
scale (optional), and provenance. Poses must be aligned T x 4 x 4 transforms in
exactly the packed clip frame/timeline. No pose is guessed from text labels.
"""
import argparse,json,runpy,sys,tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--motion-file',type=Path,required=True)
    p.add_argument('--contexts',type=Path)
    p.add_argument('--fps',type=float,help='Packed sample rate; required when meta.fps is absent')
    a=p.parse_args()
    from dex_track.motion_storage import load_motion_data, save_motion_dataset, resolve_motion_path, write_motion_folders
    from convert_motion_storage import convert_robot, segment_records
    a.motion_file=resolve_motion_path(a.motion_file)
    raw=(torch.load(a.motion_file,map_location='cpu',weights_only=False)
         if a.motion_file.suffix=='.pt' else load_motion_data(a.motion_file))
    meta=raw.setdefault('meta',{})
    if a.fps is not None:
        for recorded in (meta.get('fps'),raw.get('fps')):
            if recorded is not None and float(recorded) != a.fps:
                raise ValueError(f'Packed fps={recorded} differs from --fps={a.fps}')
        meta['fps']=a.fps
    data_api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'dex_track/data.py'))
    fps=data_api['dataset_fps'](raw)
    raw.pop('fps',None)
    contexts=json.loads(a.contexts.read_text()) if a.contexts else {}
    ids=[]
    for i,name in enumerate(raw['motion_filename']):
        held=[]
        for side in ('right','left'):
            slot=int(raw[f'motion_object_slot_{side}'][i]);oid=Path(raw[f'{side}_object_mesh_dirs'][slot]).name
            if oid not in held:held.append(oid)
        ids.append(held)
    raw['motion_scene_object_ids']=ids
    raw['scene_selection_provenance']=[{'source':'explicit hand assignments plus reviewed context trajectories','contexts':contexts.get(n,[])} for n in raw['motion_filename']]
    # First derive existing held-object states through the validated compatibility path.
    raw,_=convert_robot(raw)
    with tempfile.TemporaryDirectory(prefix='dex-scene-base-') as temporary:
        temporary=Path(temporary)
        index=write_motion_folders(raw,temporary)
        for name in ('objects','supports'):
            if (a.motion_file.parent/name).exists():
                (temporary/name).symlink_to((a.motion_file.parent/name).resolve(),target_is_directory=True)
        cat=data_api['load_scene_catalog'](index)
    all_ids=[held+[r['object_id'] for r in contexts.get(name,[]) if r['object_id'] not in held] for held,name in zip(ids,raw['motion_filename'])]
    count=max(map(len,all_ids))
    if count>8:raise ValueError(f'Maximum clip object count {count} exceeds 8')
    total=int(raw['motion_num_frames'].sum())
    state={k:torch.zeros(total,count,w) for k,w in [('pos',3),('quat',4),('lin_vel',3),('ang_vel',3)]}
    state['quat'][...,0]=1
    for k in state:state[k][:,:cat['slots']]=cat['state'][k]
    for i,name in enumerate(raw['motion_filename']):
        start=int(raw['length_starts'][i]);length=int(raw['motion_num_frames'][i]);sl=slice(start,start+length)
        for item in contexts.get(name,[]):
            if item['object_id'] in ids[i]:continue
            poses=np.load(item['poses_npz'])[item['poses_key']]
            if poses.shape!=(length,4,4):raise ValueError(f'{name}: context pose shape {poses.shape}, expected {(length,4,4)}')
            if not item.get('provenance'):raise ValueError('Context pose provenance required')
            j=all_ids[i].index(item['object_id']);r=Rotation.from_matrix(poses[:,:3,:3]);pos=poses[:,:3,3]
            q=r.as_quat()[:,[3,0,1,2]]
            angular=np.zeros((length,3));angular[:-1]=(r[1:]*r[:-1].inv()).as_rotvec()*fps;angular[-1]=angular[-2]
            for key,value in [('pos',pos),('quat',q),('lin_vel',np.gradient(pos,1/fps,axis=0)),('ang_vel',angular)]:state[key][sl,j]=torch.from_numpy(value).float()
    raw['motion_scene_object_ids']=all_ids
    raw.update({f'scene_object_{k}':v for k,v in state.items()})
    segment_records(raw)
    from dex_track.supports import SCHEMA, write_support_asset
    if 'motion_scene_supports' not in raw:
        raw['motion_scene_supports']=[[] for _ in raw['motion_filename']]
    meta['scene_support_schema']=SCHEMA
    for supports in raw['motion_scene_supports']:
        write_support_asset(a.motion_file.resolve().parent, supports)
    save_motion_dataset(raw,a.motion_file)
    print({'clips':len(all_ids),'num_object_slots':count,'counts':list(map(len,all_ids))})


if __name__=='__main__':main()
