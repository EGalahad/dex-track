"""Attach cropped complete scenes and deduplicate hand mesh slots in a packed corpus.

Run after the official retarget/pack stages. Reads scene trajectories from the
retarget outputs (already cropped), never from the original recording timeline.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source import source_root
sys.path.insert(0,str(source_root()/'src'))
from retarget.human_demo import clips as C
from retarget.build_motion_pt import angular_velocity, velocity


def deduplicate_mesh_slots(raw):
    for side in ('right','left'):
        dirs=raw[f'{side}_object_mesh_dirs']
        scales=raw[f'{side}_object_mesh_scales']
        masses=raw.get(f'{side}_object_masses',[0.]*len(dirs))
        unique={}; mapping=[]
        for key in zip(dirs,scales,masses):
            mapping.append(unique.setdefault(key,len(unique)))
        old=raw[f'motion_object_slot_{side}'].long()
        raw[f'motion_object_slot_{side}']=torch.tensor(mapping,dtype=torch.long)[old]
        raw[f'{side}_object_mesh_dirs']=[x[0] for x in unique]
        raw[f'{side}_object_mesh_scales']=[x[1] for x in unique]
        if f'{side}_object_masses' in raw:
            raw[f'{side}_object_masses']=[x[2] for x in unique]


def attach_scenes(raw, retargeted, vel_sigma=2.):
    scenes=[]; all_ids=[]; all_scales=[]; rates=set()
    for name,length in zip(raw['motion_filename'],raw['motion_num_frames'].tolist()):
        clip=C.load_clip(retargeted/f'{name}.npz')
        ids=list(map(str,clip['scene_object_ids']))
        pose=clip['scene_object_poses']
        scales=clip['scene_object_scales'].tolist()
        if len(ids)!=len(set(ids)) or not 1<=len(ids)<=8:
            raise ValueError(f'{name}: invalid complete object set {ids}')
        if pose.shape!=(length,len(ids),4,4) or not np.isfinite(pose).all():
            raise ValueError(f'{name}: retargeted scene timeline/shape mismatch')
        for side in ('right','left'):
            if not np.allclose(pose[:,ids.index(C.obj_id(clip,side))],C.obj_pose(clip,side),atol=1e-6):
                raise ValueError(f'{name}: {side} object and scene disagree')
        rates.add(float(clip['fps']));scenes.append(pose);all_ids.append(ids);all_scales.append(scales)
    if len(rates)!=1:
        raise ValueError(f'Packed corpus sample rates must agree: {rates}')
    fps=rates.pop();n=max(map(len,all_ids));total=int(raw['motion_num_frames'].sum())
    arrays={k:torch.zeros(total,n,w) for k,w in [('pos',3),('quat',4),('lin_vel',3),('ang_vel',3)]}
    arrays['quat'][...,0]=1
    for pose,start,length in zip(scenes,raw['length_starts'].tolist(),raw['motion_num_frames'].tolist()):
        sl=slice(start,start+length)
        for j in range(pose.shape[1]):
            pos,rot=pose[:,j,:3,3],pose[:,j,:3,:3].astype(np.float64)
            values={'pos':pos,'quat':Rotation.from_matrix(rot).as_quat()[:,[3,0,1,2]],
                    'lin_vel':velocity(pos,1/fps,vel_sigma),'ang_vel':angular_velocity(rot,1/fps,vel_sigma)}
            for k,v in values.items():arrays[k][sl,j]=torch.as_tensor(v,dtype=torch.float32)
    raw['motion_scene_object_ids']=all_ids
    raw['motion_scene_object_scales']=all_scales
    raw['scene_selection_provenance']=[{'source':'audited source selection; complete scene carried through retarget crop',
        'retargeted_clip':str(retargeted/f'{name}.npz')} for name in raw['motion_filename']]
    raw.update({f'scene_object_{k}':v for k,v in arrays.items()})
    raw.setdefault('meta',{}).update(fps=fps,scene_schema='complete-cropped-rigid-scenes-v1')
    deduplicate_mesh_slots(raw)
    return raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion-file',type=Path,required=True)
    p.add_argument('--retargeted',type=Path,required=True)
    args=p.parse_args()
    raw=torch.load(args.motion_file,map_location='cpu',weights_only=False)
    cfg=OmegaConf.load(source_root()/'config/retarget/kinematic_retargeting.yaml')
    attach_scenes(raw,args.retargeted,float(cfg.pack.vel_sigma))
    tmp=args.motion_file.with_suffix('.partial.pt');torch.save(raw,tmp);tmp.replace(args.motion_file)
    from dex_track.data import load_scene_catalog
    c=load_scene_catalog(args.motion_file)
    print({'clips':len(c['mask']),'slots':c['slots'],'layouts':len(c['layouts']),
           'hand_meshes':{s:len(raw[f'{s}_object_mesh_dirs']) for s in ('right','left')}})


if __name__=='__main__':main()
