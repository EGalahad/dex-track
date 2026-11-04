"""Generic full-scene adapter for the unchanged official TopoRetarget solver."""
from pathlib import Path
from dataclasses import asdict
import argparse, hashlib, json, os, time, traceback, shutil
import numpy as np
import mujoco as mj
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from toporetarget.config import load_config, COLLISION_SIDECAR
from toporetarget.solver.hand import HandSpec
from toporetarget.solver.clip import build_retarget_track, derive_input_wrist_pose, CanonicalClip
from toporetarget.solver.run import build_hand_config,world_fingertips,stack_metrics
from toporetarget.solver.session import FrameSolver
from toporetarget.solver.sampling import resolve_sequence_object_points
from toporetarget.solver.selfcol import ObjectClearance,optimize_trajectory
from toporetarget.solver.qp import PenetrationSpec
from toporetarget.solver.materialize import FloatingSceneFK
PIPELINE=Path(__file__).resolve().parent
TASK=Path(os.environ['DEX_PIPELINE_WORK']).resolve()
OLD=PIPELINE
SOURCE=Path(os.environ['DEX_SOURCE_XML']).resolve()
DATA=Path(os.environ['DEX_PREPARED_DATA']).resolve()
FINGERS=('thumb','index','middle','ring','pinky')
NAMES=sorted(f'{f}_{p}' for f in FINGERS for p in ('proximal','intermediate','distal','tip'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def mesh_asset(oid,scale):
 key=oid+'__'+format(float(scale),'.9g'); out=TASK/'assets/objects'/key
 src=DATA/'objects'/oid
 cached=json.loads((out/'source.json').read_text()) if (out/'source.json').exists() else {}
 if not (out/COLLISION_SIDECAR).exists() or cached.get('source')!=str(src):
  out.mkdir(parents=True,exist_ok=True)
  m=trimesh.load(src/'visual.obj',force='mesh',process=False);m.apply_scale(scale);m.export(out/'mesh.stl')
  parts=[]
  for p in sorted((src/'convex').glob('*.obj')):
   dst=out/'convex'/p.name;dst.parent.mkdir(exist_ok=True)

   if scale==1.:shutil.copy2(p,dst)
   else:
    m=trimesh.load(p,force='mesh',process=False);m.apply_scale(scale);m.export(dst)
   parts.append({'type':'mesh','file':str(dst.relative_to(out))})
  if not parts:raise ValueError(f'no convex parts: {src}')
  write_json(out/COLLISION_SIDECAR,{'geoms':parts})
  write_json(out/'source.json',{'source':str(src),'scale':float(scale),'sha256':{str(p.relative_to(src)):sha(p) for p in [src/'visual.obj',*sorted((src/'convex').glob('*.obj'))]}})
 return out/'mesh.stl'
def obstacle_assets(clip):
 # The existing z=0 world ground and only per-motion accepted supports.
 xy=np.concatenate([clip[f'mano_{side}_joints'][...,:2].reshape(-1,2) for side in ('right','left')]+[clip['scene_object_poses'][...,:2,3].reshape(-1,2)])
 half=np.maximum(2.,np.abs(xy).max(axis=0)+1.)
 minz=min(float(clip[f'mano_{side}_joints'][...,2].min()) for side in ('right','left'))
 halfdepth=.05 if minz>=-.05 else (-minz+.5)/2
 specs=[{'id':'ground','geometry':{'type':'box','size':[float(half[0]),float(half[1]),float(halfdepth)]},'local_position':[0,0,-float(halfdepth)],'position':[0,0,0],'quaternion_wxyz':[1,0,0,0]}]
 support=json.loads(str(clip.get('scene_supports_json','{"supports":[]}')))
 specs+=support['supports']; paths=[];poses=[]
 for spec in specs:
  g=spec['geometry']; key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:20];out=TASK/'assets/obstacles'/key
  if not (out/COLLISION_SIDECAR).exists():
   out.mkdir(parents=True,exist_ok=True)
   if g['type']=='box':size=g['size'];mesh=trimesh.creation.box(extents=np.asarray(size)*2)
   elif g['type']=='cylinder':size=[g['radius'],g['height']/2,0];mesh=trimesh.creation.cylinder(radius=g['radius'],height=g['height'])
   else:raise ValueError(g)
   local=spec.get('local_position',[0,0,0]);mesh.apply_translation(local);mesh.export(out/'mesh.stl');write_json(out/COLLISION_SIDECAR,{'geoms':[{'type':g['type'],'size':size,'pos':local,'quat':[1,0,0,0]}]})
  paths.append(out/'mesh.stl');poses.append(spec['position']+spec['quaternion_wxyz'])
 return paths,np.asarray(poses),specs

def convert(q,diagnostics,side,m):
 pre='R' if side=='right' else 'L'; poses=diagnostics['wrist_pose_w'].astype(float); root=m.body(f'{pre}_forearm_pos_x_link');R0=Rotation.from_quat(root.quat[[1,2,3,0]]).as_matrix()
 pos=(poses[:,:3]-root.pos)@R0;Rrel=np.einsum('ij,tjk->tik',R0.T,Rotation.from_quat(poses[:,[4,5,6,3]]).as_matrix());angles=np.unwrap(Rotation.from_matrix(Rrel).as_euler('ZXY'),axis=0);angles[:,2]*=-1
 for j,n in enumerate([f'{pre}_forearm_pos_{a}_joint' for a in 'xyz']+[f'{pre}_forearm_rot_{a}_joint' for a in 'zxy']):q[:,m.joint(n).qposadr[0]]=np.concatenate([pos,angles],axis=1)[:,j]
 for j,n in enumerate(diagnostics['joint_names']):q[:,m.joint(str(n)).qposadr[0]]=diagnostics['reference_qpos'][:,j]

def run(path,output,max_frames=None):
 output.mkdir(parents=True,exist_ok=True);start=time.time();clip=dict(np.load(path,allow_pickle=False));T=len(clip['qpos']);fps=float(clip['fps'])
 if max_frames:
  for k,v in list(clip.items()):
   if v.ndim and len(v)==T:clip[k]=v[:max_frames]
  T=min(T,max_frames)
 ids=list(map(str,clip['scene_object_ids']));scales=clip['scene_object_scales']; assert len(ids)==len(set(ids))
 # Preserve the prior Whiteboard's right/left object order, append all other scene objects.
 order=list(dict.fromkeys([str(clip[f'obj_{s}_id']) for s in ('right','left')]+ids));ix=[ids.index(o) for o in order]
 paths=[mesh_asset(o,float(scales[i])) for o,i in zip(order,ix)]
 mat=clip['scene_object_poses'][:,ix];poses=np.concatenate([mat[:,:,:3,3],Rotation.from_matrix(mat[:,:,:3,:3].reshape(-1,3,3)).as_quat().reshape(T,len(ids),4)[:,:,[3,0,1,2]]],axis=-1)
 obs_paths,obs_pose,specs=obstacle_assets(clip);all_paths=paths+obs_paths;all_poses=np.concatenate([poses,np.broadcast_to(obs_pose,(T,*obs_pose.shape))],axis=1)
 summary={'clip':path.stem,'source_sha256':sha(path),'frames':T,'fps':fps,'objects':order,'scales':[float(scales[i]) for i in ix],'obstacles':specs,'solver_source_modified':True,'self_collision_geometry':'existing mesh convex hulls; cross-finger pairs only; 1mm per-side distance margin','graph_object_policy':'associated-only; every other object remains collision-only; samples identical to full-scene baseline','adapter_sha256':sha(__file__),'physical_admission':False,'hands':{}}
 q=np.zeros((T,36));model=mj.MjModel.from_xml_path(str(SOURCE))
 for side in ('right','left'):
  dest=output/f'{side}_diagnostics.npz'
  if dest.exists():
   diag=dict(np.load(dest)); assert len(diag['wrist_pose_w'])==T
  else:
   config=load_config(OLD/f'config/xhand_{side}.json');hand=HandSpec.from_config(config);cfg=build_hand_config(config)
   points=np.stack([clip[f'mano_{side}_wrist_pos']]+[clip[f'mano_{side}_joints'][:,NAMES.index(f'{f}_{p}')] for f in FINGERS for p in ('proximal','intermediate','distal','tip')],axis=1).astype(np.float64)
   wrist=derive_input_wrist_pose(points,hand)
   source=build_retarget_track(points,poses[...,:3],poses[...,[4,5,6,3]],wrist[:,[4,5,6,3]],mesh_paths=list(map(str,paths)))
   samples=resolve_sequence_object_points(source,cfg.object)
   associated_index=order.index(str(clip[f'obj_{side}_id']))
   samples=[x if i==associated_index else np.empty((0,3)) for i,x in enumerate(samples)]
   track=build_retarget_track(points,all_poses[...,:3],all_poses[...,[4,5,6,3]],wrist[:,[4,5,6,3]],mesh_paths=list(map(str,all_paths)))
   pos=np.stack([o.pose.position for o in track.objects],axis=1);quat=np.stack([o.pose.quaternion_xyzw for o in track.objects],axis=1)
   solver=FrameSolver.from_paths(cfg,hand,all_paths,object_points_local=samples+[np.empty((0,3)) for _ in obs_paths]);qp=np.empty((T,18));metrics=[];st=time.time()
   for f in range(T):
    qp[f]=solver.step_palm(track.landmarks[f],pos[f],quat[f]);metrics.append(dict(solver.retargeter.runtime.optimizer.frame_np_metrics))
    if f%100==0:write_json(output/'progress.json',{'side':side,'frame':f,'frames':T,'pid':os.getpid(),'updated':time.time()})
   pre=qp.copy();clearance=ObjectClearance(hand=solver.retargeter.runtime.hand,wrist_qpos=qp[:,:6].copy(),object_position=pos,object_quaternion_xyzw=quat,spec=PenetrationSpec(enabled=True,tolerance=cfg.penetration.tolerance,query_threshold=cfg.mesh.interaction_radius_m))
   qp[:,6:],info=optimize_trajectory(qp[:,6:],hand=hand,config=cfg.selfcol,clearance=clearance)
   fk=FloatingSceneFK(hand,paths,contact_margin_m=cfg.mesh.contact_margin_m);material=fk.materialize(qp,wrist,poses)
   canonical=CanonicalClip(path.stem,side,fps,tuple(order),tuple(paths),tuple(p.parent for p in paths),points,poses,wrist,0,T,T,{})
   tips=world_fingertips(solver.retargeter.runtime.hand,qp,canonical)
   diag=dict(material,qpos_solver=qp,qpos_before_selfcol=pre,tip_error_m=np.linalg.norm(tips-points[:,[4,8,12,16,20]],axis=-1),input_wrist_pose_w=wrist,object_names=np.array(order),hand_spec_json=np.array(hand.to_json()),hand_config_json=np.array(json.dumps(asdict(cfg))),selfcol_phi_before_m=info['phi_before'],selfcol_phi_after_m=info['phi_after'],solve_time_s=np.array(time.time()-st))
   diag.update({'metric_'+k:v for k,v in stack_metrics(metrics).items()});np.savez_compressed(dest,**diag)
  assert all(np.isfinite(v).all() for v in diag.values() if np.asarray(v).dtype.kind in 'fc')
  convert(q,diag,side,model)
  data=mj.MjData(model);pre='R' if side=='right' else 'L';pe=[];re=[];limit=0.
  for j,n in enumerate(diag['joint_names']):
   joint=model.joint(str(n));values=diag['reference_qpos'][:,j];limit=max(limit,float(np.maximum(joint.range[0]-values,values-joint.range[1]).max()))
  for f in range(T):
   data.qpos[:]=q[f];mj.mj_forward(model,data);b=data.body(f'{pre}_forearm_rot_y_link');target=diag['wrist_pose_w'][f]
   pe.append(np.linalg.norm(b.xpos-target[:3]));re.append((Rotation.from_matrix(b.xmat.reshape(3,3)).inv()*Rotation.from_quat(target[[4,5,6,3]])).magnitude())
  assert max(pe)<1e-6 and max(re)<1e-6 and limit<1e-6,(max(pe),max(re),limit)
  summary['hands'][side]={'tip_error_mean_m':float(diag['tip_error_m'].mean()),'solve_time_s':float(diag['solve_time_s']),'wrist_roundtrip_pos_max_m':float(max(pe)),'wrist_roundtrip_rot_max_rad':float(max(re)),'joint_limit_violation_max_rad':limit}

 for k in list(clip):
  if k.startswith('ik_'):clip['source_flash_'+k]=clip.pop(k)
 clip['qpos']=q.astype(np.float32)
 for side,pre in [('right','R'),('left','L')]:
  target=np.load(output/f'{side}_diagnostics.npz')['wrist_pose_w'];data=mj.MjData(model);errors=[];angles=[]
  for f in range(T):
   data.qpos[:]=clip['qpos'][f];mj.mj_forward(model,data);b=data.body(f'{pre}_forearm_rot_y_link');errors.append(np.linalg.norm(b.xpos-target[f,:3]));angles.append((Rotation.from_matrix(b.xmat.reshape(3,3)).inv()*Rotation.from_quat(target[f,[4,5,6,3]])).magnitude())
  # The distance on SO(3) is bounded by the sum of the three Euler angle
  # quantization errors. Unwrapped coordinates can exceed 2*pi, so a fixed
  # 1e-6 rad bound can be smaller than ordinary float32 quantization.
  ri=[int(model.joint(f'{pre}_forearm_rot_{axis}_joint').qposadr[0]) for axis in 'zxy']
  pi=[int(model.joint(f'{pre}_forearm_pos_{axis}_joint').qposadr[0]) for axis in 'xyz']
  quant=abs(clip['qpos'].astype(np.float64)-q)
  rot_bound=quant[:,ri].sum(axis=1)+summary['hands'][side]['wrist_roundtrip_rot_max_rad']+1e-10
  pos_bound=np.linalg.norm(quant[:,pi],axis=1)+summary['hands'][side]['wrist_roundtrip_pos_max_m']+1e-10
  assert np.all(np.asarray(angles)<=rot_bound) and np.all(np.asarray(errors)<=pos_bound),(side,max(errors),max(angles),max(pos_bound),max(rot_bound))
  assert max(errors)<1e-6 and max(angles)<1e-5,(side,max(errors),max(angles))
  summary['hands'][side].update(float32_rotation_quantization_bound_max_rad=float(max(rot_bound)),float32_position_quantization_bound_max_m=float(max(pos_bound)))
  summary['export_validation_revision']='v2-per-frame-float32-quantization-bound'
  summary['hands'][side].update(export_float32_wrist_pos_max_m=float(max(errors)),export_float32_wrist_rot_max_rad=float(max(angles)))
 clip['toporetarget_provenance_json']=np.array(json.dumps(summary));np.savez_compressed(output/'retargeted.npz',**clip)

 # Explicit 50Hz view; static/full-recording context stays untouched.
 target=np.arange(max(1,int(np.floor((T-1)*50/fps+1e-9))+1))/50.;source=np.arange(T)/fps;view=dict(clip)
 for k,v in clip.items():
  if k=='qpos' or k.startswith('mano_') or k in ('obj_right_pose','obj_left_pose','scene_object_poses'):
   if not v.ndim or len(v)!=T:continue
   if v.shape[-2:]==(3,3) or v.shape[-2:]==(4,4):
    shape=v.shape;flat=v.reshape(T,-1,shape[-2],shape[-1]);out=np.broadcast_to(np.eye(shape[-1]),(len(target),flat.shape[1],shape[-2],shape[-1])).copy()
    for j in range(flat.shape[1]):
     out[:,j,:3,:3]=Slerp(source,Rotation.from_matrix(flat[:,j,:3,:3]))(target).as_matrix()
     if shape[-1]==4:out[:,j,:3,3]=np.stack([np.interp(target,source,c) for c in flat[:,j,:3,3].T],axis=-1)
    view[k]=out.reshape(len(target),*shape[1:]).astype(v.dtype)
   else:
    flat=v.reshape(T,-1);view[k]=np.stack([np.interp(target,source,c) for c in flat.T],axis=-1).reshape(len(target),*v.shape[1:]).astype(v.dtype)
 view['fps']=np.array(50.,dtype=np.float32);np.savez_compressed(output/'retargeted_50hz.npz',**view)
 summary.update(stage='completed',elapsed_s=time.time()-start,qpos_finite=bool(np.isfinite(q).all()),output_sha256=sha(output/'retargeted.npz'));write_json(output/'verification.json',summary);print(json.dumps(summary),flush=True)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('clip',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int);a=p.parse_args();run(a.clip,a.output,a.max_frames)
