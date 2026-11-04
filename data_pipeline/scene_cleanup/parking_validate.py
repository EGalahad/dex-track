import os
"""Independent folder/loader roundtrip; never publishes without physical acceptance."""
from pathlib import Path
import os,json,hashlib,time,sys,subprocess
subprocess.run([sys.executable,str(Path(__file__).with_name('parking_repair_contact_semantics.py'))],check=True)
subprocess.run([sys.executable,str(Path(__file__).with_name('parking_validate_contact.py'))],check=True)
subprocess.run([sys.executable,str(Path(__file__).with_name('parking_repair_probe_coverage.py'))],check=True)
_task=Path(os.environ['DEX_CLEANUP_WORK']).resolve();_validation=_task/'artifacts/parking-v1/validation.json'
if _validation.exists():
 _previous=json.loads(_validation.read_text());_hashes=_previous.get('revision_files_sha256',{})
 if _previous.get('status')=='passed' and _previous.get('contact_semantics_revision')=='object-local-contact-v2' and _hashes and all((_task/'staging'/rel).is_file() and hashlib.sha256((_task/'staging'/rel).read_bytes()).hexdigest()==digest for rel,digest in _hashes.items()):
  _record=_task/'artifacts/parking-v1/validation-cache-reuses.json';_events=json.loads(_record.read_text()) if _record.exists() else [];_events.append(dict(heartbeat_unix=time.time(),verified_identical_files=len(_hashes),proof='full current file SHA256 equality with previously passed object-local-v2 loader/FK revision'));_record.write_text(json.dumps(_events,indent=2)+'\n');print('Same immutable validated revision; all file SHA256 verified; loader/FK proof reused',flush=True);raise SystemExit(0)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,trimesh
from scipy.spatial import cKDTree
contact_trees={}
T=Path(os.environ['DEX_CLEANUP_WORK']).resolve();R=Path(os.environ['DEX_PROJECT_ROOT']).resolve();S=Path(os.environ['DEX_DATASET']).resolve();O=T/'staging';A=Path(os.environ['DEX_AA_ROOT']).resolve();D=Path(os.environ['DEX_PROJECT']).resolve();sys.path[:0]=[str(A),str(D),str(D)]
import torch
import active_adaptation as aa
aa.set_backend('mjlab')
from dex_track.data import load_scene_catalog
from dex_track.tasks.motion import load_motion_dataset
from dex_track.supports import validate_supports
index=json.loads((O/'index.json').read_text());source=json.loads((S/'index.json').read_text());assert len(index['motions'])==len(source['motions'])==704
report=[];hashes={};errors=[]
for i,e in enumerate(index['motions']):
 f=O/e['path'];src=S/source['motions'][i]['path'];meta=json.loads(f.read_text());old=json.loads(src.read_text());sc=dict(np.load(f.parent/'scene.npz'));before=dict(np.load(src.parent/'scene.npz'));r=json.loads((T/'artifacts/parking-v1/motions'/f'{i:04d}.json').read_text());lift=r['total_world_z_lift_m'];ids=meta['motion_scene_object_ids'][0];kept=[old['motion_scene_object_ids'][0].index(oid) for oid in ids]
 for side in ['right','left']:
  slot=int(sc['motion_object_slot_'+side][0]);assert slot==int(before['motion_object_slot_'+side][0]);oid=Path(meta[side+'_object_mesh_dirs'][slot]).name;assert oid in ids;(ii,)=np.where(np.array(ids)==oid);j=int(ii)
  assert np.allclose(sc['obj_'+side+'_pos'],sc['scene_object_pos'][:,j],atol=2e-6),(i,side,'position association')
  from scipy.spatial.transform import Rotation
  assert np.allclose(Rotation.from_quat(sc['scene_object_quat'][:,j][:,[1,2,3,0]]).as_matrix(),sc['obj_'+side+'_rotmat'],atol=2e-6),(i,side,'orientation association')
  assert np.allclose(sc['obj_'+side+'_vel'],sc['scene_object_lin_vel'][:,j],atol=2e-6),(i,side,'velocity association')
  assert meta[side+'_object_mesh_dirs']==old[side+'_object_mesh_dirs'];assert np.allclose(sc['obj_'+side+'_pos'],before['obj_'+side+'_pos']+[0,0,lift],atol=2e-7)
 for k in ['scene_object_quat','scene_object_lin_vel','scene_object_ang_vel']:assert np.array_equal(sc[k],before[k][:,kept]),(i,k)
 # Native arrays independently verify every world-position reference and unchanged source derivatives.
 robot=dict(np.load(f.parent/'robot.npz'));robot0=dict(np.load(src.parent/'robot.npz'));mano=dict(np.load(f.parent/'mano.npz'));mano0=dict(np.load(src.parent/'mano.npz'));contact=dict(np.load(f.parent/'contact.npz'));contact0=dict(np.load(src.parent/'contact.npz'))
 for side in ['right','left']:
  translation=np.array([0,0,lift])+np.array(r['parking_world_delta_m'].get(side,[0,0,0]))
  for key in mano:
   expected=mano0[key]+translation if key.startswith('mano_'+side+'_') and (key.endswith('_wrist_pos') or key.endswith('_joints')) else None
   if expected is not None:assert np.allclose(mano[key],expected,atol=2e-7),(i,key,'native MANO transform')
  key='contact_contact_pos_full_'+side
  check_tips=mano['mano_'+side+'_joints'][:,[19,3,7,15,11]];check_world=np.einsum('tij,tkj->tki',sc['obj_'+side+'_rotmat'].astype(float),contact[key].astype(float))+sc['obj_'+side+'_pos'][:,None];assert np.allclose(np.linalg.norm(check_tips-check_world,axis=-1),contact['tips_distance_'+side],atol=2e-6),(i,side,'all native object-local contact world distance')
  if side not in r['parking_world_delta_m']:assert np.array_equal(contact[key],contact0[key]),(i,key,'active object-local contact points changed')
  else:
   joint_names=sorted(f'{finger}_{part}' for finger in ['thumb','index','middle','ring','pinky'] for part in ['proximal','intermediate','distal','tip']);tip_cols=[joint_names.index(f'{finger}_tip') for finger in ['thumb','index','middle','ring','pinky']];local_points=contact[key].astype(float);world_points=np.einsum('tij,tkj->tki',sc['obj_'+side+'_rotmat'].astype(float),local_points)+sc['obj_'+side+'_pos'][:,None];distance=np.linalg.norm(world_points-mano['mano_'+side+'_joints'][:,tip_cols],axis=-1);assert np.allclose(distance,contact['tips_distance_'+side],atol=2e-6),(i,side,'parked contact local->world distance');assert np.count_nonzero(contact['contact_contact_'+side])==0
   oid=Path(meta[side+'_object_mesh_dirs'][int(sc['motion_object_slot_'+side][0])]).name;scale=float(meta['motion_scene_object_scales'][0][ids.index(oid)]);tree_key=(oid,scale)
   if tree_key not in contact_trees:
    vertices=np.asarray(trimesh.load(S/'objects'/oid/'visual.obj',force='mesh',process=False,skip_materials=True).vertices,dtype=float)*scale;contact_trees[tree_key]=(vertices,cKDTree(vertices))
   vertices,tree=contact_trees[tree_key];local_tips=np.einsum('tji,tkj->tki',sc['obj_'+side+'_rotmat'].astype(float),mano['mano_'+side+'_joints'][:,tip_cols]-sc['obj_'+side+'_pos'][:,None]);nearest_distance,nearest=tree.query(local_tips.reshape(-1,3));assert np.allclose(nearest_distance.reshape(distance.shape),contact['tips_distance_'+side],atol=2e-6);assert np.allclose(vertices[nearest].reshape(local_points.shape),local_points,atol=2e-7)


 for key in mano:
  if not (key.endswith('_wrist_pos') or key.endswith('_joints')):assert np.array_equal(mano[key],mano0[key]),(i,key,'derivative or rotation changed')
 for key in contact:
  if key.startswith('contact_contact_') and not key.startswith('contact_contact_pos_full_'):assert np.array_equal(contact[key],contact0[key]),(i,key,'source contact flags changed')
  if key.startswith('tips_distance_') and key.split('tips_distance_')[1] not in r['parking_world_delta_m']:assert np.array_equal(contact[key],contact0[key]),(i,key,'active source tip distances changed')
 assert np.array_equal(robot['joint_vel'],robot0['joint_vel']),(i,'joint velocities changed')
 validate_supports(meta['motion_scene_supports'][0]);assert all((f.parent/k).exists() for k in ['objects'])
 for name in ['robot.npz','scene.npz','mano.npz','contact.npz','metadata.json']:
  p=f.parent/name;hashes[str(p.relative_to(O))]=hashlib.sha256(p.read_bytes()).hexdigest()
 report.append(dict(index=i,associated_objects_present=True,side_slots_unchanged=True,scene_sliced_consistently=True))
cat=load_scene_catalog(O);raw=cat['raw'];assert len(raw['motion_filename'])==704;ds=load_motion_dataset(raw,scene_state=cat['state'],step_dt=.02,joint_order=list(range(36)),device='cpu');original=load_scene_catalog(S);baseline=load_motion_dataset(original['raw'],scene_state=original['state'],step_dt=.02,joint_order=list(range(36)),device='cpu');assert torch.equal(ds.lengths,baseline.lengths)
# Explicit loader qpos comparison protects the local-frame root translation transform.
import mujoco as mj
from scipy.spatial.transform import Rotation
code=Path(__file__).with_name('parking_process.py').read_text();ns={'__file__':str(Path(__file__).with_name('parking_process.py'))};exec(code.split('OUT.mkdir(exist_ok=True)')[0],ns);m=ns['hm'];data=mj.MjData(m);geoms=ns['hgeom'];verts=ns['hv'];minimum=1.;hand_minimum=1.
for i in range(704):
 start=int(ds.starts[i]);N=int(ds.lengths[i]);st=int(baseline.starts[i]);rr=json.loads((T/'artifacts/parking-v1/motions'/f'{i:04d}.json').read_text());lift=rr['total_world_z_lift_m']
 # All actual-loader 50Hz joint targets must equal source targets plus only the declared constant root transforms.
 expected=baseline.reset.joint_pos[st:st+N].numpy().copy()
 for side in ['right','left']:ns['world_qshift'](expected,side,np.array([0,0,lift])+np.array(rr['parking_world_delta_m'].get(side,[0,0,0])))
 actual=ds.reset.joint_pos[start:start+N].numpy()
 assert np.allclose(actual,expected,atol=2e-6),(i,'all actual loader reset/future joint targets')
 frame_dir=T/'artifacts/parking-v1/loader-policy-frames';frame_dir.mkdir(exist_ok=True)
 np.savez_compressed(frame_dir/f'{i:04d}.npz',qpos=actual)
 # Independent FK samples remain explicit; all-frame ground/scene audit consumes these actual loader targets.
 for frame in sorted({0,N//2,N-1}):
  data.qpos[:]=ds.reset.joint_pos[start+frame].numpy();mj.mj_kinematics(m,data)
  for g in geoms:
   z=(verts[g]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])[:,2].min();hand_minimum=min(hand_minimum,float(z))
  olddata=mj.MjData(m);olddata.qpos[:]=baseline.reset.joint_pos[st+frame].numpy();mj.mj_kinematics(m,olddata);assert np.allclose(data.geom_xpos[geoms],olddata.geom_xpos[geoms]+[0,0,lift]+np.array([rr.get('parking_world_delta_m',{}).get('right' if 'right_' in m.geom(g).name else 'left',[0,0,0]) for g in geoms]),atol=2e-6),(i,'FK transform')
 hashes['index.json']=hashlib.sha256((O/'index.json').read_bytes()).hexdigest()
result=dict(status='passed',contact_coordinate_frame='object_local',contact_semantics_revision='object-local-contact-v2',supersedes='earlier contact checks incorrectly assumed world frame',motions=704,folder_roundtrip=True,side_associations_consistent=True,loader_frames=int(ds.lengths.sum()),fk_samples=2112,sampled_hand_min_z_m=hand_minimum,physical_admission=False,publish_allowed=False,reason='geometric data verification is not physical support stability admission',revision_files_sha256=hashes)
(T/'artifacts/parking-v1/validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='revision_files_sha256'}),flush=True)
