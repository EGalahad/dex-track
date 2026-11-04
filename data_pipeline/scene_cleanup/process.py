import os
"""Conservative immutable staging. Never writes canonical data or source assets."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,copy,time,hashlib,xml.etree.ElementTree as E,shutil
import numpy as np,mujoco as mj,trimesh
from scipy.spatial.transform import Rotation as Rot
T=Path(os.environ['DEX_CLEANUP_WORK']).resolve();AUD=T.parent;R=Path(os.environ['DEX_PROJECT_ROOT']).resolve();S=Path(os.environ['DEX_DATASET']).resolve();OUT=T/'staging';A=Path(os.environ['DEX_AA_ROOT']).resolve();D=Path(os.environ['DEX_PROJECT']).resolve()
import sys
sys.path.insert(0,str(D))
import importlib.util
_spec=importlib.util.spec_from_file_location('scene_supports',D/'dex_track/supports.py');_supports=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_supports)
write_support_asset=_supports.write_support_asset;validate_supports=_supports.validate_supports;support_asset_directory=_supports.support_asset_directory
rows=json.loads((AUD/'artifacts/per-motion.json').read_text());idx=json.loads((S/'index.json').read_text());mapping=json.loads((D/'assets/collision_simplified/explicit-mesh-map.json').read_text());src=D/'third_party/FlashDexRetarget/assets/robot/xhand/bimanual.xml'
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def handtree():
 tree=E.parse(src);tree.find('compiler').set('meshdir',str(src.parent/'meshes'));root=tree.getroot();asset=tree.find('asset');added=set()
 for tag in ['actuator','contact','equality','keyframe']:
  for el in root.findall(tag):root.remove(el)
 for el in root.iter('geom'):
  el.set('contype','0');el.set('conaffinity','0')
  if el.get('name','').startswith('collision_hand_'):
   old=el.get('mesh');name='simple_'+old
   if name not in added:E.SubElement(asset,'mesh',name=name,file=str(D/'assets/collision_simplified'/mapping['hands'][old]));added.add(name)
   el.set('mesh',name);el.set('contype','1');el.set('conaffinity','1');el.set('group','3')
 return tree
hm=mj.MjModel.from_xml_string(E.tostring(handtree().getroot(),encoding='unicode'));hd=mj.MjData(hm);hgeom=[i for i in range(hm.ngeom) if hm.geom(i).name.startswith('collision_hand_')];hv={i:hm.mesh_vert[hm.mesh_vertadr[hm.geom_dataid[i]]:hm.mesh_vertadr[hm.geom_dataid[i]]+hm.mesh_vertnum[hm.geom_dataid[i]]].astype(float) for i in hgeom};delta=[]
for side in ['R','L']:
 b=hm.body(side+'_forearm_pos_x_link');rot=Rot.from_quat(b.quat[[1,2,3,0]]).as_matrix();cols=[int(hm.joint(side+'_forearm_pos_'+a+'_joint').qposadr[0]) for a in 'xyz'];delta.append((cols,rot.T@np.array([0,0,1.])))
cache={}
def objverts(oid,scale):
 key=(oid,scale)
 if key not in cache:cache[key]=np.concatenate([np.asarray(trimesh.load(p,force='mesh',process=False).vertices) for p in sorted((D/'assets/collision_simplified/objects'/oid/'convex').glob('*.obj'))])*scale
 return cache[key]
def gap(a,b):return np.linalg.norm(np.maximum(0,np.maximum(a[:,0]-b[:,1],b[:,0]-a[:,1])),axis=-1)
def support(id,xy,top,provenance,radius=.004):
 return dict(id=id,fixed=True,coordinate_frame='scene',position=[float(xy[0]),float(xy[1]),float(top/2)],quaternion_wxyz=[1.,0.,0.,0.],geometry=dict(type='cylinder',radius=radius,height=float(top)),friction=.8,provenance=provenance)
OUT.mkdir(exist_ok=True);(OUT/'motions').mkdir(exist_ok=True)
if not (OUT/'objects').exists():(OUT/'objects').symlink_to(S/'objects',target_is_directory=True)
reports=[];start=time.time()
priority=[237,326,174]+[i for i,e in enumerate(idx['motions']) if 'fac912c78b870666a08a' in e['name'] or 'ebf4620c54f1ee2c0e8b' in e['name']]
for mi in priority+[i for i in range(len(idx['motions'])) if i not in priority]:
 entry=idx['motions'][mi]
 reportfile=T/'artifacts/motions'/f'{mi:04d}.json'
 if reportfile.exists():reports.append(json.loads(reportfile.read_text()));continue
 f=S/entry['path'];meta=json.loads(f.read_text());arrays={n:{k:z[k] for k in z.files} for n in ['robot','mano','scene','contact'] for z in [np.load(f.parent/(n+'.npz'))]};scene=arrays['scene'];mano=arrays['mano'];q=arrays['robot']['joint_pos'];audit=np.load(AUD/'artifacts/frames'/f'{mi:04d}.npz');qp=audit['qpos'];N=len(qp);ids=meta['motion_scene_object_ids'][0];scales=meta['motion_scene_object_scales'][0];pos=audit['object_pos'];quat=audit['object_quat'];bounds=[];lowest=[]
 for j,(oid,scale) in enumerate(zip(ids,scales)):
  v=objverts(oid,float(scale));rot=Rot.from_quat(quat[:,j][:,[1,2,3,0]]).as_matrix();world=np.einsum('tij,vj->tvi',rot,v)+pos[:,j,None];bounds.append(np.stack([world.min(1),world.max(1)],1));tt,vv=np.unravel_index(world[:,:,2].argmin(),world[:,:,2].shape);lowest.append((world[tt,vv].copy(),tt,world[tt]))
 ob=np.stack(bounds,1);hb=np.empty((N,len(hgeom),2,3))
 gp=np.empty((N,len(hgeom),3));gm=np.empty((N,len(hgeom),3,3))
 for ti in range(N):
  hd.qpos[:]=qp[ti];mj.mj_kinematics(hm,hd);gp[ti]=hd.geom_xpos[hgeom];gm[ti]=hd.geom_xmat[hgeom].reshape(-1,3,3)
 for j,g in enumerate(hgeom):
  v=np.einsum('tij,vj->tvi',gm[:,j],hv[g])+gp[:,j,None];hb[:,j,0]=v.min(1);hb[:,j,1]=v.max(1)
 handbox=np.stack([hb[:,:,0].min(1),hb[:,:,1].max(1)],1)
 # Explicit validity evidence. An associated hand is never suppressed merely because it is static.
 validity={};inactive=[]
 for side in ['right','left']:
  slot=int(scene['motion_object_slot_'+side][0]);joints=mano['mano_'+side+'_joints'];wr=mano['mano_'+side+'_wrist_pos'];zero=bool(np.max(np.abs(joints))<1e-8 and np.max(np.abs(wr))<1e-8)
  inactive.append(slot<0 and zero);validity[side]=dict(associated_slot=slot,all_zero_source=zero,source_wrist_span_m=float(np.linalg.norm(np.ptp(wr,axis=0))),excluded_from_lift=inactive[-1])
 # Conservative pruning: only wholly separated, motionless, unassociated clutter.
 associated=set()
 for side in ['right','left']:
  slot=int(scene['motion_object_slot_'+side][0])
  if slot>=0:
   oid=Path(meta[side+'_object_mesh_dirs'][slot]).name
   assert oid in ids,(mi,side,oid,ids)
   associated.add(ids.index(oid))
 keep=set(range(len(ids)));evidence=[]
 for j,oid in enumerate(ids):
  moved=float(np.max(np.linalg.norm(pos[:,j]-pos[0,j],axis=1)));angle=float(np.max(Rot.from_quat(quat[:,j][:,[1,2,3,0]]).magnitude())) # rotation change below
  rr=Rot.from_quat(quat[:,j][:,[1,2,3,0]]);angle=float((rr[0].inv()*rr).magnitude().max());hg=float(min(gap(ob[:,j],hb[:,k]).min() for k in range(len(hgeom))));og={ids[k]:float(gap(ob[:,j],ob[:,k]).min()) for k in range(len(ids)) if k!=j};remove=j not in associated and moved<.002 and angle<.02 and hg>.03 and all(v>.02 for v in og.values())
  if remove:keep.remove(j)
  evidence.append(dict(id=oid,associated=j in associated,translation_span_m=moved,rotation_span_rad=angle,conservative_hand_AABB_gap_m=hg,object_AABB_gaps_m=og,removed=remove))
 keep=sorted(keep);lift=max(0.,-float(ob[:,keep,0,2].min()));hz=[]
 for si,key in enumerate(['right_z','left_z']):
  if not inactive[si]:hz.append(float(audit[key].min()))
 residual=max(0.,-min(hz)-lift) if hz else 0.;extra=max(0.,-min(hz)+.001-lift) if residual>.005 else 0.;total=lift+extra
 supports=copy.deepcopy(meta.get('motion_scene_supports',[[]])[0]);existing=len(supports);candidate=[]
 # Existing supports retain their contact tops, but get ground-connected pedestal extensions.
 for si,s in enumerate(supports):
  s['position'][2]+=total;g=s['geometry'];h=g['size'][2] if g['type']=='box' else g['height'];upright=np.allclose(s['quaternion_wxyz'],[1,0,0,0],atol=1e-5);bottom=s['position'][2]-h/2
  if upright and bottom>1e-5:candidate.append(support('ground_extension_'+str(si),s['position'][:2],bottom,dict(kind='existing_support_ground_extension',parent=s['id']),radius=min(.01,g.get('radius',.01))))
 # A support foot reaches the globally lowest point of each formerly grounded object.
 # This top cannot intersect the object anywhere in the reference; full collision audit below.
 if extra>0:
  for j in keep:
   point,ti,world=lowest[j];top=float(point[2]+total)
   if top<=.001 or point[2]+lift>.015:continue
   points=world[world[:,2]<point[2]+.0015];chosen=[]
   # Spread up to three tiny feet along genuinely low collision vertices.
   for v in points:
    if not chosen or min(np.linalg.norm(v[:2]-c[:2]) for c in chosen)>.02:chosen.append(v)
    if len(chosen)==3:break
   for k,v in enumerate(chosen):candidate.append(support(f'ground_foot_obj{j}_{k}',v[:2],top,dict(kind='lowest_collision_vertex_support',object_id=ids[j],reference_policy_frame=int(ti),uniform_world_z_lift_m=total,foot_top_gap_to_selected_vertex_m=float(v[2]-point[2]))))
 hb[...,2]+=total
 accepted=[];rejected=[]
 # Conservative full-trajectory hand geometry AABB clearance: safe acceptance, false negatives allowed.
 for s in candidate:
  x,y,z=s['position'];rad=s['geometry']['radius'];top=2*z;sb=np.broadcast_to(np.array([[x-rad,y-rad,0],[x+rad,y+rad,top]]),(N,2,3));clear=float(np.min([gap(hb[:,j],sb).min() for j in range(len(hgeom))]));s['provenance']['full_trajectory_hand_AABB_clearance_m']=clear
  if clear>=.001:accepted.append(s)
  else:rejected.append(dict(support=s,reason='conservative_hand_clearance_below_1mm'))
 supports+=accepted;validate_supports(supports)
 # Rewrite every positional reference, preserving derivatives and local topology coordinates.
 for cols,vec in delta:q[:,cols]+=total*vec
 for key,v in mano.items():
  if key.endswith('_wrist_pos') or key.endswith('_joints'):v[...,2]+=total
 for key in ['obj_right_pos','obj_left_pos','scene_object_pos']:scene[key][...,2]+=total
 for key,v in arrays['contact'].items():
  if key.startswith('contact_contact_pos_full_'):v[...,2]+=total
 for key in ['scene_object_pos','scene_object_quat','scene_object_lin_vel','scene_object_ang_vel']:scene[key]=scene[key][:,keep]
 # Side object slots index side mesh_dirs, not scene arrays; preserve both unchanged.
 meta['motion_scene_object_ids']=[[ids[j] for j in keep]];meta['motion_scene_object_scales']=[[scales[j] for j in keep]];meta['motion_scene_supports']=[supports];meta['meta']['physical_admission']=False
 meta['meta']['ground_support_revision']='uniform-lift-conservative-feet-v1';meta['scene_selection_provenance'].append(dict(kind='subsequence_noninteraction_cleanup',removed_ids=[ids[j] for j in range(len(ids)) if j not in keep],evidence=f'artifacts/motions/{mi:04d}.json'))
 meta['motion_scene_support_provenance']=[dict(kind='preserved_and_ground_connected',uniform_world_z_lift_m=total,existing_support_count=existing,added_support_count=len(accepted),rejected_support_count=len(rejected),physical_admission=False)]
 dst=OUT/entry['path'];dst.parent.mkdir(parents=True,exist_ok=True)
 for n,a in arrays.items():
  assert all(np.isfinite(v).all() for v in a.values());np.savez_compressed(dst.parent/(n+'.npz'),**a)
 for k in ['objects']:(dst.parent/k).symlink_to(OUT/k,target_is_directory=True)
 asset=write_support_asset(OUT,supports)
 if supports:(dst.parent/'supports').symlink_to(OUT/'supports',target_is_directory=True)
 save(dst,meta);entry['object_ids']=[ids[j] for j in keep]
 r=dict(index=mi,name=entry['name'],source_metadata_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),uniform_object_lift_m=lift,extra_hand_lift_m=extra,total_world_z_lift_m=total,hand_validity=validity,hand_depth_after_object_lift_m=residual,hand_depth_after_total_lift_m=max(0.,-min(hz)-total) if hz else 0.,object_min_z_after_m=float(ob[:,keep,0,2].min()+total),existing_supports=existing,added_supports=len(accepted),rejected_supports=rejected,object_evidence=evidence,removed_objects=[ids[j] for j in range(len(ids)) if j not in keep],physical_admission=False,status='needs_support_review' if extra>0 and not accepted and not existing else 'staged_geometry_candidate')
 save(reportfile,r);reports.append(r);save(T/'artifacts/state.json',dict(completed=len(reports),total=704,elapsed_s=time.time()-start,pid=os.getpid(),heartbeat_unix=time.time(),stage='staging'))
 if mi%20==0:print(mi,round(time.time()-start,1),flush=True)
save(OUT/'index.json',idx);save(T/'artifacts/manifest.json',reports);summary=dict(motions=len(reports),staging=str(OUT),published=False,canonical_modified=False,physical_admission=False,hand_lift_motions=sum(r['extra_hand_lift_m']>0 for r in reports),added_support_instances=sum(r['added_supports'] for r in reports),motions_with_rejected_support_candidates=sum(bool(r['rejected_supports']) for r in reports),removed_object_instances=sum(len(r['removed_objects']) for r in reports),inactive_hands=sum(v['excluded_from_lift'] for r in reports for v in r['hand_validity'].values()),max_hand_depth_after_m=max(r['hand_depth_after_total_lift_m'] for r in reports),min_object_z_m=min(r['object_min_z_after_m'] for r in reports),needs_support_review=sum(r['status']=='needs_support_review' for r in reports),stage='staged_requires_collision_physics_review',heartbeat_unix=time.time());save(T/'artifacts/summary.json',summary);save(T/'artifacts/state.json',summary);print(json.dumps(summary),flush=True)
