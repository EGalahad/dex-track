import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,csv,xml.etree.ElementTree as ET
import numpy as np,torch,mujoco as mj,trimesh
from scipy.spatial import ConvexHull
T=Path(os.environ['DEX_CLEANUP_WORK']).resolve().parent;R=Path(os.environ['DEX_PROJECT_ROOT']).resolve();A=Path(os.environ['DEX_AA_ROOT']).resolve();D=Path(os.environ['DEX_PROJECT']).resolve();S=Path(os.environ['DEX_DATASET']).resolve()
sys.path[:0]=[str(A),str(D),str(D)];torch.set_num_threads(1)
import active_adaptation as aa
aa.set_backend('mjlab')
from dex_track.data import load_scene_catalog
from dex_track.tasks.motion import load_motion_dataset
def simplified_object_collision(p):
 target=D/'assets/collision_simplified/objects'/str(p).split('/objects/',1)[1]
 assert target.is_file(),target
 return target
cat=load_scene_catalog(S);raw=cat['raw'];assert len(raw['motion_filename'])==704
orig=load_scene_catalog(Path(os.environ['DEX_DATASET']).resolve())['raw']
for k in ['motion_filename','motion_scene_supports','motion_scene_object_ids','motion_scene_object_scales']:assert raw[k]==orig[k],k
(T/'artifacts/source-manifest.json').write_text(json.dumps(dict(names=raw['motion_filename'],objects=raw['motion_scene_object_ids'],scales=raw['motion_scene_object_scales'],supports=raw['motion_scene_supports']),indent=2))
same={}
for k,v in raw.items():
 if torch.is_tensor(v):assert torch.equal(v,orig[k]),k;same[k]=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()
del orig
mapping=json.loads((D/'assets/collision_simplified/explicit-mesh-map.json').read_text());src=D/'third_party/FlashDexRetarget/assets/robot/xhand/bimanual.xml';tree=ET.parse(src);tree.find('compiler').set('meshdir',str(src.parent/'meshes'))
assets=tree.find('asset');added=set()
for geom in tree.iter('geom'):
 if (geom.get('name') or '').startswith('collision_hand_'):
  old=geom.get('mesh');name='simplified_'+old
  if name not in added:ET.SubElement(assets,'mesh',name=name,file=str(D/'assets/collision_simplified'/mapping['hands'][old]));added.add(name)
  geom.set('mesh',name)
m=mj.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'));d=mj.MjData(m);assert m.nq==36
geoms=[i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('collision_hand_')];verts={}
for i in geoms:
 mid=m.geom_dataid[i];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].astype(float);verts[i]=v[ConvexHull(v).vertices]
ds=load_motion_dataset(raw,scene_state=cat['state'],step_dt=.02,joint_order=list(range(36)),device='cpu')
O=T/'artifacts';(O/'frames').mkdir(parents=True,exist_ok=True);cache={};mesh_sha={};rows=[];start=time.time();thresholds=[.1,1,5,10,20,50]

def hull(oid):
 if oid not in cache:
  parts=sorted((S/'objects'/oid/'convex').glob('*.obj'));assert parts
  paths=[simplified_object_collision(p) for p in parts];vs=np.concatenate([np.asarray(trimesh.load(p,force='mesh',process=False).vertices) for p in paths]);cache[oid]=vs[ConvexHull(vs).vertices]
  for p in paths:mesh_sha[str(p.relative_to(D))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return cache[oid]
def stats(z):
 depth=np.maximum(0,-z)*1000;f=int(np.argmax(depth));return dict(max_mm=float(depth[f]),peak_frame=f,peak_s=f*.02,min_z_mm=float(z.min()*1000),frames={str(t):int((depth>=t).sum()) for t in thresholds})
for mi,name in enumerate(raw['motion_filename']):
 st=int(ds.starts[mi]);N=int(ds.lengths[mi]);sl=slice(st,st+N);q=ds.reset.joint_pos[sl].numpy();objpos=ds.objects.pos_w[sl].numpy();objquat=ds.objects.quat_w[sl].numpy()
 from scipy.spatial.transform import Rotation
 nr=len(raw['motion_scene_object_ids'][mi]);oz=np.empty((N,nr));ol=[]
 for j,(oid,scale) in enumerate(zip(raw['motion_scene_object_ids'][mi],raw['motion_scene_object_scales'][mi])):
  rot=Rotation.from_quat(objquat[:,j,[1,2,3,0]]).as_matrix();oz[:,j]=(rot[:,2]@(hull(oid)*float(scale)).T).min(1)+objpos[:,j,2];ol.append(dict(id=oid,scale=float(scale),min_z_mm=float(oz[:,j].min()*1000),start_min_z_mm=float(oz[0,j]*1000)))
 gp=np.empty((N,len(geoms),3));gz=np.empty_like(gp)
 for f in range(N):d.qpos[:]=q[f];mj.mj_kinematics(m,d);gp[f]=d.geom_xpos[geoms];gz[f]=d.geom_xmat[geoms,6:9]
 hz=np.full((N,2),np.inf);which=np.full((N,2),-1,dtype=int)
 for j,i in enumerate(geoms):
  low=(gz[:,j]@verts[i].T).min(1)+gp[:,j,2];side=0 if 'hand_right_' in m.geom(i).name else 1;mask=low<hz[:,side];hz[mask,side]=low[mask];which[mask,side]=i
 z=dict(right=hz[:,0],left=hz[:,1],hand=hz.min(1),object=oz.min(1));lift=max(0.,-float(z['object'].min()));after={k:v+lift for k,v in z.items()};assert after['object'].min()>=-1e-10
 residual=max(0.,-float(after['hand'].min()));extra=max(0.,residual-.001);supports=raw['motion_scene_supports'][mi]
 linear=ds.objects.lin_vel_w[sl,:nr].numpy();angular=ds.objects.ang_vel_w[sl,:nr].numpy();stationary=(np.linalg.norm(linear,axis=2)<.03)&(np.linalg.norm(angular,axis=2)<.15);near=(oz+lift<.005);rest=int(np.any(stationary&near,axis=1).sum())
 worst=stats(z['hand'])['peak_frame'];side=int(np.argmin(hz[worst]));geom=m.geom(int(which[worst,side])).name
 if residual<=.001:category='no_ground_lift_needed_with_1mm_tolerance'
 elif residual<=.005:category='minor_1_to_5mm_user_previously_accepted'
 elif supports:category='existing_support_and_retarget_need_review'
 elif rest>=10:category='lift_and_support_candidate_requires_geometry_review'
 else:category='retarget_or_height_review_support_not_established'
 row=dict(index=mi,name=name,frames=N,source_native_frames=int(raw['motion_num_frames'][mi]),object_lift_mm=lift*1000,signed_object_touch_ground_translation_mm=-float(z['object'].min())*1000,extra_lift_for_hand_1mm_tolerance_mm=extra*1000,support_count=len(supports),supports=supports,stationary_object_near_ground_frames=rest,objects=ol,before={k:stats(v) for k,v in z.items()},after_object_lift={k:stats(v) for k,v in after.items()},worst_hand_geom=geom,recommendation=category,physical_admission=False)
 rows.append(row);np.savez_compressed(O/'frames'/f'{mi:04d}.npz',right_z=z['right'],left_z=z['left'],object_z=oz,qpos=q,object_pos=objpos[:,:nr],object_quat=objquat[:,:nr],object_lift_m=lift)
 if mi%25==0:print(mi,N,round(time.time()-start,1),flush=True)
 (O/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=704,elapsed_s=time.time()-start)))
(O/'per-motion.json').write_text(json.dumps(rows,indent=2))
summary=dict(motions=704,frames=sum(r['frames'] for r in rows),fps=50,ground_z_m=0,dataset=str(S),dataset_index_sha256=hashlib.sha256((S/'index.json').read_bytes()).hexdigest(),original_and_simplified_reference_tensors_exact=True,tensor_sha256=same,existing_support_motions=sum(r['support_count']>0 for r in rows),support_recommendation_counts={c:sum(r['recommendation']==c for r in rows) for c in sorted(set(r['recommendation'] for r in rows))},source_modified=False,physical_admission=False,categories={})
for stage in ['before','after_object_lift']:
 summary['categories'][stage]={k:dict(max_mm=max(r[stage][k]['max_mm'] for r in rows),thresholds={str(th):dict(motions=sum(r[stage][k]['frames'][str(th)]>0 for r in rows),frames=sum(r[stage][k]['frames'][str(th)] for r in rows)) for th in thresholds}) for k in ['right','left','hand','object']}
for field in ['object_lift_mm','extra_lift_for_hand_1mm_tolerance_mm']:
 v=np.array([r[field] for r in rows]);summary[field]=dict(positive_motions=int((v>1e-7).sum()),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),max=float(v.max()))
summary['object_global_min_z_bins_motions']={label:sum(lo<=r['before']['object']['min_z_mm']<hi for r in rows) for label,lo,hi in [('below_minus1mm',-1e9,-1),('within_1mm',-1,1),('above1to5mm',1,5),('above5to20mm',5,20),('above20mm',20,1e9)]}
(O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'mesh-sha256.json').write_text(json.dumps(mesh_sha,indent=2))
with (O/'per-motion.csv').open('w') as f:
 fields=['index','name','object_lift_mm','extra_lift_for_hand_1mm_tolerance_mm','support_count','stationary_object_near_ground_frames','hand_before_mm','hand_after_object_lift_mm','object_before_mm','recommendation'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in rows:w.writerow({**{k:r[k] for k in fields if k in r},'hand_before_mm':r['before']['hand']['max_mm'],'hand_after_object_lift_mm':r['after_object_lift']['hand']['max_mm'],'object_before_mm':r['before']['object']['min_z_mm']})
print(json.dumps({k:v for k,v in summary.items() if k!='tensor_sha256'},indent=2),flush=True)
