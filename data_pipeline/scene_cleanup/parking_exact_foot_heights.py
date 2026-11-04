import os
"""Exact convex-polyhedron/vertical-box intersection raises each foot safely."""
from pathlib import Path
code=Path(__file__).with_name('parking_process.py').read_text();exec(code.split('OUT.mkdir(exist_ok=True)')[0])
from scipy.spatial import ConvexHull
O=T/'artifacts/parking-v1/exact-foot-heights';O.mkdir(exist_ok=True);reports=[]
def column_min(piece,xy,radius):
 w,edges,normals,offsets=piece;lo=xy-radius;hi=xy+radius;N=len(w)
 if not np.any(np.all((w[:,:,:2].min(1)<=hi)&(w[:,:,:2].max(1)>=lo),axis=-1)):return np.inf
 best=np.full(N,np.inf)
 inside=np.all((w[:,:,:2]>=lo-1e-10)&(w[:,:,:2]<=hi+1e-10),axis=-1);best=np.minimum(best,np.where(inside,w[:,:,2],np.inf).min(1));a=w[:,edges[:,0]];b=w[:,edges[:,1]]
 for axis in [0,1]:
  other=1-axis;den=b[:,:,axis]-a[:,:,axis]
  for bound in [lo[axis],hi[axis]]:
   tt=np.divide(bound-a[:,:,axis],den,out=np.full_like(den,np.nan),where=abs(den)>1e-12);oo=a[:,:,other]+tt*(b[:,:,other]-a[:,:,other]);zz=a[:,:,2]+tt*(b[:,:,2]-a[:,:,2]);valid=(tt>=-1e-10)&(tt<=1+1e-10)&(oo>=lo[other]-1e-10)&(oo<=hi[other]+1e-10);best=np.minimum(best,np.where(valid,zz,np.inf).min(1))
 nz=normals[:,:,2]
 for x in [lo[0],hi[0]]:
  for y in [lo[1],hi[1]]:
   term=normals[:,:,0]*x+normals[:,:,1]*y+offsets;z=np.divide(-term,nz,out=np.zeros_like(nz),where=abs(nz)>1e-12);lower=np.where(nz< -1e-12,z,-np.inf).max(1);upper=np.where(nz>1e-12,z,np.inf).min(1);parallel=np.all((abs(nz)>1e-12)|(term<=1e-9),axis=1);valid=parallel&(lower<=upper+1e-9);best=np.minimum(best,np.where(valid,lower,np.inf))
 return float(best.min())
selection=[int(v) for v in os.environ['EXACT_INDICES'].split(',')] if os.environ.get('EXACT_INDICES') else [326,119,174,674]+[i for i in range(704) if i not in [326,119,174,674]]
for mi in selection:
 path=OUT/idx['motions'][mi]['path'];meta=json.loads(path.read_text());sha=hashlib.sha256(path.read_bytes()).hexdigest();done=O/f'{mi:04d}.json'
 if done.exists() and json.loads(done.read_text()).get('output_metadata_sha256')==sha:reports.append(json.loads(done.read_text()));continue
 feet=[s for s in meta['motion_scene_supports'][0] if s['id'].startswith('ground_foot_')];lift=json.loads((T/'artifacts/parking-v1/motions'/f'{mi:04d}.json').read_text())['total_world_z_lift_m'];changes=[]
 if feet:
  f=parked_frames(mi);N=len(f['qpos']);oldids=[v['id'] for v in rows[mi]['objects']];pieces=[]
  for oid,scale in zip(meta['motion_scene_object_ids'][0],meta['motion_scene_object_scales'][0]):
   j=oldids.index(oid);fullq=f['object_quat'][:,j][:,[1,2,3,0]].astype(float);fullq/=np.linalg.norm(fullq,axis=1,keepdims=True);fullpos=f['object_pos'][:,j]+[0,0,lift];maxradius=float(np.linalg.norm(objverts(oid,float(scale)),axis=1).max());reps=[];epsilon=.00025
   for ti in range(N):
    if reps:
     dot=np.clip(np.abs(fullq[reps]@fullq[ti]),0,1);bound=np.linalg.norm(fullpos[reps]-fullpos[ti],axis=1)+2*maxradius*np.sqrt(np.maximum(0,1-dot*dot))
     if bound.min()<=epsilon:continue
    reps.append(ti)
   cover_error=epsilon if len(reps)<N else 0.;rot=Rot.from_quat(fullq[reps]).as_matrix();pos=fullpos[reps]
   for p in sorted((D/'assets/collision_simplified/objects'/oid/'convex').glob('*.obj')):
    v=np.asarray(trimesh.load(p,force='mesh',process=False).vertices)*scale;hull=ConvexHull(v);edges=np.array(sorted({tuple(sorted((int(a),int(b)))) for face in hull.simplices for a,b in [(face[0],face[1]),(face[1],face[2]),(face[2],face[0])]}));world=np.einsum('tij,vj->tvi',rot,v)+pos[:,None];norm=np.einsum('tij,fj->tfi',rot,hull.equations[:,:3]);off=hull.equations[:,3][None]-np.einsum('tfi,ti->tf',norm,pos);
    if len(world)>1:
     cloud=world.reshape(-1,3);swept=ConvexHull(cloud);compact=cloud[swept.vertices];hh=ConvexHull(compact);world=compact[None];edges=np.array(sorted({tuple(sorted((int(a),int(b)))) for face in hh.simplices for a,b in [(face[0],face[1]),(face[1],face[2]),(face[2],face[0])]}));norm=hh.equations[None,:,:3];off=hh.equations[None,:,3]
    pieces.append((world,edges,norm,off,cover_error))
  gp=np.empty((N,len(hgeom),3));gm=np.empty((N,len(hgeom),3,3))
  for ti in range(N):hd.qpos[:]=f['qpos'][ti];mj.mj_kinematics(hm,hd);gp[ti]=hd.geom_xpos[hgeom]+[0,0,lift];gm[ti]=hd.geom_xmat[hgeom].reshape(-1,3,3)
  hb=np.empty((N,len(hgeom),2,3))
  for gi,g in enumerate(hgeom):v=np.einsum('tij,vj->tvi',gm[:,gi],hv[g])+gp[:,gi,None];hb[:,gi,0]=v.min(1);hb[:,gi,1]=v.max(1)
  for s in feet:
   xy=np.array(s['position'][:2]);g=s['geometry'];radius=g['radius'] if g['type']=='cylinder' else g['size'][0]/2;oldtop=s['position'][2]*2;geomtop=min(column_min(piece[:4],xy,radius+piece[4])-piece[4] for piece in pieces)-.00005;mask=np.all((hb[:,:,0,:2]<=xy+radius)&(hb[:,:,1,:2]>=xy-radius),axis=-1);handtop=float(hb[:,:,0,2][mask].min())-.001 if mask.any() else np.inf;top=min(geomtop,handtop)
   # Both the original full-frame AABB bound and the new covered-frame bound are safe.
   top=max(top,oldtop);assert np.isfinite(top) and top>0;s['position'][2]=top/2;s['geometry']=dict(type='box',size=[float(radius*2),float(radius*2),float(top)]);s['provenance'].update(kind='swept_convex_column_cradle_v4b',previous_conservative_top_m=oldtop,safe_swept_envelope_column_min_m=geomtop+.00005,hand_bound_top_m=handtop if np.isfinite(handtop) else None,object_clearance_m=.00005,rigid_frame_cover_tolerance_m=.00025,frame_cover_convention='translation_plus_2radius_sin_halfangle_bound; XY inflation and Z subtraction',temporal_envelope='convex hull of covered world poses; conservative contains true sweep');changes.append(dict(id=s['id'],old_top_m=oldtop,new_top_m=top,raised_m=top-oldtop))
  meta['motion_scene_support_provenance']=[dict(kind='swept_convex_column_cradle_v4b',physical_admission=False)];validate_supports(meta['motion_scene_supports'][0]);write_support_asset(OUT,meta['motion_scene_supports'][0]);save(path,meta)
 result=dict(index=mi,feet=len(feet),changes=changes,output_metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physical_admission=False);save(done,result);reports.append(result);save(T/'artifacts/parking-v1/exact-foot-state.json',dict(completed=len(reports),total=704,pid=os.getpid(),heartbeat_unix=time.time()))
 if mi%25==0:print(mi,flush=True)
save(T/'artifacts/parking-v1'/('exact-foot-priority-summary.json' if os.environ.get('EXACT_INDICES') else 'exact-foot-summary.json'),dict(motions=704,feet=sum(r['feet'] for r in reports),raised_feet=sum(c['raised_m']>.0001 for r in reports for c in r['changes']),max_raise_m=max([c['raised_m'] for r in reports for c in r['changes']] or [0.]),physical_admission=False));print('exact columns completed',flush=True)
