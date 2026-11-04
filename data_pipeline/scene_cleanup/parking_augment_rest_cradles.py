import os
"""One bounded fallback: support convex-piece centres of failed unheld rest poses."""
from pathlib import Path
code=Path(__file__).with_name('parking_exact_foot_heights.py').read_text();exec(code.split('selection=[int(v)')[0])
import subprocess
subprocess.run([sys.executable,str(Path(__file__).with_name('parking_repair_object_lift_coverage.py'))],check=True)
archive=V/'settle-before-grid-fallback'
if not archive.exists():shutil.copytree(V/'settle',archive)
for name in ['settle-summary.json','validation.json']:
 if (V/name).exists() and not (V/('pre-grid-'+name)).exists():shutil.copy2(V/name,V/('pre-grid-'+name))
O=T/'artifacts/parking-v1/cradle-augmentation';O.mkdir(exist_ok=True);selection=[int(x) for x in os.environ['AUGMENT_INDICES'].split(',')] if os.environ.get('AUGMENT_INDICES') else list(range(704));reports=[]
for mi in selection:
 probe=json.loads((T/'artifacts/parking-v1/settle'/f'{mi:04d}.json').read_text());fail=[v for v in probe['probes'] if not v['passed']];ref=json.loads((T/'artifacts/parking-v1/refinement-rest-v3'/f'{mi:04d}.json').read_text());anchors={d['object_id']:d.get('rest_anchor_policy_frames',[]) for d in ref.get('objects',[])};eligible=[v for v in fail if anchors.get(v['object_id'])];path=OUT/idx['motions'][mi]['path'];meta=json.loads(path.read_text());add=[];rejected=[]
 previous_path=O/f'{mi:04d}.json';coverage_time=json.loads((V/'object-lift-support-coverage-repaired.json').read_text())['heartbeat_unix'];previously_searched=[]
 if previous_path.exists() and previous_path.stat().st_mtime>=coverage_time:
  previous=json.loads(previous_path.read_text());previously_searched=previous.get('completed_grid_search_objects',[oid for oid in previous.get('failed_probe_objects',[]) if oid not in previous.get('excluded_no_unheld_rest_anchor',[])])
 eligible=[v for v in eligible if v['object_id'] not in previously_searched]
 capped=[v['object_id'] for v in eligible if sum('_rest_centre_' in support['id'] and support.get('provenance',{}).get('object_id')==v['object_id'] for support in meta['motion_scene_supports'][0])>=16]
 eligible=[v for v in eligible if v['object_id'] not in capped]
 if eligible:
  f=parked_frames(mi);N=len(f['qpos']);oldids=[v['id'] for v in rows[mi]['objects']];lift=json.loads((T/'artifacts/parking-v1/motions'/f'{mi:04d}.json').read_text())['total_world_z_lift_m'];pieces=[];fullpieces=[];pieceids=[];byobject={}
  for oid,scale in zip(meta['motion_scene_object_ids'][0],meta['motion_scene_object_scales'][0]):
   j=oldids.index(oid);rot=Rot.from_quat(f['object_quat'][:,j][:,[1,2,3,0]]).as_matrix();pos=f['object_pos'][:,j]+[0,0,lift];representatives=np.sort(np.unique(np.c_[f['object_pos'][:,j],f['object_quat'][:,j]],axis=0,return_index=True)[1]);byobject[oid]=[]
   for p in sorted((D/'assets/collision_simplified/objects'/oid/'convex').glob('*.obj')):
    v=np.asarray(trimesh.load(p,force='mesh',process=False).vertices)*scale;h=ConvexHull(v);edges=np.array(sorted({tuple(sorted((int(a),int(b)))) for face in h.simplices for a,b in [(face[0],face[1]),(face[1],face[2]),(face[2],face[0])]}));world=np.einsum('tij,vj->tvi',rot,v)+pos[:,None];norm=np.einsum('tij,fj->tfi',rot,h.equations[:,:3]);off=h.equations[:,3][None]-np.einsum('tfi,ti->tf',norm,pos);fullpieces.append((world,edges,norm,off));pieces.append((world[representatives],edges,norm[representatives],off[representatives]));pieceids.append(oid);byobject[oid].append(world)
  gp=np.empty((N,len(hgeom),3));gm=np.empty((N,len(hgeom),3,3))
  for ti in range(N):hd.qpos[:]=f['qpos'][ti];mj.mj_kinematics(hm,hd);gp[ti]=hd.geom_xpos[hgeom]+[0,0,lift];gm[ti]=hd.geom_xmat[hgeom].reshape(-1,3,3)
  hb=np.empty((N,len(hgeom),2,3))
  for gi,g in enumerate(hgeom):v=np.einsum('tij,vj->tvi',gm[:,gi],hv[g])+gp[:,gi,None];hb[:,gi,0]=v.min(1);hb[:,gi,1]=v.max(1)
  for failure in eligible:
   oid=failure['object_id'];ti=failure['policy_frame'];existing=[np.array(s['position'][:2]) for s in meta['motion_scene_supports'][0] if s.get('provenance',{}).get('object_id')==oid];count=sum('_rest_centre_' in s['id'] and s.get('provenance',{}).get('object_id')==oid for s in meta['motion_scene_supports'][0])
   if count>=16:continue
   cloud=np.concatenate([w[ti] for w in byobject[oid]]);lo=cloud[:,:2].min(0);hi=cloud[:,:2].max(0)
   candidates=[cloud[:,:2].mean(0)]+[world[ti,:,:2].mean(0) for world in byobject[oid]]+[np.array([x,y]) for x in np.linspace(lo[0],hi[0],7) for y in np.linspace(lo[1],hi[1],7)]
   scored=[]
   for xy in candidates:
    for radius in [.012,.006,.003]:
     top=min(column_min(p,xy,radius) for p in pieces)-.00005
     if not np.isfinite(top) or top<=.001:continue
     mask=np.all((hb[:,:,0,:2]<=xy+radius)&(hb[:,:,1,:2]>=xy-radius),axis=-1);handtop=float(hb[:,:,0,2][mask].min())-.001 if mask.any() else np.inf
     if top>handtop:continue
     anchor_bottom=min(column_min((w[ti:ti+1],p[1],p[2][ti:ti+1],p[3][ti:ti+1]),xy,radius) for object_id,p in zip(pieceids,fullpieces) if object_id==oid for w in [p[0]])
     if not np.isfinite(anchor_bottom):continue
     scored.append((max(0.,anchor_bottom-top),-radius,xy.copy(),top,radius));break
   ordered=[]
   for candidate in sorted(scored,key=lambda x:(x[0],x[1])):
    if candidate[0]>.003:rejected.append(dict(object_id=oid,reason='safe_full_reference_column_cannot_reach_rest_surface',rest_gap_m=candidate[0]));continue
    if any(np.linalg.norm(candidate[2]-other[2])<.008 for other in ordered):continue
    ordered.append(candidate)
   for rest_gap,_,xy,top,radius in ordered:
    if count>=16:break
    if any(np.linalg.norm(xy-c)<.008 for c in existing):continue
    mask=np.all((hb[:,:,0,:2]<=xy+radius)&(hb[:,:,1,:2]>=xy-radius),axis=-1);handtop=float(hb[:,:,0,2][mask].min())-.001 if mask.any() else np.inf
    if top<=.001 or top>handtop:rejected.append(dict(object_id=oid,reason='no_clear_ground_column_at_piece_centre'));continue
    s=dict(id=f'ground_foot_{oid}_rest_centre_{count}',fixed=True,coordinate_frame='scene',position=[float(xy[0]),float(xy[1]),float(top/2)],quaternion_wxyz=[1.,0.,0.,0.],geometry=dict(type='box',size=[float(2*radius),float(2*radius),float(top)]),friction=.5,provenance=dict(kind='failed_rest_piece_and_grid_cradle_parking_v1',object_id=oid,policy_frame=ti,exact_full_reference_frame_object_clearance_m=.00005,rest_surface_gap_m=float(rest_gap),continuous_interpolation_certified=False,exact_duplicate_object_poses_collapsed=True,conservative_hand_clearance_m=.001));add.append(s);existing.append(xy);count+=1
    if count>=16:break
  if add:meta['motion_scene_supports'][0]+=add;meta['motion_scene_support_provenance']=[dict(kind='rest_cradle_v4_with_bounded_v5_centre_fallback',physical_admission=False)];validate_supports(meta['motion_scene_supports'][0]);write_support_asset(OUT,meta['motion_scene_supports'][0]);save(path,meta)
 result=dict(index=mi,search_revision='grid49x3-after-objectlift-coverage',completed_grid_search_objects=sorted(set(previously_searched)|{v['object_id'] for v in eligible}|set(capped)),bounded_16_foot_cap_reached_objects=capped,failed_probe_objects=[v['object_id'] for v in fail],excluded_no_unheld_rest_anchor=[v['object_id'] for v in fail if not anchors.get(v['object_id'])],added_feet=len(add),rejected=rejected,physical_admission=False);save(O/f'{mi:04d}.json',result);reports.append(result)
 if add:print(mi,len(add),flush=True)
save(T/'artifacts/parking-v1'/('cradle-augmentation-priority-summary.json' if os.environ.get('AUGMENT_INDICES') else 'cradle-augmentation-summary.json'),dict(motions=len(reports),added_feet=sum(v['added_feet'] for v in reports),physical_admission=False))
