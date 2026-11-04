import os
"""Refine terrain-connected feet using per-piece swept bounds and settle probes."""
from pathlib import Path
code=Path(__file__).with_name('parking_process.py').read_text();exec(code.split('OUT.mkdir(exist_ok=True)')[0])
import subprocess
subprocess.run([sys.executable,str(Path(__file__).with_name('parking_reconcile_validity.py'))],check=True)
from scipy.spatial import ConvexHull
REF=T/'artifacts/parking-v1/refinement-rest-v3';REF.mkdir(exist_ok=True);manifest=json.loads((T/'artifacts/parking-v1/manifest.json').read_text());byi={r['index']:r for r in manifest};summary=[]
selection=[int(v) for v in os.environ['REFINE_INDICES'].split(',')] if os.environ.get('REFINE_INDICES') else [326,119,174,674]+[i for i in range(704) if i not in [326,119,174,674]]
for mi in selection:
 r=byi[mi];done=REF/f'{mi:04d}.json'
 if done.exists():summary.append(json.loads(done.read_text()));continue
 path=OUT/idx['motions'][mi]['path'];meta=json.loads(path.read_text());supports=[s for s in meta['motion_scene_supports'][0] if not s['id'].startswith('ground_foot_')]
 for s in supports:s['friction']=.5
 if r['total_world_z_lift_m']<=1e-6 and not rows[mi]['supports']:
  result=dict(index=mi,needed=False,feet=0,physical_admission=False)
 else:
  replaced_supports=copy.deepcopy(rows[mi]['supports']);witness_ids={s.get('provenance',{}).get('witness',{}).get('object_id') for s in replaced_supports};supports=[]
  f=parked_frames(mi);ids=meta['motion_scene_object_ids'][0];original=rows[mi];oldids=[v['id'] for v in original['objects']];N=len(f['qpos']);total=r['total_world_z_lift_m'];pieces=[];obounds=[];objinfo={};allpoints={}
  for oid,scale in zip(ids,meta['motion_scene_object_scales'][0]):
   j=oldids.index(oid);rot=Rot.from_quat(f['object_quat'][:,j][:,[1,2,3,0]]).as_matrix();pp=[]
   for p in sorted((D/'assets/collision_simplified/objects'/oid/'convex').glob('*.obj')):
    v=np.asarray(trimesh.load(p,force='mesh',process=False).vertices)*scale;wv=np.einsum('tij,vj->tvi',rot,v)+f['object_pos'][:,j,None]+[0,0,total];b=np.stack([wv.min(1),wv.max(1)],1);pieces.append((oid,b));obounds.append(b);pp.append(wv)
   zmin=min(float(v[:,:,2].min()) for v in pp);objinfo[oid]=dict(bottom=zmin,grounded_before_extra=zmin-total<.015 or oid in witness_ids);allpoints[oid]=pp
  pb=np.stack(obounds,1);gp=np.empty((N,len(hgeom),3));gm=np.empty((N,len(hgeom),3,3))
  for ti in range(N):hd.qpos[:]=f['qpos'][ti];mj.mj_kinematics(hm,hd);gp[ti]=hd.geom_xpos[hgeom]+[0,0,total];gm[ti]=hd.geom_xmat[hgeom].reshape(-1,3,3)
  hb=np.empty((N,len(hgeom),2,3))
  for gi,g in enumerate(hgeom):v=np.einsum('tij,vj->tvi',gm[:,gi],hv[g])+gp[:,gi,None];hb[:,gi,0]=v.min(1);hb[:,gi,1]=v.max(1)
  details=[]
  for oid,info in objinfo.items():
   if not info['grounded_before_extra']:continue
   candidates=[]
   jj=oldids.index(oid);velocity=np.linalg.norm(np.gradient(f['object_pos'][:,jj],.02,axis=0),axis=-1);qq=Rot.from_quat(f['object_quat'][:,jj][:,[1,2,3,0]]);omega=np.r_[0.,(qq[:-1].inv()*qq[1:]).magnitude()/.02];stable=(velocity<.03)&(omega<.15);anchors=set()
   for ti in range(0,N,10):
    if stable[max(0,ti-5):min(N,ti+6)].mean()<.8:continue
    box=np.array([np.min([wv[ti].min(0) for wv in allpoints[oid]],axis=0),np.max([wv[ti].max(0) for wv in allpoints[oid]],axis=0)])
    clearance=min(float(gap(hb[ti:ti+1,gi],box[None])[0]) for gi in range(len(hgeom)))
    if clearance>.03:anchors.add(ti)
   for oldsupport in replaced_supports:
    witness=oldsupport.get('provenance',{}).get('witness',{})
    if witness.get('object_id')==oid:
     for nativeframe in [witness.get('start_frame',0),witness.get('end_frame',0)]:anchors.add(min(N-1,int(nativeframe*50/60)))
   reduced=[]
   for ti in sorted(anchors):
    if not any(np.linalg.norm(f['object_pos'][ti,jj]-f['object_pos'][tj,jj])<.005 and (qq[tj].inv()*qq[ti]).magnitude()<.05 for tj in reduced):reduced.append(ti)
   anchors=set(reduced)
   for wv in allpoints[oid]:
    gt,gv=np.unravel_index(wv[:,:,2].argmin(),wv[:,:,2].shape)
    for ti in sorted(anchors|{gt}):
     lowest=wv[ti,:,2].min();objbottom=min(float(v[ti,:,2].min()) for v in allpoints[oid])
     if lowest>objbottom+.02:continue
     pool=wv[ti][wv[ti,:,2]<lowest+.002]
     for v in pool:
      if not candidates or min(np.linalg.norm(v[:2]-c[:2]) for c in candidates)>.015:candidates.append(v)
   accepted=[];reasons=[]
   # Widely spaced candidates first; avoid taking three vertices on one leg.
   if candidates:
    candidates=np.array(candidates);ordered=[int(np.argmin(candidates[:,0]))]
    while len(ordered)<len(candidates):ordered.append(max((j for j in range(len(candidates)) if j not in ordered),key=lambda j:min(np.linalg.norm(candidates[j,:2]-candidates[k,:2]) for k in ordered)))
   else:ordered=[]
   for ci in ordered:
    v=candidates[ci]
    for radius in [.01,.006,.003]:
     xylo=v[:2]-radius;xyhi=v[:2]+radius;overlap=np.all((pb[:,:,0,:2]<=xyhi)&(pb[:,:,1,:2]>=xylo),axis=-1);z=pb[:,:,0,2][overlap];top=float(z.min()) if len(z) else float(v[2]);top-=.00005
     if top<=.001:reasons.append('no_positive_clear_column');continue
     box=np.array([[*xylo,0.],[*xyhi,top]]);clear=float(min(gap(hb[:,gi],np.broadcast_to(box,(N,2,3))).min() for gi in range(len(hgeom))))
     if clear<.001:reasons.append('hand_swept_collision_bound');continue
     s=support(f'ground_foot_{oid}_{len(accepted)}',v[:2],top,dict(kind='per_convex_piece_swept_bound_column',object_id=oid,uniform_world_z_lift_m=total,full_trajectory_hand_AABB_clearance_m=clear,object_AABB_vertical_safety_margin_m=.00005),radius=radius);s['friction']=.5;accepted.append(s);break
    if len(accepted)>=12:break
   supports+=accepted;hull_area=0.
   if len(accepted)>=3:
    try:hull_area=float(ConvexHull(np.array([s['position'][:2] for s in accepted])).volume)
    except Exception:pass
   details.append(dict(object_id=oid,minimum_geometry_z_m=info['bottom'],has_reference_ground_contact=info['bottom']<=.001,accepted_feet=len(accepted),candidate_count=len(candidates),support_xy_hull_area_m2=hull_area,rejection_reasons=reasons,requires_physics=True,rest_anchor_policy_frames=sorted(anchors)))
  result=dict(index=mi,needed=True,replaced_existing_supports=replaced_supports,feet=sum(d['accepted_feet'] for d in details),objects=details,missing_foot_objects=[d['object_id'] for d in details if not d['accepted_feet'] and not d['has_reference_ground_contact']],degenerate_support_polygon_objects=[d['object_id'] for d in details if d['accepted_feet'] and d['support_xy_hull_area_m2']<1e-6],physical_admission=False)
 if len(meta.get('scene_selection_provenance',[]))>1:meta['scene_selection_provenance']=[dict(source=meta['scene_selection_provenance'][0],cleanup=meta['scene_selection_provenance'][1])]
 validate_supports(supports);meta['motion_scene_supports']=[supports];meta['motion_scene_support_provenance']=[dict(kind='rest_anchors_swept_bound_columns_v3',physical_admission=False,friction=.5)];write_support_asset(OUT,supports);save(path,meta)
 if supports and not (path.parent/'supports').exists():(path.parent/'supports').symlink_to(OUT/'supports',target_is_directory=True)
 save(done,result);summary.append(result);save(T/'artifacts/parking-v1/refinement-state.json',dict(completed=len(summary),total=len(selection),pid=os.getpid(),heartbeat_unix=time.time()))
 if mi%25==0:print(mi,flush=True)
stage_index=json.loads((OUT/'index.json').read_text())
for ent in stage_index['motions']:ent['object_ids']=json.loads((OUT/ent['path']).read_text())['motion_scene_object_ids'][0]
save(OUT/'index.json',stage_index)
summary=[json.loads(p.read_text()) for p in REF.glob('*.json')]
save(T/'artifacts/parking-v1/refinement-summary.json',dict(motions=704,feet=sum(v['feet'] for v in summary),missing_foot_objects=sum(len(v.get('missing_foot_objects',[])) for v in summary),degenerate_support_polygons=sum(len(v.get('degenerate_support_polygon_objects',[])) for v in summary),physical_admission=False,stage='requires_settle_and_collision_checks'));print('refinement complete',flush=True)
