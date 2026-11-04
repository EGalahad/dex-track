import os
"""Remove isolated clutter components; precise geometry for requested M4/M8."""
from pathlib import Path
code=Path(__file__).with_name('process.py').read_text();exec(code.split('OUT.mkdir(exist_ok=True)')[0])
O=T/'artifacts/component-cleanup';O.mkdir(exist_ok=True);index=json.loads((OUT/'index.json').read_text());reports=[]
selection=[int(v) for v in os.environ['CLEANUP_INDICES'].split(',')] if os.environ.get('CLEANUP_INDICES') else list(range(704))
for mi in selection:
 e=index['motions'][mi]
 path=OUT/e['path'];meta=json.loads(path.read_text());r=json.loads((T/'artifacts/motions'/f'{mi:04d}.json').read_text());ev={v['id']:v for v in r['object_evidence']};ids=meta['motion_scene_object_ids'][0];handdist={oid:ev[oid]['conservative_hand_AABB_gap_m'] for oid in ids};pairdist={(a,b):ev[a]['object_AABB_gaps_m'][b] for a in ids for b in ids if a!=b};precise=False
 if mi in [119,674]:
  precise=True;f=np.load(AUD/'artifacts/frames'/f'{mi:04d}.npz');oldids=[v['id'] for v in rows[mi]['objects']];tree=handtree();asset=tree.find('asset');world=tree.find('worldbody')
  for oid,scale in zip(ids,meta['motion_scene_object_scales'][0]):
   j=oldids.index(oid);body=E.SubElement(world,'body',name='object_'+str(j),mocap='true')
   for k,p in enumerate(sorted((D/'assets/collision_simplified/objects'/oid/'convex').glob('*.obj'))):name=f'object{j}_piece{k}';E.SubElement(asset,'mesh',name=name,file=str(p),scale=' '.join([str(scale)]*3));E.SubElement(body,'geom',name=name,type='mesh',mesh=name,contype='1',conaffinity='1')
  model=mj.MjModel.from_xml_string(E.tostring(tree.getroot(),encoding='unicode'));data=mj.MjData(model);hg=[i for i in range(model.ngeom) if model.geom(i).name.startswith('collision_hand_')];og={oid:[i for i in range(model.ngeom) if model.geom(i).name.startswith(f'object{oldids.index(oid)}_piece')] for oid in ids};queryhand={oid for oid in ids if not ev[oid]['associated'] and ev[oid]['translation_span_m']<.002 and ev[oid]['rotation_span_rad']<.05 and handdist[oid]<=.03};querypair={(a,b) for a,b in pairdist if a<b and pairdist[a,b]<=.02};exacthand={oid:.031 for oid in queryhand};exactpair={pair:.021 for pair in querypair}
  for ti in range(len(f['qpos'])):
   data.qpos[:]=f['qpos'][ti]
   for oid in ids:j=oldids.index(oid);mid=model.body('object_'+str(j)).mocapid[0];data.mocap_pos[mid]=f['object_pos'][ti,j];data.mocap_quat[mid]=f['object_quat'][ti,j]
   mj.mj_forward(model,data)
   for oid in queryhand:
    for ga in hg:
     for gb in og[oid]:exacthand[oid]=min(exacthand[oid],float(mj.mj_geomDistance(model,data,ga,gb,.031,None)))
   for a,b in querypair:
    for ga in og[a]:
     for gb in og[b]:exactpair[a,b]=min(exactpair[a,b],float(mj.mj_geomDistance(model,data,ga,gb,.021,None)))
  handdist.update(exacthand)
  for (a,b),v in exactpair.items():pairdist[a,b]=v;pairdist[b,a]=v
 roots={oid for oid in ids if ev[oid]['associated'] or ev[oid]['translation_span_m']>=.002 or ev[oid]['rotation_span_rad']>=.05 or handdist[oid]<=.03};keep=set(roots)
 while True:
  nxt=keep|{b for a in keep for b in ids if b!=a and pairdist[a,b]<=.02}
  if nxt==keep:break
  keep=nxt
 removed=[oid for oid in ids if oid not in keep];kept=[j for j,oid in enumerate(ids) if oid in keep];assert keep
 if removed:
  scene=dict(np.load(path.parent/'scene.npz'))
  for k in ['scene_object_pos','scene_object_quat','scene_object_lin_vel','scene_object_ang_vel']:scene[k]=scene[k][:,kept]
  np.savez_compressed(path.parent/'scene.npz',**scene);meta['motion_scene_object_ids']=[[ids[j] for j in kept]];meta['motion_scene_object_scales']=[[meta['motion_scene_object_scales'][0][j] for j in kept]];meta['motion_scene_supports']=[[s for s in meta['motion_scene_supports'][0] if s.get('provenance',{}).get('object_id') not in removed]];write_support_asset(OUT,meta['motion_scene_supports'][0]);save(path,meta);e['object_ids']=meta['motion_scene_object_ids'][0]
 result=dict(index=mi,roots=sorted(roots),retained=sorted(keep),additional_removed=removed,hand_distance_m=handdist,pair_distance_m={a+'|'+b:v for (a,b),v in pairdist.items() if a<b},precise_mujoco_all_policy_frames=precise,physical_admission=False);save(O/f'{mi:04d}.json',result);reports.append(result)
 if removed:print(mi,removed,flush=True)
save(OUT/'index.json',index);save(T/'artifacts'/('component-cleanup-priority-summary.json' if os.environ.get('CLEANUP_INDICES') else 'component-cleanup-summary.json'),dict(motions=704,additional_removed_instances=sum(len(r['additional_removed']) for r in reports),physical_admission=False))
