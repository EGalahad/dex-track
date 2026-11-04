import os
"""Confirm strict source candidates against all retained object collision geometry."""
from pathlib import Path
code=Path(__file__).with_name('parking_process.py').read_text();exec(code.split('OUT.mkdir(exist_ok=True)')[0])
import ast
node=next(n for n in ast.walk(ast.parse(Path(__file__).with_name('scene_geometry.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='build');exec(compile(ast.Module(body=[node],type_ignores=[]),'<source-validity-build>','exec'))
O=V/'validity';O.mkdir(exist_ok=True);results=[]
for mi,audit in parking_audit.items():
 candidates=[side for side,metrics in audit['sides'].items() if metrics['strict_noninteraction_parking_candidate']]
 if not candidates:continue
 done=O/f'{mi:04d}.json'
 if done.exists():results.append(json.loads(done.read_text()));continue
 side=candidates[0];old=json.loads((S/idx['motions'][mi]['path']).read_text());meta=copy.deepcopy(old);meta['motion_scene_supports']=[[]];keep=[j for j,oid in enumerate(old['motion_scene_object_ids'][0]) if oid in cleanup[mi]['retained']];m,d=build(meta,keep);f=np.load(AUD/'artifacts/frames'/f'{mi:04d}.npz');hg=[g for g in range(m.ngeom) if m.geom(g).name.startswith('collision_hand_'+side+'_')];og=[g for g in range(m.ngeom) if m.geom(g).name.startswith('object')];pairs=np.array([(a,b) for a in hg for b in og]);minimum=.03;witness=None;queries=0
 for ti,q in enumerate(f['qpos']):
  d.qpos[:]=q
  for j in keep:mid=m.body(f'object_{j}').mocapid[0];d.mocap_pos[mid]=f['object_pos'][ti,j];d.mocap_quat[mid]=f['object_quat'][ti,j]
  mj.mj_forward(m,d);near=np.linalg.norm(d.geom_xpos[pairs[:,0]]-d.geom_xpos[pairs[:,1]],axis=-1)<=m.geom_rbound[pairs[:,0]]+m.geom_rbound[pairs[:,1]]+.03
  for pi in np.flatnonzero(near):
   ga,gb=map(int,pairs[pi]);distance=float(mj.mj_geomDistance(m,d,ga,gb,.03,None));queries+=1
   if distance<minimum:minimum=distance;witness=dict(frame=ti,pair=[m.geom(ga).name,m.geom(gb).name])
 result=dict(index=mi,side=side,source_metrics=audit['sides'][side],policy_frames=len(f['qpos']),all_retained_scene_object_min_distance_m_capped_30mm=minimum,witness=witness,queries=queries,confirmed=minimum>=.01,decision='confirmed_strict_source_noninteraction_and_no_scene_contact' if minimum>=.01 else 'ambiguous_scene_proximity_requires_review',source_contact_labels_not_validity_ground_truth=True)
 save(done,result);results.append(result);save(V/'validity-state.json',dict(completed=len(results),total=114,pid=os.getpid(),heartbeat_unix=time.time()));print(mi,side,round(minimum,5),flush=True)
save(V/'validity-summary.json',dict(candidates=len(results),confirmed=sum(v['confirmed'] for v in results),ambiguous=[v['index'] for v in results if not v['confirmed']],minimum_separation_threshold_m=.01,source_semantics='strict zero-contact + tips>100mm + wrist<30mm + max joints<20mm; all retained scene geometry no contact within10mm',user_approved=True,published=False))
