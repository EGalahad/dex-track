import os
"""Independent actual-loader full-frame signed geometry checks. No admission/publication."""
from pathlib import Path
code=Path(__file__).with_name('parking_process.py').read_text();exec(code.split('OUT.mkdir(exist_ok=True)')[0])
import ast
node=next(n for n in ast.walk(ast.parse(Path(__file__).with_name('scene_geometry.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='build');exec(compile(ast.Module(body=[node],type_ignores=[]),'<independent-geometry-build>','exec'))
def only_contact_metadata_changed(old_sha,current):
 history=V/'metadata-history'/(str(old_sha)+'.json')
 if not history.exists():return False
 old=json.loads(history.read_text());new=copy.deepcopy(current)
 for item in [old,new]:
  for key in ['contact_semantics_revision','contact_position_coordinate_frame','contact_semantics_provenance']:item['meta'].pop(key,None)
  item['meta'].get('inactive_hand_parking',{}).pop('tips_distance_semantics',None)
 return old==new
def support_ground(meta):
  bottoms=[]
  for support in meta['motion_scene_supports'][0]:
   geometry=support['geometry'];rotation=Rot.from_quat(np.array(support['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
   extent=float(np.abs(rotation[2])@(np.array(geometry['size'])/2)) if geometry['type']=='box' else float(abs(rotation[2,2])*geometry['height']/2+geometry['radius']*np.sqrt(max(0,1-rotation[2,2]**2)))
   bottoms.append(float(support['position'][2]-extent))
  support_ground_connected=all(abs(z)<1e-7 for z in bottoms)
  return bottoms,support_ground_connected
O=V/'full-geometry';O.mkdir(exist_ok=True);results=[]
selection=[int(v) for v in os.environ['GEOMETRY_INDICES'].split(',')] if os.environ.get('GEOMETRY_INDICES') else [326,119,674,268]+[i for i in range(704) if i not in [326,119,674,268]]
for mi in selection:
 old=json.loads((S/idx['motions'][mi]['path']).read_text());meta=json.loads((OUT/idx['motions'][mi]['path']).read_text());sha=hashlib.sha256((OUT/idx['motions'][mi]['path']).read_bytes()).hexdigest();done=O/f'{mi:04d}.json'
 if done.exists() and (json.loads(done.read_text()).get('metadata_sha256')==sha or only_contact_metadata_changed(json.loads(done.read_text()).get('metadata_sha256'),meta)):
  cached=json.loads(done.read_text());old_sha=cached['metadata_sha256'];cached.update(metadata_sha256=sha,cache_reuse_contact_only_metadata_proof=(old_sha!=sha),cache_source_metadata_sha256=old_sha);bottoms,connected=support_ground(meta);cached.update(support_ground_bottom_range_m=[min(bottoms or [0.]),max(bottoms or [0.])],all_support_bottoms_at_ground=connected,new_ground_support_geometry_passed=cached['new_ground_support_geometry_passed'] and connected);save(done,cached);results.append(cached);continue
 keep=[old['motion_scene_object_ids'][0].index(oid) for oid in meta['motion_scene_object_ids'][0]];r=json.loads((V/'motions'/f'{mi:04d}.json').read_text());f=np.load(AUD/'artifacts/frames'/f'{mi:04d}.npz');q=np.load(V/'loader-policy-frames'/f'{mi:04d}.npz')['qpos'];N=len(q);assert N==len(f['qpos']);m,d=build(meta,keep)
 cats={k:[] for k in ['hand','object','support']};verts={};mins={k:float('inf') for k in ['right','left','object']}
 for gi in range(m.ngeom):
  name=m.geom(gi).name;key='hand' if name.startswith('collision_hand_') else 'object' if name.startswith('object') else 'support' if name.startswith('support_') else None
  if key:cats[key].append(gi)
  if key in ['hand','object']:
   mesh=m.geom_dataid[gi];verts[gi]=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]].astype(float)
 bottoms,support_ground_connected=support_ground(meta)
 pairs=[];kinds=[]
 for ca,cb,kind in [('hand','support','hand_support'),('object','support','object_support'),('object','object','object_object'),('hand','object','hand_object'),('hand','hand','cross_hand')]:
  for ga in cats[ca]:
   for gb in cats[cb]:
    if ca==cb and (ga>=gb or m.geom_bodyid[ga]==m.geom_bodyid[gb]):continue
    if kind=='cross_hand' and ('hand_right_' in m.geom(ga).name)==('hand_right_' in m.geom(gb).name):continue
    pairs.append((ga,gb));kinds.append(kind)
 pairs=np.array(pairs,dtype=int).reshape(-1,2);peak={k:dict(depth_m=0.,frame=None,pair=None) for k in set(kinds)};query_count=0
 for ti in range(N):
  d.qpos[:]=q[ti]
  for j in keep:mid=m.body(f'object_{j}').mocapid[0];d.mocap_pos[mid]=f['object_pos'][ti,j]+[0,0,r['total_world_z_lift_m']];d.mocap_quat[mid]=f['object_quat'][ti,j]
  mj.mj_forward(m,d)
  for gi,v in verts.items():
   key='right' if 'hand_right_' in m.geom(gi).name else 'left' if 'hand_left_' in m.geom(gi).name else 'object';minimum=float((v@d.geom_xmat[gi].reshape(3,3).T+d.geom_xpos[gi])[:,2].min());mins[key]=min(mins[key],minimum)
  if len(pairs):
   near=np.linalg.norm(d.geom_xpos[pairs[:,0]]-d.geom_xpos[pairs[:,1]],axis=-1)<=m.geom_rbound[pairs[:,0]]+m.geom_rbound[pairs[:,1]]+.001
   for pi in np.flatnonzero(near):
    ga,gb=map(int,pairs[pi]);dist=float(mj.mj_geomDistance(m,d,ga,gb,.001,None));query_count+=1;kind=kinds[pi]
    if -dist>peak[kind]['depth_m']:peak[kind]=dict(depth_m=-dist,frame=ti,pair=[m.geom(ga).name,m.geom(gb).name])
 new_geometry_passed=support_ground_connected and mins['object']>=-2e-6 and min(mins['right'],mins['left'])>=-.005001 and peak.get('hand_support',{}).get('depth_m',0)<1e-6 and peak.get('object_support',{}).get('depth_m',0)<1e-6
 result=dict(index=mi,revision='parking-v1',metadata_sha256=sha,actual_loader_policy_frames=N,minimum_geometry_z_m=mins,support_ground_bottom_range_m=[min(bottoms or [0.]),max(bottoms or [0.])],all_support_bottoms_at_ground=support_ground_connected,max_signed_penetration=peak,distance_queries=query_count,new_ground_support_geometry_passed=new_geometry_passed,physical_admission=False,kinematic_only=True,existing_hand_object_and_scene_overlap_not_hidden=True)
 save(done,result);results.append(result);save(V/'full-geometry-state.json',dict(completed=len(results),total=704,index=mi,pid=os.getpid(),heartbeat_unix=time.time()))
 if mi%20==0:print(mi,new_geometry_passed,flush=True)
summary=dict(hand_ground_motion_counts_source_after_object_lift={str(mm):sum(r['after_object_lift']['hand']['max_mm']>mm for r in rows) for mm in [5,20,50]},hand_ground_motion_counts_after={str(mm):sum(min(r['minimum_geometry_z_m']['right'],r['minimum_geometry_z_m']['left'])< -mm/1000 for r in results) for mm in [5,20,50]},max_hand_ground_depth_after_m=max(max(0.,-min(r['minimum_geometry_z_m']['right'],r['minimum_geometry_z_m']['left'])) for r in results),min_object_geometry_z_m=min(r['minimum_geometry_z_m']['object'] for r in results),all_support_bottoms_at_ground=all(r['all_support_bottoms_at_ground'] for r in results),scope='all actual-loader 50Hz reference poses; continuous interpolation collision not certified; static probes separate from policy dynamics',motions=len(results),actual_loader_policy_frames=sum(r['actual_loader_policy_frames'] for r in results),failed_new_geometry=[r['index'] for r in results if not r['new_ground_support_geometry_passed']],max_depth_m={k:max(r['max_signed_penetration'].get(k,{}).get('depth_m',0.) for r in results) for k in ['hand_support','object_support','object_object','hand_object','cross_hand']},physical_admission=False,publish_allowed=False)
save(V/('geometry-priority-summary.json' if os.environ.get('GEOMETRY_INDICES') else 'full-geometry-summary.json'),summary);print(summary,flush=True)
