import os
"""Bounded collision-height correction based on independent signed distance witnesses."""
from pathlib import Path
code=Path(__file__).with_name('parking_process.py').read_text();exec(code.split('OUT.mkdir(exist_ok=True)')[0])
summary=json.loads((V/'full-geometry-summary.json').read_text());changes=[];unresolved=[]
for mi in summary['failed_new_geometry']:
 report=json.loads((V/'full-geometry'/f'{mi:04d}.json').read_text());path=OUT/idx['motions'][mi]['path'];meta=json.loads(path.read_text());assert report['metadata_sha256']==hashlib.sha256(path.read_bytes()).hexdigest();amounts={}
 for kind in ['hand_support','object_support']:
  witness=report['max_signed_penetration'].get(kind,{})
  if witness.get('depth_m',0)<=1e-6:continue
  name=next((n for n in witness['pair'] if n.startswith('support_')),None)
  if name is None:continue
  si=int(name.split('_')[1])
  if witness['depth_m']<=.001:
   foot=meta['motion_scene_supports'][0][si];oid=foot.get('provenance',{}).get('object_id');settle=V/'settle'/f'{mi:04d}.json';settle_record=json.loads(settle.read_text()) if settle.exists() else {};probes=settle_record.get('probes',[]) if settle_record.get('staged_metadata_sha256')==hashlib.sha256(path.read_bytes()).hexdigest() else [];real_rest_fail=any(v['object_id']==oid and v.get('reference_frame_kind')=='unheld_stationary_anchor' and not v['passed'] for v in probes)
   if not real_rest_fail:unresolved.append(dict(index=mi,support_id=foot['id'],signed_depth_m=witness['depth_m'],reason='minor contact-level signed overlap; preserve stable or untested support rather than chase numerical zero'));continue
  amounts[si]=max(amounts.get(si,0),witness['depth_m']+.0002)
 for si,amount in amounts.items():
  support=meta['motion_scene_supports'][0][si];g=support['geometry'];height=g['size'][2] if g['type']=='box' else g['height'];bottom=support['position'][2]-height/2
  if not support['id'].startswith('ground_foot_') or amount>.002 or height-amount<=.001 or abs(bottom)>.00001:unresolved.append(dict(index=mi,support_id=support['id'],requested_lower_m=amount,reason='outside bounded ground-foot correction'));continue
  newheight=height-amount;support['position'][2]=newheight/2
  if g['type']=='box':g['size'][2]=newheight
  else:g['height']=newheight
  support['provenance'].update(signed_distance_height_correction_m=amount,previous_height_m=height,reason='independent actual-loader signed collision witness; thresholds unchanged',static_test_required=True)
  changes.append(dict(index=mi,support_id=support['id'],lower_m=amount,old_height_m=height,new_height_m=newheight))
 if amounts:validate_supports(meta['motion_scene_supports'][0]);write_support_asset(OUT,meta['motion_scene_supports'][0]);save(path,meta)
p=V/'signed-height-corrections.json';previous=json.loads(p.read_text()) if p.exists() else [];previous.append(dict(changes=changes,unresolved=unresolved,heartbeat_unix=time.time(),physical_admission=False));save(p,previous);print('corrected',len(changes),'unresolved',len(unresolved),flush=True)
