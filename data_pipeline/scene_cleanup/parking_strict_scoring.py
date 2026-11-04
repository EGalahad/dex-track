import os
from pathlib import Path
import json,time
T=Path(os.environ['DEX_CLEANUP_WORK']).resolve();V=T/'artifacts/parking-v1';out={}
for kind in ['settle','scene-rest-probes']:
 probes=[]
 for p in (V/kind).glob('*.json'):
  r=json.loads(p.read_text())
  for v in r.get('probes',[]):
   v['passed_legacy_depth3mm']=v['passed'];v['passed_depth1mm']=v['position_displacement_m']<.01 and v['rotation_change_rad']<.1 and v['final_speed_m_s']<.01 and v['max_contact_depth_m']<.001;probes.append(dict(index=r['index'],**v))
  p.write_text(json.dumps(r,indent=2)+'\n')
 chosen=[p for p in probes if kind!='settle' or p.get('reference_frame_kind')=='unheld_stationary_anchor'];out[kind]=dict(unheld_probes=len(chosen),failed_depth1mm=sum(not v['passed_depth1mm'] for v in chosen),failed_legacy_depth3mm=sum(not v['passed_legacy_depth3mm'] for v in chosen),failed=[dict(index=v['index'],object_id=v['object_id'],position_displacement_m=v['position_displacement_m'],rotation_change_rad=v['rotation_change_rad'],final_speed_m_s=v['final_speed_m_s'],max_contact_depth_m=v['max_contact_depth_m'],initial_contact_depth_m=v.get('initial_contact_depth_m'),passed_legacy_depth3mm=v['passed_legacy_depth3mm'],initial_penetrating_pairs=v.get('initial_penetrating_pairs',[])) for v in chosen if not v['passed_depth1mm']])
old_probes={};new_probes={}
for folder,target in [(V/'superseded-pre-objectlift-repair/scene-rest-probes',old_probes),(V/'scene-rest-probes',new_probes)]:
 for file in folder.glob('*.json'):
  r=json.loads(file.read_text())
  for v in r.get('probes',[]):target[(r['index'],v['object_id'],v['policy_frame'])]=v
strict=lambda v:v['position_displacement_m']<.01 and v['rotation_change_rad']<.1 and v['final_speed_m_s']<.01 and v['max_contact_depth_m']<.001
common=old_probes.keys()&new_probes.keys();out['common_scene_probes']=dict(matched=len(common),new_coverage_only=len(new_probes.keys()-old_probes.keys()),old_strict_failed=sum(not strict(old_probes[k]) for k in common),new_strict_failed=sum(not strict(new_probes[k]) for k in common),improved=[list(k) for k in sorted(common) if not strict(old_probes[k]) and strict(new_probes[k])],regressed=[list(k) for k in sorted(common) if strict(old_probes[k]) and not strict(new_probes[k])])
metrics=['position_displacement_m','rotation_change_rad','final_speed_m_s','max_contact_depth_m'];gates=[.01,.1,.01,.001]
out['common_scene_probes']['regression_details']=[dict(index=k[0],object_id=k[1],policy_frame=k[2],old={field:old_probes[k][field] for field in metrics},new={field:new_probes[k][field] for field in metrics},failed_gates=[field for field,gate in zip(metrics,gates) if new_probes[k][field]>=gate],contact_depth_only=all(new_probes[k][field]<gate for field,gate in zip(metrics[:3],gates[:3]))) for k in sorted(common) if strict(old_probes[k]) and not strict(new_probes[k])]
g=json.loads((V/'full-geometry-summary.json').read_text());substantial=[];minor=[]
for i in g['failed_new_geometry']:
 r=json.loads((V/'full-geometry'/f'{i:04d}.json').read_text());depth=max(r['max_signed_penetration'].get(k,{}).get('depth_m',0) for k in ['hand_support','object_support']);(substantial if depth>.001 else minor).append(dict(index=i,max_signed_support_depth_m=depth))
out.update(dynamic_depth_gate_m=.001,other_gates=dict(position_displacement_m=.01,rotation_change_rad=.1,final_speed_m_s=.01),substantial_support_overlaps=substantial,minor_signed_support_contacts=minor,physical_admission=False,published=False,heartbeat_unix=time.time());(V/'strict-admission-summary.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['settle','scene-rest-probes']},flush=True)
