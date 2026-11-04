import os
"""Repair support coverage for object-only common lifts before the bounded grid fallback."""
from pathlib import Path
import json,os,sys,subprocess,shutil,time
T=Path(os.environ['DEX_CLEANUP_WORK']).resolve();V=T/'artifacts/parking-v1';done=V/'object-lift-support-coverage-repaired.json'
if done.exists():raise SystemExit(0)
archive=V/'before-object-lift-coverage-repair';archive.mkdir(exist_ok=True)
for name in ['refinement-rest-v3','settle']:
 if not (archive/name).exists():shutil.copytree(V/name,archive/name)
for name in ['refinement-summary.json','settle-summary.json','validation.json']:
 if (V/name).exists() and not (archive/name).exists():shutil.copy2(V/name,archive/name)
manifest=json.loads((V/'manifest.json').read_text());source_rows=json.loads((T.parent/'artifacts/per-motion.json').read_text());idx=json.loads((T/'staging/index.json').read_text());selected=[]
for r in manifest:
 i=r['index'];ref=json.loads((V/'refinement-rest-v3'/f'{i:04d}.json').read_text());meta=json.loads((T/'staging'/idx['motions'][i]['path']).read_text());covered={v['object_id'] for v in ref.get('objects',[])};expected={v['id'] for v in source_rows[i]['objects'] if v['min_z_mm']<15 and v['id'] in meta['motion_scene_object_ids'][0]}
 if (r['total_world_z_lift_m']>1e-6 or source_rows[i]['supports']) and (not ref['needed'] or not expected.issubset(covered)):selected.append(i);(V/'refinement-rest-v3'/f'{i:04d}.json').unlink()
(V/'object-lift-support-coverage-selection.json').write_text(json.dumps(dict(motions=selected,reason='object-only common lift can float other originally grounded objects; test total lift not hand extra',published=False),indent=2)+'\n')
if selected:
 for name,key in [('refine_rest_supports','REFINE_INDICES'),('exact_foot_heights','EXACT_INDICES')]:
  env=os.environ.copy();env[key]=','.join(map(str,selected));subprocess.run([sys.executable,str(T/'scripts'/('parking_'+name+'.py'))],env=env,check=True)
 reports=[json.loads(p.read_text()) for p in (V/'exact-foot-heights').glob('*.json')]
 (V/'exact-foot-summary.json').write_text(json.dumps(dict(motions=len(reports),feet=sum(r['feet'] for r in reports),raised_feet=sum(c['raised_m']>.0001 for r in reports for c in r['changes']),max_raise_m=max([c['raised_m'] for r in reports for c in r['changes']] or [0]),physical_admission=False),indent=2)+'\n')
 for name in ['validate','settle_rest']:
  env=os.environ.copy()
  if name=='validate' and os.environ.get('LD_PRELOAD'):env['LD_PRELOAD']=os.environ['LD_PRELOAD']
  subprocess.run([sys.executable,str(T/'scripts'/('parking_'+name+'.py'))],env=env,check=True)
done.write_text(json.dumps(dict(completed=True,recomputed_motions=selected,physical_admission=False,published=False,heartbeat_unix=time.time()),indent=2)+'\n')
