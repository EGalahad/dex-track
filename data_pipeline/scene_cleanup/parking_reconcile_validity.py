import os
from pathlib import Path
import json,time,subprocess,sys,shutil,hashlib
T=Path(os.environ['DEX_CLEANUP_WORK']).resolve();V=T/'artifacts/parking-v1';start=time.time()
while not (V/'validity-summary.json').exists():
 if time.time()-start>3600:raise RuntimeError('source validity audit not completed; stop before supports')
 time.sleep(30)
summary=json.loads((V/'validity-summary.json').read_text());reprocess=[]
for i in summary['ambiguous']:
 p=V/'motions'/f'{i:04d}.json';r=json.loads(p.read_text())
 if r['parking_world_delta_m']:
  archive=V/'ambiguous-parking-rejected';archive.mkdir(exist_ok=True);shutil.copy2(p,archive/p.name);p.unlink();reprocess.append(i)
# Earlier generation may have preceded the support-envelope expansion. Rebuild only affected motions.
source_index=json.loads(((Path(os.environ['DEX_DATASET'])/'index.json')).read_text())
for i,e in enumerate(source_index['motions']):
 p=V/'motions'/f'{i:04d}.json';r=json.loads(p.read_text())
 if r['parking_world_delta_m'] and not r.get('safe_parking_includes_source_support_bounds',False):
  src=Path(os.environ['DEX_DATASET'])/e['path']
  if json.loads(src.read_text()).get('motion_scene_supports',[[]])[0]:p.unlink();reprocess.append(i)
subprocess.run([sys.executable,str(T/'scripts/parking_process.py')],check=True)
idx=json.loads((T/'staging/index.json').read_text());allr=json.loads((V/'manifest.json').read_text())
for r in allr:
 i=r['index'];p=V/'validity'/f'{i:04d}.json'
 if not p.exists():continue
 evidence=json.loads(p.read_text());r['source_scene_validity_confirmation']=evidence;target=V/'motions'/f'{i:04d}.json';target.write_text(json.dumps(r,indent=2)+'\n');m=T/'staging'/idx['motions'][i]['path'];meta=json.loads(m.read_text());meta['meta']['inactive_hand_parking']['source_scene_validity_confirmation']=evidence;meta['meta']['inactive_hand_parking']['reference_semantics']='constant world translation of confirmed inactive hand, not hand disabling; source labels preserved';m.write_text(json.dumps(meta,indent=2)+'\n')
(V/'manifest.json').write_text(json.dumps(allr,indent=2)+'\n');(V/'validity-reconciliation.json').write_text(json.dumps(dict(reprocessed_without_parking=reprocess,confirmed=summary['confirmed'],ambiguous=summary['ambiguous'],published=False),indent=2)+'\n')
