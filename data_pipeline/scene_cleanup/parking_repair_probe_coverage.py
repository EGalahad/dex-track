import os
"""Do not let unchanged metadata skip newly selected floor/rest probe objects."""
from pathlib import Path
import json,sys,os,time,subprocess,hashlib
T=Path(os.environ['DEX_CLEANUP_WORK']).resolve();V=T/'artifacts/parking-v1'
if not (V/'object-lift-support-coverage-repaired.json').exists():raise SystemExit(0)
missing=[];expected=actual=0
for i in range(704):
 ref=json.loads((V/'refinement-rest-v3'/f'{i:04d}.json').read_text());p=V/'settle'/f'{i:04d}.json';probe=json.loads(p.read_text()) if p.exists() else dict(probes=[]);target={v['object_id'] for v in ref.get('objects',[])};tested={v['object_id'] for v in probe['probes']};expected+=len(target);actual+=len(tested)
 if target!=tested or probe.get('refinement_sha256')!=hashlib.sha256((V/'refinement-rest-v3'/f'{i:04d}.json').read_bytes()).hexdigest():missing.append(i)
if not missing:raise SystemExit(0)
def state(stage,**kwargs):
 p=V/'probe-coverage-catchup-state.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),expected_probes=expected,previous_probes=actual,motions=missing,heartbeat_unix=time.time(),published=False,**kwargs),indent=2)+'\n');tmp.replace(p)
for name in ['settle_rest','augment_rest_cradles']:
 child=subprocess.Popen([sys.executable,str(T/'scripts'/('parking_'+name+'.py'))])
 while child.poll() is None:state('running',script=name,child_pid=child.pid);time.sleep(30)
 if child.returncode:state('failed',script=name,error=f'exit{child.returncode}');raise RuntimeError(f'probe catchup {name}:exit{child.returncode}')
state('complete_ready_for_native_validation');(V/'probe-coverage-catchup-completion.json').write_text(json.dumps(dict(expected_probes=expected,previous_probes=actual,retested_motions=missing,completed=True,published=False,heartbeat_unix=time.time()),indent=2)+'\n')
