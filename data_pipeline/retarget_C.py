import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from dataclasses import replace,asdict
import argparse,sys,hashlib
T=Path(__file__).resolve().parent;sys.path.insert(0,str(T/'solver'))
p=argparse.ArgumentParser(description='C-weight mesh-selfcollision TopoRetarget')
p.add_argument('clip',type=Path);p.add_argument('--output',type=Path,required=True)
p.add_argument('--prepared-data',type=Path,required=True)
p.add_argument('--source-xml',type=Path,required=True)
p.add_argument('--work',type=Path,required=True)
a=p.parse_args()
os.environ['DEX_PREPARED_DATA']=str(a.prepared_data.resolve())
os.environ['DEX_SOURCE_XML']=str(a.source_xml.resolve())
os.environ['DEX_PIPELINE_WORK']=str(a.work.resolve())
import retarget_impl as adapter
from toporetarget.solver import selfcol
original=adapter.build_hand_config
def configured(config):
 c=original(config)
 return replace(c,wrist=replace(c.wrist,zero_translation_weight=1.),mesh=replace(c.mesh,laplacian_weight=200.),penetration=replace(c.penetration,hybrid=replace(c.penetration.hybrid,slack_weight=100.)))
adapter.build_hand_config=configured
selfcol._HARD_OPTS=replace(selfcol._HARD_OPTS,hybrid_slack_weight=100.)
selfcol._HYBRID_OPTS=replace(selfcol._HYBRID_OPTS,hybrid_slack_weight=100.)
a.output.mkdir(parents=True,exist_ok=True)
cfg=configured(adapter.load_config(adapter.OLD/'config/xhand_right.json'));fingerprint=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
adapter.write_json(a.output/'effective-config.json',dict(preset='C',hand=asdict(cfg),post_hard=asdict(selfcol._HARD_OPTS),post_hybrid=asdict(selfcol._HYBRID_OPTS),wrapper_sha256=fingerprint))
v=adapter.run(a.clip,a.output);v.update(preset='C',wrapper_sha256=fingerprint,cost_weights=dict(wrist_translation=1.,laplacian=200.,collision_slack=100.),post_hard_slack=100.,post_hybrid_slack=100.)
adapter.write_json(a.output/'verification.json',v)
