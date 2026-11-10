"""Retain audited scene/MANO metadata and store current free-root robot qpos."""
from pathlib import Path
import copy,hashlib,json,sys,runpy
import numpy as np
import torch
from convert_motion_storage import convert_robot, segment_records
import argparse
p=argparse.ArgumentParser()
p.add_argument('--work',type=Path,required=True)
a=p.parse_args()
TASK=a.work.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 src=TASK/'inputs/platforms';raw=torch.load(src/'motion.pt',map_location='cpu',weights_only=False);names=raw['motion_filename'];q=[];manifest=[]
 for name,length in zip(names,raw['motion_num_frames'].tolist()):
  out=TASK/'outputs'/name;check=json.loads((out/'verification.json').read_text());assert check['stage']=='completed';clip=np.load(out/'retargeted.npz');x=clip['qpos'];assert len(x)==length and np.isfinite(x).all()
  q.append(x);manifest.append(check)
 raw['joint_pos']=torch.from_numpy(np.concatenate(q));raw.pop('qpos',None)
 raw,_=convert_robot(raw);segment_records(raw)
 raw={k:v for k,v in raw.items() if not k.startswith(('contact_','tips_'))}
 raw['meta']=dict(raw['meta'],retarget_method='associated-object-graph-mesh-selfcollision-xhand',retarget_preset='C',retarget_cost_weights={'wrist_translation':1.,'laplacian':200.,'collision_slack':100.},retarget_commit='039c54022016a882ef8d3a6d34b4578aee7cb51f',source_motion_sha256=sha(src/'motion.pt'),physical_admission=False)
 native=TASK/'dataset-60hz';native.mkdir(exist_ok=True)
 for name in ['objects','supports']:
  if not (native/name).exists():(native/name).symlink_to(src/name,target_is_directory=True)
 rd=native/'retargeted';rd.mkdir(exist_ok=True)
 for name in names:
  if not (rd/f'{name}.npz').exists():(rd/f'{name}.npz').symlink_to(TASK/'outputs'/name/'retargeted.npz')
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
 from dex_track.motion_storage import write_motion_folders
 write_motion_folders(raw,native)
 (native/'manifest.json').write_text(json.dumps({'clips':manifest,'source_manifest':json.loads((src/'manifest.json').read_text())},indent=2))
 report={}
 for path,data in [(native,raw)]:
  tensors=[x for x in data.values() if torch.is_tensor(x)];assert all(torch.isfinite(x).all() for x in tensors)
  validator=runpy.run_path(str(Path(__file__).resolve().parents[1]/'dex_track/data.py'))
  contract=validator['validate_motion'](path/'index.json',require_sim_assets=True)
  validator['validate_training_object_limit'](path/'index.json')
  report[path.name]={'training_contract':contract,'clips':len(names),'frames':int(data['motion_num_frames'].sum()),'fps':data['meta']['fps'],'finite_tensors':len(tensors),'motion_sha256':sha(path/'index.json')}
 baseline=torch.load(src/'motion.pt',map_location='cpu',weights_only=False);unchanged=[]
 for k,value in baseline.items():
  if torch.is_tensor(value) and k not in ('joint_pos','joint_vel') and not k.startswith(('contact_','tips_')):
   assert torch.equal(value,raw[k]),k;unchanged.append(k)
 report['unchanged_native_source_tensors']=unchanged
 (TASK/'artifacts/pack-verification.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
