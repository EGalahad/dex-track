import os
from pathlib import Path
import os,json,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
T=Path(os.environ['DEX_CLEANUP_WORK']).resolve();R=Path(os.environ['DEX_PROJECT_ROOT']).resolve();S=Path(os.environ['DEX_DATASET']).resolve();idx=json.loads((S/'index.json').read_text());audit=json.loads((T.parent/'artifacts/per-motion.json').read_text());rows=[]
for i,e in enumerate(idx['motions']):
 path=S/e['path'];mano=np.load(path.parent/'mano.npz');contact=np.load(path.parent/'contact.npz');scene=np.load(path.parent/'scene.npz');sides={}
 for side in ['right','left']:
  joints=mano['mano_'+side+'_joints'];wrist=mano['mano_'+side+'_wrist_pos'];ws=float(np.linalg.norm(np.ptp(wrist,axis=0)));js=float(np.linalg.norm(np.ptp(joints,axis=0),axis=-1).max());frac=float(np.mean(contact['contact_contact_'+side]>0));tip=float(contact['tips_distance_'+side].min());inactive=frac==0 and tip>.1 and ws<.03 and js<.02
  sides[side]=dict(contact_fraction=frac,min_tip_object_distance_m=tip,wrist_span_m=ws,max_joint_span_m=js,source_hand_min_z_m=float(min(joints[...,2].min(),wrist[...,2].min())),source_hand_max_z_m=float(max(joints[...,2].max(),wrist[...,2].max())),strict_noninteraction_parking_candidate=bool(inactive))
 row=dict(index=i,name=e['name'],sides=sides,hand_depth_after_object_lift_mm=audit[i]['after_object_lift']['hand']['max_mm']);rows.append(row)
summary=dict(motions=704,thresholds=dict(contact_fraction=0,min_tip_distance_m=.1,max_wrist_span_m=.03,max_joint_span_m=.02),candidate_hands=sum(v['strict_noninteraction_parking_candidate'] for r in rows for v in r['sides'].values()),candidate_motions=sum(any(v['strict_noninteraction_parking_candidate'] for v in r['sides'].values()) for r in rows),severe50mm_motions_with_candidate=sum(r['hand_depth_after_object_lift_mm']>50 and any(v['strict_noninteraction_parking_candidate'] for v in r['sides'].values()) for r in rows),both_hands_candidates=sum(all(v['strict_noninteraction_parking_candidate'] for v in r['sides'].values()) for r in rows),source_modified=False,physical_hand_disabling_authorized=False)
(T/'artifacts/inactive-hand-audit.json').write_text(json.dumps(dict(summary=summary,motions=rows),indent=2)+'\n');print(json.dumps(summary),flush=True)
