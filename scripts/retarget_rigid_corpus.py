"""Prepare collisions and reset poses; audit every clip before joint admission.

CPU-only stages can run durably while downloads/reviews finish. No training is
launched here, and a successful retarget does not imply physical scene QA passed.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from _source import source_root
SOURCE=source_root();sys.path.insert(0,str(SOURCE/'src'))
from omegaconf import OmegaConf
from retarget.human_demo import clips as C
from retarget.human_demo.preprocess import preprocess_clips
from retarget.object.obj_convex_decompose import decompose_objects
from retarget.robot_hand.hand_retarget import retarget_clips
from retarget.build_motion_pt import build
from pack_rigid_corpus import attach_scenes
import torch


def report(path,rows):
    tmp=path.with_suffix('.partial.csv')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(sorted(rows,key=lambda x:x['clip']))
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();root=a.dataset.resolve()
    cfg=OmegaConf.load(SOURCE/'config/retarget/kinematic_retargeting.yaml')
    cfg.convex=OmegaConf.merge(OmegaConf.load(SOURCE/'config/retarget/convex/coacd.yaml'),cfg.convex)
    fps=C.clip_fps(C.list_clips(root/'human_demo'))
    rows,kept=preprocess_clips(root/'human_demo',root,cfg,fps,a.workers)
    report(root/'corpus-report.csv',rows)
    if not kept:raise RuntimeError('No clips pass human-reference validity filters')
    ids=sorted({str(oid) for clip in kept for oid in C.load_clip(clip)['scene_object_ids']})
    print(f'Complete scenes: {len(kept)} clips, {len(ids)} object meshes',flush=True)
    decompose_objects(root,ids,cfg.convex,a.workers)
    result={r['clip']:r for r in retarget_clips(list(kept.items()),root,root/'retargeted',
        'xhand',cfg.retarget,fps,a.workers)}
    for row in rows:row.update(result.get(row['clip'],{}))
    report(root/'corpus-report.csv',rows)
    accepted={r['clip'] for r in rows if r['clip'] in result and not r.get('reject')
              and not r.get('error') and not r.get('flags')}
    decisions={'retarget_complete':True,'training_admission':False,'fps':fps,
        'preprocess_kept':len(kept),'retarget_quality_pass':len(accepted),
        'quality_pass_names':sorted(accepted),'objects':ids,'rows':rows,
        'pending':['USD collision conversion','SDF cache','whole-scene physical replay',
                   'merge eligible corpus views','joint environment and memory smoke']}
    (root/'retarget-admission.json').write_text(json.dumps(decisions,indent=2)+'\n')
    if not accepted:raise RuntimeError('No reset trajectories pass the pinned collision/tracking quality flags')
    build(root/'retargeted',root,fps,cfg.pack,names=accepted)
    raw=torch.load(root/'motion.pt',map_location='cpu',weights_only=False)
    attach_scenes(raw,root/'retargeted',float(cfg.pack.vel_sigma))
    tmp=root/'motion.partial.pt';torch.save(raw,tmp);tmp.replace(root/'motion.pt')
    print(f'Reset data prepared: {len(accepted)} clips; physical scene QA remains pending.',flush=True)


if __name__=='__main__':main()
