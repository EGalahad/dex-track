"""Select complete motions by scene-object count; never truncate their objects."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-objects',type=int,default=8)
    a=p.parse_args()
    if a.max_objects < 1: raise ValueError('max_objects must be positive')
    stats=json.loads(a.statistics.read_text())
    keys={'oakink2':'oakink2_scene_unique_from_raw_obj_list', 'taco':'taco_scene_pose_identity', 'hot3d':'hot3d_prepared_unique_uid'}
    out={'max_objects':a.max_objects,'table_counted':False,'count_scope':'complete_motion_scene_identity_union',
         'overflow':'exclude_whole_motion','source_statistics':str(a.statistics.resolve()),'source_time':stats['time'],'datasets':{}}
    for dataset,key in keys.items():
        rows=stats['records'][key]
        accepted=[];rejected=[]
        for row in rows:
            record={**row,'reason':'within_limit' if 0 < row['count'] <= a.max_objects else 'too_many_objects' if row['count'] > a.max_objects else 'empty_scene'}
            (accepted if record['reason']=='within_limit' else rejected).append(record)
        out['datasets'][dataset]={'coverage':'partial_prepared_only' if dataset=='hot3d' else 'complete',
            'accepted_count':len(accepted),'rejected_count':len(rejected),'accepted':accepted,'rejected':rejected}
        print(f'{dataset}: keep {len(accepted)}, exclude {len(rejected)}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': main()
