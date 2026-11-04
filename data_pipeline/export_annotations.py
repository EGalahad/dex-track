"""Export hand associations, scene instances, supports and topology checksums."""
import argparse,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    index=a.dataset/'index.json';manifest=json.loads(index.read_text());rows=[]
    for item in manifest['motions']:
        path=index.parent/item['path'];meta=json.loads(path.read_text());topology=path.parent/'topology.npz'
        if not topology.is_file():raise FileNotFoundError(topology)
        sides={}
        for side in ('right','left'):
            # Single-motion metadata has slot arrays stored in scene.npz; use
            # the loader to resolve the physical mesh id rather than guessing.
            import runpy
            api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'dex_track/motion_storage.py'))
            raw=api['load_motion_data'](path)
            slot=int(raw[f'motion_object_slot_{side}'][0])
            sides[side]={'mesh_directory':raw[f'{side}_object_mesh_dirs'][slot],
                         'scale':raw[f'{side}_object_mesh_scales'][slot]}
        rows.append({'metadata':item['path'],'hands':sides,
          'scene_object_ids':raw['motion_scene_object_ids'][0],
          'supports':raw.get('motion_scene_supports',[[]])[0],
          'metadata_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'topology_sha256':hashlib.sha256(topology.read_bytes()).hexdigest()})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),
      'motions':rows},indent=2)+'\n')

if __name__=='__main__':main()
