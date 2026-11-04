"""Prepare a small immutable checksum/annotation manifest; never copy the corpus."""
import argparse,hashlib,json,runpy
from pathlib import Path


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.dataset.resolve();index=root/'index.json'
    if (args.output.resolve()==root or root in args.output.resolve().parents):
        raise ValueError('Release manifests belong outside the immutable dataset.')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(index.read_text());paths={index};rows=[]
    if manifest.get('schema')!='dex-track-motion-folders-v1':raise ValueError('NPZ folder dataset required')
    import numpy as np
    for item in manifest['motions']:
        metadata=root/item['path']
        if root not in metadata.resolve().parents:raise ValueError('Metadata outside dataset')
        paths.add(metadata)
        for group in ['robot','mano','scene','contact','topology']:
            path=metadata.parent/(group+'.npz')
            if not path.is_file():raise FileNotFoundError(path)
            with np.load(path,allow_pickle=False) as arrays:
                for value in arrays.values():
                    if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError(path)
                if group=='topology':
                    if str(arrays['schema'])!='dex-track-topology-v1' or float(arrays['fps'])!=50:raise ValueError(path)
                    bits=arrays['edge_bits'];points=arrays['object_points']
                    if bits.dtype!=np.uint8 or bits.shape[1:]!=(2,71,9) or points.shape!=(2,50,3):raise ValueError(path)
                    edges=np.unpackbits(bits,axis=-1,bitorder='little')
                    if edges[...,71:].any():raise ValueError('Topology padding bits')
                    edges=edges[...,:71]
                    if not np.array_equal(edges,edges.swapaxes(-1,-2)) or np.diagonal(edges,axis1=-2,axis2=-1).any():raise ValueError('Topology adjacency')
            paths.add(path)
        rows.append({'metadata':item['path'],'topology_sha256':sha(metadata.parent/'topology.npz')})
    for name in ['objects','supports']:
        paths.update(p for p in (root/name).rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(root).parts))
    records=[{'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)]
    # Actual associations, not only a high-level description.
    import subprocess,sys
    subprocess.run([sys.executable,str(Path(__file__).with_name('export_annotations.py')),
      '--dataset',str(root),'--output',str(args.output/'associations.json')],check=True)
    release={'published':False,'hf_revision':None,'dataset_index_sha256':sha(index),
      'motions':len(rows),'topology':rows,'files':records,'physical_admission':False}
    (args.output/'release.json').write_text(json.dumps(release,indent=2)+'\n')
    (args.output/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in records))
    print(f"Prepared {len(rows)} motions, {len(records)} files; no corpus copy or upload")

if __name__=='__main__':main()
