"""Publish only a checksum-matched, explicitly approved dataset; default is dry-run."""
import argparse,hashlib,json
from pathlib import Path
from ground_support import require_publishable


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--manifest-dir',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--repo',default='elijahgalahad/oakink2')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();approved=require_publishable(args.review)
    if approved.get('rest_quarantine_motion_indices'):raise ValueError('Unresolved quarantined motions')
    release=json.loads((args.manifest_dir/'release.json').read_text())
    if approved.get('dataset_index_sha256')!=release['dataset_index_sha256']:
        raise ValueError('Approval must identify this exact dataset index SHA256')
    for row in release['files']:
        path=args.dataset/row['path']
        with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=row['sha256']:raise ValueError(f"Dataset changed: {row['path']}")
    if not args.publish:
        print(f"Approved checksum-matched {release['motions']} motions; dry-run only")
        return
    from huggingface_hub import HfApi,CommitOperationAdd
    operations=[CommitOperationAdd(path_in_repo='training/C/'+row['path'],path_or_fileobj=str(args.dataset/row['path'])) for row in release['files']]
    for name in ['release.json','SHA256SUMS','associations.json']:
        operations.append(CommitOperationAdd(path_in_repo='annotations/C/'+name,path_or_fileobj=str(args.manifest_dir/name)))
    annotations=Path(__file__).parent/'annotations'
    for path in sorted(annotations.glob('*.json')):
        operations.append(CommitOperationAdd(path_in_repo='annotations/C/review/'+path.name,path_or_fileobj=str(path)))
    info=HfApi().create_commit(repo_id=args.repo,repo_type='dataset',operations=operations,
      commit_message='Publish reviewed C NPZ references and topology')
    (args.manifest_dir/'upload-result.json').write_text(json.dumps({'repo':args.repo,'revision':info.oid,'url':info.commit_url,'index_sha256':release['dataset_index_sha256']},indent=2)+'\n')
    print(info.commit_url)

if __name__=='__main__':main()
