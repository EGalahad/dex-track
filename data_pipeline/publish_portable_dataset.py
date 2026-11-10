"""Seal and publish the explicitly selected pose-only v3 corpus, without generated caches.

This does not promote the separately quarantined scene-cleanup candidate.
Preparation modifies only README/checksums/release metadata and tools in the
supplied local publication copy. Publishing/deleting requires explicit flags.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

EXCLUDED = {'.sdf_cache', '.isaac_usd', '.cache', '__pycache__', '.git'}


def digest(path, algorithm='sha256'):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def files(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
                  and not EXCLUDED.intersection(p.relative_to(root).parts))


def prepare(root, expected):
    import numpy as np
    index = json.loads((root/'index.json').read_text())
    if index.get('schema') != 'dex-track-motion-folders-v3' or digest(root/'index.json') != expected:
        raise ValueError('Expected the explicitly selected pose-only v3 index')
    for row in index['motions']:
        metadata = root/row['path']
        if not metadata.resolve().is_relative_to(root):
            raise ValueError('Motion path outside dataset')
        for name in ('robot', 'mano', 'objects'):
            with np.load(metadata.parent/(name+'.npz'), allow_pickle=False) as arrays:
                for array in arrays.values():
                    if array.dtype.kind in 'fc' and not np.isfinite(array).all():
                        raise ValueError(f'Non-finite {metadata}: {name}')
                if name == 'robot' and arrays['qpos'].shape != (row['frames'],38):
                    raise ValueError(f'Invalid robot shape: {metadata}')
    project = Path(__file__).resolve().parents[1]
    tools = root/'tools'
    (tools/'reference').mkdir(parents=True, exist_ok=True)
    for name in ('generate_sdf_cache.py','generate_isaac_usd.py','download_tracking_data.py'):
        shutil.copyfile(project/'scripts'/name, tools/name)
    for name in ('sdf.py','collision_geometry.py'):
        shutil.copyfile(project/'dex_track/tasks/reference'/name, tools/'reference'/name)
    (tools/'reference/__init__.py').write_text('')
    shutil.copyfile(project/'dex_track/supports.py', tools/'supports.py')
    shutil.copyfile(project/'data_pipeline/convert_pose_only_v3.py', tools/'convert_pose_only_v3.py')
    source = project/'third_party/FlashDexRetarget'
    shutil.copyfile(source/'src/simulator/isaacsim/convert_assets.py', tools/'convert_assets.py')
    shutil.copyfile(source/'LICENSE', tools/'FlashDexRetarget-LICENSE')
    shutil.copyfile(project/'docs/portable-dataset-card.md', root/'README.md')
    payload = [p for p in files(root) if p.name not in ('release.json','checksums.sha256')]
    records = [{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in payload]
    release = {'schema':'dex-track-portable-release-v3','dataset_index_sha256':expected,
               'source_revision':'2026-11-10-pose-only-v3','segments':len(index['motions']),
               'frames':sum(row['frames'] for row in index['motions']),
               'excluded_generated_directories':sorted(EXCLUDED),
               'physical_admission':False,'files':records,
               'asset_links':'Each motions/<segment>/objects and supports links to ../../objects and ../../supports; downloader restores them.'}
    (root/'release.json').write_text(json.dumps(release,indent=2)+'\n')
    payload.append(root/'release.json')
    (root/'checksums.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(root).as_posix()}\n' for p in sorted(payload)))
    return release


def verify_published(api, repo, revision, root):
    info = api.dataset_info(repo,revision=revision,files_metadata=True)
    local = {p.relative_to(root).as_posix():p for p in files(root)}
    remote = {p.rfilename:p for p in info.siblings}
    extra = set(remote)-set(local)-{'.gitattributes'}
    missing = set(local)-set(remote)
    if extra or missing:
        raise ValueError(f'Published paths differ: extra={sorted(extra)}, missing={sorted(missing)}')
    for name,path in local.items():
        row = remote[name]
        if row.size != path.stat().st_size:
            raise ValueError(f'Published size differs: {name}')
        if row.lfs:
            if row.lfs.sha256 != digest(path):
                raise ValueError(f'LFS checksum differs: {name}')
        else:
            blob = path.read_bytes()
            actual = hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
            if actual != row.blob_id:
                raise ValueError(f'Git blob checksum differs: {name}')
    if any(EXCLUDED.intersection(Path(name).parts) for name in remote):
        raise ValueError('Generated cache was uploaded')
    return {'verified_files':len(local),'bytes':sum(p.stat().st_size for p in local.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--expected-index-sha256',required=True)
    parser.add_argument('--repo',default='elijahgalahad/oakink2')
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--replace-repo',action='store_true',help='Delete/recreate old repository, removing old history')
    parser.add_argument('--expected-old-revision',help='Required when deleting the old repository')
    parser.add_argument('--verify-revision',help='Only verify a previously uploaded revision')
    parser.add_argument('--receipt',type=Path,required=True,help='Outside the dataset')
    args = parser.parse_args(); root=args.dataset.resolve()
    if args.receipt.resolve().is_relative_to(root):
        parser.error('--receipt must be outside dataset')
    if args.verify_revision:
        if digest(root/'index.json') != args.expected_index_sha256:
            raise ValueError('Local index differs from expected SHA256')
        from huggingface_hub import HfApi
        result=verify_published(HfApi(),args.repo,args.verify_revision,root)
        print(json.dumps(result));return
    if args.replace_repo and (not args.publish or not args.expected_old_revision):
        parser.error('--replace-repo requires --publish and --expected-old-revision')
    release=prepare(root,args.expected_index_sha256)
    print(f"Sealed {release['segments']} segments, {len(files(root))} files, {sum(p.stat().st_size for p in files(root))} bytes",flush=True)
    if not args.publish:
        return
    from huggingface_hub import HfApi
    api=HfApi(); receipt={'repo':args.repo,'dataset_index_sha256':args.expected_index_sha256}
    if args.receipt.exists():
        previous=json.loads(args.receipt.read_text())
        if previous.get('repo') == args.repo and previous.get('dataset_index_sha256') == args.expected_index_sha256:
            receipt.update(previous)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    def save():args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    if args.replace_repo:
        old=api.dataset_info(args.repo)
        if old.sha != args.expected_old_revision:
            raise ValueError('Old repository revision changed; refusing deletion')
        receipt['deleted_old_revision']=old.sha;receipt['old_file_count']=len(old.siblings);save()
        api.delete_repo(args.repo,repo_type='dataset')
        receipt['old_repo_deleted']=True;save()
        api.create_repo(args.repo,repo_type='dataset',private=old.private,exist_ok=False)
    else:
        api.create_repo(args.repo,repo_type='dataset',private=False,exist_ok=True)
    info=api.upload_folder(repo_id=args.repo,repo_type='dataset',folder_path=root,
        allow_patterns=[p.relative_to(root).as_posix() for p in files(root)],
        commit_message='Publish canonical free-root v2 corpus without SDF or USD caches')
    revision=info.oid
    receipt['revision']=revision;receipt['url']=f'https://huggingface.co/datasets/{args.repo}/tree/{revision}';save()
    receipt.update(verify_published(api,args.repo,revision,root));receipt['verified']=True;save()
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
