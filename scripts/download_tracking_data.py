"""Download and verify the current portable v3 corpus, then restore asset links."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def verify_dataset(root):
    root = Path(root).resolve()
    for line in (root / 'checksums.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Checksum path outside dataset: {name}')
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    index = json.loads((root / 'index.json').read_text())
    if index.get('schema') != 'dex-track-motion-folders-v3':
        raise ValueError('Expected pose-only v3; convert v2 with data_pipeline/convert_pose_only_v3.py')
    for item in index['motions']:
        parent = (root / item['path']).parent
        if not parent.resolve().is_relative_to(root):
            raise ValueError('Metadata path outside dataset')
        for name in ('objects', 'supports', 'collision_simplified'):
            target = root / name
            if not target.is_dir():
                continue
            link = parent / name
            if link.is_symlink():
                if link.resolve() != target:
                    raise ValueError(f'Wrong existing asset link: {link}')
            elif link.exists():
                raise ValueError(f'Expected asset symlink, found existing directory/file: {link}')
            else:
                link.symlink_to(os.path.relpath(target, parent), target_is_directory=True)
    return len(index['motions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='elijahgalahad/oakink2')
    parser.add_argument('--revision', default='main', help='Use a commit SHA for reproducibility')
    parser.add_argument('--output', type=Path, required=True, help='Use a fresh directory when changing releases')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only:
        snapshot_download(args.repo, repo_type='dataset', revision=args.revision, local_dir=args.output)
    count = verify_dataset(args.output)
    print(f'Verified {count} segments: {args.output.resolve()}')


if __name__ == '__main__':
    main()
