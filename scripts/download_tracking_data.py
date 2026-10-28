"""Download the published tracking references or prepared single-task packs."""
import argparse
from pathlib import Path
import tarfile
from huggingface_hub import snapshot_download

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',default='elijahgalahad/oakink2')
p.add_argument('--revision',default='main')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--part',choices=['reference','annotations','training','all'],default='training')
a=p.parse_args()
patterns=None if a.part=='all' else [f'{a.part}/**','README.md','release.json','SHA256SUMS']
snapshot_download(a.repo,repo_type='dataset',revision=a.revision,
                  local_dir=a.output,allow_patterns=patterns)
if a.part in ('reference','all'):
    for name in ('object_repair','program'):
        with tarfile.open(a.output/'reference'/f'{name}.tar.gz') as archive:
            archive.extractall(a.output/'reference',filter='data')
print(a.output.resolve())
