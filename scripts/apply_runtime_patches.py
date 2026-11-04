"""Apply the audited collision files to the documented isolated environment."""
import hashlib,importlib.metadata,json,shutil,sysconfig
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'runtime_patches'
manifest=json.loads((root/'manifest.json').read_text())
if importlib.metadata.version('mujoco-warp')!=manifest['version']:
    raise RuntimeError('Install the exact documented mujoco-warp version first.')
target=Path(sysconfig.get_paths()['purelib'])/'mujoco_warp/_src'
for name,digest in manifest['files_sha256'].items():
    source=root/name
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    destination=target/name
    if hashlib.sha256(destination.read_bytes()).hexdigest()==digest:continue
    backup=destination.with_suffix('.py.before-dex-track')
    if not backup.exists():shutil.copy2(destination,backup)
    shutil.copy2(source,destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
print('Audited collision runtime installed. Restart Python before training.')
