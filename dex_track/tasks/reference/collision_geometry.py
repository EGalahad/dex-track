"""Object geometry shared by collision SDF and scene point observations."""
from functools import lru_cache
from pathlib import Path
import hashlib
import trimesh


def collision_digest(directory):
    paths = sorted(Path(directory).glob('*.obj'))
    if not paths:
        raise FileNotFoundError(f'No collision hulls: {directory}')
    digest = hashlib.sha256(b'convex-union-manifold-v1')
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


@lru_cache(maxsize=128)
def _load_union(directory, digest):
    # Both backends cook each OBJ as a convex hull. Union removes internal faces
    # of overlapping pieces; concatenation would give false internal surfaces.
    hulls = [trimesh.load(p, force='mesh').convex_hull
             for p in sorted(Path(directory).glob('*.obj'))]
    mesh = hulls[0] if len(hulls) == 1 else trimesh.boolean.union(hulls, engine='manifold')
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight or mesh.volume <= 0:
        raise ValueError(f'Invalid collision union: {directory}')
    return mesh


def collision_mesh(directory, scale=1.):
    directory = str(Path(directory).resolve())
    mesh = _load_union(directory, collision_digest(directory)).copy()
    mesh.apply_scale(scale)
    return mesh
