"""Optional, fixed scene supports stored with each motion (metres, wxyz)."""
import json
import math
import hashlib

SCHEMA = 'fixed-scene-supports-v1'
CLIP_KEY = 'scene_supports_json'


def validate_supports(value):
    """Validate without inferring supports or modifying the supplied poses."""
    if not isinstance(value, list):
        raise ValueError('Scene supports must be a list; use [] for no support')
    seen = set()
    for support in value:
        if not isinstance(support, dict):
            raise ValueError('Each support must be a dictionary')
        name = support.get('id')
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError('Support IDs must be nonempty and unique within a motion')
        seen.add(name)
        if support.get('fixed') is not True:
            raise ValueError(f'{name}: scene supports must be fixed')
        if support.get('coordinate_frame') != 'scene':
            raise ValueError(f'{name}: support pose must use the motion scene coordinate frame')
        for field, count in [('position', 3), ('quaternion_wxyz', 4)]:
            vec = support.get(field)
            if not isinstance(vec, (list, tuple)) or len(vec) != count or not all(
                isinstance(x, (int, float)) and math.isfinite(x) for x in vec
            ):
                raise ValueError(f'{name}: invalid finite {field}')
        if abs(sum(x*x for x in support['quaternion_wxyz']) - 1.) > 1e-5:
            raise ValueError(f'{name}: quaternion_wxyz must be normalized')
        geometry = support.get('geometry', {})
        if geometry.get('type') == 'box':
            dimensions = geometry.get('size', [])
            count = 3  # Full XYZ dimensions, not half-extents.
        elif geometry.get('type') == 'cylinder':
            dimensions = [geometry.get('radius'), geometry.get('height')]
            count = 2  # Local Z axis, position is the centre, not the top.
        else:
            raise ValueError(f'{name}: expected box or cylinder support geometry')
        if len(dimensions) != count or not all(
            isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in dimensions
        ):
            raise ValueError(f'{name}: invalid positive support dimensions')
        friction = support.get('friction')
        if not isinstance(friction, (int, float)) or not math.isfinite(friction) or friction < 0:
            raise ValueError(f'{name}: invalid friction')
        if not isinstance(support.get('provenance'), dict) or not support['provenance']:
            raise ValueError(f'{name}: support provenance is required')
    return value


def read_clip_supports(clip):
    """Legacy clips have no enabled support; legacy support_disks are proposals."""
    if CLIP_KEY not in clip:
        return []
    payload = json.loads(str(clip[CLIP_KEY].item()))
    if payload.get('schema') != SCHEMA:
        raise ValueError('Unsupported scene support schema')
    return validate_supports(payload['supports'])


def encode_clip_supports(supports):
    return json.dumps({'schema': SCHEMA, 'supports': validate_supports(supports)},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def support_layout_key(supports):
    """Only physical geometry/pose/material affect environment compatibility."""
    fields = ('position', 'quaternion_wxyz', 'geometry', 'friction')
    return json.dumps([{k: s[k] for k in fields} for s in validate_supports(supports)],
                      sort_keys=True, separators=(',', ':'))


def support_asset_directory(supports):
    if not supports:
        return None
    digest = hashlib.sha256(support_layout_key(supports).encode()).hexdigest()
    return f'supports/{digest}'


def write_support_asset(root, supports):
    """Compile dataset metadata to portable meshes and a fixed Isaac USD."""
    from pathlib import Path
    directory = support_asset_directory(supports)
    if directory is None:
        return None
    folder = Path(root)/directory
    if (folder/'definition.json').is_file():
        if json.loads((folder/'definition.json').read_text()) != json.loads(support_layout_key(supports)):
            raise ValueError(f'Support asset hash collision: {folder}')
        return directory
    (folder/'convex').mkdir(parents=True, exist_ok=True)
    (folder/'.isaac_usd').mkdir(exist_ok=True)
    import trimesh
    meshes = [support_mesh(s) for s in supports]
    trimesh.util.concatenate(meshes).export(folder/'visual.obj')
    # USDA is portable and can be authored without importing Isaac/pxr while
    # packing data. The backend runtime opens and validates this same asset.
    usd = ['#usda 1.0', '(defaultPrim = "object"\n metersPerUnit = 1\n upAxis = "Z")',
           'def Xform "object" {',
           'def Xform "baseLink" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {',
           'bool physics:rigidBodyEnabled = true', 'bool physics:kinematicEnabled = true',
           'float physics:mass = 1']
    for i, (support, mesh) in enumerate(zip(supports, meshes)):
        mesh.export(folder/'convex'/f'{i:03d}.obj')
        vertices = ','.join('('+','.join(format(float(x), '.9g') for x in v)+')' for v in mesh.vertices)
        usd += [f'def Mesh "support_{i}" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {{',
                f'point3f[] points = [{vertices}]',
                f'int[] faceVertexCounts = {json.dumps([3]*len(mesh.faces))}',
                f'int[] faceVertexIndices = {json.dumps(mesh.faces.reshape(-1).tolist())}',
                'uniform token subdivisionScheme = "none"',
                'color3f[] primvars:displayColor = [(0.32, 0.39, 0.46)]',
                'bool physics:collisionEnabled = true',
                'token physics:approximation = "convexHull"', '}']
    usd += ['}', '}']
    (folder/'.isaac_usd/object.usd').write_text('\n'.join(usd)+'\n')
    (folder/'definition.json').write_text(support_layout_key(supports)+'\n')
    return directory


def support_mesh(support):
    """Scene-local closed mesh; the entity root remains at the scene origin."""
    import trimesh
    from scipy.spatial.transform import Rotation
    geometry = support['geometry']
    if geometry['type'] == 'box':
        mesh = trimesh.creation.box(extents=geometry['size'])
    else:
        mesh = trimesh.creation.cylinder(radius=geometry['radius'],
                                        height=geometry['height'], sections=64)
    q = support['quaternion_wxyz']
    mesh.vertices = mesh.vertices @ Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T
    mesh.apply_translation(support['position'])
    return mesh
