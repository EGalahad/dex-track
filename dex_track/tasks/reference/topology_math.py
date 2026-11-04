"""Source-only hand/object graph geometry and reward kernels (metres, wxyz).

The proxcurv sampler follows TopoRetarget's area*(1+curvature+30*proximity)
surface sampling. No robot/retarget reference participates in graph construction.
"""
import itertools
import numpy as np
import torch

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
PARTS = ('proximal', 'intermediate', 'distal', 'tip')
EDGES = np.array([(0 if j == 0 else 1+4*f+j-1, 1+4*f+j)
                  for f in range(5) for j in range(4)], dtype=np.int64)


def mano_indices(names):
    # The serialized MANO tensor contains 20 named finger landmarks; wrist is
    # stored separately. Never assume its alphabetical order is MediaPipe order.
    return [list(names).index(f'{finger}_{part}') for finger in FINGERS for part in PARTS]


def landmark_specs(side):
    """Exact C Topo link1 mapping; DIP proxies are fixed offsets on distal links."""
    root = f'{"R" if side == "right" else "L"}_forearm_rot_y_link'
    rows = [(root, (0., 0., 0.))]  # replaced with semantic palm position by caller
    thumb = [f'{side}_hand_thumb_{suffix}' for suffix in ('bend_link', 'rota_link1', 'rota_link2')]
    rows += [(n, (0., 0., 0.)) for n in thumb]
    rows += [(thumb[-1], (.04188, -.00004 if side == 'right' else .00004, -.00746))]
    for finger, short in (('index', 'index_rota'), ('middle', 'mid'), ('ring', 'ring'), ('pinky', 'pinky')):
        links = [f'{side}_hand_{short}_link{i}' for i in (1, 2)]
        rows += [(n, (0., 0., 0.)) for n in links]
        sign = 1 if side == 'right' else -1
        rows += [(links[-1], (0., sign*.00401, -.01824)),
                 (links[-1], (0., sign*.00802, -.03648))]
    return rows


def bone_directions(points, rotation):
    """Unit bones in the hand's own palm axes; preserve leading batch dims."""
    a, b = EDGES.T
    bones = points[..., b, :] - points[..., a, :]
    bones = torch.matmul(bones, rotation)
    return torch.nn.functional.normalize(bones, dim=-1, eps=1e-8)


def bone_error(actual, reference):
    return (1-(actual*reference).sum(-1).clamp(-1, 1)).mean(-1)


def sample_proxcurv(mesh, tips_local, count=50, seed=0):
    def normalize(x):
        return x/x.max() if len(x) and x.max() > 0 else x
    curvature = np.zeros(len(mesh.faces))
    if len(mesh.face_adjacency):
        for column in (0, 1):
            np.add.at(curvature, mesh.face_adjacency[:, column], np.abs(mesh.face_adjacency_angles))
    else:
        curvature[:] = 1
    nearest = np.full(len(mesh.faces), np.inf)
    for start in range(0, len(tips_local), 512):
        d = mesh.triangles_center[:, None]-tips_local[None, start:start+512]
        nearest = np.minimum(nearest, np.square(d).sum(-1).min(-1))
    proximity = normalize(np.exp(-.5*nearest/.01**2))
    weights = mesh.area_faces*(1+normalize(curvature)+30*proximity)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError('Invalid object surface sampling weights')
    rng = np.random.default_rng(seed)
    faces = rng.choice(len(mesh.faces), size=count, replace=True, p=weights/weights.sum())
    uv = rng.random((count, 2)); u = np.sqrt(uv[:, 0])
    bary = np.column_stack((1-u, u*(1-uv[:, 1]), u*uv[:, 1]))
    return np.einsum('nf,nfd->nd', bary, mesh.triangles[faces])


def graph_connections(vertices):
    """Reference Delaunay adjacency only; no distance/weight precomputation."""
    from scipy.spatial import Delaunay
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape != (71, 3) or not np.isfinite(vertices).all():
        raise ValueError('Expected 71 finite MANO/object vertices')
    tri = Delaunay(vertices)
    pairs = np.unique(np.concatenate([tri.simplices[:, [a, b]]
                      for a, b in itertools.permutations(range(4), 2)]), axis=0)
    row, col = pairs.T
    degree = np.bincount(row, minlength=71)
    if (degree == 0).any():
        raise ValueError('Degenerate reference graph: isolated vertices')
    return row.astype(np.uint8), col.astype(np.uint8)


def graph_edges(vertices):
    """Weighted reference implementation for numerical A/B checks."""
    row, col = graph_connections(vertices)
    vertices = np.asarray(vertices, dtype=np.float64)
    weight = np.exp(-30*np.linalg.norm(vertices[row]-vertices[col], axis=-1))
    denominator = np.bincount(row, weights=weight, minlength=71)
    weight /= denominator[row]
    return row, col, weight.astype(np.float32)


def laplacian_error(delta, rows, columns, weights):
    """RMS of L*(actual-reference), with flattened global vertex indices."""
    flat = delta.reshape(-1, 3)
    neighbour_mean = torch.zeros_like(flat)
    neighbour_mean.index_add_(0, rows, weights[:, None]*flat[columns])
    return (flat-neighbour_mean).reshape_as(delta).square().sum(-1).mean(-1).sqrt()
