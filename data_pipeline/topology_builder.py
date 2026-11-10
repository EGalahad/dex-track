"""Offline deterministic MANO/object sample and Delaunay adjacency generation."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from dex_track.assets import body_name
from dex_track.tasks.reference import topology_math as geometry

class TopologyBuilder:

    def __init__(self, command, cache=None):
        import trimesh
        from filelock import FileLock
        self.c = c = command
        self.indices = [geometry.mano_indices(c.dataset.mano_joint_names[s]) for s in c.hand_sides]
        motion = c.dataset.mano
        fingers = torch.stack([motion.mano_joint_pos_w[:, si, ids] for si, ids in enumerate(self.indices)], 1)
        self.mano = torch.cat((motion.mano_wrist_pos_w[:, :, None], fingers), 2)
        directions = geometry.bone_directions(self.mano, motion.mano_wrist_rotmat_w)
        if not torch.isfinite(directions).all() or (directions.norm(dim=-1) < 0.99).any():
            raise ValueError('MANO reference contains nonfinite or zero-length bones')
        self.bones = directions
        self.body_ids, offsets = ([], [])
        for side in c.hand_sides:
            specs = geometry.landmark_specs(side)
            self.body_ids.append([c.robot.body_names.index(body_name(n, c.env.backend)) for n, _ in specs])
            positions = [list(p) for _, p in specs]
            positions[0] = c.palm_position_root[side]
            offsets.append(positions)
        self.offsets = torch.tensor(offsets, device=c.device, dtype=torch.float32)
        catalog = c.scene_objects.catalog
        cache = Path(cache) if cache else catalog['path'].parent / '.topology-reward-cache-v1'
        cache.mkdir(parents=True, exist_ok=True)
        raw = catalog['raw']
        native_lengths = raw['motion_num_frames'].tolist()
        native_starts = raw['length_starts'].tolist()
        self.object_points = torch.empty(len(c.dataset.lengths), 2, 50, 3, device=c.device)
        self.graphs = []
        meshes = {}
        mano_cpu = self.mano.cpu().numpy()
        object_pos, object_rot = c.dataset.hand_object_trajectory()
        obj_p, obj_r = object_pos.cpu().numpy(), object_rot.cpu().numpy()
        code = hashlib.sha256(Path(geometry.__file__).read_bytes() + Path(__file__).read_bytes()).digest()
        graph_fields = ('ptr', 'rows', 'cols')
        for mi, (start, length) in enumerate(zip(c.dataset.starts.tolist(), c.dataset.lengths.tolist())):
            graphs = []
            for si, side in enumerate(c.hand_sides):
                slot = catalog['hand_slots'][mi, si].item()
                directory = catalog['path'].parent / catalog['directories'][mi][slot]
                scale = catalog['scales'][mi][slot]
                key = (str(directory), float(scale))
                if key not in meshes:
                    mesh = trimesh.load(directory / 'visual.obj', force='mesh')
                    mesh.apply_scale(scale)
                    meshes[key] = mesh
                mesh = meshes[key]
                a, n = (native_starts[mi], native_lengths[mi])
                ids = geometry.mano_indices(raw[f'mano_{side}_joint_names'])
                tips = raw[f'mano_{side}_joints'][a:a + n, [ids[i] for i in (3, 7, 11, 15, 19)]].cpu().numpy()
                pos = raw['scene_object_pos'][a:a + n,slot].cpu().numpy()
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_quat(raw['scene_object_quat'][a:a + n,slot].cpu().numpy(),scalar_first=True).as_matrix()
                tips_local = np.einsum('tpi,tij->tpj', tips - pos[:, None], rot).reshape(-1, 3)
                points = geometry.sample_proxcurv(mesh, tips_local)
                self.object_points[mi, si] = torch.as_tensor(points, device=c.device, dtype=torch.float32)
                object_vertices = np.einsum('tij,pj->tpi', obj_r[start:start + length, si], points)
                object_vertices += obj_p[start:start + length, si, None]
                vertices = np.concatenate((mano_cpu[start:start + length, si], object_vertices), 1)
                fingerprint = hashlib.sha256(code + np.ascontiguousarray(vertices, dtype='<f8').tobytes()).hexdigest()
                root = cache / fingerprint
                with FileLock(str(root) + '.lock'):
                    if not (root / 'complete.json').is_file() or not all(((root / f'{k}.npy').is_file() for k in graph_fields)):
                        self._write_graphs(root, vertices, fingerprint, include_weights='weights' in graph_fields)
                graphs.append({name: np.load(root / f'{name}.npy', mmap_mode='r') for name in graph_fields})
            self.graphs.append(graphs)
        print(f'[dex_track][topology] {len(self.graphs)} clips; 21+50 vertices/hand; source-only sparse cache {cache}', flush=True)
        self.laplacian_bank_bytes = 0
        self.adjacency_bank_bytes = 0
        self._prepare_adjacency_bank()

    def _prepare_adjacency_bank(self):
        c = self.c
        bits = np.zeros((len(c.dataset.mano), 2, 71, 9), dtype=np.uint8)
        for mi, graphs in enumerate(self.graphs):
            start = int(c.dataset.starts[mi])
            for si, graph in enumerate(graphs):
                ptr = graph['ptr']
                for frame, (lo, hi) in enumerate(zip(ptr[:-1], ptr[1:])):
                    rows, cols = (graph['rows'][lo:hi], graph['cols'][lo:hi])
                    np.bitwise_or.at(bits[start + frame, si], (rows, cols // 8), (1 << cols % 8).astype(np.uint8))
        self.edge_bits = torch.as_tensor(bits, device=c.device)
        self.adjacency_bank_bytes = self.edge_bits.numel() * self.edge_bits.element_size()
        print(f'[dex_track][topology] GPU topology-only bank {self.adjacency_bank_bytes / 2 ** 20:.2f} MiB; on-the-fly reference weights', flush=True)

    @staticmethod
    def _write_graphs(root, vertices, fingerprint, include_weights):
        root.mkdir(parents=True, exist_ok=True)
        rows, cols, weights, ptr = ([], [], [], [0])
        for frame, value in enumerate(vertices):
            try:
                if include_weights:
                    r, c, w = geometry.graph_edges(value)
                    weights.append(w)
                else:
                    r, c = geometry.graph_connections(value)
            except Exception as error:
                raise ValueError(f'MANO/object Delaunay failed at reference frame {frame}: {root}') from error
            rows.append(r)
            cols.append(c)
            ptr.append(ptr[-1] + len(r))
        arrays = dict(ptr=np.asarray(ptr, dtype=np.int64), rows=np.concatenate(rows), cols=np.concatenate(cols))
        if include_weights:
            arrays['weights'] = np.concatenate(weights)
        for name, value in arrays.items():
            temporary = root / f'{name}.partial.npy'
            np.save(temporary, value, allow_pickle=False)
            temporary.replace(root / f'{name}.npy')
        (root / 'complete.json').write_text(json.dumps({'fingerprint': fingerprint, 'frames': len(vertices), 'vertices': 71, 'source': 'MANO+associated physical object only', 'sampler': 'proxcurv50-seed0', 'graph': 'reference-frame Delaunay; no jitter or cutoff', 'optional_weights_for_ab_test': include_weights}))
