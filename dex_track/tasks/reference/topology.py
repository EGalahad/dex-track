"""Runtime MANO reference state and preprocessed adjacency bank; no graph building."""
import numpy as np
import torch
from . import topology_math as geometry
from ..transforms import matrix_from_quat
from ..transforms import rotate
from dex_track.assets import body_name


class TopologyReference:
    def __init__(self, command):
        self.c = c = command
        self.indices = [geometry.mano_indices(c.dataset.mano_joint_names[s]) for s in c.hand_sides]
        motion = c.dataset.mano
        fingers = torch.stack([motion.mano_joint_pos_w[:, si, ids]
                               for si, ids in enumerate(self.indices)], 1)
        self.mano = torch.cat((motion.mano_wrist_pos_w[:, :, None], fingers), 2)
        directions = geometry.bone_directions(self.mano, motion.mano_wrist_rotmat_w)
        if not torch.isfinite(directions).all() or (directions.norm(dim=-1) < .99).any():
            raise ValueError('MANO reference contains nonfinite or zero-length bones')
        self.bones = directions
        self.body_ids, offsets = [], []
        for side in c.hand_sides:
            specs = geometry.landmark_specs(side)
            self.body_ids.append([c.robot.body_names.index(body_name(n, c.env.backend)) for n, _ in specs])
            positions = [list(p) for _, p in specs]
            positions[0] = c.palm_position_root[side]
            offsets.append(positions)
        self.offsets = torch.tensor(offsets, device=c.device, dtype=torch.float32)
        self._load_dataset_topology(c.scene_objects.catalog)

    def _load_dataset_topology(self, catalog):
        """Load dataset adjacency once; command indexes the GPU bank per step."""
        from dex_track.topology_storage import validate_topology_payload, raw_motion_digest
        c = self.c
        payloads = catalog['raw'].get('motion_topology', [])
        if len(payloads) != len(c.dataset.lengths) or any((x is None for x in payloads)):
            raise ValueError('Topology task requires topology.npz in every motion folder; run prepare_motion_topology.py first')
        self.object_points = torch.empty(len(payloads), 2, 50, 3, device=c.device)
        banks = []
        object_pos, object_rot = c.dataset.hand_object_trajectory()
        for mi, payload in enumerate(payloads):
            start, length = (int(c.dataset.starts[mi]), int(c.dataset.lengths[mi]))
            validate_topology_payload(payload, self.mano[start:start + length].cpu().numpy(), object_pos[start:start + length].cpu().numpy(), object_rot[start:start + length].cpu().numpy(), c.dataset.fps, raw_motion_digest(catalog['raw'], mi))
            self.object_points[mi] = torch.as_tensor(payload['object_points'], device=c.device)
            bits = payload['edge_bits']
            banks.append(bits)
        self.edge_bits = torch.as_tensor(np.concatenate(banks), device=c.device)
        self.adjacency_bank_bytes = self.edge_bits.numel() * self.edge_bits.element_size()
        self.laplacian_bank_bytes = 0
        print(f'[dex_track][topology] dataset adjacency GPU bank {self.adjacency_bank_bytes / 2 ** 20:.2f} MiB; no runtime Delaunay', flush=True)

    def actual_landmarks(self):
        def compute():
            d = self.c.robot.data
            p = d.body_link_pos_w[:, self.body_ids]
            q = d.body_link_quat_w[:, self.body_ids]
            return p+rotate(q, self.offsets.expand_as(p))
        return self.c._tracking_value('topology_actual_landmarks', compute)


    def bone_errors(self):
        def compute():
            actual = geometry.bone_directions(self.actual_landmarks(), matrix_from_quat(self.c.robot_palm_quat_w))
            return geometry.bone_error(actual, self.bones[self.c.frame_indices])
        return self.c._tracking_value('topology_bone_errors', compute)


    def laplacian_errors(self):
        """Reference distances, exp weights, normalization and residual all GPU."""
        def compute():
            c = self.c
            frames = c.frame_indices
            points = self.object_points[c.motion_ids]
            reference_object = torch.matmul(points, matrix_from_quat(c.ref_object_quat_w).transpose(-1, -2))
            reference_object += c.ref_object_pos_w[:, :, None] - c.env.scene.env_origins[:, None, None]
            reference = torch.cat((self.mano[frames], reference_object), 2)
            actual_object = torch.matmul(points, matrix_from_quat(c.object_quat_w).transpose(-1, -2))
            actual_object += c.object_pos_w[:, :, None]-c.env.scene.env_origins[:, None, None]
            actual_hand = self.actual_landmarks()-c.env.scene.env_origins[:, None, None]
            from .laplacian_kernel import laplacian_on_the_fly
            return laplacian_on_the_fly(reference, torch.cat((actual_hand, actual_object), 2), c.current_topology_edges(), graph_mode=c.laplacian_graph_mode)
        return self.c._tracking_value('topology_laplacian_errors_on_the_fly', compute)


    def future_fingers(self, offsets):
        c = self.c
        ref = c.ref_motion_future(offsets)
        points = torch.stack([ref.mano_joint_pos_w[:, :, si, ids]
                              for si, ids in enumerate(self.indices)], 2)
        return torch.matmul(points-ref.mano_wrist_pos_w[..., None, :], ref.mano_wrist_rotmat_w)
