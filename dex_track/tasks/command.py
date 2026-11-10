# Hand/object reference accessors ported from FlashDexRetarget 6f95e13 (MIT); see LICENSE.
"""Reference sampling and kinematic state in AA's command lifecycle."""
from pathlib import Path
import math
import torch
from active_adaptation.envs.mdp.commands.base import Command
from dex_track.assets import description, robot_model, body_name
from .motion import ManoData, load_motion_dataset
from .reference.sdf import bake_object_sdf_grid as _bake_object_sdf_grid, object_sdf_extent
from .transforms import matrix_from_quat, quat_from_matrix, quat_apply_inverse
from .transforms import inverse, multiply, rotate, log
from .transforms import pose_error_in_anchor, compute_tracking_errors


class MotionTracking(Command, namespace='dex_track'):
    def __init__(self, motion_file, sampling='uniform', object_tracking_radius=.4, palm_position_root=None, reset_noise=None, palm_quaternion_root=None):
        Command.__init__(self)
        # Training-only uniform reset noise on the hands (objects and supports untouched).
        # Keys: finger_pos [rad], finger_vel [rad/s], wrist_pos [m per axis], wrist_rot [rad per axis].
        # All zero (default) leaves reset bit-identical to the noiseless path; eval (env.training False) never perturbs.
        noise = dict(finger_pos=0., finger_vel=0., wrist_pos=0., wrist_rot=0.)
        unknown = set(reset_noise or {}) - set(noise)
        if unknown:
            raise ValueError(f'Unknown reset_noise keys: {sorted(unknown)}')
        noise.update({k: float(v) for k, v in (reset_noise or {}).items()})
        if not all(math.isfinite(v) and v >= 0 for v in noise.values()):
            raise ValueError('reset_noise magnitudes must be finite and non-negative')
        self.reset_noise = noise
        self._tracking_cache = {}
        self._future_offsets = {}
        self.motion_file, self.sampling = str(motion_file), sampling
        self._palm_position_override = palm_position_root
        self._palm_quaternion_override = palm_quaternion_root
        self.object_tracking_radius = float(object_tracking_radius)
        if not math.isfinite(self.object_tracking_radius) or self.object_tracking_radius <= 0:
            raise ValueError('Object tracking radius must be finite and positive')
        if sampling not in ('uniform', 'start'):
            raise ValueError(sampling)

    def _initialize(self, env):
        Command._initialize(self, env)
        from dex_track.tasks.transforms import select_kernel, site_kinematics
        self._site_kinematics = select_kernel(env, site_kinematics)
        self.robot = self.asset
        desc, model = description(), robot_model()
        self.finger_names = desc['body_names']['fingers']
        from dex_track.data import load_scene_catalog
        catalog = load_scene_catalog(self.motion_file)
        raw = catalog['raw']
        self.dataset = load_motion_dataset(raw, scene_state=catalog['state'],
            hand_slots=catalog['hand_slots'], step_dt=env.step_dt,
            joint_names=self.robot.joint_names, device=self.device)
        print(f'[dex_track][motion_dataset] {self.dataset.source_fps:g} -> {self.dataset.fps:g} Hz, '
              f'{len(self.dataset.lengths)} segments, {len(self.dataset.mano)} frames', flush=True)
        self.hand_sides = self.dataset.hand_sides
        self._tip_mano_ids = {side: [self.dataset.mano_joint_names[side].index(f'{f}_tip')
                                   for f in self.finger_names] for side in self.hand_sides}
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_steps = torch.zeros_like(self.motion_ids)
        self._refresh_reference()
        from dex_track.wrist_frames import WRIST_BODY
        self.wrist_body_ids = [self.robot.body_names.index(body_name(WRIST_BODY[side], env.backend))
                               for side in self.hand_sides]
        calibration = raw['robot']['palm_in_root']
        positions = {side:calibration[side]['position_xyz'] for side in self.hand_sides}
        quaternions = {side:calibration[side]['quaternion_wxyz'] for side in self.hand_sides}
        for kind, override in [('position',self._palm_position_override),('quaternion',self._palm_quaternion_override)]:
            if override is not None:
                print(f'[dex_track][palm override] {kind}: dataset default overridden by {dict(override)}',flush=True)
                if kind == 'position': positions = dict(override)
                else: quaternions = dict(override)
        self.palm_position_root = positions
        self.palm_quaternion_root = quaternions
        self.semantic_palm_p = torch.tensor([positions[side] for side in self.hand_sides], device=self.device)
        self.semantic_palm_q = torch.tensor([quaternions[side] for side in self.hand_sides], device=self.device)
        if self.semantic_palm_p.shape != (2,3) or self.semantic_palm_q.shape != (2,4) or not torch.isfinite(self.semantic_palm_p).all() or not torch.isfinite(self.semantic_palm_q).all() or not torch.allclose(self.semantic_palm_q.norm(dim=-1),torch.ones(2,device=self.device),atol=1e-5):
            raise ValueError('Invalid palm calibration/override')
        self._tip_site_ids = {}
        sites = []
        for side in self.hand_sides:
            ids = [model.site(n).id for n in desc['site_names']['tip'][side]]
            self._tip_site_ids[side] = list(range(len(sites), len(sites)+len(ids)))
            sites.extend(ids)
        self.site_body_ids = [self.robot.body_names.index(body_name(model.body(int(i)).name, env.backend)) for i in model.site_bodyid[sites]]
        self.site_pos_b = torch.as_tensor(model.site_pos[sites], device=self.device, dtype=torch.float32)
        self.site_quat_b = torch.as_tensor(model.site_quat[sites], device=self.device, dtype=torch.float32)
        # SDF-bank indices are derived caches, not stored hand/scene mappings.
        paths, scales, bank_indices = {}, {}, []
        for si, side in enumerate(self.hand_sides):
            entries, indices = [], []
            for mi, row in enumerate(catalog['directories']):
                slot = int(catalog['hand_slots'][mi,si])
                entry = (str(Path(self.motion_file).resolve().parent/row[slot]/'convex'),catalog['scales'][mi][slot])
                if entry not in entries: entries.append(entry)
                indices.append(entries.index(entry))
            paths[side] = [p for p,_ in entries]
            scales[side] = [v for _,v in entries]
            bank_indices.append(indices)
        self._sdf_bank_indices = torch.tensor(bank_indices,device=self.device,dtype=torch.long).T
        self._init_object_sdf(paths, scales)
        from .scene_objects import SceneObjects
        self.scene_objects = SceneObjects(self, catalog)
        self.update()

    def _urdf_wrist_state(self):
        def compute():
            if getattr(self.robot, 'is_free_hand_pair', False):
                return self.robot.direct_wrist_state()
            d = self.robot.data
            ids = self.wrist_body_ids
            return (d.body_link_pos_w[:, ids], d.body_link_quat_w[:, ids],
                    d.body_link_lin_vel_w[:, ids], d.body_link_ang_vel_w[:, ids])
        return self._tracking_value('urdf_wrist_state', compute)

    def _semantic_palm_state(self):
        def compute():
            p, q, v, w = self._urdf_wrist_state()
            return self._site_kinematics(p, q, v, w,
                self.semantic_palm_p.expand_as(p), self.semantic_palm_q.expand_as(q))
        return self._tracking_value('semantic_palm_state', compute)

    @property
    def robot_wrist_pos_w(self):
        return self._urdf_wrist_state()[0]

    @property
    def robot_wrist_quat_w(self):
        return self._urdf_wrist_state()[1]

    @property
    def robot_wrist_lin_vel_w(self):
        return self._urdf_wrist_state()[2]

    @property
    def robot_wrist_ang_vel_w(self):
        return self._urdf_wrist_state()[3]

    @property
    def robot_palm_pos_w(self):
        return self._semantic_palm_state()[0]

    @property
    def robot_palm_quat_w(self):
        return self._semantic_palm_state()[1]

    @property
    def robot_palm_lin_vel_w(self):
        return self._semantic_palm_state()[2]

    @property
    def robot_palm_ang_vel_w(self):
        return self._semantic_palm_state()[3]

    @property
    def robot_tip_pos_w(self) -> torch.Tensor:
        """Robot fingertip site positions (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self.hand_sides:
            ids = self._tip_site_ids[side]
            parts.append(self.site_pos_w[:, ids])
        return torch.stack(parts, dim=1)

    @property
    def ref_mano_wrist_pos_w(self):
        return self.current_ref_motion.mano_wrist_pos_w + self.env.scene.env_origins[:, None]

    @property
    def ref_mano_wrist_quat_w(self):
        return quat_from_matrix(self.current_ref_motion.mano_wrist_rotmat_w)

    @property
    def ref_mano_tip_pos_w(self):
        return self._mano_tip_pos_w(self.current_ref_motion)

    def _mano_tip_pos_w(self, motion):
        return torch.stack([motion.mano_joint_pos_w[:, si, self._tip_mano_ids[side]]
                            + self.env.scene.env_origins[:, None]
                            for si, side in enumerate(self.hand_sides)], 1)

    def ref_mano_tip_pos_w_future(self, offsets):
        motion = self.ref_motion_future(offsets)
        return torch.stack([motion.mano_joint_pos_w[:, :, si, self._tip_mano_ids[side]]
                            + self.env.scene.env_origins[:, None, None]
                            for si, side in enumerate(self.hand_sides)], 2)

    def ref_mano_wrist_pos_w_future(self, offsets):
        return self.ref_motion_future(offsets).mano_wrist_pos_w + self.env.scene.env_origins[:, None, None]

    def ref_mano_wrist_rotmat_w_future(self, offsets):
        return self.ref_motion_future(offsets).mano_wrist_rotmat_w

    @property
    def ref_mano_tip_object_distance(self):
        return torch.stack([self.sdf_query(self.ref_mano_tip_pos_w[:, si], side,
            object_pos_w=self.ref_object_pos_w[:, si], object_quat_w=self.ref_object_quat_w[:, si])[0]
            for si, side in enumerate(self.hand_sides)], 1)

    def active_obj_slot(self, side: str) -> torch.Tensor:
        """Active object slot per env for ``side``. Shape: (B,), long."""
        return self._sdf_bank_indices[self.motion_ids, self.hand_sides.index(side)]

    @property
    def object_pos_w(self) -> torch.Tensor:
        """Sim object positions. Shape: (B, n_sides, 3)."""
        return torch.stack(
            [self._sim_obj_field(s, "body_link_pos_w") for s in self.hand_sides],
            dim=1,
        )

    @property
    def object_quat_w(self) -> torch.Tensor:
        """Sim object quaternions. Shape: (B, n_sides, 4)."""
        return torch.stack(
            [self._sim_obj_field(s, "body_link_quat_w") for s in self.hand_sides],
            dim=1,
        )

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        """Sim object velocities. Shape: (B, n_sides, 3)."""
        return torch.stack(
            [self._sim_obj_field(s, "body_link_lin_vel_w") for s in self.hand_sides],
            dim=1,
        )

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        """Sim object angular velocities. Shape: (B, n_sides, 3)."""
        return torch.stack(
            [self._sim_obj_field(s, "body_link_ang_vel_w") for s in self.hand_sides],
            dim=1,
        )

    def _init_object_sdf(self, paths, scales):
        self._obj_sdf_grids = {}
        self._obj_sdf_extents = {}
        self._obj_sdf_n = 128
        for side, mesh_paths in paths.items():
            extents = [object_sdf_extent(path, scale)
                       for path, scale in zip(mesh_paths, scales[side])]
            grids = [_bake_object_sdf_grid(path, scale, extent,
                      self._obj_sdf_n, device=str(self.device))
                     for path, scale, extent in zip(mesh_paths, scales[side], extents)]
            self._obj_sdf_grids[side] = grids[0] if len(grids) == 1 else torch.stack(grids)
            self._obj_sdf_extents[side] = torch.tensor(extents, device=self.device)

    def sdf_query(
        self,
        points_w: torch.Tensor,
        side: str,
        object_pos_w: torch.Tensor | None = None,
        object_quat_w: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the per-side baked SDF + gradient at world-frame points (B, K, 3) or (B, 3).
        Returns (sdf, grad_world); obj pose/env_ids override the sim pose (all envs)."""
        if side not in self._obj_sdf_grids:
            raise KeyError(
                f"sdf_query: no SDF grid baked for side {side!r}. Set "
                f"collision mesh directories for {side!r} in the motion dataset."
            )
        grid = self._obj_sdf_grids[side]  # (4, N, N, N) or (S, 4, N, N, N)
        si = self.hand_sides.index(side)
        if object_pos_w is None:
            object_pos_w = self.object_pos_w[:, si]  # (B, 3)
        if object_quat_w is None:
            object_quat_w = self.object_quat_w[:, si]  # (B, 4)

        squeeze_K = points_w.dim() == 2
        if squeeze_K:
            points_w = points_w.unsqueeze(1)  # (B, 1, 3)

        B, K = points_w.shape[0], points_w.shape[1]
        delta = points_w - object_pos_w[:, None, :]  # (B, K, 3)
        quat_b = object_quat_w[:, None, :].expand(B, K, 4)
        from .transforms import quat_apply, quat_apply_inverse

        local = quat_apply_inverse(quat_b, delta)  # (B, K, 3) in object frame

        extents = self._obj_sdf_extents[side]
        slot = None
        if grid.dim() == 5:
            slot = self.active_obj_slot(side)
            if env_ids is not None:
                slot = slot[env_ids]
            extents = extents[slot, None, None]
        norm = local / extents  # (B, K, 3)
        norm_zyx = norm.flip(-1)  # (B, K, 3) reordered to (z, y, x)
        if grid.dim() == 5:
            # Multi-object: pack envs by active slot into ONE batched grid_sample (batch=S);
            # object-count-flat and bit-identical vs the old per-slot launch loop.
            S = grid.shape[0]
            order = torch.argsort(slot)  # env ids grouped by slot
            counts = torch.bincount(slot, minlength=S)
            n_max = int(counts.max().item())
            starts = torch.cumsum(
                torch.cat([counts.new_zeros(1), counts[:-1]]), 0
            )  # (S,) first sorted-position of each slot
            within = torch.arange(B, device=self.device) - starts[slot[order]]
            packed = norm_zyx.new_zeros(S, n_max, K, 3)
            packed[slot[order], within] = norm_zyx[order]
            o = torch.nn.functional.grid_sample(
                grid,
                packed.reshape(S, n_max * K, 1, 1, 3),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (S, 4, n_max*K, 1, 1)
            o = o.view(S, 4, n_max, K).permute(0, 2, 3, 1)  # (S, n_max, K, 4)
            out = norm_zyx.new_empty(B, K, 4)
            out[order] = o[slot[order], within]
        else:
            sample_grid = norm_zyx.reshape(1, B * K, 1, 1, 3)
            grid_5d = grid.unsqueeze(0)  # (1, 4, N, N, N)
            out = torch.nn.functional.grid_sample(
                grid_5d,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (1, 4, B*K, 1, 1)
            out = out.view(4, B, K).permute(1, 2, 0)  # (B, K, 4)
        sdf = out[..., 0]  # (B, K)
        grad_local = out[..., 1:]  # (B, K, 3) in object frame
        grad_local = grad_local / (
            grad_local.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        )
        grad_world = quat_apply(quat_b, grad_local)  # (B, K, 3)

        if squeeze_K:
            sdf = sdf.squeeze(1)
            grad_world = grad_world.squeeze(1)
        return sdf, grad_world

    @property
    def ref_object_pos_w(self):
        slots = self.dataset.hand_slots[self.motion_ids]
        return self.dataset.objects.pos_w[self.frame_indices[:, None], slots] + self.env.scene.env_origins[:, None]

    @property
    def ref_object_quat_w(self):
        slots = self.dataset.hand_slots[self.motion_ids]
        return self.dataset.objects.quat_w[self.frame_indices[:, None], slots]

    def ref_object_pos_w_future(self, offsets):
        frames = self.dataset.frame_indices(self.motion_ids[:, None], self.motion_steps[:, None] + offsets[None])
        slots = self.dataset.hand_slots[self.motion_ids]
        return self.dataset.objects.pos_w[frames[:, :, None], slots[:, None]] + self.env.scene.env_origins[:, None, None]

    def ref_object_rotmat_w_future(self, offsets):
        frames = self.dataset.frame_indices(self.motion_ids[:, None], self.motion_steps[:, None] + offsets[None])
        slots = self.dataset.hand_slots[self.motion_ids]
        return matrix_from_quat(self.dataset.objects.quat_w[frames[:, :, None], slots[:, None]])


    def prepare_velocity_reference(self, kind):
        """Immutable derivatives; shared by velocity rewards and diagnostics."""
        if not hasattr(self, '_velocity_references'):
            self._velocity_references = {}
        if kind not in self._velocity_references:
            from .transforms import pose_velocities
            if kind == 'wrist':
                p = self.dataset.mano.mano_wrist_pos_w
                q = quat_from_matrix(self.dataset.mano.mano_wrist_rotmat_w)
            elif kind == 'finger':
                from .reference.finger_bodies import finger_body_reference
                p, q = finger_body_reference(self).world_trajectory()
            else:
                raise ValueError(kind)
            self._velocity_references[kind] = pose_velocities(
                p, q, self.dataset.starts, self.dataset.lengths, self.env.step_dt)
        return self._velocity_references[kind]

    def velocity_tracking_errors(self, kind):
        """World-frame MANO wrist or retarget finger-link velocity errors."""
        def compute():
            reference = self.prepare_velocity_reference(kind)
            target_v, target_w = (v[self.frame_indices] for v in reference)
            if kind == 'wrist':
                actual_v, actual_w = self.robot_palm_lin_vel_w, self.robot_palm_ang_vel_w
            else:
                from .reference.finger_bodies import finger_body_reference
                ids = [row[1:] for row in finger_body_reference(self).body_ids]
                d = self.robot.data
                actual_v = d.body_link_lin_vel_w[:, ids]
                actual_w = d.body_link_ang_vel_w[:, ids]
            return ((actual_v-target_v).norm(dim=-1),
                    (actual_w-target_w).norm(dim=-1))
        return self._tracking_value(('velocity', kind), compute)

    def _invalidate_tracking(self):
        self._tracking_cache.clear()

    def _tracking_value(self, key, compute):
        if key not in self._tracking_cache:
            self._tracking_cache[key] = compute()
        return self._tracking_cache[key]

    def future_offsets(self, steps):
        key = tuple(map(int, steps))
        if key not in self._future_offsets:
            self._future_offsets[key] = torch.tensor(key, device=self.device)
        return self._future_offsets[key]

    def wrist_tracking_errors(self, offsets=None):
        """Return world/local position delta, local rotation matrix, position/orientation magnitudes."""
        def compute():
            p, q = self.robot_palm_pos_w, self.robot_palm_quat_w
            if offsets is None:
                target_p, target_q = self.ref_mano_wrist_pos_w, self.ref_mano_wrist_quat_w
            else:
                target_p = self.ref_mano_wrist_pos_w_future(offsets)
                target_q = quat_from_matrix(self.ref_mano_wrist_rotmat_w_future(offsets))
                p, q = p[:, None], q[:, None]
            delta, error_q, position_error, orientation_error = compute_tracking_errors(p, q, target_p, target_q)
            return (delta, quat_apply_inverse(q.expand_as(target_q), delta),
                    matrix_from_quat(error_q), position_error, orientation_error)
        return self._tracking_value(('wrist', offsets), compute)

    def mano_tip_tracking_errors(self, offsets=None):
        """Return world/local tip position delta and distance."""
        def compute():
            actual = self.robot_tip_pos_w
            if offsets is None:
                target, q = self.ref_mano_tip_pos_w, self.robot_palm_quat_w[:, :, None]
            else:
                target = self.ref_mano_tip_pos_w_future(offsets)
                actual = actual[:, None]
                q = self.robot_palm_quat_w[:, None, :, None]
            delta = target-actual
            return (delta,
                quat_apply_inverse(q.expand(*delta.shape[:-1], 4), delta),
                delta.norm(dim=-1))
        return self._tracking_value(('mano_tip', offsets), compute)

    def finger_body_state(self):
        """Return world position/quaternion, then wrist-local position/quaternion."""
        def compute():
            from .reference.finger_bodies import finger_body_reference, local_poses
            reference = finger_body_reference(self)
            d = self.robot.data
            p, q = d.body_link_pos_w[:, reference.body_ids], d.body_link_quat_w[:, reference.body_ids]
            lp, lq = local_poses(p, q)
            return p[:, :, 1:], q[:, :, 1:], lp, lq
        return self._tracking_value('finger_body_state', compute)

    def finger_body_tracking_errors(self, reference='retarget', key=None):
        """Historical reward metric: each hand's own wrist-local finger poses.

        Kept distinct from full world-reference pose errors: wrist displacement
        is already rewarded by wrist tracking and must not be added here.
        """
        def compute():
            from .reference.finger_bodies import finger_body_reference
            _, _, actual_p, actual_q = self.finger_body_state()
            p, q = finger_body_reference(self).target(reference, key)
            return ((actual_p-p).norm(dim=-1),
                    log(multiply(actual_q, inverse(q))).norm(dim=-1))
        return self._tracking_value(('finger_body_tracking', reference, key), compute)

    def finger_body_pose_error_future(self, offsets):
        def compute():
            from .reference.finger_bodies import finger_body_reference
            reference = finger_body_reference(self)
            indices = self.dataset.frame_indices(self.motion_ids[:, None], self.motion_steps[:, None]+offsets)
            p, q = (v[indices] for v in reference.retarget)
            root = self.dataset.robot
            root_q = root.root_quat_w[indices][..., None, :].expand_as(q)
            target_p = root.root_pos_w[indices][..., None, :] + rotate(root_q, p)
            target_q = multiply(root_q, q)
            position_w, quaternion_w, _, _ = self.finger_body_state()
            actual_p = position_w[:, None]-self.env.scene.env_origins[:, None, None, None]
            actual_r = matrix_from_quat(quaternion_w[:, None])
            return pose_error_in_anchor(actual_p, actual_r, target_p, matrix_from_quat(target_q), actual_r)
        return self._tracking_value(('finger_body_future', offsets), compute)

    def object_pose_error_future(self, offsets):
        def compute():
            return pose_error_in_anchor(self.object_pos_w[:, None], matrix_from_quat(self.object_quat_w)[:, None],
                self.ref_object_pos_w_future(offsets), self.ref_object_rotmat_w_future(offsets),
                matrix_from_quat(self.robot_wrist_quat_w)[:, None])
        return self._tracking_value(('object_pose_future', offsets), compute)


    def object_pose_tracking_errors(self):
        """All valid scene objects once, not duplicated per associated hand."""
        def compute():
            scene = self.scene_objects
            delta, _, position_error, orientation_error = compute_tracking_errors(
                scene.sim('pos'), scene.sim('quat'), scene.ref('pos'), scene.ref('quat'))
            return delta, position_error, orientation_error
        return self._tracking_value('object_pose_current', compute)

    @property
    def frame_indices(self):
        return self.dataset.frame_indices(self.motion_ids, self.motion_steps)

    def _refresh_reference(self):
        self._invalidate_tracking()
        # Dataset world axes; env-origin translation is applied by *_pos_w accessors.
        self.current_ref_motion: ManoData = self.dataset.mano[self.frame_indices]

    @property
    def retarget_joint_pos(self):
        """Opt-in robot joint reference at the current MANO/reward frame.

        The historical ``reset`` container retains its serialized interface;
        these full-frame positions can also be read by finger tracking rewards.
        Columns have already been mapped to this backend's robot joint order.
        """
        return self.dataset.robot.joint_pos[self.frame_indices]

    def ref_motion_future(self, offsets: torch.Tensor) -> ManoData:
        return self._tracking_value(('future_reference', offsets),
            lambda: self.dataset.get_slice(self.motion_ids, self.motion_steps, offsets))

    def _sim_obj_field(self, side, attr):
        return self.scene_objects.hand_field(side, attr)

    def update(self):
        self._invalidate_tracking()
        d = self.robot.data
        p, q, v, omega = self._site_kinematics(
            d.body_link_pos_w[:, self.site_body_ids], d.body_link_quat_w[:, self.site_body_ids],
            d.body_link_lin_vel_w[:, self.site_body_ids], d.body_link_ang_vel_w[:, self.site_body_ids],
            self.site_pos_b.expand(self.num_envs, -1, -1), self.site_quat_b.expand(self.num_envs, -1, -1))
        self.site_pos_w, self.site_quat_w = p, q
        self.site_lin_vel_w, self.site_ang_vel_w = v, omega

    def reference_control_wrist_pose_w(self, env_ids):
        """MANO palm reference converted once to the controlled URDF wrist frame."""
        from .transforms import inverse, multiply, rotate
        p, q = self.reference_wrist_pose_w(env_ids)
        root_q = multiply(q, inverse(self.semantic_palm_q).expand_as(q))
        root_p = p-rotate(root_q, self.semantic_palm_p.expand_as(p))
        return root_p, root_q

    def reset(self, env_ids, tensordict):
        # Scene state was reset/refreshed before this callback; update site feedback too.
        self.update()

    def reference_wrist_pose_w(self, env_ids, sides=('right', 'left')):
        """Current sampled MANO roots, world position and wxyz quaternion.

        Motion/frame selection and environment-origin translation belong to the
        command. Consumers must not reconstruct this reference from robot state.
        """
        indices = [self.hand_sides.index(side) for side in sides]
        return (self.ref_mano_wrist_pos_w[env_ids][:, indices],
                self.ref_mano_wrist_quat_w[env_ids][:, indices])

    def step(self):
        self.motion_steps.add_(1)
        self._refresh_reference()

    def sample_init(self, env_ids, reset_td=None):
        self.motion_ids[env_ids] = self.scene_objects.sample_ids(env_ids)
        lengths = self.dataset.lengths[self.motion_ids[env_ids]]
        self.motion_steps[env_ids] = (torch.rand(len(env_ids), device=self.device)*lengths).long() if self.sampling == 'uniform' else 0
        self._refresh_reference()
        origins = self.env.scene.env_origins[env_ids]
        self.env.episode_origin[env_ids] = origins
        root = self.init_root_state[env_ids].clone()
        root[:, :3] += origins
        noisy = self.env.training and any(v > 0 for v in self.reset_noise.values())
        if getattr(self.robot, 'is_free_hand_pair', False):
            reference = self.dataset.robot[self.frame_indices[env_ids]]
            state = torch.cat((reference.root_pos_w, reference.root_quat_w,
                               reference.root_lin_vel_w, reference.root_ang_vel_w), -1)
            state[:,:,:3]+=origins[:,None]
            if noisy:
                state = self._noisy_wrist_state(state)
            self.robot.write_wrist_states(state[:,:,:3],state[:,:,3:7],state[:,:,7:10],state[:,:,10:13],env_ids)
        else:
            self._write_initial_states({'robot': root}, env_ids)
        joint_pos = self.dataset.robot.joint_pos[self.frame_indices[env_ids]]
        joint_vel = self.dataset.robot.joint_vel[self.frame_indices[env_ids]]
        if noisy:
            joint_pos, joint_vel = self._noisy_finger_state(joint_pos, joint_vel, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.scene_objects.reset(env_ids)
        if 'support' in self.env.scene.entities:
            # Support vertices already contain the scene-local support pose.
            # Move only the entity root to this parallel environment's origin.
            support = self.env.scene['support'].data.default_root_state[env_ids].clone()
            support[:, :3] += origins
            self._write_initial_states({'support': support}, env_ids)

    def _uniform(self, shape, scale):
        return (torch.rand(shape, device=self.device)*2-1)*scale

    def _noisy_wrist_state(self, state):
        """Perturb wrist position and orientation only; reference velocities are kept.
        The wrist controller's PD target is still initialised from the reference pose."""
        from .transforms import exp, multiply
        state = state.clone()
        state[..., :3] += self._uniform(state[..., :3].shape, self.reset_noise['wrist_pos'])
        delta = exp(self._uniform(state[..., :3].shape, self.reset_noise['wrist_rot']))
        state[..., 3:7] = torch.nn.functional.normalize(multiply(delta, state[..., 3:7]), dim=-1)
        return state

    def _noisy_finger_state(self, joint_pos, joint_vel, env_ids):
        limits = self.robot.data.joint_pos_limits[env_ids]
        joint_pos = (joint_pos + self._uniform(joint_pos.shape, self.reset_noise['finger_pos'])).clamp(limits[..., 0], limits[..., 1])
        joint_vel = joint_vel + self._uniform(joint_vel.shape, self.reset_noise['finger_vel'])
        return joint_pos, joint_vel

    def edit_spec(self, scene_config):
        import active_adaptation as aa
        if aa.get_backend() == 'isaaclab':
            scene_config.replicate_physics = False

    def object_tracking_error(self, top_k=3):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= self.scene_objects.points.shape[-2]:
            raise ValueError('Object top_k must be an integer within the sample count')
        return self._tracking_value(('object_point_error', top_k),
            lambda: self.scene_objects._compute_tracking_error(top_k))


class TopologyMotionTracking(MotionTracking, namespace='dex_track'):
    """MANO/object-only tracking; robot trajectory is an initialization source."""
    def __init__(self, motion_file, palm_position_root=None, laplacian_graph_mode='full', **kwargs):
        super().__init__(motion_file, palm_position_root=palm_position_root, **kwargs)
        if laplacian_graph_mode not in ('full', 'hand_object'):
            raise ValueError(laplacian_graph_mode)
        self.laplacian_graph_mode = laplacian_graph_mode

    def _initialize(self, env):
        super()._initialize(env)
        if not getattr(self.robot, 'is_free_hand_pair', False):
            raise ValueError('Topology task requires the audited free-hand pair')
        positions = torch.tensor([self.palm_position_root[s] for s in self.hand_sides],
                                 device=self.device, dtype=torch.float32)
        if positions.shape != (2, 3) or not torch.isfinite(positions).all():
            raise ValueError('Expected two finite semantic palm offsets')
        torch.testing.assert_close(self.semantic_palm_p, positions)
        from .reference.topology import TopologyReference
        self.topology = TopologyReference(self)

    def current_topology_edges(self):
        """Current reference-frame adjacency, GPU indexing without CPU transfers."""
        return self._tracking_value('current_topology_edges',
            lambda: self.topology.edge_bits[self.frame_indices])
