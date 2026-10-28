# Ported from FlashDexRetarget 6f95e13 (MIT); see LICENSE.
"""Associated object state and collision-SDF queries."""
import torch
from .sdf import bake_object_sdf_grid as _bake_object_sdf_grid

class ObjectPropertiesMixin:

    def active_obj_slot(self, side: str) -> torch.Tensor:
        """Active object slot per env for ``side``. Shape: (B,), long."""
        return self.dataset.object_indices[self.motion_ids, self.hand_sides.index(side)]

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
        self._obj_sdf_extent = .3
        self._obj_sdf_n = 128
        for side, mesh_paths in paths.items():
            grids = [_bake_object_sdf_grid(path, scale, self._obj_sdf_extent,
                      self._obj_sdf_n, device=str(self.device))
                     for path, scale in zip(mesh_paths, scales[side])]
            self._obj_sdf_grids[side] = grids[0] if len(grids) == 1 else torch.stack(grids)

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
        from .math import quat_apply, quat_apply_inverse

        local = quat_apply_inverse(quat_b, delta)  # (B, K, 3) in object frame

        norm = local / self._obj_sdf_extent  # (B, K, 3)
        norm_zyx = norm.flip(-1)  # (B, K, 3) reordered to (z, y, x)
        if grid.dim() == 5:
            # Multi-object: pack envs by active slot into ONE batched grid_sample (batch=S);
            # object-count-flat and bit-identical vs the old per-slot launch loop.
            S = grid.shape[0]
            slot = self.active_obj_slot(side)  # (B,)
            if env_ids is not None:
                slot = slot[env_ids]
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
        return self.current_ref_motion.object_pos_w + self.env.scene.env_origins[:, None]

    @property
    def ref_object_quat_w(self):
        from .math import quat_from_matrix
        return quat_from_matrix(self.current_ref_motion.object_rotmat_w)

    def ref_object_pos_w_future(self, offsets):
        return self.ref_motion_future(offsets).object_pos_w + self.env.scene.env_origins[:, None, None]

    def ref_object_rotmat_w_future(self, offsets):
        return self.ref_motion_future(offsets).object_rotmat_w
