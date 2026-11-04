"""Reference sampling and kinematic state in AA's command lifecycle."""
from pathlib import Path
import math
from types import SimpleNamespace
import torch
from active_adaptation.envs.mdp.commands.base import Command
from dex_track.assets import description, robot_model, body_name
from .motion import MotionData, load_motion_dataset
from .reference.hand import HandPropertiesMixin
from .reference.objects import ObjectPropertiesMixin
from .tracking_errors import TrackingErrorsMixin


class MotionTracking(TrackingErrorsMixin, HandPropertiesMixin, ObjectPropertiesMixin, Command, namespace='dex_track'):
    def __init__(self, motion_file, sampling='uniform', object_tracking_radius=.4, palm_position_root=None):
        Command.__init__(self)
        self._tracking_cache = {}
        self._future_offsets = {}
        self.motion_file, self.sampling = str(motion_file), sampling
        if palm_position_root is not None:
            self.palm_position_root = dict(palm_position_root)
        self.object_tracking_radius = float(object_tracking_radius)
        if not math.isfinite(self.object_tracking_radius) or self.object_tracking_radius <= 0:
            raise ValueError('Object tracking radius must be finite and positive')
        if sampling not in ('uniform', 'start'):
            raise ValueError(sampling)

    def _initialize(self, env):
        Command._initialize(self, env)
        from dex_track.tasks.tensor_kernels import select_kernel, site_kinematics
        self._site_kinematics = select_kernel(env, site_kinematics)
        self.robot = self.asset
        desc, model = description(), robot_model()
        self.finger_names = desc['body_names']['fingers']
        from dex_track.data import load_scene_catalog
        catalog = load_scene_catalog(self.motion_file)
        raw = catalog['raw']
        # Packed reset joint columns follow the asset's MJCF order.
        source_names = [model.joint(i).name for i in range(model.njnt)]
        order = [source_names.index(n) for n in self.robot.joint_names]
        self.dataset = load_motion_dataset(raw, scene_state=catalog['state'],
            step_dt=env.step_dt, joint_order=order, device=self.device)
        if getattr(self.robot, 'is_free_hand_pair', False):
            # Preserve the retargeted reset pose/twist from the original packed
            # virtual-joint representation; only its simulator encoding changes.
            import numpy as np
            import mujoco
            full = load_motion_dataset(raw, scene_state=catalog['state'],
                step_dt=env.step_dt, joint_order=list(range(model.njnt)), device='cpu')
            md = mujoco.MjData(model)
            ids = [model.body(n).id for n in ('R_forearm_rot_y_link','L_forearm_rot_y_link')]
            roots = []
            jp=np.zeros((3,model.nv)); jr=np.zeros_like(jp)
            for qp,qv in zip(full.reset.joint_pos.numpy(),full.reset.joint_vel.numpy()):
                md.qpos[:]=qp;md.qvel[:]=qv;mujoco.mj_forward(model,md)
                row=[]
                for body in ids:
                    mujoco.mj_jacBody(model,md,jp,jr,body)
                    row.append(np.concatenate((md.xpos[body],md.xquat[body],jp@qv,jr@qv)))
                roots.append(row)
            self.free_root_states=torch.tensor(np.asarray(roots),device=self.device,dtype=torch.float32)
        print(f'[dex_track][motion_dataset] {self.dataset.source_fps:g} -> {self.dataset.fps:g} Hz, '
              f'{len(self.dataset.lengths)} clips, {len(self.dataset.frames)} frames', flush=True)
        self.hand_sides = self.dataset.hand_sides
        self._tip_mano_ids = {side: [self.dataset.mano_joint_names[side].index(f'{f}_tip')
                                   for f in self.finger_names] for side in self.hand_sides}
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_steps = torch.zeros_like(self.motion_ids)
        self._refresh_reference()
        from dex_track.wrist_frames import WRIST_BODY, PALM_QUAT_ROOT
        self.wrist_body_ids = [self.robot.body_names.index(body_name(WRIST_BODY[side], env.backend))
                               for side in self.hand_sides]
        positions = getattr(self, 'palm_position_root', {side: [0., 0., 0.] for side in self.hand_sides})
        self.semantic_palm_p = torch.tensor([positions[side] for side in self.hand_sides], device=self.device)
        self.semantic_palm_q = torch.tensor([PALM_QUAT_ROOT[side] for side in self.hand_sides], device=self.device)
        self._tip_site_ids = {}
        sites = []
        for side in self.hand_sides:
            ids = [model.site(n).id for n in desc['site_names']['tip'][side]]
            self._tip_site_ids[side] = list(range(len(sites), len(sites)+len(ids)))
            sites.extend(ids)
        self.site_body_ids = [self.robot.body_names.index(body_name(model.body(int(i)).name, env.backend)) for i in model.site_bodyid[sites]]
        self.site_pos_b = torch.as_tensor(model.site_pos[sites], device=self.device, dtype=torch.float32)
        self.site_quat_b = torch.as_tensor(model.site_quat[sites], device=self.device, dtype=torch.float32)
        paths = {side: [str(Path(self.motion_file).resolve().parent / d / 'convex')
                        for d in raw[f'{side}_object_mesh_dirs']] for side in self.hand_sides}
        scales = {side: raw[f'{side}_object_mesh_scales'] for side in self.hand_sides}
        self._init_object_sdf(paths, scales)
        from .scene_objects import SceneObjects
        self.scene_objects = SceneObjects(self, catalog)
        self.update()

    @property
    def frame_indices(self):
        return self.dataset.frame_indices(self.motion_ids, self.motion_steps)

    def _refresh_reference(self):
        self._invalidate_tracking()
        # Dataset world axes; env-origin translation is applied by *_pos_w accessors.
        self.current_ref_motion: MotionData = self.dataset.frames[self.frame_indices]

    @property
    def retarget_joint_pos(self):
        """Opt-in robot joint reference at the current MANO/reward frame.

        The historical ``reset`` container retains its serialized interface;
        these full-frame positions can also be read by finger tracking rewards.
        Columns have already been mapped to this backend's robot joint order.
        """
        return self.dataset.reset.joint_pos[self.frame_indices]

    def ref_motion_future(self, offsets: torch.Tensor) -> MotionData:
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
        self._sites = SimpleNamespace(
            site_pos_w=p,
            site_quat_w=q,
            site_lin_vel_w=v,
            site_ang_vel_w=omega,
        )

    def reference_control_wrist_pose_w(self, env_ids):
        """MANO palm reference converted once to the controlled URDF wrist frame."""
        from .wrist_math import inverse, multiply, rotate
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
        if getattr(self.robot, 'is_free_hand_pair', False):
            state=self.free_root_states[self.frame_indices[env_ids]].clone()
            state[:,:,:3]+=origins[:,None]
            self.robot.write_wrist_states(state[:,:,:3],state[:,:,3:7],state[:,:,7:10],state[:,:,10:13],env_ids)
        else:
            self._write_initial_states({'robot': root}, env_ids)
        self.robot.write_joint_state_to_sim(self.dataset.reset.joint_pos[self.frame_indices[env_ids]], self.dataset.reset.joint_vel[self.frame_indices[env_ids]], env_ids=env_ids)
        self.scene_objects.reset(env_ids)
        if 'support' in self.env.scene.entities:
            # Support vertices already contain the scene-local support pose.
            # Move only the entity root to this parallel environment's origin.
            support = self.env.scene['support'].data.default_root_state[env_ids].clone()
            support[:, :3] += origins
            self._write_initial_states({'support': support}, env_ids)

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
    def __init__(self, motion_file, palm_position_root, laplacian_graph_mode='full', **kwargs):
        super().__init__(motion_file, **kwargs)
        self.palm_position_root = dict(palm_position_root)
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
