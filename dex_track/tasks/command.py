"""Reference sampling and kinematic state in AA's command lifecycle."""
from pathlib import Path
from types import SimpleNamespace
import torch
from active_adaptation.envs.mdp.commands.base import Command
from dex_track.assets import description, robot_model, body_name
from .motion import MotionData, load_motion_dataset
from .reference.hand import HandPropertiesMixin
from .reference.objects import ObjectPropertiesMixin


class MotionTracking(HandPropertiesMixin, ObjectPropertiesMixin, Command, namespace='dex_track'):
    def __init__(self, motion_file, sampling='uniform', voc=None):
        Command.__init__(self)
        self.motion_file, self.sampling = str(motion_file), sampling
        self.voc_cfg = dict(voc or {})
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
        print(f'[dex_track][motion_dataset] {self.dataset.source_fps:g} -> {self.dataset.fps:g} Hz, '
              f'{len(self.dataset.lengths)} clips, {len(self.dataset.frames)} frames', flush=True)
        self.hand_sides = self.dataset.hand_sides
        self._tip_mano_ids = {side: [self.dataset.mano_joint_names[side].index(f'{f}_tip')
                                   for f in self.finger_names] for side in self.hand_sides}
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_steps = torch.zeros_like(self.motion_ids)
        self._refresh_reference()
        self._palm_site_ids, self._tip_site_ids = {}, {}
        for side in self.hand_sides:
            self._palm_site_ids[side] = model.site(desc['site_names']['palm'][side]).id
            self._tip_site_ids[side] = [model.site(n).id for n in desc['site_names']['tip'][side]]
        self.site_body_ids = [self.robot.body_names.index(body_name(model.body(int(i)).name, env.backend)) for i in model.site_bodyid]
        self.site_pos_b = torch.as_tensor(model.site_pos, device=self.device, dtype=torch.float32)
        self.site_quat_b = torch.as_tensor(model.site_quat, device=self.device, dtype=torch.float32)
        paths = {side: [str(Path(self.motion_file).resolve().parent / d / 'convex')
                        for d in raw[f'{side}_object_mesh_dirs']] for side in self.hand_sides}
        scales = {side: raw[f'{side}_object_mesh_scales'] for side in self.hand_sides}
        self._init_object_sdf(paths, scales)
        from .scene_objects import SceneObjects
        self.scene_objects = SceneObjects(self, catalog)
        from dex_track.tasks.voc import VirtualObjectController
        self.voc = VirtualObjectController(self, **self.voc_cfg)
        self.update()

    @property
    def frame_indices(self):
        return self.dataset.frame_indices(self.motion_ids, self.motion_steps)

    def _refresh_reference(self):
        # Dataset world axes; env-origin translation is applied by *_pos_w accessors.
        self.current_ref_motion: MotionData = self.dataset.frames[self.frame_indices]

    def ref_motion_future(self, offsets: torch.Tensor) -> MotionData:
        return self.dataset.get_slice(self.motion_ids, self.motion_steps, offsets)

    def _sim_obj_field(self, side, attr):
        return self.scene_objects.hand_field(side, attr)

    def update(self):
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

    def pre_step(self, substep):
        self.voc.apply(substep)

    def reset(self, env_ids, tensordict):
        self.voc.reset(env_ids)

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
        self._write_initial_states({'robot': root}, env_ids)
        self.robot.write_joint_state_to_sim(self.dataset.reset.joint_pos[self.frame_indices[env_ids]], self.dataset.reset.joint_vel[self.frame_indices[env_ids]], env_ids=env_ids)
        self.scene_objects.reset(env_ids)

    def edit_spec(self, scene_config):
        import active_adaptation as aa
        if aa.get_backend() == 'isaaclab':
            scene_config.replicate_physics = False

    def object_tracking_error(self):
        return self.scene_objects.tracking_error()
