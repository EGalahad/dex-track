"""Short, per-episode object pose assistance through AA's physics callback."""
import math
import torch


class VirtualObjectController:
    def __init__(self, command, enabled=False, duration=1., kp_pos=200., kd_pos=10.,
                 kp_rot=.2, kd_rot=.02, max_force=20., max_torque=1.,
                 gravity_compensation=True):
        self.c, self.env = command, command.env
        from dex_track.tasks.tensor_kernels import select_kernel, object_wrench
        self._object_wrench = select_kernel(self.env, object_wrench)
        self.enabled, self.duration = enabled, duration
        self.gains = (kp_pos, kd_pos, kp_rot, kd_rot, max_force, max_torque)
        if any(not math.isfinite(x) or x <= 0 for x in (duration, *self.gains)):
            raise ValueError('VOC duration, gains and limits must be positive and finite')
        self.gravity_compensation = gravity_compensation
        slots = command.scene_objects
        self.names = slots.names
        self.force = torch.zeros(command.num_envs, len(self.names), 3, device=command.device)
        self.torque = torch.zeros_like(self.force)
        self.active = torch.zeros(command.num_envs, device=command.device, dtype=torch.bool)
        self.masses = []
        if enabled:
            for name in self.names:
                obj = self.env.scene[name]
                if len(obj.body_names) != 1:
                    raise ValueError('VOC currently requires single-body rigid object entities')
                if self.env.backend == 'isaaclab':
                    mass = obj.root_physx_view.get_masses().to(command.device).reshape(command.num_envs, 1)
                else:
                    mass = self.env.sim.model.body_mass[:, obj.indexing.body_ids].reshape(-1, 1)
                self.masses.append(mass)

    def indicator(self):
        """Assistance for the next action, including immediately after reset."""
        active = (self.env.episode_length_buf * self.env.step_dt < self.duration - 1e-7) & self.enabled
        active = active & self.c.scene_objects.mask.any(-1)
        return active.float().unsqueeze(-1)

    def _write(self, index, force, torque, env_ids=None):
        obj = self.env.scene[self.names[index]]
        if self.env.backend == 'isaaclab':
            obj.set_external_force_and_torque(force[:, None], torque[:, None],
                                             body_ids=[0], env_ids=env_ids, is_global=True)
        else:
            obj.write_external_wrench_to_sim(force[:, None], torque[:, None],
                                            body_ids=[0], env_ids=env_ids)

    def reset(self, env_ids):
        self.active[env_ids] = False
        self.force[env_ids] = 0
        self.torque[env_ids] = 0
        if self.enabled:
            for i in range(len(self.names)):
                self._write(i, self.force[env_ids, i], self.torque[env_ids, i], env_ids)

    def apply(self, substep):
        if not self.enabled:
            return
        c, env = self.c, self.env
        # Episode age, not reference time: middle-frame resets also receive 1 s.
        age = env.episode_length_buf * env.step_dt + substep * env.physics_dt
        self.active.copy_(age < self.duration - 1e-7)
        slots = c.scene_objects
        target_p, target_q, target_v, target_w = [slots.ref(k) for k in ('pos','quat','lin_vel','ang_vel')]
        mask = slots.mask
        for i, name in enumerate(self.names):
            d = env.scene[name].data
            p, q = d.root_link_pos_w, d.root_link_quat_w
            weight = (self.active & mask[:, i])[:, None]
            self.force[:, i], self.torque[:, i] = self._object_wrench(
                p, q, d.root_link_lin_vel_w, d.root_link_ang_vel_w, d.root_com_pos_w,
                target_p[:, i], target_q[:, i], target_v[:, i], target_w[:, i],
                self.masses[i], weight, self.gains, self.gravity_compensation)
            # Explicit zero writes after cutoff prevent stale backend wrenches.
            self._write(i, self.force[:, i], self.torque[:, i])
