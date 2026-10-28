"""Backend-neutral scene recording, sampled after physics and before autoreset."""
from pathlib import Path
import torch
from omegaconf import OmegaConf


class SceneRecorder:
    def __init__(self, env, cfg, checkpoint):
        self.env = env
        self.cfg = OmegaConf.to_container(cfg, resolve=True)
        self.checkpoint = str(checkpoint)
        self.frames = []
        self.names = {name: list(entity.body_names) for name, entity in env.scene.entities.items()}
        self.joint_names = {name: list(getattr(entity, 'joint_names', []))
                            for name, entity in env.scene.entities.items()}

    def capture(self, result):
        def cpu(x): return x.detach().cpu().clone()
        entities = {}
        for name, entity in self.env.scene.entities.items():
            d = entity.data
            state = {key: cpu(getattr(d, attr)) for key, attr in (
                ('position', 'body_link_pos_w'), ('quaternion', 'body_link_quat_w'),
                ('linear_velocity', 'body_link_lin_vel_w'), ('angular_velocity', 'body_link_ang_vel_w'))}
            if self.joint_names[name]:
                state.update(joint_position=cpu(d.joint_pos), joint_velocity=cpu(d.joint_vel))
            entities[name] = state
        c = self.env.command_manager
        self.frames.append(dict(entities=entities, episode_id=cpu(self.env.episode_id),
            episode_step=cpu(self.env.episode_length_buf), done=cpu(result['done']),
            motion_id=cpu(c.motion_ids), motion_frame=cpu(c.motion_steps - 1)))

    def save(self, path):
        if not self.frames: raise ValueError('No scene frames recorded')
        entities = {name: dict(body_names=self.names[name], joint_names=self.joint_names[name],
            **{key: torch.stack([f['entities'][name][key] for f in self.frames])
               for key in self.frames[0]['entities'][name]}) for name in self.names}
        payload = dict(format='aa_scene_trajectory_v1', backend=self.env.backend,
            sample_phase='post_physics_pre_autoreset', quaternion_order='wxyz',
            dt=float(self.env.step_dt), num_envs=self.env.num_envs,
            config=self.cfg, checkpoint=self.checkpoint, entities=entities,
            **{key: torch.stack([f[key] for f in self.frames]) for key in
               ('episode_id', 'episode_step', 'done', 'motion_id', 'motion_frame')})
        path=Path(path); temporary=path.with_suffix('.partial.pt')
        torch.save(payload, temporary); temporary.replace(path)
        return payload
