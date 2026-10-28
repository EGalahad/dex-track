"""Play a saved AA policy using its observation/action schema and current termination."""
import argparse
import json
from pathlib import Path
import sys
import torch
from omegaconf import OmegaConf
import active_adaptation as aa

# Register the project-owned PPO before loading its saved configuration.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dex_track_learning


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--motion-file', required=True)
    p.add_argument('--backend', default='mjlab', choices=['mjlab', 'isaaclab'])
    p.add_argument('--usd-path')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--render-gpu', type=int)
    p.add_argument('--camera-scale', type=float, default=1.0)
    p.add_argument('--camera-target', nargs=3, type=float, help='Explicit world-space camera look-at')
    p.add_argument('--num-envs', type=int, default=1)
    p.add_argument('--nconmax', type=int, help='mjlab contact buffer capacity override')
    p.add_argument('--njmax', type=int, help='mjlab constraint buffer capacity override')
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--render', action='store_true')
    p.add_argument('--save-scene', action='store_true', help='Save every entity link pose before autoreset for Viser replay')
    p.add_argument('--contacts', action='store_true', help='Isaac: overlay native PhysX contact/force reports and save raw contacts')
    p.add_argument('--output', default='outputs/dex-play')
    a = p.parse_args()
    if a.contacts and (a.backend != 'isaaclab' or not a.render or a.num_envs != 1):
        p.error('--contacts requires --backend isaaclab --render --num-envs 1')
    checkpoint = Path(a.checkpoint).resolve()
    cfg = OmegaConf.load(checkpoint.parent/'cfg.yaml')
    OmegaConf.set_struct(cfg, False)
    cfg.backend = a.backend
    cfg.device = a.device
    cfg.headless = a.headless
    cfg.app = {'headless':a.headless, 'enable_cameras':a.render}
    if a.render_gpu is not None:
        cfg.app.kit_args = f'--/renderer/multiGpu/enabled=false --/renderer/activeGpu={a.render_gpu}'
    cfg.wandb.mode = 'disabled'
    cfg.task.motion_file = str(Path(a.motion_file).resolve())
    cfg.task.command.motion_file = cfg.task.motion_file
    for obj in cfg.task.objects.values():
        if 'motion_file' in obj: obj.motion_file = cfg.task.motion_file
    raw = torch.load(cfg.task.motion_file, map_location='cpu', weights_only=False)
    centers = [raw[k][0] for k in ('obj_right_pos', 'obj_left_pos') if k in raw]
    if centers:
        center = torch.tensor(a.camera_target) if a.camera_target is not None else torch.stack(centers).mean(0)
        cfg.task.viewer.lookat = center.tolist()
        cfg.task.viewer.eye = (center + a.camera_scale * torch.tensor([.65, .8, .65])).tolist()
    cfg.task.num_envs = a.num_envs
    if a.backend == 'mjlab':
        if a.nconmax is not None:
            cfg.task.sim.nconmax = a.nconmax
        if a.njmax is not None:
            cfg.task.sim.njmax = a.njmax
    cfg.task.robot.usd_path = a.usd_path
    cfg.task.command.sampling = 'start'
    cfg.task.enable_cameras = a.render
    cfg.task.viewer.contact_reports = a.contacts
    cfg.task.max_episode_length = a.steps
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'dex_track/data.py'))['configure_scene_slots'](cfg)
    OmegaConf.resolve(cfg)
    aa.init(cfg, auto_rank=False)
    # Register this project's assets and MDP terms even on a fresh AA install.
    import dex_track
    from active_adaptation.helpers import make_env_policy, evaluate
    from active_adaptation.learning.modules.vecnorm import VecNorm
    env, policy = make_env_policy(cfg.task, cfg.algo, seed=0, headless=a.headless,
                                 device=cfg.device, checkpoint_path=str(checkpoint))
    if a.render:
        env.base_env.sensor_render_enabled = True
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    import active_adaptation.helpers as helpers
    import time, shutil
    recorder = None
    contacts = None
    if a.contacts:
        from active_adaptation.envs.backends.isaaclab.contact_visualization import ContactVisualization
        contacts = ContactVisualization(env.base_env, out)
        original_render = env.base_env.render
        def render_contacts(mode='human'):
            frame = original_render(mode)
            return contacts.overlay(frame) if mode == 'rgb_array' else frame
        env.base_env.render = render_contacts
    if a.save_scene:
        from dex_track.scene_recording import SceneRecorder
        recorder = SceneRecorder(env.base_env, cfg, checkpoint)
    if recorder is not None or contacts is not None:
        original_step = env.base_env._step
        def record_step(td):
            result = original_step(td)
            if recorder is not None:
                recorder.capture(result)
            if contacts is not None:
                contacts.capture(result)
            return result
        env.base_env._step = record_step
    render_start = time.time()
    try:
        with torch.inference_mode(), VecNorm.freeze():
            info, trajectory, stats = evaluate(env, policy.get_rollout_policy('eval'),
                                              render=a.render, seed=0)
        if a.render:
            videos = [p for p in Path(helpers.__file__).parent.glob('recording-*.mp4') if p.stat().st_mtime >= render_start]
            if videos:
                shutil.copy2(max(videos, key=lambda p:p.stat().st_mtime), out/'policy.mp4')
        info.update(checkpoint=str(checkpoint), backend=a.backend,
                    termination='current: object top3 error >0.2m for 5 consecutive steps; tip >0.2m after 15 steps; velocity guard')
        (out/'metrics.json').write_text(json.dumps(info, indent=2))
        torch.save(trajectory, out/'trajectory.pt')
        if recorder is not None:
            recorder.save(out/'scene_trajectory.pt')
            info['scene_trajectory'] = str(out/'scene_trajectory.pt')
        print(json.dumps(info, indent=2), flush=True)
    finally:
        if contacts is not None:
            contacts.close()
        env.close()

if __name__ == '__main__':
    main()
