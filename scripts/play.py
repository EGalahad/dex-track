"""Play policies with the same Hydra configuration tree as train.py."""
import hydra
import json
from pathlib import Path
import sys
import torch
from omegaconf import OmegaConf, DictConfig
import active_adaptation as aa

# Register the project-owned PPO before loading its saved configuration.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dex_track_learning


PROJECT_PATH = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_PATH / 'cfg'


def prepare_config(cfg: DictConfig) -> DictConfig:
    """Resolve paths without replacing the selected task/algo or overrides."""
    from active_adaptation.utils.wandb import parse_checkpoint_path
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import to_absolute_path

    OmegaConf.set_struct(cfg, False)
    overrides = HydraConfig.get().overrides.task if HydraConfig.initialized() else []
    explicit_keys = {x.lstrip('+').split('=', 1)[0] for x in overrides}
    # Training snapshots contain headless=true and thousands of environments.
    # Apply playback defaults even with --config-path, while honoring CLI overrides.
    if 'headless' not in explicit_keys:
        cfg.headless = False
    if 'task.num_envs' not in explicit_keys:
        cfg.task.num_envs = 16
    # Saved training snapshots also contain a distant camera.
    if 'task.viewer.eye' not in explicit_keys:
        cfg.task.viewer.eye = [0.55, -0.65, 0.48]
    if 'task.viewer.lookat' not in explicit_keys:
        cfg.task.viewer.lookat = [0.0, 0.0, 0.18]
    # Playback: policy twist integrates the pose target; D is pure damping.
    if 'task.input.wrist.desired_velocity_feedforward' not in explicit_keys:
        cfg.task.input.wrist.desired_velocity_feedforward = False
    if not cfg.get('checkpoint_path'):
        raise ValueError('Playback requires checkpoint_path=...; select task/algo as in train.py.')
    checkpoint_spec = str(cfg.checkpoint_path).strip()
    if not checkpoint_spec.startswith(('run:', 'http://', 'https://')):
        # Saved train configs may enable hydra.job.chdir. CLI file paths are
        # relative to the directory where the user invoked play, not Hydra's run dir.
        checkpoint_spec = to_absolute_path(str(Path(checkpoint_spec).expanduser()))
    checkpoint = parse_checkpoint_path(checkpoint_spec)
    if checkpoint is None:
        raise FileNotFoundError(cfg.checkpoint_path)
    cfg.checkpoint_path = str(Path(checkpoint).expanduser().absolute())
    motion = None if OmegaConf.is_missing(cfg.task, 'motion_file') else cfg.task.get('motion_file')
    motion = Path(to_absolute_path(str(Path(motion).expanduser()))) if motion else None
    if motion is not None and motion.is_dir():
        motion = motion / 'index.json'
    explicit_motion = 'task.motion_file' in explicit_keys
    if (motion is None or not motion.is_file()) and not explicit_motion:
        motion = Path(cfg.checkpoint_path).parent / 'motion.pt'
    if motion is None or not motion.is_file():
        raise FileNotFoundError('Set task.motion_file=... or place motion.pt beside the checkpoint.')
    cfg.task.motion_file = str(motion.resolve())
    cfg.task.command.motion_file = cfg.task.motion_file
    cfg.setdefault('app', {})
    cfg.app.headless = bool(cfg.headless)
    cfg.app.enable_cameras = bool(cfg.get('record_video', False))
    cfg.task.enable_cameras = cfg.app.enable_cameras
    if 'wandb' in cfg:
        cfg.wandb.mode = 'disabled'
    import runpy
    runpy.run_path(str(PROJECT_PATH/'dex_track/data.py'))['configure_scene_slots'](cfg, allow_play_subset=True)
    OmegaConf.resolve(cfg)
    return cfg


@hydra.main(config_path=str(CONFIG_PATH), config_name='play', version_base=None)
def main(cfg: DictConfig):
    cfg = prepare_config(cfg)
    from dex_track_learning.config import normalize_policy_config
    normalize_policy_config(cfg.algo)
    steps = cfg.get('eval_steps')
    record_video = bool(cfg.get('record_video', False))
    realtime = bool(cfg.get('realtime', True))
    save_scene = bool(cfg.get('save_scene', False))
    contact_reports = bool(cfg.get('contact_reports', False))
    if steps is not None and int(steps) <= 0:
        raise ValueError('eval_steps must be positive; leave it null for continuous playback.')
    if steps is None and (save_scene or contact_reports):
        raise ValueError('Full scene/contact diagnostics require eval_steps=N.')
    if contact_reports and (cfg.backend != 'isaaclab' or not record_video or cfg.task.num_envs != 1):
        raise ValueError('contact_reports requires backend=isaaclab record_video=true task.num_envs=1.')
    cfg.task.viewer.contact_reports = contact_reports
    checkpoint = Path(cfg.checkpoint_path)
    aa.init(cfg, auto_rank=False)
    # Register this project's assets and MDP terms even on a fresh AA install.
    import dex_track
    from active_adaptation.helpers import make_env_policy, evaluate
    from active_adaptation.learning.modules.vecnorm import VecNorm
    env, policy = make_env_policy(cfg.task, cfg.algo, seed=cfg.seed, headless=cfg.headless,
                                 device=cfg.device, checkpoint_path=str(checkpoint))
    wrist = env.base_env.input_managers['wrist']
    print('[Playback] wrist PD desired velocity: '
          + ('policy twist' if wrist.desired_velocity_feedforward else 'ZERO (linear and angular); policy twist only integrates pose'),
          flush=True)
    if record_video:
        env.base_env.sensor_render_enabled = True
    if steps is None:
        from active_adaptation.playback import play_policy
        try:
            with VecNorm.freeze():
                play_policy(env, policy, Path(cfg.get('output_dir', 'outputs/dex-play')).resolve() / 'policy.mp4',
                            record_video=record_video, realtime=realtime)
        finally:
            env.close()
        return
    out = Path(cfg.get('output_dir', 'outputs/dex-play')).resolve(); out.mkdir(parents=True, exist_ok=True)
    import active_adaptation.helpers as helpers
    import time, shutil
    recorder = None
    contacts = None
    if contact_reports:
        from active_adaptation.envs.backends.isaaclab.contact_visualization import ContactVisualization
        contacts = ContactVisualization(env.base_env, out)
        original_render = env.base_env.render
        def render_contacts(mode='human'):
            frame = original_render(mode)
            return contacts.overlay(frame) if mode == 'rgb_array' else frame
        env.base_env.render = render_contacts
    if save_scene:
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
    if realtime:
        original_eval_step = env.step_and_maybe_reset
        def realtime_step(td):
            start = time.monotonic()
            result = original_eval_step(td)
            time.sleep(max(0., float(env.base_env.step_dt) - (time.monotonic() - start)))
            return result
        env.step_and_maybe_reset = realtime_step
    render_start = time.time()
    try:
        with torch.inference_mode(), VecNorm.freeze():
            info, trajectory, stats = evaluate(env, policy.get_rollout_policy('eval'),
                                              render=record_video, seed=cfg.seed, max_steps=int(steps))
        if record_video:
            videos = [p for p in Path(helpers.__file__).parent.glob('recording-*.mp4') if p.stat().st_mtime >= render_start]
            if videos:
                shutil.copy2(max(videos, key=lambda p:p.stat().st_mtime), out/'policy.mp4')
        info.update(checkpoint=str(checkpoint), backend=cfg.backend,
                    termination=OmegaConf.to_container(cfg.task.termination, resolve=True))
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
