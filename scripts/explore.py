"""Watch training-time exploration without optimizing (same Hydra tree and overrides as train.py).

The env stays in training mode, so reset noise, training termination, uniform start sampling and the
termination curriculum behave exactly as in a rollout of train.py. Actions are sampled from the policy
distribution (ExplorationType.RANDOM). Without checkpoint_path the policy is freshly initialized, i.e. the
iteration-0 policy of a new run with the same seed; with checkpoint_path it explores around that policy.
No PPO update, no logger, no checkpoints.

Two modes:
  live (default)     viser viewer at http://localhost:8080. Tensor compilation reduces rollout overhead
                     after the initial warm-up; 4 envs by default, one GUI update per policy step.
                     realtime=true caps playback at 50 Hz.
  record_seconds=T   run headless for T simulated seconds, record every physics substep of the first
                     record_envs envs, then write a smooth real-time (or slowed) mp4 next to the run.

Extra keys (cfg/explore.yaml): realtime, explore_steps, print_every, record_seconds, record_envs,
video_slowdown (1 = real time, 4 = every physics substep at 50 fps), video_size [h, w], label.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import sys
import time
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.utils import set_exploration_type, ExplorationType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dex_track_learning  # noqa: F401  registers the project PPO
import active_adaptation as aa

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'cfg'


class Window:
    """Statistics over one print interval."""
    def __init__(self):
        self.n = 0; self.err = 0.; self.out = 0.; self.speed = 0.; self.vmax = 0.
        self.ends = []; self.reasons = {}; self.t0 = time.perf_counter()

    def report(self, step, step_dt):
        n = max(self.n, 1); rate = self.n / max(time.perf_counter() - self.t0, 1e-9)
        lines = [f"[explore] step {step}   {rate:.1f} policy steps/s = {rate*step_dt:.2f}x real time",
                 f"  finger  |pd_target - q|      {self.err/n:6.3f} rad",
                 f"          target out of range  {100*self.out/n:6.0f} %",
                 f"          speed rms / max      {self.speed/n:6.1f} / {self.vmax:.0f} rad/s"]
        if self.ends:
            mean = sum(self.ends) / len(self.ends); total = sum(self.reasons.values()) or 1
            lines.append(f"  episodes {len(self.ends):4d}   mean length {mean:5.1f} steps ({mean*step_dt:.2f} s)")
            lines.append('  ended by ' + ',  '.join(f'{k} {100*v/total:.0f}%' for k, v in sorted(self.reasons.items(), key=lambda kv: -kv[1])))
        else:
            lines.append('  episodes    0')
        print('\n'.join(lines) + '\n', flush=True)


def write_video(cfg, base, frames, dones, out, label):
    """Render recorded substep qpos of each env with CPU MuJoCo and tile them into one mp4."""
    import mujoco, mediapy
    from PIL import Image, ImageDraw
    m = base.sim._sim.mj_model; d = mujoco.MjData(m)
    T, E = frames.shape[:2]
    origins = base.scene.env_origins[:E].cpu().numpy()
    free = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    wrists = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and 'wrist' in m.joint(j).name]
    h, w = [int(x) for x in cfg.get('video_size', [360, 480])]
    renderer = mujoco.Renderer(m, h, w)
    slow = max(1, int(cfg.get('video_slowdown', 1))); sub = int(base.decimation); dt = float(base.physics_dt)
    stride = max(1, sub // slow); fps = int(round(1 / (dt * stride) / slow))
    cams = []
    for e in range(E):
        centre = np.mean([frames[:, e, a:a+3] - origins[e] for a in wrists], axis=(0, 1)) if wrists else np.zeros(3)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = centre; cam.distance = 0.9; cam.azimuth = 135.; cam.elevation = -25.
        cams.append(cam)
    cols = 2 if E > 1 else 1; rows = (E + cols - 1) // cols
    flash = np.zeros(E); hold = max(1, int(0.06 / (dt * stride)))  # short red border on each reset
    video = []
    for t in range(0, T, stride):
        tiles = []
        for e in range(E):
            d.qpos[:] = frames[t, e]
            for a in free:  # GPU worlds carry env-origin offsets; the CPU model is one world at the origin
                d.qpos[a:a+3] -= origins[e]
            mujoco.mj_forward(m, d); renderer.update_scene(d, camera=cams[e])
            img = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(img)
            if dones[t:t+stride, e].any():
                flash[e] = hold
            draw.text((8, 6), f'env {e}   t = {t*dt:5.2f} s', fill=(0, 0, 0))
            if flash[e] > 0:
                draw.rectangle([0, 0, w-1, h-1], outline=(220, 40, 40), width=4); draw.text((8, 20), 'reset', fill=(220, 40, 40)); flash[e] -= 1
            tiles.append(np.asarray(img))
        while len(tiles) < rows * cols:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.concatenate([np.concatenate(tiles[r*cols:(r+1)*cols], 1) for r in range(rows)], 0)
        banner = Image.new('RGB', (grid.shape[1], 22), (255, 255, 255))
        ImageDraw.Draw(banner).text((8, 5), label + (f'   ({slow}x slow motion)' if slow > 1 else '   (real time)'), fill=(0, 0, 0))
        video.append(np.concatenate([np.asarray(banner), grid], 0))
    mediapy.write_video(str(out), video, fps=fps)
    print(f'[explore] video written: {out}  ({len(video)} frames at {fps} fps)', flush=True)


@hydra.main(config_path=str(CONFIG_PATH), config_name='explore', version_base=None)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    from hydra.utils import to_absolute_path
    from hydra.core.hydra_config import HydraConfig
    record = cfg.get('record_seconds')
    cfg.headless = True if record else bool(cfg.get('headless', False)); cfg.app.headless = cfg.headless
    if 'wandb' in cfg:
        cfg.wandb.mode = 'disabled'
    realtime = bool(cfg.get('realtime', True)) and not record
    steps = cfg.get('explore_steps'); every = float(cfg.get('print_every', 2.0))
    checkpoint = cfg.get('checkpoint_path')
    checkpoint = to_absolute_path(str(Path(checkpoint).expanduser())) if checkpoint else None
    from dex_track.data import configure_scene_slots
    configure_scene_slots(cfg)
    OmegaConf.resolve(cfg)
    aa.init(cfg, auto_rank=False)
    import dex_track  # noqa: F401
    from active_adaptation.helpers import make_env_policy
    env, policy = make_env_policy(cfg.task, cfg.algo, seed=cfg.seed, headless=cfg.headless, device=cfg.device, checkpoint_path=checkpoint)
    base = env.base_env
    base.train()
    assert base.training
    step_dt = float(base.step_dt)
    finger = base.input_managers['finger']; robot = finger.asset
    low, high = robot.data.joint_pos_limits[0, finger.joint_ids].unbind(-1)
    pd_target = lambda: getattr(finger, 'q_bar', finger.joint_target)  # reference-model action filters the raw target
    r = cfg.task.robot
    action = type(finger).__name__ + (f' omega {finger.omega:g} zeta {finger.zeta:g}' if hasattr(finger, 'omega') else ' (no filter)')
    label = cfg.get('label') or (f"armature {r.get('finger_armature')}  Kp {r.get('finger_stiffness')}  Kd {r.get('finger_damping')}  "
                                 f"passive {r.get('finger_passive_damping')}  |  {action}  |  std {cfg.algo.init_noise_scale:g}  scale {finger.action_scale:g}")
    noise = dict(cfg.task.command.get('reset_noise') or {})
    mode = f'record {record} s (headless)' if record else ('live viewer http://localhost:8080' if not cfg.headless else 'headless')
    print('\n'.join(['', '[explore] setup',
                     f'  finger   {action}, action_scale {finger.action_scale:g}, init std {cfg.algo.init_noise_scale:g}',
                     f"  PD       armature {r.get('finger_armature')}, Kp {r.get('finger_stiffness')}, Kd {r.get('finger_damping')}, "
                     f"passive damping {r.get('finger_passive_damping')} (None = XML 1.0)",
                     f'  reset    {noise}', f'  policy   {checkpoint or "fresh (iteration 0)"}', f'  mode     {mode}', '']), flush=True)
    E = min(int(cfg.get('record_envs', 4)), env.num_envs); rec_frames, rec_dones = [], []
    if record:
        sim = base.sim._sim; original_step = sim.step
        def recording_step(*a, **k):
            out = original_step(*a, **k)
            rec_frames.append(sim.data.qpos[:E].detach().double().cpu().numpy().copy()); rec_dones.append(np.zeros(E, bool))
            return out
        sim.step = recording_step
        steps = int(round(float(record) / step_dt))
    rollout_policy = policy.get_rollout_policy('train')
    if cfg.task.get('tensor_compile') or cfg.task.get('rollout_compile'):
        print('[explore] compiling on first use; initial steps may be slow', flush=True)
    carry = env.reset()
    win = Window(); ages = torch.zeros(env.num_envs, device=env.device); step = 0
    try:
        with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
            while steps is None or step < int(steps):
                t0 = time.perf_counter()
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                q = robot.data.joint_pos[:, finger.joint_ids]; v = robot.data.joint_vel[:, finger.joint_ids]; raw = finger.joint_target
                win.n += 1; win.err += float((pd_target() - q).abs().mean()); win.speed += float(v.square().mean().sqrt())
                win.out += float(((raw < low) | (raw > high)).float().mean()); win.vmax = max(win.vmax, float(v.abs().max()))
                ages += 1; done = td['next', 'done'].flatten()
                if record and rec_dones:
                    rec_dones[-1][:] = done[:E].cpu().numpy()
                for i in torch.nonzero(done).flatten().tolist():
                    win.ends.append(int(ages[i])); ages[i] = 0
                    for key in base.termination_funcs:
                        if float(td['next', 'stats', 'termination', key][i]) > 0:
                            win.reasons[key] = win.reasons.get(key, 0) + 1
                step += 1
                if time.perf_counter() - win.t0 >= every:
                    win.report(step, step_dt); win = Window()
                if realtime:
                    time.sleep(max(0., step_dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print('[explore] stopped', flush=True)
    finally:
        if win.n:
            win.report(step, step_dt)
        if record and rec_frames:
            out = Path(HydraConfig.get().runtime.output_dir) / 'explore.mp4'; out.parent.mkdir(parents=True, exist_ok=True)
            write_video(cfg, base, np.stack(rec_frames), np.stack(rec_dones), out, label)
        env.close()


if __name__ == '__main__':
    main()
