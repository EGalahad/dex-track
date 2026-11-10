# Changelog

Runtime / training changes only (data-pipeline work omitted).

## Unreleased

### Action presets
- Added `task/action=filtered_position` (default, scale0.5, omega20/zeta1) and
  `task/action=velocity` (no JRM, integrated velocity scale2 rad/s,
  PD velocity target0 and absolute joint-target observation).
- Both presets use actuator Kp20/Kd0.5, armature0.01 and passive damping0;
  wrist control is shared in behavior and reward definitions are unchanged.
- Removed duplicated inline action blocks from the two task definitions.


### Logging
- Default experiment logger is local **Trackio** (`logger.backend=trackio`,
  `logger.mode=offline`) via `dex_track_learning/experiment_run.py`.
- `scripts/train.py` uses the shared run facade; W&B remains available with
  `logger.backend=wandb`.
- Checkpoints stay under the Hydra job cwd:
  `outputs/<date>/<run>/trackio/offline-run-*/files/`, with
  `checkpoint_latest.pt` symlinked in the Hydra run directory (W&B-style layout).

### Actions / control
- Added `FingerFilteredPosition`: second-order joint reference prefilter
  (`omega` / `zeta`) over the same nominal + `action_scale` finger action space.
- `HandPair.set_joint_velocity_target` so the reference model can feed PD
  velocity targets.
- `task=tracking` and `task=tracking_topology` both use that action for the
  fingers (`omega: 20`, `zeta: 1`); the separate `trackingJRM` task is removed.
- Finger actuators are now mjlab `BuiltinPdActuator` (`ActuatorCfg(mode='explicit')`:
  native `<position>` kp + `<velocity>` kd, effort as a ±100 sum-clamp), so PD
  velocity targets are applied (the previous `<position>` actuator ignored
  them). With a zero velocity target the dynamics equal the old actuator
  bit-for-bit. The XML `<default><position>` class is neutralised in
  `xhand_free` so the added `<velocity>` elements do not inherit `kp` as an
  extra bias stiffness.
- Finger armature `0.1 → 0.01` everywhere (ActuatorCfg and the XML joint
  default); the later action presets set passive damping0 and actuator Kd0.5.
- Added `joint_acc_l2` / `joint_vel_l2` regularization terms on finger joints
  (`task=tracking`).

- Finger PD knobs on `task.robot`: `finger_stiffness` (Kp, 20), `finger_damping`
  (Kd, 0.5), `finger_armature` (0.01), `finger_passive_damping` (0.0);
  both hands receive them through the selected action preset.

### Reset
- `MotionTracking(reset_noise=...)`: training-only uniform hand reset noise
  (finger pos/vel, wrist pos/rot; objects untouched; eval never perturbs).
  `tracking` and `tracking_topology` default to finger ±0.025 rad, ±0.025
  rad/s, wrist ±2.5 mm, ±0.015 rad.

### Task configs
- `tracking` now carries the former `tracking_ov` rewards (coarse/fine object,
  object linear/angular velocity; object weight 1) and `tracking_topology`
  the former `tracking_topov2_ov` settings (adds object velocity and MANO
  wrist velocity, bone sigma 0.1, hand-object Laplacian graph sigma 0.05).
  Both keep the finger reference-model action; `tracking` keeps the finger
  `joint_acc_l2` / `joint_vel_l2` terms. The two `_ov` cfgs are removed.

### Tools
- `scripts/explore.py` (`cfg/explore.yaml`): rollout-only exploration viewer with the
  training-mode env (reset noise, training termination, sampled actions), no PPO update or logging;
  prints finger target error, out-of-range targets, finger speed and termination counts.
  `record_seconds=T` runs headless and writes a smooth real-time (or `video_slowdown`) mp4 of the first
  `record_envs` envs, since the live viewer runs below real time (per-step CPU overhead, ~40 ms on a laptop).
- `scripts/visualize_motion.py`: Viser reference-motion viewer with motion
  picker and visual / collision geom checkboxes.
