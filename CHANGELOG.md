# Changelog

Runtime / training changes only (data-pipeline work omitted).

## Unreleased

### Logging
- Default experiment logger is local **Trackio** (`logger.backend=trackio`,
  `logger.mode=offline`) via `dex_track_learning/experiment_run.py`.
- `scripts/train.py` uses the shared run facade; W&B remains available with
  `logger.backend=wandb`.
- Checkpoints stay under the Hydra job cwd:
  `outputs/<date>/<run>/trackio/offline-run-*/files/`, with
  `checkpoint_latest.pt` symlinked in the Hydra run directory (W&B-style layout).

### Actions / control
- Added `FingerNominalReferenceModel`: second-order joint reference prefilter
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
  default); passive joint damping 1.0 and actuator Kd 2.5 unchanged.
- Added `joint_acc_l2` / `joint_vel_l2` regularization terms on finger joints
  (`task=tracking`).

### Task configs
- `tracking` now carries the former `tracking_ov` rewards (coarse/fine object,
  object linear/angular velocity; object weight 1) and `tracking_topology`
  the former `tracking_topov2_ov` settings (adds object velocity and MANO
  wrist velocity, bone sigma 0.1, hand-object Laplacian graph sigma 0.05).
  Both keep the finger reference-model action; `tracking` keeps the finger
  `joint_acc_l2` / `joint_vel_l2` terms. The two `_ov` cfgs are removed.

### Tools
- `scripts/visualize_motion.py`: Viser reference-motion viewer with motion
  picker and visual / collision geom checkboxes.
