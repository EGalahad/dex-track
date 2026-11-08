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
- New task cfg `cfg/task/trackingJRM.yaml` (`task=trackingJRM`) switches fingers
  to that action (`omega: 20`, `zeta: 1`).
- Tracking baseline: finger armature `0.1 → 0.01`; added `joint_acc_l2` /
  `joint_vel_l2` regularization terms on finger joints.

### Tools
- `scripts/visualize_motion.py`: Viser reference-motion viewer with motion
  picker and visual / collision geom checkboxes.
