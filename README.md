# dex-track

Bimanual XHand tracking of human MANO and rigid-object demonstrations, built on
[Active Adaptation (AA)](https://github.com/Agent-3154/active-adaptation/tree/dev/dex-track).
AA provides the environment interface and switchable Isaac Lab / mjlab backends.
This project owns its MDP components, training loop and PPO implementation.
**Installing or importing mimic-lite is not required.**

## Layout and dependencies

```text
cfg/train.yaml                 training defaults and project-owned PPO config
cfg/task/tracking.yaml         action, observation, reward, termination and VOC
scripts/train.py               rollout buffer, PPO updates, logging and checkpoints
scripts/play.py                deterministic checkpoint evaluation
scripts/replay_scene.py        recorded full-scene playback with Viser
scripts/*oakink2*.py            annotation download, compacting and stage audits
scripts/*rigid_corpus.py        MANO reference, collision and reset-retarget pipeline
dex_track_learning/ppo.py      PPO, entropy schedule and adaptive KL learning rate
dex_track_learning/common.py   local policy helpers
dex_track/tasks/command.py     motion sampling, reference state and reset lifecycle
dex_track/tasks/motion.py      MotionData / ResetData tensorclass containers
dex_track/tasks/actions.py     separate wrist and finger action components
dex_track/tasks/observations/  proprioception, scene and intent terms
dex_track/tasks/rewards/       one class per reward term
dex_track/tasks/terminations.py independent termination classes
dex_track/tasks/scene_objects.py dataset-sized object slots and padding
```

Required project dependencies are AA, `any4hdmi` for tensor interpolation, and
`manifold3d` for collision geometry. The selected simulator and the normal AA
PyTorch/TorchRL stack must already be installed. Data preparation additionally
uses the pinned [FlashDexRetarget](https://github.com/Holiday-Robot/FlashDexRetarget)
source, CoACD and a locally licensed MANO model. Upstream source and robot assets
are checked against `upstream-lock.json`. `learning-provenance.json` records the
origin of the copied PPO implementation; it is provenance, not a dependency.

## Installation

Use Python 3.11 and an AA-compatible simulator environment. The commands below
run from the AA checkout; replace `PYTHON` with that environment's interpreter.
AA must include `dev/dex-track` commit `108cde6925787b4a66a3fea5178d92631d8bbeb4`
or a descendant containing its named-action and contact-playback changes.

```bash
git clone https://github.com/EGalahad/dex-track projects/dex-track
export PYTHON=/absolute/path/to/simulator/environment/bin/python
uv pip install --python "$PYTHON" -e 'projects/dex-track[data]'
"$PYTHON" projects/dex-track/scripts/setup_source.py
"$(dirname "$PYTHON")/aa-discover-projects"
"$PYTHON" projects/dex-track/scripts/train.py --cfg job
```

Enable the `dex_track` environment/learning project in AA's project manifest if
project discovery leaves it disabled. Existing unrelated AA projects can retain
their own dependencies; a dex-track-only installation needs no mimic-lite checkout.
The project uses editable installation so its scripts and Hydra config remain
next to the Python packages.

## Download the tracking data

The derived [oakink2 dataset](https://huggingface.co/datasets/elijahgalahad/oakink2)
contains compact MANO/object references, object meshes, stage-object audit metadata
and prepared single-task packs. It omits RGB recordings, SMPL-X arrays and MANO
model weights. Download directly on the training/data server:

```bash
export DATA=/absolute/path/to/oakink2
"$PYTHON" projects/dex-track/scripts/download_tracking_data.py \
  --part all --output "$DATA"
(cd "$DATA" && sha256sum --check --ignore-missing SHA256SUMS)
```

`--part training` downloads only prepared packs; `reference` downloads the full
compact source with meshes/program annotations; `annotations` downloads stage
object decisions. Pass `--revision <HF-commit>` to reproduce a fixed release.

```text
reference/*.npz                one compact file per source recording
reference/object_repair.tar.gz shared object meshes and descriptions
reference/program.tar.gz       stage intervals, action labels and descriptions
annotations/stage_objects.json all stages, required/candidate objects and evidence
annotations/rigid_candidates.json preparation candidates, not physics admission
training/{whiteboard,pour_milk,flask}/motion.pt
training/<task>/objects/<ID>/  visual and collision assets
release.json                  counts, readiness and package metadata
SHA256SUMS                    file integrity manifest
```

The first release covers 627 source recordings and 2840 stages. Object-set and
rigid eligibility labels are automated, evidence-backed audits, not a claim of
complete human verification. The full-corpus physics-ready training pack is still
pending retarget/collision/support checks. Liquid transfer and surface cleaning
are represented by hand/object pose tracking; fluid or ink dynamics are not simulated.

## Reproduce preparation from the original source

The scripts use the official OakInk2 annotation distribution and preserve source
frame IDs. A recording can include several task stages and background objects;
stage selection must not blindly retain every recorded object or discard every
unmentioned one.

```bash
export RAW=/absolute/path/to/oakink2-raw
export COMPACT=/absolute/path/to/oakink2-compact
export AUDIT=/absolute/path/to/oakink2-audit
export MANO=/absolute/path/to/licensed/MANO/models
export PREPARED=/absolute/path/to/oakink2-prepared

# Download annotation recordings and shared meshes/programs; no RGB video.
"$PYTHON" projects/dex-track/scripts/download_oakink2.py --all --output "$RAW"
# Losslessly store MANO arrays and object poses, without pickle object overhead.
"$PYTHON" projects/dex-track/scripts/compact_oakink2.py --raw "$RAW" --output "$COMPACT"

# Reconstruct each stage's objects from labels, descriptions and geometric evidence.
"$PYTHON" projects/dex-track/scripts/consolidate_oakink2_stage_objects.py \
  --raw "$RAW" --compact "$COMPACT" --output "$AUDIT/initial"
"$PYTHON" projects/dex-track/scripts/refine_oakink2_stage_geometry.py \
  --manifest "$AUDIT/initial/manifest.json" --raw "$COMPACT" \
  --output "$AUDIT/geometry" --full-rate-borderline
"$PYTHON" projects/dex-track/scripts/label_oakink2_simulation_eligibility.py \
  --manifest "$AUDIT/geometry/manifest.json" --output "$AUDIT/rigid"
"$PYTHON" projects/dex-track/scripts/audit_oakink2_stage_objects_v2.py \
  --manifest "$AUDIT/rigid/manifest.json" --raw "$COMPACT" --output "$AUDIT/v2"
```

For a published, consolidated candidate list, use
`$DATA/annotations/rigid_candidates.json` in the next step. To reproduce new
selection decisions, merge the baseline with audit-v2 additions using
`build_oakink2_release.py` and retain the complete exclusions/review list.

```bash
# MANO forward kinematics, stage slicing, scene coordinates and fixed hand roles.
"$PYTHON" projects/dex-track/scripts/prepare_rigid_corpus.py \
  --output "$PREPARED" --workers 4 oakink2 \
  --raw "$RAW" --compact "$COMPACT" --mano "$MANO" \
  --stage-manifest "$DATA/annotations/rigid_candidates.json"
# CoACD for every scene object, robot reset retarget, quality flags and packing.
"$PYTHON" projects/dex-track/scripts/retarget_rigid_corpus.py \
  --dataset "$PREPARED" --workers 16
"$PYTHON" projects/dex-track/scripts/validate_data.py "$PREPARED/motion.pt"
```

MANO_LEFT.pkl and MANO_RIGHT.pkl must be obtained under their own license. They
are never bundled in this repository or the derived dataset. Preparation records
failed/unsupported stages explicitly; it does not silently treat them as ready.
After preparation, replay human + object and robot + object, inspect all collision
meshes, and run physical reset/rollout checks before admitting a new training pack.
Retarget success alone does not establish stable contacts or adequate table support.

`motion.pt` uses relative `objects/<ID>/` paths, per-clip lengths, scene masks and
`meta.fps`. Source OakInk2 annotations are 120 Hz; prepared packs are currently
60 Hz. At load time each clip is independently interpolated to **50 Hz policy**;
physics runs at **200 Hz**. Positions/joints use linear interpolation, orientations
use shortest-arc SLERP. The source files are not rewritten and clips are not blended.

## Task MDP

- Retargeted robot joints are used **only at reset**. Observations, rewards and
  termination use MANO/object references, not robot-joint imitation targets.
- Actions: 12 wrist velocity components and 24 absolute finger joint targets.
  Wrist velocities integrate a pose setpoint; a wrench PD controller tracks it.
  Hands have gravity disabled; objects retain gravity. Reset setpoints come from
  command-managed MANO roots plus env origins and a fixed model frame transform.
- Object slots are allocated from the largest scene in the loaded dataset.
  More than 8 objects raises an error. Padding is placed below the scene and masked.
  Each wrist currently observes one fixed associated object per clip.
- Optional VOC stabilizes objects for the first second. `voc_active` is observed.
- Joint/link order is explicitly mapped across backends. Orientations in observations
  use rot6d; positions and velocities retain physical units before PPO VecNorm.

| Observation group | Contents | Dimensions |
|---|---|---:|
| proprioception | wrist pose/velocity, joint state, raw previous actions, wrist controller state, VOC flag | 163 |
| scene | associated-object surface points and nearest point + distance for 13 links per hand, in wrist frame | 872 |
| intent | future MANO wrist/tip and object spatial errors in current wrist frame, steps 1–10 | 660 |

| Reward | Weight | Form |
|---|---:|---|
| object tracking | 1.0 | `exp(-error / 0.1)` |
| fingertip/object distance | 0.3 | `exp(-max(distance - reference_distance, 0) / 0.05)` |
| wrist position | 0.15 | `exp(-error / 0.2)` |
| wrist orientation | 0.15 | `exp(-angle / 0.3)` |
| fingertip tracking | 0.3 | `exp(-error / 0.2)` |
| action regularization | -0.1 | wrist action L2 + finger action-change L2 |

Reward implementation defines the point/hand reductions; rewards are multiplied
by policy dt. Termination classes independently check object top-3-point error
(mean > 0.2 m for 5 consecutive steps), fingertip error (> 0.2 m after 15 steps),
physical velocity guards and motion end. Thresholds belong to each termination's
configuration. Motion end is a timeout; other conditions are failures.

## Train, inspect exploration, and play

Run these commands from the AA checkout. The defaults use the built-in
`dex_track_ppo`: rollout length 32, VecNorm, entropy coefficient 0.002 throughout,
MuON/AdamW optimization, desired KL 0.01 → 0.005 and adaptive actor learning rate.
The training loop uses a reusable rollout buffer. Evaluation is a separate script.

```bash
# Formal single-task run; W&B must be authenticated and online.
"$PYTHON" projects/dex-track/scripts/train.py \
  task.motion_file="$DATA/training/whiteboard/motion.pt" \
  backend=mjlab task.num_envs=4096 total_iters=1000 \
  algo.train_every=32 checkpoint_interval=250 upload_interval=250 \
  wandb.mode=online

# Inspect exploration: actual stochastic TRAINING, 16 visible environments.
"$PYTHON" projects/dex-track/scripts/train.py \
  task.motion_file="$DATA/training/pour_milk/motion.pt" \
  task.command.voc.enabled=true backend=mjlab \
  task.num_envs=16 headless=false wandb.mode=disabled
# Add checkpoint_path=/absolute/path/to/checkpoint_250.pt to load a checkpoint.

# Deterministic policy playback; no exploration noise.
"$PYTHON" projects/dex-track/scripts/play.py \
  --checkpoint /absolute/path/to/checkpoint_1000.pt \
  --motion-file "$DATA/training/whiteboard/motion.pt" \
  --backend mjlab --num-envs 1 --steps 1500 --save-scene \
  --output outputs/whiteboard-play
```

For Isaac training, use `backend=isaaclab` and
`task.robot.usd_path=/absolute/path/to/xhand_bimanual_spec.usd`. Generate the robot
USD with `scripts/convert_robot.sh` in an existing Isaac installation; object USD
conversion is handled by the asset loader. Isaac playback accepts `--usd-path`;
add `--render --contacts --headless` for contact-point/force overlays.
`scripts/replay_scene.py --help` describes Viser replay of the saved full scene,
including playback speed from 0.2× to 1.0× in 0.1 increments.

For joint training, point `task.motion_file` at a complete, physically checked
corpus pack and use `total_iters=4000`. The three prepared example packs are not a
substitute for the full eligible OakInk2 corpus. Old checkpoints targeting removed
observation schemas or external PPO classes are not supported.

## Attribution

The PPO was initialized from the user's validated mimic-lite implementation and
is now maintained here. AA remains responsible for simulator abstractions and
shared learning primitives. FlashDexRetarget-derived utilities retain their MIT
notice in `dex_track/tasks/reference/LICENSE`. OakInk2 data and derived stage
metadata retain the source dataset's CC-BY-SA-4.0 attribution; see the dataset card.
