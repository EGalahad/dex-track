# Dex-Track

XHand bimanual manipulation tracking on Active Adaptation/mjlab. The project owns
its MDP terms and shared-PointNet PPO; it has no runtime dependency on mimic-lite
or a local any4hdmi development checkout.

## Installation

Use Python 3.11 and an isolated CUDA environment. Install AA from its
`dev/dex-track` branch, then this project in editable mode:

```bash
git clone --branch dev/dex-track git@github.com:Agent-3154/active-adaptation.git
cd active-adaptation
git submodule update --init projects/dex-track
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . -e 'projects/dex-track[mjlab]'
.venv/bin/python projects/dex-track/scripts/apply_runtime_patches.py
.venv/bin/python projects/dex-track/scripts/setup_source.py
```

The collision runtime is pinned to mjlab 1.6.0, MuJoCo 3.11.0, MuJoCo-Warp 3.8.1
and Warp 1.12.0. `runtime_patches/manifest.json` locks the three audited collision
files (pair-local convex coordinates, original GJK, invalid mesh-ID guard). Apply
them only inside this isolated environment. Restart Python afterward; existing
training snapshots and environments must never be patched in place.
`setup_source.py` fetches the locked FlashDexRetarget asset/preprocessing source;
`upstream-lock.json` verifies the audited files. MANO models are separately licensed
and must be supplied by their owner for preprocessing.

## Dataset status

**The latest support/ground/parking/object-cleanup candidate is not published.**
It has 212 quarantined motions and `publish_allowed=false`. Do not infer physical
admission from a valid NPZ or finite rollout. See
[data_pipeline/README.md](data_pipeline/README.md) and
[data_pipeline/annotations/publication-status.json](data_pipeline/annotations/publication-status.json).
No new HF revision is claimed by this code release. Use an explicitly agreed,
frozen original training dataset until the candidate is approved.

Runtime input is a portable NPZ dataset's `index.json`, or one motion's
`metadata.json`. All paths inside metadata refer to dataset assets. The runtime
accepts only this NPZ format; original source PT is an offline conversion input.
Never rewrite a live run's input or any historical snapshot. Retain one canonical
original source and one canonical simplified training corpus; preprocessing
workspaces/caches are temporary and must not become duplicate full corpora.

## Train and play

Run from the AA checkout, replacing DATASET with an absolute NPZ folder:

```bash
uv run --python .venv/bin/python projects/dex-track/scripts/train.py \
  task=tracking task.motion_file=/absolute/DATASET/index.json
uv run --python .venv/bin/python projects/dex-track/scripts/train.py \
  task=tracking_topology task.motion_file=/absolute/DATASET/index.json
uv run --python .venv/bin/python projects/dex-track/scripts/play.py \
  task=tracking task.motion_file=/absolute/DATASET/index.json \
  checkpoint_path=/absolute/checkpoint.pt headless=true eval_steps=8
```

Formal training defaults to W&B online. A construction/PPO check uses
`task.num_envs=16 total_iters=3 wandb.mode=disabled`. Full-corpus default length
is 4000; single-motion studies use their requested length. Reference and policy
sampling rates come from metadata and `task.sim.step_dt` (normally 50 Hz).

## Current configuration surface

| Task | Purpose |
|---|---|
| `tracking` | 67/69 controller/reward settings: 18D controller state, wrist rotation D .3, position PD 350/25, four world velocity rewards; current shared semantic-palm frame convention |
| `tracking_topology` | MANO future fingers in semantic palm, bone sigma .2, full Laplacian sigma .1; retarget finger position/orientation weights .5, disabled by default |
| `tracking_ov` | Active coarse/fine object + object-velocity ablation |
| `tracking_topov2_ov` | Active topo-v2 + coarse/fine/object-velocity ablation: MANO wrist velocity, bone sigma .1, hand-object graph sigma .05 |

No historical observation-schema branches are shipped. The two ablation cfgs
remain because the current 13-run experiment actively uses them.

- Velocity: set individual `task.reward.tracking.*_velocity.enabled=false`; terms
  still compute/log. Object V terms are present in the O/V cfgs.
- Controller state: remove `task.observation.proprioception.wrist_controller_state`
  with Hydra `~task.observation.proprioception.wrist_controller_state`. This changes
  policy input width and requires a matching checkpoint.
- Collision: `task.robot.collision_variant=simplified` (default) or `original` for
  an explicit ablation; this setting propagates to both hands and scene objects.
- Retarget finger rewards: toggle `finger_position.enabled` and
  `finger_orientation.enabled`; topology defaults false, weights remain .5.
- Object O: sigmas .2/.04, weights 1/1 in O/V cfgs, versus .2/weight2 baseline.
- T: add `+task.termination.object_tracking_error.curriculum={final_threshold:0.1,start_fraction:0.25,end_fraction:0.75}`.
  Evaluation uses the common base .2 m / five consecutive steps; remove the
  curriculum when composing comparisons.
- Laplacian: `task.reward.tracking.laplacian.enabled=false` still computes/logs.

`enabled=false` excludes a reward only from training totals and critic heads.
It does not skip its diagnostics. Metrics live in a disabled group.

## Code layout

`dex_track/tasks/command.py` owns sampling/reset/reference state; `motion.py` and
`resampling.py` own the timeline. `actions/wrist.py` and `actions/finger.py` own
control. `observations/{proprioception,scene,intent}.py` only expose observations.
`rewards/{tracking,regularization,metrics}.py` only score command state;
`terminations.py` owns failure counters/curricula. Shared reference geometry is in
`tasks/reference/`; asset construction is outside the MDP in `assets.py`.

VOC and its observation/wrench, old controller/observation migration branches,
absolute joint-limit action mapping, old task configs, project tests, experiment
records, outputs, debug renderers and profilers are absent from this release.
The old runtime palm sites are removed; semantic palm is a fixed mathematical
transform. The source-asset loader strips upstream sites without changing the
locked upstream files. Data-processing XHand adapters may contain solver-only
landmark sites; they are never training controller or observation sites.

Removing VOC removes one constant-zero proprioception channel. Archived weights
must use their archived source or an explicit offline weight migration; the runtime
does not silently reinterpret historical observation schemas. New checkpoints
save the current observation contract.

Full preprocessing and review annotations live in `data_pipeline/`. Tests, logs,
checkpoints and validation evidence belong outside both published repositories.
