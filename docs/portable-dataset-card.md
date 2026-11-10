---
license: cc-by-sa-4.0
language:
- en
tags:
- robotics
- manipulation
- mano
- oakink2
- dex-track
size_categories:
- n<1K
---
# OakInk2 — dex-track pose-only v3 dataset

Candidate release; publish only after receipt approval. The converted corpus: **704 stage segments, 367,678 native frames at
60 Hz, 72 scene objects**, with verified simplified object/hand collision meshes.
This replaces the old compact-reference/example-pack release at this repository
name. Use `index.json` at the repository root; old `reference/`, `annotations/`,
and `training/` paths no longer apply. Pin the new commit SHA for reproducibility.

Derived from [OakInk-v2](https://huggingface.co/datasets/kelvin34501/OakInk-v2),
prepared with [dex-track](https://github.com/EGalahad/dex-track) and
[FlashDexRetarget](https://github.com/Holiday-Robot/FlashDexRetarget) source
revision `6f95e133aa4b7ca6a784eac93725df3de4d41509`.
Dataset license remains CC-BY-SA-4.0; the included upstream conversion/SDF code
carries its MIT license in `tools/FlashDexRetarget-LICENSE`.
No MANO model weights, RGB videos, or simulator binaries are included.

## Files and format

Each `motions/<segment>/` contains:

| File | Contents |
|---|---|
| `robot.npz` | `qpos[T,38]`: right then left hand; each has xyz + wxyz quaternion + 12 named finger joints. Derive velocities per segment after resampling. |
| `mano.npz` | Both hands' wrist transforms and 20 finger landmarks; poses only. |
| `objects.npz` | Scene position/quaternion poses and hand slots into the scene ID list. No stored velocities or duplicated obj_* arrays. |
| `metadata.json` | NPZ field references, object assignments, scene membership/scales and stage provenance; dataset_file points to shared dataset.json. |

Coordinates are simulator z-up, metres, with convention labels in metadata.
Each entry is one cropped stage with fixed hand-object assignment, not a complete
recording. Source stage bounds are retained; the additional retarget crop offset
is unknown and explicitly null. Policy-rate interpolation normally uses 50 Hz.
There are no contact-label archives, stored robot joint velocities, or topology
sidecars in this release. Topology training needs separately generated graphs.

Shared `objects/<id>/` retains original visual meshes, optional materials/textures,
and original convex reference pieces used for inertia and geometric references.
`collision_simplified/` contains the verified simulation collision geometry.
`supports/` contains fixed support definitions and meshes.
`dataset.json`, `conversion-verification.json`, `release.json`, and
`checksums.sha256` record provenance and integrity.

**No `.sdf_cache` or `.isaac_usd` content is distributed.** These are generated
assets. Original source and existing server caches are retained separately; their
absence here does not remove required trajectories or physical mesh inputs.

## Download

Use `tools/download_tracking_data.py` from this release (also provided in the
project's `scripts/` directory), with `huggingface_hub` installed:

```bash
python tools/download_tracking_data.py --repo elijahgalahad/oakink2 \
  --revision COMMIT_SHA --output /absolute/path/C-full-simplified-v3
```

It verifies every SHA-256 and creates each segment's relative `objects` and
`supports` links. Raw `snapshot_download` alone does not restore these links.
Alternatively clone/download the files, then run the same command with
`--verify-only`. Use a fresh destination when changing releases.

## Rebuild generated assets

SDF grids need Python with NumPy, Torch, Trimesh, manifold3d and warp-lang
(the runtime was validated with Warp 1.12.0). The CPU default avoids starting a
simulator; `--device cuda:0` enables GPU baking:

```bash
python tools/generate_sdf_cache.py --dataset /absolute/path/C-full-simplified-v3
```

Only assigned hand-object/scale pairs need grids. Each grid is float32
`[4,128,128,128]` (distance plus XYZ gradient), built from the union of original
convex reference pieces. Geometry, scale, extent and resolution determine the
cache key. Optional `--object OBJECT_ID` selects a subset. The training command
also regenerates missing grids during initialization; exclusion saves distribution
space, not necessarily runtime cache space. Rebuilds need not be bitwise identical
to old caches across geometry/kernel/library versions; use the current generator
consistently when comparing runs.

Object USD conversion requires an IsaacLab/Isaac Sim environment (the original
converter targets Isaac Sim 5.1). Run with that environment's Python:

```bash
python tools/generate_isaac_usd.py --dataset /absolute/path/C-full-simplified-v3
```

Use `--supports-only` to regenerate support USDA without Isaac, or `--urdf-only`
to author object URDFs without producing object USD. `--force` rebuilds existing
outputs. Meshes and trajectories are not modified. Object exports reproduce the
old original-convex USD convention and inertial properties; they do not implement
Isaac support for the current free-hand/simplified-collision training task.
MuJoCo/mjlab does not need these USD exports.

## Validation and limits

The format migration checked all native frames, preserving retargeted finger
coordinates and hand-body poses (maximum discrepancy 1.8e-8 m). Reference arrays
and assignments round-trip exactly. Robot velocities are now finite differences
after resampling; root rotations use quaternion SLERP.

This release preserves the established scene geometry. It is **not** promotion
of the separate quarantined scene-cleanup/parking candidate, and does not certify
physical cleanliness or policy success. Existing representative penetration of
about 5.7 mm remains. Tasks track hand/object motion; pouring does not model fluids,
and surface interactions do not model ink removal.

## Pose-only and palm contract

Schema `dex-track-motion-folders-v3`, revision `2026-11-10-pose-only-v3`.
The loader resamples poses first and derives all required velocities at policy
rate, using shortest-arc rotations and per-segment boundaries. No NPZ velocity
arrays are stored. Hand object slots index `motion_scene_object_ids`, with
scales exclusively from `motion_scene_object_scales`. Shared metadata,
provenance, MANO joint ordering and `robot.palm_in_root` are in `dataset.json`.
The fixed palm transform is calibrated against the unchanged v2 corpus;
its source/method/residuals are recorded and it is not labelled an original
retarget constant. Explicit runtime overrides are logged. This does not claim
physical task success or zero retarget residual. Frozen v2 runs remain valid
with their frozen loaders; the new loader rejects v2 and requests conversion.
