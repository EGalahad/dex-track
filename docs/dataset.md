# Pose-only v3 dataset contract (candidate)

The new loader contract is `dex-track-motion-folders-v3`, revision
`2026-11-10-pose-only-v3`. The laptop candidate is
`datasets/dex-track/C-full-simplified-v3/`: 704 segments, 367,678 native frames,
60 Hz, 72 scene objects, up to eight objects/segment. V2 poses and geometry are
preserved exactly. **Canonical RP14/HF remain v2 until the user approves the v3
receipts.** Do not overwrite running experiments or their frozen code/data.

`dataset.json` is the single source for robot convention, MANO joint names,
shared metadata/provenance and `robot.palm_in_root`. Each segment references it
with `dataset_file`. The palm transform is a frame-equal corpus calibration
(translation least squares and SO(3) mean), explicitly distinguished from the
original retarget transform. Task `palm_position_root` and
`palm_quaternion_root` default to null (dataset); explicit overrides are logged.
Residuals are recorded, not removed by changing trajectory poses.

The original source remains `C-original-source-v1/motion.pt`. The unmodified v2
is the conversion input and historical reference until promotion. Legacy aliases
must not be retargeted during healthy runs.

## One entry is one cropped stage segment

Each name ends in `__stNN`; it is not an entire recording with changing hand
assignments. Each segment has one fixed associated object per hand. Its
`metadata.json` contains `motion_segments[0]` with:

- `source_recording`, `stage_index`, and source stage interval `[start,end)` at
  `source_stage_fps` (120 Hz for this corpus).
- `stored_fps` (60 Hz), `frames`, and explicit `hand_objects.right/left` IDs.
- `scene_objects`, including required unassigned receivers/collision objects.
- Original semantic annotation and program checksum.

Retargeting cropped some stages again. The retained records do not establish
the exact crop offset, so `retarget_crop_source_interval` is null. A source stage
interval must not be interpreted as the exact stored trajectory interval.
The stored arrays are aligned to one another and have explicit lengths/rate.

## Per-segment files

| File | Contents |
|---|---|
| `robot.npz` | `qpos[T,38]`: right hand then left; each has root xyz, root quaternion wxyz, and 12 finger positions. Named joint ordering is in metadata. |
| `mano.npz` | Both hands' wrist position/rotation and 20 landmarks only; no velocity arrays. |
| `objects.npz` | Complete required scene position/quaternion poses, scene-index hand slots, lengths and support arrays; no velocities or duplicated associated-hand arrays. |
| `metadata.json` | Segment provenance, assignments, scene membership/scales, fixed supports and array references; shared fields reside in dataset.json. |
| `topology.npz` (optional) | Policy-rate adjacency and object-local samples; required by `tracking_topology`. The canonical 704-segment corpus currently has no graph sidecars; use `prepare_motion_topology.py` when preparing that task. |

There are no stored velocity arrays (robot, MANO or object), `obj_{side}_*`
duplicates, or per-hand mesh-directory/scale lists. A `motion_object_slot_{side}`
is an index into that segment's `motion_scene_object_ids[0]`; it must agree
with `motion_segments[0].hand_objects[side]`. Scales come only from
`motion_scene_object_scales`. Associated object poses are selected from the
scene tensors at runtime.

All poses are first resampled to the policy rate (normally 50 Hz): linear
positions/joints, shortest-arc SLERP orientations. Robot and object velocities
are then derived on that timeline: centered differences internally, one-sided
at endpoints, no cross-segment differences. World angular velocity is
`log(q_next * inverse(q_prev)) / elapsed`, matching simulator root velocity
writes. MANO wrist/finger velocity rewards derive from the same resampled poses
on demand. Stored q/-q hemisphere flips are valid and do not cause spikes.

The loader rejects v1/v2 with the converter command. Topology sidecars use the
new digest contract `dex-track-topology-v2`; regenerate them from v3 poses.

`objects/` retains original visuals and convex reference geometry. The portable
HF/laptop corpus excludes `.isaac_usd` and `.sdf_cache`; RP14 may retain these generated caches. `collision_simplified/` bundles verified simplified meshes; current
object factories prefer this bundle. Original visual/inertial inputs are
unchanged. Robot model/configuration remains versioned in code. `supports/`
contains the supplied fixed scene supports. Contact annotations removed here
were geometric MANO-to-visual-mesh labels, not simulator contacts. The historical
parking-candidate scripts require frozen v1 inputs/code if rerun.

### Shared object directory

Each of the 72 `objects/<object-id>/` directories has the following layout (generated directories are omitted from HF):

```
visual.obj                  original visual mesh
material.mtl, material_0.png optional material/texture
convex/*.obj                original convex hull pieces, retained as references
.isaac_usd/                 converted URDF/USD, physics layers, textures and config
.sdf_cache/*.pt             regeneratable signed-distance grids
```

Object trajectories live in each segment's `objects.npz`, not in this shared
directory. The simulation's verified simplified collision pieces are under
`collision_simplified/objects/<object-id>/convex/`.

The 69 SDF caches contain float32 tensors `[4,128,128,128]`: signed distance and
three spatial-gradient channels, about 32 MiB each (2.16 GiB total). They are
rebuilt by `bake_object_sdf_grid` from the union of the original convex reference
meshes, scale, extent and resolution. The cache key hashes these inputs.
They are derived geometry acceleration data, not recorded motion or contact
annotations. `tip_object_distance` is the consumer, but is disabled in the
default `tracking` config and absent from `tracking_topology`; command
initialization currently still loads them. The gradient channels are not used
by current reward consumers. The initial migration retained and inventoried them. The portable publication now
excludes them, while runtime initialization can still regenerate them as needed.

## Conversion and validation

From the AA checkout, create a new destination first:

```bash
uv run projects/dex-track/data_pipeline/convert_motion_storage.py \
  --source /absolute/old/index.json \
  --destination /absolute/new-dataset \
  --segment-manifest /absolute/source-stage-manifest.json
```

The converter checks every hand body's native-frame pose against the current
free-hand model, preserves finger coordinates, verifies the NPZ round trip,
and checks reference arrays/assignments/topology payloads. Migration is a format
change, not a new retarget solve or approval of the quarantined scene-cleanup
candidate. The 2026-11-09 migration verified all 367,678 frames; maximum hand-body
position discrepancy was 1.8e-8 m. Both compiled tasks passed reset/64-step smoke
checks. The representative replay retains existing ~5.7 mm contact penetration;
these checks do not establish physical cleanliness or policy success.

## Portable publication and local/server workflow

The current HF destination is `elijahgalahad/oakink2`, replacing the previous
reference/example-pack release under the same name. Publication omits all SDF
and Isaac USD directories; `release.json` and `checksums.sha256` describe the
portable contents. The server’s original checksum manifest additionally covers
its retained caches; the `index.json` and trajectory/mesh hashes are identical.

Keep the complete latest portable dataset and current code on the laptop. Sync
server-originated dataset edits back to the laptop and verify checksums. Before
running dex-track code remotely, run the appropriate server-family sync script
with the intended host selected and verify completion. Preserve healthy runs’
frozen workspaces/inputs. RP14’s old oakink2 HF cache is removed; local filesystem
inputs do not need a second server HF snapshot.

From the AA checkout:

```bash
.venv/bin/python projects/dex-track/scripts/download_tracking_data.py \
  --repo elijahgalahad/oakink2 --revision COMMIT_SHA \
  --output /absolute/path/C-full-simplified-v3
.venv/bin/python projects/dex-track/scripts/generate_sdf_cache.py \
  --dataset /absolute/path/C-full-simplified-v3 --device cpu
```

The downloader verifies hashes and restores segment asset links. Missing SDFs
are also built automatically at runtime. For original-convex Isaac object USD,
use an IsaacLab environment’s Python with
`scripts/generate_isaac_usd.py --dataset /absolute/path/C-full-simplified-v3`.
`--supports-only` works without Isaac; `--urdf-only` authors object URDFs without
object USD. This does not add Isaac free-hand/simplified-collision backend support.
The same tools and required helper sources are bundled in the HF release’s
`tools/` directory. See [release documentation](portable-dataset-card.md).

`data_pipeline/publish_portable_dataset.py` seals the explicitly selected current
v3 candidate and publishes only with `--publish`. Replacing a repository additionally
requires `--replace-repo --expected-old-revision SHA`. It verifies all uploaded
Git/LFS object hashes. The older `prepare_release.py`/`upload_dataset.py` workflow
belongs to the separately quarantined candidate and retains its admission gate.

Verified public revision (2026-11-09):
[`74219062eaea69324a4be9d6334c3305d4a67a9b`](https://huggingface.co/datasets/elijahgalahad/oakink2/tree/74219062eaea69324a4be9d6334c3305d4a67a9b).
All 5,063 release files were checked by Git-blob/LFS hashes from RP14 and independently
from the laptop. Portable payload: 588,206,454 bytes (560.96 MiB), plus HF’s generated
`.gitattributes`. Original server trajectories and geometry remain unchanged.

## V2 to v3 conversion and verification

```bash
cd /home/elijah/Documents/projects/simple-tracking/active-adaptation
uv run --project venv/mjlab --no-sync projects/dex-track/data_pipeline/convert_pose_only_v3.py \
  --source /absolute/path/C-full-simplified-v2 \
  --destination /absolute/path/C-full-simplified-v3 \
  --palm-calibration /absolute/path/palm-calibration.json \
  --receipt /absolute/path/conversion-receipt.json
```

The converter refuses in-place/existing destinations, proves retained pose
bytes and associated-pose equivalence before deletion, validates scene slots,
checks NPZ round trips and unchanged assets, and regenerates checksums. The
v3 handoff receipts and remaining promotion gate are under
`tasks/dex-track/dataset/pose-only-v3-20261110/` in the parent workspace.
