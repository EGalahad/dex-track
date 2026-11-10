# OakInk2 → C retarget → simplified NPZ

This folder contains dataset preparation, the exact C solver/adaptation, collision
simplification, reference topology generation and candidate scene cleanup. It
contains code and review decisions, not a second dataset. Pipeline dependencies
are in `requirements.txt`; install them into a separate preprocessing environment.
The bundled TopoRetarget numerical solver retains its upstream notices and the
mesh-based cross-finger self-collision modifications used by the C corpus.

## 1. Raw recording and subsequences

Raw OakInk2 `anno_preview/<sequence>.pkl` contains contiguous mocap frames at
120 Hz; `program/program_info/<sequence>.json` supplies task stages. Preserve the
program entry order and name each stage `<sequence>__stNN`. Parse the stage's
interval tuple, ignore `None`, use `[min(start), max(end))`, and select only actual
mocap frame IDs inside that half-open interval. Never slice a stage by an assumed
number of frames. Missing intervals, too few frames, unsupported multi-object
hand assignments and non-rigid mechanisms remain explicit review/exclusion rows.

`compact_oakink2.py` creates pickle-free recording NPZs. Use
`consolidate_oakink2_stage_objects.py` to assemble stage semantic candidates and
`refine_oakink2_stage_geometry.py` for geometric refinement. These are candidate
annotations, not proof of irrelevance. `prepare_rigid_corpus.py oakink2` consumes
the reviewed stage manifest; source-reference preparation keeps full-recording
context separately from per-stage physical scene instances.

## 2. Hands, task objects and background instances

Each stage has a complete `required_object_ids` scene and one associated object
per hand. `prepare_rigid_corpus.associate` counts contact-bearing frames for the
16 MANO skinning links against each eligible object's surface. It selects the
largest count; with zero contacts it falls back to the stage annotation, or mean
wrist-to-surface distance if no annotated object is available. The method,
contact counts and fallback evidence are recorded. A fallback is not evidence
that an inactive hand interacts. No associated object outside the retained scene
is invented, and physical instance IDs must not be replaced by category aliases.

Preserve action-required mechanisms, object-object contact chains and supports.
Static background by itself is not sufficient for removal. Reviewed deletions
are in `annotations/final-object-manifest.json` (704 motions; 99 motions remove
180 instances). M4 changes 7→4, M8 6→3. These edits are still part of the blocked
candidate, not silently applied to the historical training corpus. Seven further
isolated objects with legacy hand associations remain until nullable/validity
support is established; never alias them to an active object.

`export_annotations.py` exports actual hand mesh/scale identities, complete scene
IDs, fixed supports, metadata and topology SHA256 from a prepared NPZ dataset.
For the topology-enabled cleanup candidate, retain the reviewed stage manifest,
review decisions and topology beside the trajectories. The current canonical v2
release instead embeds assignments/provenance in each metadata.json and has no
topology sidecars; topology training requires generating them first.

## 3. XHand TopoRetarget and the C choice

`retarget_C.py` and `retarget_impl.py` adapt both XHands using
`config/xhand_{right,left}.json` and `assets/`. A hand has 6 floating wrist DoFs
and 12 articulated finger DoFs; export maps these to the named 36-joint bimanual
legacy reset representation before conversion to free-root qpos. MCP/thumb-CMC proxies use link1; fixed distal proxies
supply MANO-compatible landmarks. Wrist conversion accounts for source root
position/rotation, ZXY convention and float32 quantization bounds. The associated
object alone supplies topology samples; other scene objects and accepted supports
remain collision obstacles. The solver uses physical meshes, not the reward's
normalized object points.

C costs are exactly **wrist translation 1, Laplacian 200, collision slack 100**.
Both hard/hybrid post-trajectory self-collision slack settings are also 100.
The wrapper records the complete effective config and source checksums. C was
selected to reduce excessive high-frequency wrist/finger jitter from the former
strong wrist/slack settings while retaining hand-object topology. It is a
tradeoff: lower jitter did not establish collision-free retargets; some clips
have substantial hand-object/ground penetration. Do not label C physically valid.

```bash
python data_pipeline/retarget_C.py /absolute/prepared/retargeted/CLIP.npz \
  --prepared-data /absolute/prepared --source-xml /absolute/FlashDexRetarget/assets/robot/xhand/bimanual.xml \
  --work /absolute/temporary/C-work --output /absolute/temporary/C-work/outputs/CLIP
```

`pack_C.py --work ...` updates robot retarget coordinates in the retained
reference before conversion to free-root qpos. The current v2 files store no robot
velocity or contact arrays; runtime derives robot velocities per segment after
policy-rate resampling. MANO/object/source association arrays are retained.
It writes NPZ training folders directly and does not maintain a redundant
permanent 50 Hz corpus. Its work layout expects `inputs/platforms` to reference
the prepared source; temporary retarget outputs are not a second canonical source.

## 4. Ground, support and clutter cleanup

Source conversion rotates y-up to z-up and estimates the scene floor using
full-recording physical object context: lowest hull point over stationary frames
(object speed < .1 m/s). Human and object trajectories share that alignment.
Support-disk proposals are not automatically simulator supports. Only accepted
`motion_scene_supports` entries create fixed collision geometry.

Candidate cleanup first measures every native-frame collision vertex. Common
object lift is `max(0, -min_object_z)` over the whole motion; a common scene lift
preserves active hand/object relative trajectories. Additional supports require
actual resting geometry, not an arbitrary thick platform hiding a retarget error.
Supports use narrow signed-height feet/rest cradles, and preserve required contact
objects. A hand may be parked by a constant translation only after strict
no-contact/motion and full-frame source-collision clearance checks. The 114 reviewed
parking candidates passed a 10 mm minimum object gap; active references are not
moved independently. Scene lift/supports must then be recomputed.

`contact_contact_pos_full` is an **object-local nearest visual-mesh vertex**,
not a world position: active-side values are source-exact; parked sides require
new nearest local vertices and tip distances. `scene_cleanup/` retains the actual
candidate generation and independent loader/FK/contact/geometry/rest gates.
It is explicitly staging-only. Common world translations do not translate local
contact vertices. Rebuild/validate topology for changed MANO/object references.

Publication is blocked: 212 quarantined motions, 269 with sampled unheld anchors
all passing, 223 without unheld anchors (unverified, not admitted). Scene probes
are 451 pass, 201 source-relative fixture overlap, 61 stable but depth >1 mm,
21 unstable. The strict gate is depth <1 mm, displacement <1 cm, rotation <.1 rad,
final speed <.01 m/s. Legacy depth <3 mm scoring is recorded separately and is
not a strict pass. Finite/FK/kinematic replay is not dynamic admission.
`annotations/publication-status.json` is the current hold; no auto-upload or
canonical replacement is permitted by these processing tools.

## 5. Native 60 → policy 50 Hz

Prepared MANO/object/C-reset data retains native 60 Hz (read `meta.fps`; some
selected historical clips are already 50 Hz). Policy resampling is per clip:
`floor((N-1)*50/source_fps)+1` frames at `arange(M)/50`. Scalars, positions,
velocities and scalar joint coordinates use linear interpolation; quaternions
use normalized shortest-arc SLERP, with matrix/quaternion conversion where
needed. Never interpolate across a clip boundary. The same plan samples robot,
MANO and all scene objects. `dex_track/tasks/resampling.py` contains the exact
validated packed implementation with explicit quaternion slices.

## 6. NPZ and reference adjacency

Current storage is `dex-track-motion-folders-v2`:

```
index.json
motions/NNNN-NAME/{metadata.json,robot.npz,mano.npz,objects.npz}
objects/ID/{visual.obj,convex/...,...}
collision_simplified/...
supports/...
```

`metadata.json` encodes nested metadata and array references
`{"__array__": KEY, "file": "GROUP.npz", "kind": "torch"}`. Loading uses
`allow_pickle=False` and preserves dtype. Robot contains free-root `qpos[T,38]`
(right then left, each xyz+wxyz+12 fingers); velocities are derived per segment.
MANO contains both hands' wrist poses, named finger landmarks and retained
reference derivatives. Objects contains complete scene position/quaternion/
linear/angular velocity, clip lengths and hand slot mappings. Contact arrays
are not stored. Positions are metres, angular velocity rad/s. Assets and fixed
supports are scene-local; the command adds parallel environment origins.

The following optional graph sidecar is required for topology training but is
absent from the current full-corpus canonical release.

`topology.npz` uses `dex-track-topology-v1` at explicit **policy FPS**:

| Field | Contract |
|---|---|
| `edge_bits` | uint8 `[policy_frames,2,71,9]`, little-endian packed binary adjacency; padding bit zero |
| `object_points` | float32 `[2,50,3]`, fixed physical object-local samples |
| `fps`, `schema` | explicit timeline and format |
| `reference_sha256` | native MANO/object arrays, names and source FPS |
| `points_sha256` | contiguous object sample bytes |

Per hand, vertices 0–20 are palm+20 MANO finger landmarks; vertices 21–70 are 50
associated-object surface samples. Deterministic seed0 proxcurv sampling weights
triangle area × (1 + normalized curvature + 30 × proximity), proximity sigma
.01 m. At each reference frame, transform object points to world and compute
3D Delaunay tetrahedra. Every pair of tetrahedron vertices is an edge; symmetrize,
remove self edges and pack bits. Adjacency is categorical and never interpolated.
`prepare_motion_topology.py --motion-file DATASET --fps 50 --cache BUILD_CACHE`
computes this sidecar offline. Runtime only indexes the bank; it never runs
Delaunay. Current distance weights are exp(-30d), normalized over each row's
neighbors. Baseline uses the full graph; topo-v2 restricts to hand-object edges
and 21 hand output rows, with residual zero for a row without eligible neighbors.
It does not regenerate or replace the stored full reference adjacency.

## 7. Collision simplification

`simplify_collision.py` works **per existing convex piece**, preserving its mesh
frame and physical scale. Try vertex budgets 64/128/256; choose the smallest with
sampled support gap ≤1 mm at the largest physical object scale and convex volume
ratio ≥.98. Test 4096 Fibonacci directions + six axes. If none passes, retain the
original hull. Export explicit OBJ hull vertices, preserving convex-piece counts.
This finite directional test is not a rigorous Hausdorff or grasp-quality bound.
Never recompute mass, COM, inertia, visual meshes, friction or collision filters
from reduced pieces. For hands, preserve original explicit body inertials and
replace collision geometry only; original collision remains an ablation switch.

## 8. Releasing data and retention

Keep exactly one canonical original source and one simplified NPZ training
corpus. `convert_motion_storage.py` converts explicitly and checks tensor identity;
its original-source alias points to the same canonical source, not a second PT.
Never alter active inputs or historical snapshots. All preprocessing exports and
HF caches are transient/managed; no second original-collision converted corpus.

`prepare_release.py` checks all topology/annotation files and writes checksums.
`upload_dataset.py` requires an approved review manifest and matching SHA list
before uploading. It does not treat `published=false` as approval. For this
candidate it refuses publication, so **HF revision is null**. The original source
SHA is `2f3ce7b9200f7af8560cfd7ec58329b79e0a56e553c0fe5209fa271f9bf4b852`.

## Current canonical v2 publication

Use `publish_portable_dataset.py` for the explicitly selected unchanged canonical
corpus. It excludes SDF/USD caches, includes generator tools, seals all files, and
verifies HF blob hashes. `prepare_release.py` and `upload_dataset.py` above are
the legacy reviewed-candidate workflow; do not use its v1 layout for current v2.
The canonical release is not approval of the quarantined cleanup candidate.
See [current dataset contract](../docs/dataset.md) for qpos storage, derived
robot velocities, removed contact arrays, download and regeneration commands.
