# Candidate scene cleanup (not admitted for publication)

These scripts describe the historical v1 parking candidate and require its
frozen v1 inputs/contact annotations. They are not a v2 training-data pipeline.
The canonical v2 dataset deliberately omits contact arrays and uses free-root
qpos; do not point these legacy candidate scripts at that dataset. Re-running
this candidate requires its frozen code snapshot as well as frozen data.

These are the actual numeric generation/admission stages used for the parking-v1
candidate. They are kept separately from the runtime MDP. The candidate remains
quarantined: this directory is not a command to publish or overwrite training data.
No renderer, profiler, experiment launcher or historical training output is shipped.

Set explicit paths in a disposable preprocessing workspace before invoking any file:

```
DEX_PROJECT=/absolute/dex-track
DEX_AA_ROOT=/absolute/active-adaptation
DEX_PROJECT_ROOT=/absolute/workspace
DEX_DATASET=/absolute/canonical-simplified
DEX_CLEANUP_WORK=/absolute/work/audit/cleanup
```

`DEX_DATASET` is read-only input. `DEX_CLEANUP_WORK/staging` is the sole candidate;
keep it outside the input. Place the reviewed `final-object-manifest.json` in
`$DEX_CLEANUP_WORK/artifacts/` (from `../annotations`). The working directory's
`scripts` symlink must point to this directory for subprocess stage references.
`build_frame_cache.py` computes native full-frame robot/object collision caches in
`../artifacts/frames` and per-motion summaries. `audit_inactive_hands.py` derives
strict parking candidates; manual acceptance is still necessary.

Generation order: `parking_process.py`, `parking_refine_rest_supports.py`,
`parking_exact_foot_heights.py`, `parking_repair_object_lift_coverage.py`,
`parking_augment_rest_cradles.py`, `parking_correct_signed_heights.py`.
Then `parking_validate.py` (including object-local contact repair/independent NN),
`parking_geometry_audit.py`, `parking_settle_rest.py`,
`parking_scene_rest_probe.py`, `parking_strict_scoring.py`.
Refinement is bounded and cached by geometry/content hashes; never reuse a
metadata-only cache after editing feet or omitting floor objects.

Prerequisites include the source dataset's physical object collision assets,
full-frame cache, reviewed instance manifest and no-contact parking decisions.
Some scene filters explicitly identify the original corpus's motion indices;
these are part of that 704-motion dataset contract, not a generic automatic
cleanup algorithm for other corpora. For another index, use its motion identities
and review decisions before running. No new physics/replay has been run here.
Publication remains blocked by `../annotations/publication-status.json`.
