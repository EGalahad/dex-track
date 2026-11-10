# What dex-track adds to mimic-lite's structure

Both projects use Active Adaptation for the environment loop, simulator adapters,
MDP component lifecycle, observation/reward groups, and rollout collection. Their
core task flow is the same: sample a reference, read the simulated state, build
observations, run the policy, apply controls, and score tracking errors.

## What the policies learn

| | Mimic-lite | Dex-track |
|---|---|---|
| Controlled system | A humanoid robot, including balance and ground contact | Two floating XHands, their fingers, and manipulated rigid objects |
| Reference | Motion already retargeted to the robot's bodies/joints | Human MANO hand motion, object trajectories, and robot retarget states |
| Action | Robot joint targets | 12 wrist twist components plus 24 finger targets; wrist wrench PD and filtered finger targets/PD apply them |
| Tracking objective | Robot body/joint/root pose and velocity, with task-dependent regularization | Hand/wrist motion and object pose/velocity; optional robot-finger tracking or human-hand/object topology |
| Scene input | Primarily robot/reference state in the current tracking task | Robot/reference state plus object surfaces and hand-to-surface relationships |
| Scene variation | Robot-specific motion datasets | Different meshes, object counts, hand-object assignments, and fixed supports per scene |

Dex-track does not directly command object trajectories. The policy has to move
objects through physical contact. Its object tracking rewards measure motion
reproduction; they do not by themselves establish semantic success such as
whether a pouring task transferred liquid. Mimic-lite's manipulation-motion
evaluation likewise measures robot reference tracking rather than object-task
success in this code path.

The two dex-track task configs use the same control loop. `tracking` can compare
robot finger links against retargeted robot poses. `tracking_topology` emphasizes
MANO bone directions and hand-object spatial relationships, reducing dependence
on a joint-angle correspondence between human and robot hands. Retargeted states
still initialize episodes and remain available for diagnostics.

## Follow the runtime in this order

```text
scripts/train.py, play.py, explore.py
    cfg/task/tracking.yaml or tracking_topology.yaml
    dex_track/data.py                 prepare scene layout before env creation
    dex_track/assets.py               build hands, objects, supports
    dex_track/tasks/
        command.py                   sample/reset/reference and current state
        motion.py + resampling.py    one aligned reference timeline
        actions.py                   wrist control and filtered finger position targets
        transforms.py                shared tensor math and tracking errors
        observations/                proprioception, scene, intent
        rewards/                     tracking, regularization, diagnostics
        terminations.py              failure counters and motion timeout
    dex_track_learning/
        ppo.py + common.py           PointNet features and PPO
```

`MotionTracking` directly inherits AA's `Command`. It owns hand/object accessors
and tracking caches. `TopologyMotionTracking` adds the optional topology state;
the property-only hand/object mixins and the former error mixin are removed.
Observation and reward terms consume command state, as in mimic-lite.

The motion buffer now has three direct groups: `dataset.mano`, `dataset.robot`,
and `dataset.objects`. The former `frames` and `reset` names are removed.
Stored robot `qpos` already matches the floating-hand roots, so loading no longer
runs the old virtual-wrist FK/Jacobian conversion. Robot velocities are derived
within each segment after policy-rate resampling. See [dataset.md](dataset.md)
for the server authority, format and segment assignment contract.

## Why the additional modules remain

| Module(s) | Why they exist | Status |
|---|---|---|
| `free_hands.py` | Exposes two independent floating entities through the bimanual robot interface that the MDP uses; maps joint/body indices and state buffers | Needed for the current two-entity model; no physics stepping is hidden here |
| `wrist_frames.py` | Defines robot wrist/body names and the fixed MANO semantic-palm convention | Required to keep reference, control, and geometry in consistent frames |
| `data.py`, `motion_storage.py` | Validate and load portable NPZ motions, scene membership, scales, and object assignments | Needed dataset contract; mimic-lite delegates much of motion storage/loading to any4hdmi |
| `tasks/scene_objects.py` | Manages object slots, masks, reference/sim states, and reset/parking for variable scenes | Manipulation-specific state; cannot be replaced by a robot-only motion buffer |
| `supports.py` | Loads and constructs fixed tables/rest supports supplied with a scene | Required when those supports are part of the physical task |
| `tasks/reference/finger_bodies.py` | Builds/caches robot finger-link FK from retargeted joints or a nominal pose | Required by robot-finger targets; human MANO joints are not robot joints |
| `tasks/reference/collision_geometry.py`, `sdf.py`, `object_points.py` | Load collision surfaces, bake distance fields, sample surface points, and normalize tracking samples | Used by scene observations and contact-distance/tracking terms; point-cloud and reward normalizations are distinct |
| `tasks/reference/topology.py`, `topology_math.py`, `laplacian_kernel.py` | Load precomputed graph connectivity, compare hand bones and hand-object relationships, and evaluate the GPU graph residual | Required for the topology objective; not needed by a pure robot-pose imitation objective |
| `topology_storage.py` | Checks that graph payloads match the motion, object geometry, and policy timeline | Prevents training against stale/misaligned precomputed graphs |
| `dex_track_learning/common.py` | Encodes two wrist-frame object point clouds with shared PointNet weights and routes one action distribution into wrist/finger inputs | Needed by the current geometry-conditioned policy; routing is now colocated here |
| `scene_recording.py` | Captures all moving scene entities after physics and before automatic reset | Playback/evaluation support, outside reward computation |
| `source.py`, `assets/`, `runtime_patches/` | Resolve locked upstream assets and apply pinned collision-runtime fixes | Packaging/dependency requirements; not extra policy abstractions |

There is a real asymmetry in repository boundaries: dex-track also contains
`data_pipeline/` (61 Python files at this audit), including conversion,
retargeting, mesh simplification, topology preprocessing, and scene cleanup.
Mimic-lite's README sends dataset conversion/validation to the separate
any4hdmi project. Those offline tools are not 61 modules executed each RL step.
They should be read separately from the runtime. Moving them into another
repository would change packaging and deployment, not the task computation.

## Cleanup performed

- Three action-package files became `tasks/actions.py`.
- `wrist_math.py`, `tensor_kernels.py`, and reference math/error/velocity helpers
  became `tasks/transforms.py`.
- `HandPropertiesMixin` and `ObjectPropertiesMixin` methods moved into
  `MotionTracking`; the two files and inheritance layers were deleted.
- The termination counter is stored directly in its termination term; the
  one-use `ConsecutiveExceed` class/file was removed.
- Object tracking-point normalization moved beside other object-point helpers.
- Action routing moved into policy `common.py`; the validation-only policy
  config wrapper was removed because PPO already validates the same setting.
- The topology Laplacian pass-through method was removed.

This pass reduced `dex_track/` from 46 to 36 Python files and
`dex_track_learning/` from 6 to 4. Mimic-lite's runtime package has 23 Python
files, but more lines in its runtime package at this checkout (4,693 versus
3,641 after consolidation and the cold-start fix). File count measures fragmentation, not
task complexity or speed. The remaining geometry and scene modules carry
functionality absent from the robot-only imitation path.

Registry names, policy tensor dimensions, checkpoint state-dict keys, motion
formats, reward formulas and physical gains remain unchanged. Internal Python
import paths were consolidated; external scripts importing the deleted internal
modules need to follow the new locations.

## Compilation boundary

Explore enables `task.rollout_compile=[observations,rewards,control]`. AA's
existing `_apply_action` now contains force clearing, controller updates,
pre-step hooks, and simulator-buffer writes as one compilable function. Physics
integration remains in the backend. The current Warp CUDA context call cannot
be traced into a single `torch.compile(fullgraph=True)` graph.

The final-code benchmark measured 1.577 → 0.443 ms per control substep and
35.84 → 30.31 ms per full policy step (16 environments, four substeps, headless,
50 warm-up and 300 timed rollout steps). The baseline is the average of two runs
bracketing the compiled modes. The earlier prototype measured 1.293 → 0.268 ms
and 30.2 → 26.7 ms respectively; absolute laptop timings varied between passes.
Both passes preserved buffers within `1.91e-6` across 3 target cases and 20
consecutive substeps per case: second-order filter state, both hands' position/velocity targets,
external wrenches, and simulator control buffers. Rollout timing is not a
matched-trajectory training-convergence comparison.

Fresh starts of both `tracking` and `tracking_topology` passed 30 exact legacy
error checks and 18 compiled observation/reward checks per task. mjlab's lazy
zero-copy COM/control Torch views are created during initialization so the first
compiled control call does not attempt a Warp-to-Torch conversion. This
optimization preserves the current controller; switching to simulator-native
wrist servos is a separate modeling change.

## Native wrist PD in the two backends

| Layer | Capability | Current integration gap |
|---|---|---|
| MuJoCo / MuJoCo Warp | A wrist site measured relative to a reference site supplies six translational/rotational actuator coordinates. Paired position and velocity actuators can evaluate PD inside the simulator. | A free-joint or body transmission alone has no useful position-servo length. The model needs site/refsite actuators and target-buffer wiring. |
| mjlab 1.6.0 | Joint/tendon native PD is already supported; this is what the fingers use. | `BuiltinPdActuatorCfg` explicitly rejects `TransmissionType.SITE`. The current site interface is effort control, not a floating-body pose/velocity target API. |
| PhysX | D6 joint drives accept target pose, linear/angular velocity, stiffness and damping; the angular SLERP drive handles orientation. | A world-to-wrist driven constraint and a suitable batched target-update path must be integrated and validated. This is not the same interface as articulation joint targets. |
| IsaacLab | `ImplicitActuator` submits joint position/velocity targets to PhysX. | Root pose/velocity writes set physical state; they are not servo targets. Current dex-track's `xhand_free` asset supports mjlab only, so no IsaacLab dex-track rollout was tested. |

The installed MuJoCo 3.11.0 and MuJoCo Warp 3.8.1 were checked with an isolated
free-body model, six site/refsite axes, and twelve position/velocity actuators.
With nonzero desired velocities, Warp actuator forces matched the expected PD
formula within `9.54e-8`. With desired velocities then set to zero, all six pose
coordinates settled within `6.43e-8` of their targets after two simulated seconds.
This verifies the engine mechanism, not a completed dex-track integration or a
performance gain in the hand/object scene. IsaacLab/PhysX findings are from
source and API documentation; no runtime verification is claimed there.

Replacing the current wrist controller would also require matching wrist/COM
frames, desired angular-velocity conventions, rotation errors, and force/torque
limits. In particular, current control clamps vector norms; independent native
actuator limits clamp axes. Native integration/damping can also change dynamics.
The compilation optimization keeps the existing controller and avoids those
modeling changes.

Sources: [MuJoCo actuator site/refsite semantics](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general-refsite),
[PhysX D6 drives](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/Joints.html#drives),
and [IsaacLab actuator API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html).
