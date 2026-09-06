# CAD / URDF reference review — 2026-09-06

The current CAD is recoverable. A whole-robot redesign is not a prerequisite
for continuing simulation. The highest-value export change is to preserve the
mechanism's rigid bodies, named joint frames, and loop closures instead of
exporting one merged assembly. The most concrete newly measured geometry issue
is the **foot collider's missing middle volume**, described below.

This review is not a successful training report. Current execution belongs in
[STATUS.md](../../../STATUS.md); generated descriptions and CPU geometry checks
cannot establish PPO convergence, hardware accuracy, or terrain competence.

## What the official references actually establish

Sources were retrieved on 2026-09-06. Exact repository revisions, inspected
paths, byte lengths and SHA-256 hashes are in
[reference_sources.json](reference_sources.json). These are useful maintained
descriptions, **not measured ground truth for the hexapod or guarantees that
every reference parameter is calibrated**. “B603” could not be identified as a
specific official robot-description project from the supplied name; no guessed
robot was substituted.

| Property | Hexapod MKII source / physical candidate | Official reference observation | Practical lesson |
| --- | --- | --- | --- |
| Structure | Source linkage URDF: 31 links, 30 revolutes, six loops cut into a tree. Physical v5: 18 actuators and 12 native bilateral relations. | SO101 new-calibration URDF: 8 links, 6 revolutes and one fixed frame joint. OpenArm v1 arm macros describe serial arm links with separate parameter files. | A serial arm avoids the hexapod's loop-solver problem; copying its URDF layout does not remove the real linkage. |
| Dynamics | All 31 source tensors are positive definite and satisfy inertia triangle inequalities; total 8.260811322 kg. v5 separately checks reconstructed USD tensors against these source tensors. | OpenArm supplies mass, COM and full inertia tensors in YAML; its macro handles mirrored COM and off-diagonal inertia signs. SO101 supplies per-link tensors. | Preserve full tensors and coordinate transformations; passing this check verifies representation, not the measured mass distribution. |
| Geometry | 1,927 visual instances, 77 unique meshes; 171 collision primitives: 75 boxes, 84 cylinders, 12 spheres. | OpenArm v1 uses one visual DAE and one separate simplified collision STL per arm link. SO101 has 17 visual and 17 mesh-collision instances, using 13 visual mesh files. | Aggregate visuals by rigid body and design collision envelopes separately. Fasteners need mass accounting, not thousands of visual objects in a learning scene. |
| Limits / motors | Source active-joint limits now say 5.5 N·m and 50.26548246 rad/s; the physical task adds a separate provisional speed/overload model. | OpenArm keeps per-joint effort/velocity/range values in YAML. SO101 URDF uses effort 10 and velocity 10 throughout, while its separate motor-default XML uses a ±3.35 force range. | A URDF limit is not an identified actuator model. Do not copy another robot's effort, damping, friction or armature. |

Reference links: [OpenArm v1 macro](https://github.com/enactic/openarm_description/blob/1fba2cbc05001f05b4514120b70130b4ac06f409/assets/robot/openarm_v1.0/urdf/arm/openarm_macro.xacro),
[OpenArm inertia data](https://github.com/enactic/openarm_description/blob/1fba2cbc05001f05b4514120b70130b4ac06f409/assets/robot/openarm_v1.0/config/arm/inertials.yaml),
[OpenArm limits](https://github.com/enactic/openarm_description/blob/1fba2cbc05001f05b4514120b70130b4ac06f409/assets/robot/openarm_v1.0/config/arm/joint_limits.yaml),
[SO101 URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/eecbe3e0a9ebb23e25ad7b2759b03884c6660903/Simulation/SO101/so101_new_calib.urdf),
[SO101 motor defaults](https://github.com/TheRobotStudio/SO-ARM100/blob/eecbe3e0a9ebb23e25ad7b2759b03884c6660903/Simulation/SO101/joints_properties.xml).

The SO101 maintainer documents simulator-specific edits, two calibration
conventions, and an unresolved gripper-coordinate mapping. Its README says base
collisions were removed: the inspected MJCF omits them, while the inspected
URDF still contains them. Its tiny fixed gripper frame also has a zero tensor;
this is a frame/merge concern, not a pattern to use for moving physical bodies.
These differences are precisely why “official URDF” is not a blanket dynamics
certificate. [SO101 description notes](https://github.com/TheRobotStudio/SO-ARM100/blob/eecbe3e0a9ebb23e25ad7b2759b03884c6660903/Simulation/SO101/README.md).

## Concrete MKII findings

**1. The feet need a better contact envelope before terrain qualification.**
Each silicone pad currently has two spheres of radius 15.682 mm, approximately
30.297 mm apart. Their union has only a 4.055 mm radius at the midpoint. After
applying the URDF visual transform to the actual pad STL, 3,592 of 4,479 unique
vertices lie outside that union by more than 1 µm, with a maximum outward
distance of 9.767 mm. All six legs reproduce this result. This count measures
tessellation vertices, not area, volume or contact probability. It establishes
an incomplete pad envelope, particularly relevant to oblique rock/edge contact;
it does **not** establish the cause of a flat-plane solver instability.

Fit a continuous sole envelope to the actual CAD, evaluate its support geometry
across foot orientations, and compare a compact convex approximation with a
small compound shape. Simply connecting the existing spheres into a capsule
fills their waist but does not prove a correct pad fit. Measure silicone
load–deflection, hysteresis, and friction before assigning compliant-contact
parameters. NVIDIA explicitly supports simple approximations and explains
their fidelity/performance tradeoff; it also provides compliant material
contacts. More triangles alone do not identify those physical properties.
[NVIDIA physics fundamentals](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html).

**2. The export discarded the mechanism information that would have prevented
most reconstruction work.** The assembly report records one merged source
link, 1,927 parts and no mate structure. The importer consequently registers an
older leg record against current geometry, assigns parts by mesh/position and
uses attachment overrides. This is recoverable but fragile when plates,
fasteners or the motor internals move between CAD revisions. Export named
rigid-body subassemblies directly, keeping every part's ownership explicit.

**3. The original cut-frame mismatch was not a reason to redesign the linkage.**
The recovered CAD pins define a 30 mm / 77.5 mm parallelogram. The existing
[pin recovery](../../mkii_step2_2026-09-04/physical_fourbar_reference/README.md)
records a maximum transverse regularization of 0.000966 mm and axis change of
0.000549°. Those are numerical CAD adjustments, not manufacturing tolerances.
The physical v5 asset retains the moving pushlever/pushrod and transmits
forces through native constraints. The historical serial URDF welds their
inertias into the femur; that reduction is not physically interchangeable.

**4. The dynamics are internally consistent but not yet 1:1 hardware.**
The vendor motor CAD contributes only a 62.2 g shell per actuator. The importer
tops each up to 191 g, placing the missing mass as a housing-sized solid
cylinder. This corrects 2.318 kg of missing mass, but its COM/inertia placement
is an approximation. A full link-level mass/COM/inertia ledger should include
battery, Jetson, CAN/power boards, cables, navigation sensors and survey payload.
The current description has no separately named sensor/payload frames or
itemized electronics bodies establishing that those planned additions are
included in the 8.261 kg figure.

The simulator also lacks measured joint stops, drivetrain friction/backlash,
reflected inertia, structural compliance, thermal recovery and bus/control
latency. The [physical motor contract](../../../docs/MKII_FOURBAR_TRAINING.md#rs05-v2)
labels its assumptions, including 48 V and symmetric braking limits. Metal
mounting alone does not validate continuous torque or regenerative braking.

**5. Documentation had a real routing regression.** The package README still
called the 19-link serial model the training contract and said physical loop
authoring was open. This review corrects that routing to the intended 31-body
physical candidate, distinguishes source kinematic mimics from native physical
constraints, and labels serial-v2 stance/instructions as historical. No runtime
asset, motor parameter or acceptance criterion is changed by that correction.

## CAD handoff for the redesign

| Priority / owner | Deliverable | Acceptance evidence |
| --- | --- | --- |
| P0 — mechanical CAD | Dedicated simulation assembly: body plus five rigid bodies per leg; every physical rigid attachment represented once. Name the coxa/femur/pushlever motors, passive hinges, and both sides of each loop closure. | Export graph has 31 physical bodies, 36 physical hinges, 18 active coordinates; no orphan parts or inferred ownership. If topology changes, replace these expected counts explicitly. |
| P0 — mechanical + controls | Common zero fixture, axis/sign definitions, encoder offsets, measured stops and a usable nonsingular workspace; retain separate hardware and software limits. | Named-joint sweeps match the fixture, loop geometry, cable clearances and mechanical stops. No implicit action index order. |
| P0 — mechanical + electrical | Loaded mass budget and thermal/torque budget using planned battery voltage and real payload. Place battery/payload close to the chosen body COM; reduce distal mass where structural testing permits. | Weighed components and link COMs; leg-stand force/torque/speed/voltage/current/temperature traces. Evaluate continuous support and impact margins, not peak torque alone. |
| P1 — simulation software | Versioned visual simplification and separate collision layer; fix the pad envelope first. Preserve full mass tensors when omitting cosmetic geometry. | Reproducible geometry/support comparison, immutable source hashes, no stale offsets; fresh simulator contact and solver-convergence validation. |
| P1 — mechanical | Predictable feet and bearings: replaceable sole, known compliance, controlled backlash and lateral stiffness. Compare a retained four-bar with a direct serial knee only if mass, thermal and mechanical advantage justify the trade. | Measured rail drop/stance/sweep response; off-axis loading tests beyond the vertically constrained stand. No claim that eliminating a loop automatically improves the robot. |
| P1 — CAD + perception teams | Named rigid navigation IMU/LiDAR/camera frames and a separate survey-payload interface with mass/COM bounds. Reserve cable routing and leg-sweep visibility. | Extrinsic calibration fiducials, sensor occlusion sweeps and synchronized timestamps; payload variation remains explicit even when its data processing belongs to another team. |

For `onshape-to-robot`, use top-level rigid-body instances with `dof_` mates,
`fix_` rigid attachments, `link_` names and `frame_` reference frames; its joint
axes follow mate-frame Z, and `_inv` reverses direction. Put operating limits
on the appropriate mates. Preserve an immutable Onshape version/document ID,
exporter version, settings, part identities and units with each release.
[Exporter design conventions](https://onshape-to-robot.readthedocs.io/en/latest/design.html).

Use `closing_` mates for the actual loop pins: the exporter can emit frames on
both participating bodies. A URDF tree still needs simulator-side closure
constraints; naming those frames does not make URDF itself a closed-loop
dynamics format. [Exporter loop documentation](https://onshape-to-robot.readthedocs.io/en/0.3.27/kinematic_loops.html).

## Reproduce the new measurements

From the repository root:

```sh
.venv/bin/python artifacts/mkii_fourbar_2026-09-06/cad_reference_review/audit_geometry.py
```

Optionally pass the pinned SO101 URDF path as the sole argument to reproduce
the reference XML counts/tensors. [geometry_audit.json](geometry_audit.json)
records the results, exact source URDF and foot STL hashes, all principal
moments, and per-leg pad measurements. It is CPU-only and reads the current
files without rewriting them. No speculative pad candidate has been installed
into the frozen physical asset.
