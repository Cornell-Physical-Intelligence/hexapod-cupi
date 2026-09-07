# Onshape and mechanical-team handoff

Reviewed 7 September 2026. **Keep the present four-bar mechanism as the baseline. First make its export reproducible, then improve the measured model and evaluate mechanical changes against load, mass and workspace requirements.** The current evidence does not justify redesigning the robot merely because URDF cannot represent a closed loop directly.

The mechanical team can start the deliverables below while simulation continues on its frozen asset. The simulation team owns the import adapter, constraint implementation and qualification of every revised asset. [Current execution status](../STATUS.md) is separate from this engineering checklist. The [detailed review and evidence](../artifacts/mkii_fourbar_2026-09-07/cad_engineer_handoff_v2/review_notes.md) includes numerical findings, source hashes and reference-robot comparisons.

## First deliverable: one explicit leg export

**CAD owner + simulation owner:** export one fully named leg and its chassis interface as a proof before repeating the changes across all six legs. The current source has 1,927 instances in one merged link, with no exported joint graph. Software reconstructs the mechanism from geometry and earlier leg records; that makes future CAD revisions unnecessarily fragile.

Represent the chassis as `body` and each leg as five physical rigid bodies: `<leg>_coxa`, `<leg>_femur`, `<leg>_tibia_push_lever`, `<leg>_tibia_pushrod`, and `<leg>_tibia`, for `lf lm lr rf rm rr`. Every part must belong to exactly one rigid body. Supply a BOM-to-body ownership table with persistent Onshape instance IDs and an exploded view.

| Connection per leg | Current software identity | Role |
| --- | --- | --- |
| Body → coxa | `<leg>_coxa_yaw` | Actuated |
| Coxa → femur | `<leg>_femur_pitch` | Actuated |
| Femur → pushlever, pin A | `<leg>_tibia_lever_pivot` | Actuated |
| Pushlever → pushrod, pin B | `<leg>_tibia_rod_pivot` | Passive |
| Pushrod → tibia, pin C | `<leg>_tibia_loop_closure` | Passive closure |
| Femur → tibia, pin D | `<leg>_tibia_pitch` | Passive knee |

Use the tested exporter's `link_`, `dof_`, `fix_`, `frame_` and `closing_` conventions. Joint motion follows the mate-frame Z axis; export both local frames of each cut closing hinge. A fixed sensor-frame link adds no physical body. Pin the actual exporter version and verify its loop-frame behavior on the one-leg proof. [Exporter design conventions](https://onshape-to-robot.readthedocs.io/en/latest/design.html), [documented paired closure frames](https://onshape-to-robot.readthedocs.io/en/0.3.27/kinematic_loops.html).

**Acceptance:** the full mechanism has 31 physical bodies, 36 hinges, 18 actuators and six closure-frame pairs; its cut tree has 30 moving coordinates. There are no orphan-part warnings, missing parts or inferred ownership assignments. Paired hinge frames reconstruct the same intended axis and common pin datum in assembly coordinates. Keep +Z up, forward −Y, left +X, and the body origin on the bottom plate. Export CAD pose, radial-yaw zero and hardware calibration zero as distinct definitions.

Two ownership details must survive re-export: yaw output flanges/hubs remain on the chassis while motor housings and attached hardware rotate with the coxae; the named `tibia_pushrod_cover` belongs to the pushlever. Keep the user-verified screw caps on the coxa. The [override ledger](../robot/hexapod_mkii_assy/part_overrides.json) records these attachments.

**Software work required:** the present importer explicitly accepts only one merged link without joints. A structured export needs a new versioned adapter and round-trip verification. Do not overwrite the current input or assume the old reconstruction accepts the improved graph.

## Parallel engineering deliverables

| Priority / owner | Deliver | Acceptance evidence |
| --- | --- | --- |
| P0 — Mechanical + controls | A physically attainable calibration fixture; per-motor axis, sign, encoder offset, transmission relation, hard stops and narrower operating limits | Marked-fixture photos and slow sweeps of all 18 motors; simultaneous-joint clearance sweeps including cables and four-bar toggle margin. CAD limits remain provisional until measured. |
| P0 — Mechanical + electrical + payload teams | Loaded BOM; per-body mass, COM and full inertia tensor with units/frame/material; motor housing/output allocation; explicit battery, Jetson, CAN, wiring, sensor and payload allowances | Every part counted once; transformed sums reconcile with assembly properties; completed motors/links weighed and attainable COMs measured with uncertainty. Record measured deviations rather than forcing agreement with CAD. |
| P0 — Mechanical/test | Structural load cases for continuous support, peak actuation, impacts, lateral foot force, rod compression/buckling and near-toggle configurations | Calculated and tested margins for the actual materials, fasteners, bearing spacing and geometry. Vertical-rail results alone cannot establish lateral or torsional strength. |
| P1 — CAD + simulation | Exact assembled foot/tibia solids and a separate collision layer preserving exposed contact features and cavities | Surface/support error and concavity overfill measured over relevant contact directions; cooked simulation shapes inspected; unchanged physical gates rerun after a versioned change. |
| P1 — Mechanical + perception + survey | Rigid IMU, navigation LiDAR/camera and separate survey-payload datums; mass/COM ranges, mount stiffness, cable routing and reserved volume | Measurable fiducials and agreed frame conventions; sensor visibility checked through forward, reverse, lateral and tilted poses; owners assigned for extrinsics and timestamps. |

For the mass ledger, **8.260811322 kg is the present modeled assembly mass**, not confirmation that all planned electronics and payload are included. The RS05 CAD shell weighs about 62.2 g; the 191 g override adds 2.318 kg across 18 motors using an approximate housing-sized inertia allocation. Check material assignments too: `machined_tibia` currently carries a density near 1180 kg/m³. A part name does not establish its manufactured material. [Assembly mass report](../robot/hexapod_mkii_assy/assembly_report.md).

For structural sizing, the ideal near-limit linkage estimate is about 328 N rod force at 1.6 N·m and 1.1 kN at 5.5 N·m. These are load-case prompts derived from geometry, not measured forces or approved operating limits. [Derivation and assumptions](../artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/README.md).

For foot contact, the two-sphere collider has up to **9.816 mm** CAD-surface undercoverage. A proposed 64-vertex hull reduces outer undercoverage to **0.158 mm**, but fills a real empty region: a sampled witness lies **7.807 mm** from the silicone and outside all other LF tibia-part bounding boxes. That candidate is uninstalled and unqualified. Investigate a small compound representation that preserves the cavity; do not redesign the silicone merely to suit one convex hull. [Fit comparison](../artifacts/mkii_fourbar_2026-09-06/pad_collision_candidate_v1/README.md), [assembled occupancy check](../artifacts/mkii_fourbar_2026-09-06/assembled_pad_accessibility_v1/README.md).

## Leg-stand measurements that improve simulation

The rig's hip/body attachment moves vertically on the rail. Model and measure the moving carriage mass and rail friction separately. Log synchronized motor targets and feedback, rail position, independent foot normal force, bus voltage/current, and motor/mount temperatures. Preserve timestamps, units, firmware, CAN configuration, gains, geometry and load with each raw trace.

Begin with static load balance and slow sweeps, then bounded reversals/steps, loaded motion, stance holds and controlled contact loading. Identify actuator bandwidth/delay, torque scale, friction, backlash, effective inertia, structural deflection and sole load–deflection/hysteresis. Reserve separate traces for checking the fitted model. Motor-reported torque cannot independently validate its own calibration.

The **5.5 N·m rating is peak torque**. Vendor continuous stall is **1.2 N·m**; the **1.6 N·m** rotating rating is at **100 rpm** with the specified cooling plate. Metal mounts do not by themselves establish sustained torque or thermal recovery. Battery voltage, braking/regeneration and motor protection need measured electrical inputs. The vertical rig also leaves off-axis stiffness and free-body balance for additional tests. [Detailed leg-stand plan](../artifacts/project_review_2026-09-04/LEG_TEST_STAND.md), [actuator assumptions](MKII_FOURBAR_TRAINING.md#rs05-v2).

## Release package and division of work

Each handoff should include an immutable Onshape document/version/element/configuration reference, exporter version/commit and settings, raw export, named graph/closure frames, BOM ownership and mass-property ledgers, units, mesh hashes, calibration tables and a change log. Preserve dynamics and limits in export settings; pin a version instead of the live workspace. [Exporter configuration documentation](https://onshape-to-robot.readthedocs.io/en/latest/config.html).

Simulation already implements separate moving linkage bodies, an 18-motor/30-coordinate adapter, coupled resets, corrected inertia conversion, a provisional actuator envelope and physical checks on every substep. Those fixes do not substitute for measured dynamics. A geometry, collider or dynamics change creates a new model release and requires fresh qualification; checkpoints may be transferred only with explicit compatibility checks and new evaluation.

OpenArm and SO101 are useful examples of structured robot-description packaging, separate collision assets and calibration inputs. They are not physical ground truth for this mechanism. The requested “B603” reference remains unidentified; no guessed robot is used. [Pinned official-reference audit](../artifacts/mkii_fourbar_2026-09-06/cad_reference_review/README.md).

Prioritize the one-leg export proof, full ownership graph, loaded mass ledger and measured workspace first. Weight reduction and alternate knee mechanisms remain trade studies evaluated against strength, thermal, transmission and terrain requirements. The survey team can change instruments through a defined interface, but the resulting payload mass, COM and vibration requirements still affect locomotion.
