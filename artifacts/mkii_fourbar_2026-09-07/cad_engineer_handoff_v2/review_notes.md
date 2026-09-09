# CAD-engineer handoff review — 2026-09-07

**The first CAD deliverable is a reproducible mechanism export, not a new robot.** The current six-leg geometry supports reconstructing the intended four-bar. The immediate engineering gaps are explicit rigid-body ownership, measured dynamics, and a faithful foot-contact envelope. Continue simulation development against its frozen asset while the mechanical team prepares the next release; a revised CAD export must not silently replace an admitted model or checkpoint lineage.

This is an independent review for the final team handoff. It changes no CAD, URDF, USD, runtime, gates, or published evidence. Existing documents are evidence to assess, not design authority. Execution status belongs in [STATUS.md](../../../STATUS.md).

## What the evidence establishes

| Finding | Classification and consequence |
| --- | --- |
| The source export contains 1,927 part instances in one merged link, with no exported mate graph. The importer reconstructs ownership using the older leg record, geometric registration, keyword assignment and overrides. | **Verified export deficiency.** It creates avoidable reconstruction work and revision risk. It does not mean the underlying assembly lacks correct mechanical mates. |
| Recovered physical pins form a 30 mm / 77.5 mm parallelogram. The largest numerical transverse regularization was 0.000966 mm and axis adjustment 0.000549°. | **Verified CAD geometry.** These are numerical reconstruction differences, not achievable manufacturing tolerances. The older approximately 0.5 mm cut-frame discrepancy is not a demonstrated bore mismatch. |
| The imported linkage has 31 positive-definite, triangle-consistent inertia tensors and mass 8.260811322 kg. Eighteen vendor motor models contain about 62.2 g each; software adds mass to reach 191 g per motor. | **Internally checked representation with physical uncertainty.** The extra 2.318 kg is real mass accounting, but housing-cylinder allocation of the missing mass is approximate. The loaded robot's measured COM and inertia remain unknown. |
| Current two-sphere sole geometry misses exact CAD surface by as much as 9.816 mm. | **Verified collider deficiency**, especially for nonplanar contact. It is not evidence that the physical silicone design is wrong or the cause of observed flat-plane instability. |
| The proposed 64-vertex hull bounds outer-support undercoverage to 0.158 mm but fills concavity. Its worst sampled empty-volume witness is 7.807 mm from silicone and at least 15.951 mm from every other LF tibia visual's bounding box. | **Measured limitation of an uninstalled proposal.** The other assembled tibia parts do not fill this witness. One convex hull cannot reproduce that external empty region. Terrain accessibility along a complete approach path and other legs' occupancy were not established. |

Sources: [assembly report](../../../robot/hexapod_mkii_assy/assembly_report.md), [independent reference audit](../../mkii_fourbar_2026-09-06/cad_reference_review/README.md), [CAD pin recovery](../../mkii_step2_2026-09-04/physical_fourbar_reference/README.md), [pad fit and bounds](../../mkii_fourbar_2026-09-06/pad_collision_candidate_v1/README.md), [assembled pad occupancy](../../mkii_fourbar_2026-09-06/assembled_pad_accessibility_v1/README.md).

## Prioritized deliverables and acceptance evidence

### P0 — Named assembly, joint frames and release package

**CAD owner:** Create a dedicated export configuration with `body` and, for each `lf`, `lm`, `lr`, `rf`, `rm`, `rr`, five physical rigid bodies: `coxa`, `femur`, `tibia_push_lever`, `tibia_pushrod`, `tibia`. Keep every rigidly attached component owned exactly once. Include an exploded ownership view and a BOM-to-link table with persistent Onshape part/instance IDs.

For each leg, the mechanism ledger must identify:

| Connection | Name retained in the current software | Physical role |
| --- | --- | --- |
| Body → coxa | `<leg>_coxa_yaw` | Active motor |
| Coxa → femur | `<leg>_femur_pitch` | Active motor |
| Femur → pushlever, A | `<leg>_tibia_lever_pivot` | Third physical actuator per leg, selected by name; the knee is passive |
| Pushlever → pushrod, B | `<leg>_tibia_rod_pivot` | Passive hinge |
| Pushrod → tibia, C | `<leg>_tibia_loop_closure` | Passive closing hinge; paired local frames |
| Femur → tibia, D | `<leg>_tibia_pitch` | Passive hinge |

Use `link_` connectors to name links, `dof_` mates for tree joints, `fix_` only for truly rigid attachments, and `closing_<leg>_tibia_loop_closure` for each cut hinge. Joint motion follows mate-frame Z; `_inv` reverses the axis. Place `body` first and retain a floating body at runtime. Named `frame_` connectors can define sensors and feet; exported dummy frame links do not count as additional physical rigid bodies. These conventions are documented by the exporter; pin its actual tested version instead of relying on the moving `latest` page. [Exporter design conventions](https://onshape-to-robot.readthedocs.io/en/latest/design.html), [paired loop-frame export](https://onshape-to-robot.readthedocs.io/en/0.3.27/kinematic_loops.html).

**Accept when:** the intended mechanism graph has 31 physical bodies, 36 physical hinges, 18 motors, and six closure-frame pairs, with no orphan warnings or inferred part ownership. The URDF tree has 30 moving coordinates plus possible fixed frame links; the closure metadata is separate. Verify both local frames of every hinge reconstruct the same physical axis and chosen common pin datum in assembly coordinates. Record units, full rigid transforms, CAD zero and radial-yaw zero as separate named poses. Preserve an immutable Onshape document/version/element/configuration reference, exporter version/commit, settings, BOM, mesh hashes, mass-property export and raw output together. If topology intentionally changes, review and version the new expected counts.

Preserve the existing body datum explicitly: +Z up, forward −Y, left +X, origin on the bottom plate rather than at COM. Positive coxa yaw is about body +Z; the original femur/knee convention raises the femur/opens the knee. The pushlever's per-leg phase bridge must be exported separately instead of assuming its CAD zero is the old knee zero or a hardware encoder zero.

**Software dependency:** `import_onshape_hexapod.py` currently rejects an input with multiple links or any joints. The improved structured export therefore requires a new import path/adapter and round-trip tests; do not feed it into the old merged-export reconstruction and assume compatibility. First deliver one fully named leg plus body interface as the export proof, then propagate to all six.

**Ownership details to preserve:** the yaw output flange and hub are fixed to the chassis; the housing/top plate and attached hardware rotate with the coxa. Screw-head caps belong to the coxa. The cover named `tibia_pushrod_cover` is bonded to the pushlever. Its name is not its ownership. [Recorded overrides](../../../robot/hexapod_mkii_assy/part_overrides.json), [importer implementation](../../../robot/tools/import_onshape_hexapod.py).

### P0 — Zero fixture, operating workspace and structural load cases

**Mechanical + controls owners:** Supply a drawing of an attainable calibration fixture and a joint table containing axis direction, positive motion, CAD-zero transform, encoder-zero offset, transmission relation, physical stops and deliberately narrower software limits. Record these per leg where assembly indexing differs. Include cable routing, foot/leg/body clearance, bearing spacing and the minimum operating distance from four-bar toggle positions.

**Accept when:** marked-fixture photos and timestamped low-speed sweeps establish all 18 motor signs and offsets; each end stop and interference boundary is documented. CAD swept volumes cover simultaneous multi-joint motion, not just independent slider sweeps. Source-derived limits remain provisional until hardware measurements replace them in a new version. No runtime array index is a joint identity.

For structural review, include near-toggle rod compression/tension, lateral foot loading, impact, stall and continuous support. The existing ideal quasistatic calculation near the lower provisional limit gives roughly 328 N rod force at 1.6 N·m and 1.1 kN at 5.5 N·m. These estimates flag load cases, not measured forces or approved limits; validate the actual angle, moment arm, fasteners and buckling assumptions. The vertical rail alone cannot establish lateral stiffness, torsional strength or free-body stability. [Pin geometry and load calculation](../../mkii_step2_2026-09-04/physical_fourbar_reference/README.md).

### P0 — Loaded mass, materials and inertia ledger

**Mechanical owner, with electrical/payload inputs:** Export mass, local COM and the complete symmetric inertia tensor about that COM for every physical body, with units and reference frame. Include motor housing/output allocation and material/density assumptions. Itemize battery, Jetson Orin Nano, CAN/power boards, wiring/connectors, navigation sensors, survey payload and mounts, including absent items as explicit allowances with COM ranges. CAD names do not prove materials: the report assigns `machined_tibia` about 1180 kg/m³, whereas pushlever and pushrod are approximately steel density; confirm the intended manufactured parts before optimizing them.

**Accept when:** sums reconcile with the same loaded configuration and a consistent override ledger; each part contributes once; transforms and parallel-axis accumulation reproduce assembly COM/inertia; source and simulator tensors agree within the separately recorded numerical export tolerance. Weigh actual motors and completed links and measure attainable COMs, reporting instrument uncertainty and differences from CAD. Do not require a fictitious exact hardware match. Do not infer mass from a collision hull, whose filled cavities are not material.

The report's 34.1% inertia-term difference compares motor-corrected assembly properties against the uncorrected fused export. It is not, by itself, proof of an inertia-transform bug. Keep uncorrected CAD, motor-corrected model and measured hardware ledgers distinguishable. Lightening the distal lever/rod or relocating heavy parts is a proposed trade study, contingent on stress, buckling, deflection, thermal and workspace margins.

### P1 — Faithful contact surfaces and efficient visual export

**CAD + simulation owners:** Deliver exact silicone and complete assembled foot/tibia solids in a common documented frame, a section through the side/socket concavity, replaceable-sole interface dimensions, and collision-relevant exposed features. Keep a separate collision layer and a simplified visual mesh per rigid body where useful. Omit cosmetic fastener detail from rendering while retaining its mass and any genuinely contact-relevant protrusions.

**Accept when:** exact-source hashes, transforms and scale survive export; no visible empty contact region is silently filled; numerical surface/support errors are reported over directions and proposed workspace, including undercoverage and overfill. Select an error budget based on the contact features and terrain to resolve. The measured 0.158 mm fit is evidence for comparison, not an automatically adopted acceptance threshold. A small compound/convex decomposition may be appropriate, but must be built and checked before claiming it fixes the current issue.

The current 64-vertex candidate remains **unqualified and uninstalled**. Review its [orthographic comparison](../../mkii_fourbar_2026-09-06/pad_collision_candidate_v1/pad_comparison.png) and the assembled occupancy audit before choosing a replacement. More vertices in one hull cannot restore concavity. After a versioned collider change, the simulation team must inspect the cooked shapes, rederive reset/contact geometry, and repeat unchanged physical/solver gates. The mechanical team need not redesign the silicone shape merely to make a single hull convenient.

### P1 — Navigation and interchangeable survey-payload interfaces

**Mechanical + perception + survey-team owners:** Add rigid datum frames for navigation IMU, LiDAR and camera mounts, and a separately named survey-payload mount. Proposed names such as `frame_nav_imu`, `frame_nav_lidar`, `frame_nav_camera` and `frame_survey_payload` are new deliverables, not claims that these frames already exist. Specify mounting tolerances, stiffness, mass/COM bounds, reserved volume and cable strain relief. Provide scan/camera occlusion sweeps through the intended leg and body motion range.

**Accept when:** the exported frame transforms match measurable fiducials and the sensor team's coordinate conventions; extrinsic calibration and timestamp responsibilities have named owners; visibility and cabling clear the approved workspace. Swapping the survey instrument should preserve a documented mechanical/electrical interface, but its mass, COM, drag and vibration demands still affect locomotion. Payload data processing can remain with the survey team while navigation sensing stays independently usable.

### Parallel — Leg-stand dynamics measurements

**Test + controls + electrical owners:** Record the carriage/rail moving mass and friction separately from the leg. Synchronize commanded/applied motor targets, encoder position/velocity, rail position, foot normal force, bus voltage/current and motor/mount temperatures. Prefer independent force/position measurements where feasible; motor-reported torque alone cannot independently calibrate its own torque scale. Retain raw timestamps, units, firmware, CAN topology/rate, controller gains and exact load/geometry/configuration for every test.

Start with static force balance and slow sweeps, then bounded reversals/steps, unloaded and loaded motion, stance holds and controlled contact loading. Estimate friction/backlash, bandwidth/delay, effective inertia, structural deflection, foot load–deflection/hysteresis and surface friction with uncertainty. Braking/regeneration and thermal recovery need their own electrical/thermal observations. Metal mounts suggest a heat path but do not establish continuous torque. The simulation's 48 V reference is not a confirmed battery choice. Its 5.5 N·m peak must not be treated as continuous support torque. [Provisional actuator contract](../../../docs/MKII_FOURBAR_TRAINING.md#rs05-v2).

**Accept when:** the measured response and a separately fitted model agree on held-out test traces within an explicitly chosen error budget, with residuals and repeatability reported. Match the stand's vertical constraint, carriage mass and rail friction in the test simulation. Additional off-axis tests are required before transferring claims to a freely moving full body. Update dynamics as a new lineage; retraining/fine-tuning and evaluation can reuse experience only through an explicit compatible transfer plan, never by relabeling old admission evidence.

## What does not require redesign, and what software has already addressed

- A closed four-bar is a legitimate mechanism. URDF's tree restriction requires exported loop metadata and a suitable simulator constraint formulation; it does not require deleting the pushrod or converting to a direct-drive knee.
- Native v5 uses 31 bodies, 30 tree coordinates, 18 named actuators and 12 bilateral physical mimic constraints, with zero external closure joints. The older pin-recovery document proposed six external closures; that is historical implementation guidance. Do not combine both constraint recipes or add a second drive to the passive knee. Physical mimics are distinct from purely visual source-URDF mimic tags.
- Software has implemented CAD-derived pin frames, the explicit 18-motor/30-coordinate adapter, coupled resets, motor mass top-up, a provisional speed/overload actuator model, corrected stance geometry, and every-substep physical checks before reset. These are implemented corrections, not certification that the robot is ready for hardware or terrain.
- Coarse primitives and separate simplified collision meshes are legitimate tools. The measurable local foot-envelope error is actionable; blanket replacement of every collider by a dense visual mesh is not justified. OpenArm and SO101 show useful packaging patterns, not transferable calibrated dynamics. [Pinned official-reference review](../../mkii_fourbar_2026-09-06/cad_reference_review/README.md).
- Current self-collision is initially disabled; absence of simulated self-contact is not evidence of real clearance. CAD interference sweeps and a separately qualified simulation collision policy remain necessary.
- Retain the archival serial model and its results for provenance. It welds moving linkage inertia into the femur and is not physically interchangeable with the current four-bar.

The recommended work order is: prove one named-leg export; extend the graph/ownership/frames to six legs; complete the loaded mass and workspace ledgers; supply the assembled foot geometry and navigation/payload interfaces; identify dynamics as the stand becomes available. These parallel deliverables improve the next asset release without making cosmetic CAD cleanup a prerequisite for learning on the current frozen candidate.

## Review provenance and checks

Reviewed working-tree HEAD: `a9019966328dbea0ac1bd065d39e0cd629dff4f9`. All three prior artifact manifests were verified read-only with zero mismatches: reference audit 4 entries, pad candidate 12 entries, assembled occupancy 7 entries. No geometry was regenerated and no test-run result is invented here. Exporter primary documentation above was re-opened on 2026-09-07; the loop source is specifically version 0.3.27, and the design source is `latest`, so the engineering release must pin its own exporter version.

| Input | SHA-256 |
| --- | --- |
| `cad_reference_review/SHA256SUMS` | `76685b11877cd38ab068b48d5d7ee4ad7177182e364f6418006fd9e4a9c72553` |
| `pad_collision_candidate_v1/SHA256SUMS` | `82a6c4e7e5e2efa01201556a6947551057be917e3956a5bf57b827799b0e1fb5` |
| `assembled_pad_accessibility_v1/SHA256SUMS` | `01ffbf0c7467a70e351dded8321cdec6f34ae318b0b2f7ee0bab8f1b54b559e1` |
| `robot/tools/import_onshape_hexapod.py` | `8fe2d708948432ca1bf8d28aed4ae1ecc3cb03bd75536c9fddced1ad3c5d8318` |
| `robot/hexapod_mkii_assy/assembly_report.json` | `4e229aa82b26ef5e3b6a779574fa990003f3529a48b09750937c6c44331b5a07` |
| `robot/hexapod_mkii_assy/part_overrides.json` | `27dd9ccb51a05fd2da6adbebc300e98c700c130e44826df6672357ee08d2f146` |
| `robot/hexapod_mkii_assy/joint_limits.json` | `37bfd70b59c12349bd0906f32d618c07dc910fbcf6ddb1b513f131ce7f4c19a8` |
| `robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf` | `2dae7165852b419f0bd6edbe03f7d398b403f1410c5b6bdc2c287b6834f543a6` |
| `docs/MKII_FOURBAR_TRAINING.md` | `51b2438bb9cecdc90c2bfa27e76c07cbcf90701b8ae947a291ddfc65e6522452` |
