# CAD evidence and proposed physical four-bar reference

The current full assembly contains enough geometric evidence to construct a physical tibia linkage without another CAD clarification. Its pins define a **30 mm / 77.5 mm planar parallelogram**. The old cut-frame mismatch comes from carried reference frames, not a comparable mismatch between the actual pins in the current CAD assembly.

This directory contains a **proposal and CPU geometric evidence**, not a physical USD or a successful Isaac/PhysX validation. The full-turn test prescribes an ideal kinematic solution; it does not show that a passive solver maintains closure, chooses the correct branch, stands, or tracks motor commands.

## Reproduce the evidence

From the repository root:

```sh
uv run --no-project --python 3.12 --with 'numpy==2.2.6' python \
  artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/extract_cad_pins.py \
  --export '/Users/andreboufama/Downloads/export (1)' \
  --out artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/cad_pin_frames.json
```

`cad_pin_frames.json` records the external export's SHA-256, repository input hashes, mesh hashes, source visual indices, original visual transforms, recovered rigid-body transforms, measured body-local pin centers and normals, and proposed joint frames. The external export path is provenance, not a runtime dependency: the measured and proposed frames are included in the JSON. Re-extraction requires the original export. All matrices multiply **column vectors**, with translation in meters; no USD/Gf row-vector matrix is stored here.

The extraction recovers each body's actual assembly pose from its own structural mesh and the matching source visual. Existing URDF FK and CAD angles only identify the correct one of six source instances. They do not determine the pin lines. Each pin visual is independently matched back to the original assembly export.

The washer mesh is centered at its geometric origin. Its measured extents are 10 x 10 x 1.1 mm, and its thin-axis covariance eigenvector agrees with CAD local Z. This supports use of each washer visual's origin and local Z as the actual bore line. Lever and knee-bearing CAD origins corroborate the four pin centers. The motor output hub on the lever and motor bearing on the femur independently coincide with hinge A to below 0.0008 mm transversely, supporting actuation at A.

## Measured versus proposed geometry

In the following table, B compares the lever and proximal rod washer lines; C compares the distal rod and tibia washer lines. Distances are perpendicular to the common motor-axis direction.

| Leg | B line separation (mm) | C line separation (mm) | Largest proposed transverse correction (mm) |
|---|---:|---:|---:|
| lf | 0.000203 | 0.000320 | 0.000513 |
| lm | 0.000725 | 0.000095 | 0.000966 |
| lr | 0.000370 | 0.000156 | 0.000884 |
| rf | 0.000220 | 0.000075 | 0.000280 |
| rm | 0.000017 | 0.000479 | 0.000620 |
| rr | 0.000284 | 0.000313 | 0.000393 |

Across the six assemblies, projected AB is 29.999863–30.000301 mm; BC is 77.499632–77.500001 mm; DC is 29.999957–30.000227 mm; AD is 77.499116–77.500295 mm. The old approximate 30.18/78.21/30.50/77.52 mm values measured distances between arbitrary points along different hinge axes in 3D. Those values should not be used as planar linkage lengths.

The two B washer centers are about **3.4 mm apart along their common pin**, and the two C washers about **5.1 mm apart along their pin**. An arbitrary point on a hinge line can be moved along that line without changing the joint geometry. This is why defining both joint frames at the same world point does not require translating the tibia or pushrod by the old 0.5 mm cut-point discrepancy.

The proposal takes the measured A axis, signed to agree with the existing lever's positive rotation, and projects all points into a plane perpendicular to that axis through A. It retains measured ground and crank directions, sets AB=DC=30 mm and AD=BC=77.5 mm, and uses C=B+D−A. This is an explicit numerical idealization: maximum transverse change **0.000966 mm**, maximum axis change **0.000549 degrees**. These are below one micrometer and ten microradians respectively. They are not manufacturing-accuracy claims. The extraction rejects changes exceeding 0.005 mm or 0.005 degrees, a guard on CAD regularization rather than permission to relax other project gates.

Proposed A/B/C/D frames are stored in **both participating links' local coordinates**. Both sides reconstruct the same world origin and axis at the reference pose. A 721-sample −π…π prescribed kinematic sweep for each leg closes to at most **2.84e−16 m** in double precision. Injecting a 1 mm transverse shift or a 0.01 rad axis error fails the extraction checks. These numerical tests verify the frame construction and guardrails only; the flattening positions traversed in this mathematical sweep are not approved operating configurations.

## Physical topology and motor contract

| Hinge | Body 0 → Body 1, per leg | Proposed name | Role |
|---|---|---|---|
| A | femur → tibia_push_lever | `<leg>_tibia_lever_pivot` | RS05 actuator |
| B | tibia_push_lever → tibia_pushrod | `<leg>_tibia_rod_pivot` | Passive tree revolute |
| C | tibia_pushrod → tibia | `<leg>_tibia_loop_closure` | Passive regular revolute, excluded from articulation |
| D | femur → tibia | `<leg>_tibia_pitch` | Passive tree revolute |

Coxa yaw and femur pitch remain actuated. Preserve the 31 separate link bodies: body mass 1.8428742 kg; per-leg coxa 0.37089293 kg, femur 0.45918125 kg, tibia 0.15195908 kg, lever 0.054741785 kg, rod 0.032881142 kg. Total from the current linkage URDF is **8.260811322 kg**. Copy each complete local COM/inertia tensor as well as its mass; the serial asset's merged femur is inappropriate here. The current motor mass allocation is a CAD-derived approximation with the 191 g override, not measured reflected rotor/gear inertia.

Remove all 12 mimic relationships before adding six physical C closures. Add drive/actuator properties to exactly the 18 named motor joints. D, B, and C must not receive position drives, hidden importer drives, motor armature, or motor torque limits by a broad regex. Their passive bearing friction can be introduced separately when measured. Do not constrain D to follow A with a mimic in addition to the physical loop, and do not drive D and A together.

The resulting model has **31 bodies, 30 articulation joint coordinates, and six external closure joints**. It has 18 independent actuated mechanism DOFs away from singularities, plus its floating base; Isaac Lab still exposes the 30 tree joint coordinates. A physical revolute C joint should have `UsdPhysics.RevoluteJoint` type and `CreateExcludeFromArticulationAttr(True)`. NVIDIA requires the reduced-coordinate articulation itself to remain a tree and documents regular excluded joints for loop closure. Closed loops need separate solver stability checks. [NVIDIA articulation documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/rigid_bodies_articulations/articulations.html), [NVIDIA closed-loop example](https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/simulation_setup/articulations.html#closed-loops).

All hinge Z axes and both joint-frame orientations agree at the raw assembly CAD pose, so **q=0 in this proposal is that pose**, not the old URDF zero and not a hardware encoder zero. For the ideal parallelogram branch, changes satisfy Δq_D=Δq_A and Δq_B=−Δq_A. These relationships initialize and audit the loop; they must not replace passive dynamics. The old CAD knee angles in each leg's JSON provide only an approximate phase bridge. A builder must derive and record the exact bridge, rederive the physical stance, transform every joint limit to the new frame convention, and identify runtime indices by name. It must not silently reuse the serial reset angles.

One construction that preserves existing upstream frames is to retain the serial body's/coxae/femora's chosen FK pose, attach each four-bar using the proposed local frames relative to that femur, and derive tibia/lever/rod initial poses from the new joint coordinates. The source CAD transforms are a consistent reference for the four-bar; they must not be mixed with old distal link transforms as though their zeros were interchangeable.

The ideal mechanism flattens around **old tibia coordinates −1.1134 and +2.0282 rad** (approximate phase bridge). Existing limits −0.95…+1.75 stay outside those toggles but approach within about 9.4 and 15.9 degrees. Establish a measured operating margin. With a 30 mm crank, a quasistatic motor torque τ creates rod force magnitude approximately |τ|/(0.030 |sin φ|), where φ is crank-to-ground angle. Near the lower software limit, 1.6 N·m alone corresponds to roughly 328 N in the ideal rod model. Using the published 5.5 N·m motor peak in the same ideal calculation raises that estimate to approximately 1.1 kN. Include peak-torque and near-toggle cases in the mechanical/stand load analysis. These are quasistatic geometric estimates, not measured forces or validated structural limits.

## Bounded implementation path

1. **CPU physical asset builder and manifest:** add `tools/prepare_mkii_fourbar_usd.py`, reading this proposal and the 31-link URDF into a new versioned bundle. Preserve all 31 visual/collision and inertia assignments; author corrected joint frames and six closures; clear all mimics and passive drives. Reuse verified inertia math where appropriate, but do not route the asset through the 19/18 serial geometry validator. Add a dedicated graph/frame/primitive/closure validator, portable dependency hashing, and the same relocation/immutability checks.
2. **Single mechanism live qualification:** constrain one femur for an initial reference test; step only A slowly through a conservative interval under no load and known loads. Measure B/C/D angle response, closure point/axis residual, motor torque, kinetic energy and numerical drift. Repeat with gravity and rail stand configuration. Do not add a fixed joint between tibia and rod or use a mimic to hide a failed loop. Choose timestep/solver iterations from convergence measurements, not a visual animation.
3. **Explicit 30-to-18 simulation adapter:** create a versioned physical-linkage contract in `packages/hexapod_core/hexapod_core/` and a corresponding asset in `packages/hexapod_env/hexapod_env/assets/`. Resolve the 30 articulation names and 18 active names separately. Gather active motor position/velocity for the policy, scatter targets only to active joints, and compute torque/energy/limits from motor coordinates. Keep all 30 states for reset, closure diagnostics and passive-joint checks. Reset all states to one closed configuration; apply randomization to the 18 independent coordinates, then derive passive positions/velocities consistently. Never independently jitter all 30 coordinates.
4. **New task and observations:** add a separate `mkii_fourbar_v1` task under `packages/hexapod_env/hexapod_env/tasks/`, with matching Isaac Lab registration/factory entry. `HexapodEnv` currently requires 18 runtime joints and uses all-joint arrays throughout its action/observation/reward path, so merely overriding the USD path will fail or give the wrong contract. Refactor through an explicit joint adapter or use a dedicated environment; test every action, joint-limit, acceleration, torque, reset and observation slice. If the policy retains the same observation width, still assign a new semantic contract because knee angle/torque and motor-lever angle/torque are different signals. Record the observed runtime order and checkpoint contract.
5. **Full-body live gate:** add `isaaclab/validate_mkii_fourbar.py` and a deployment wrapper isolated from serial v2. Require the actual 31/30/6 topology and 18 actuator subset, closed resets, continuous closure residual bounds, branch/singularity margins, passive-drive absence, self-collision filtering/clearance checks, feet/contact validity, finite state, motor torque/velocity envelopes, and stability/convergence under support and command tests. Extend contact diagnostics to lever and rod bodies, which current femur/tibia shaft checks miss. Only a successful new gate admits this task to training.
6. **Transfer evidence:** compare motor-angle/velocity/current/temperature, rail motion and foot reaction against stand data. Update motor latency, saturation, gearing/reflected inertia, friction/backlash, pad/contact parameters, moving stand mass and rail friction. Train or fine-tune a new physics-version policy after this calibration. Preserve serial experiments as characterized baselines.

## Remaining inputs and confidence

**High confidence from existing geometry:** hinge placement, centered washer bore direction, planar parallelogram dimensions, active lever location, body ownership, and a reproducible consistent set of physical joint frames. No further user answer is required to begin implementing the reference USD and simulation adapter.

**Still to establish in code/live simulation:** correct SDK parsing of excluded revolute closures, passive drive removal, runtime topology/index order, exact coordinate/limit conversion, branch-preserving reset, full-body initial stance, collider filters, and solver convergence. The mathematical full-turn test is not evidence for these.

**Still requires the single-leg stand or hardware specification:** encoder zeros/signs, measured hard stops and toggle margin, actual rotor/gear reflected inertia, torque/current calibration and thermal limits, motor bandwidth and delay, joint/rail friction and backlash, actual link mass/inertia uncertainty, structural deflection, foot contact/friction and payload distribution. Geometry from an accurate CAD file cannot determine these dynamic properties.
