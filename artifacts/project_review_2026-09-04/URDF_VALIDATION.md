# Hexapod MKII: asset and repository review

Reviewed 4 September 2026. **The serial URDF passes the static checks performed. The deployed USD is not approved for new training: its principal inertia axes do not reproduce the URDF tensors.** This review does not certify hardware accuracy or arbitrary-terrain capability. No model, training config, checkpoint, or importer was changed during the review.

## Scope and provenance

| Item | Reviewed state |
|---|---|
| Local working branch | `codex/isaaclab-training-artifacts`, `d85698c` |
| Fetched main | `81d7c6f`; reviewed from an isolated `git archive` snapshot |
| Divergence | Local HEAD has 2 unique commits; main has 23 unique commits |
| Training URDF | `robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf` |
| URDF SHA-256 | `6109956e9e3a7a648bc4afa7310904c2fd3f7cb790e157d83212339a73b7fbc0` |
| Spark | `spark-e26c`, project mirror `/home/orionh/HEXAPOD`, not a Git checkout |
| Deployed USD | `robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda` |
| USD root-layer SHA-256 | `01e5a12b234d62bb94edf8a77529a1dde20df2e485c6b4d5c2e095292e2b032c` |

The root-layer hash alone does not identify the complete USD. All nine used layers and their hashes are recorded in `evidence/spark_usd.json`. The local and Spark serial URDF hashes match. Spark's stance/config differs from the local branch. Existing local viewer changes, `HANDOFF_SHOWCASE.md`, and unrelated PDFs were preserved. Main was fetched and inspected, not merged into the dirty checkout.

## Spark blank slate

Stopped the project training queue, its active launch wrapper, the identified `hexapod-rl-training` container, and project log-tail followers. The interrupted run was the `q_crouch_s52` scratch Phase-1 run, near iteration 418 of 500. Checkpoints, queue files, and logs were preserved; this was a compute reset, not deletion of research history.

Follow-up inspection found no remaining hexapod training process or loaded hexapod system/user service. GPU utilization returned to 0%, with desktop display processes remaining. The unrelated `isim-web-viewer-1` container was left running. No new simulation/training run was launched. Short CPU-only USD inspections were used for this audit.

## What passed

- Exactly 19 unique links and 18 independent revolute joints in one connected tree: six legs, each with coxa yaw, femur pitch, and tibia pitch. No fixed or mimic joint in the training variant.
- Positive masses, finite transforms, normalized axes, positive-definite inertia tensors satisfying the principal-moment triangle inequality, and valid limits consistent with the input JSON.
- All 77 unique mesh files exist and contain finite geometry. There are 1,927 visual instances and 171 collision primitives: 75 boxes, 84 cylinders, and 12 foot spheres.
- CAD-derived total mass is **8.26081134 kg**, including the specified 191 g override for each RS05. Root `body` is 1.8428742 kg. This is the modeled assembly, not a measured fully equipped robot.
- USD units are metres, up axis is Z, and it has one floating articulation, 19 bodies, 18 joints, and 171 colliders. Mesh references resolve. Masses, COMs, and joint limits survive import to floating-point precision. Contact-report metadata is authored.
- Local branch: **288 unit tests passed**. Fetched main, using its pinned Python 3.12/uv environment: **520 tests passed**, and all 112 entries in its archived pipeline manifest matched. These suites do not establish correct USD inertia or hardware readiness.

Evidence: `evidence/static_audit.json`, `evidence/spark_usd.json`, `evidence/main_tests_pinned.txt`. `audit_urdf.py` reproduces the static/kinematic audit with NumPy and SciPy from the repository root. It writes a fresh report beside itself; it does not edit the robot.

## Findings requiring action

### 1. Blocking: deployed USD principal-axis convention is inverted

For every link, reconstructing the tensor from the USD's diagonal moments and principal-axis orientation fails to recover the URDF tensor. The moments themselves match. For example:

| Link | URDF Izz, kg m² | Imported Izz, kg m² | Difference |
|---|---:|---:|---:|
| `lf_tibia` | 0.0012473405 | 0.0003400958 | about 73% lower |
| `lf_femur` | 0.0012475636 | 0.0007187981 | about 42% lower |

The installed `urdf_usd_converter` 0.1.3 routine passes NumPy's column eigenvectors directly as the rows of `Gf.Matrix3d` before extracting a quaternion. This produces the inverse orientation. Conjugating that orientation in the comparison recovers the tensors to floating-point precision. An independent check using `Gf.Rotation.TransformDir` confirmed the convention; this is not just a SciPy matrix interpretation.

Inertia expressed in the link frame must be `R * diag(I) * R.T`, where R takes principal-axis vectors into the link frame. See the [OpenUSD mass schema](https://openusd.org/release/api/class_usd_physics_mass_a_p_i.html). Evidence includes the installed routine in `evidence/spark_importer_inertia.txt`; `compare_inertia.py` reproduces the mismatch from the saved dump and intentionally exits 1 on this asset.

**Required:** correct the conversion path in a versioned implementation, regenerate a new USD, and require a numerical tensor round trip for all links. Then rerun standing, joint sweeps, and dynamic checks. Do not globally invert unrelated quaternions or hand-edit the URDF. Do not admit existing runs on this USD as evidence for a corrected asset.

### 2. Reset geometry and standing evidence disagree with the stated baseline

At the local reset angles (femur -0.25, tibia -0.55), the lowest foot spheres are 0.137927–0.137964 m below the plate origin. Resetting the plate at 0.130 m starts the feet about **7.9–8.0 mm inside the ground**. The advertised nominal 0.124 m is lower still. The minimum non-foot collider clearance at the 0.130 m reset is about 17.3 mm. The feet are mutually level within 0.0374 mm, so this is a vertical-offset issue rather than leg asymmetry.

The historical 32-environment, 1,000-step standing log passed its implemented gate and settled at 0.135596 m. It reported zero post-settle non-foot contact and saturation, but **2.829457 N m maximum raw computed torque** during the run, with applied torque limited to 1.6 N m. The validator does not require all transient raw demand to remain below 1.6. The runbook's stronger wording is inaccurate.

Spark later used a different crouch (femur +0.5, tibia -0.8, reset 0.084 m), which passed its historical standing gate and settled at 0.077994 m. That does not validate the local reset or the corrected-inertia asset. The crouch femur target also requires checking against the articulation's soft limits before adoption.

**Required:** derive reset height from foot geometry plus a documented clearance, reconcile stance JSON and config on the chosen baseline, and record both startup and steady-state torque. Historical logs are `evidence/spark_baseline_validate.txt` and `evidence/spark_crouch_validate.txt`. They were inspected, not rerun here.

### 3. Blocking for deployment: frame and asset contracts need explicit v1 wiring

The CAD frame is left = +X, forward = -Y, up = +Z. On fetched main, `HexapodMkiiV1FlatEnvCfg` inherits `command_frame = "body"`; the class is described as forward walking but positive command X is anatomical left in this frame. The existing navigation convention must be applied explicitly and tested for forward, reverse, both lateral directions, and both yaw signs.

Main correctly introduced a separate CAD task ID, `Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0`, while preserving the mock tasks. However, the default `ActionPipeline` still selects mock Stage2C joint positions. That default is not a v1 hardware configuration. Add a validated asset-specific runtime factory/manifest, explicit joint mapping and soft limits, and a checkpoint compatibility check before any motor execution.

Observed Spark articulation order was coxa joints for `lf lm lr rf rm rr`, then femur joints in the same leg order, then tibia joints in that order. Record this as an observation tied to the asset hash; resolve by joint name at runtime rather than assuming indices forever.

The local branch's old Phase-2 stance includes tibia 2.170 rad, beyond the CAD upper limit of 1.75. Main retains those profiles for the archived mock, which is appropriate. Do not accidentally reuse them for v1 during synchronization.

### 4. Model accuracy still depends on physical measurements

The serial model merges the push lever and pushrod into the femur and approximates the tibia linkage dynamically. The visual linkage model uses parallelogram mimic relations. Confirm those relations, motor zero/sign, encoder-to-joint angle mapping, transmission ratio, friction/backlash, and effective torque limits against a real leg. If the mapping is not exactly unit ratio, position, velocity, and torque conversion all need the measured relation.

The documented femur/tibia limits are CAD-derived, not measured stops. Mass distribution comes from CAD plus a motor approximation. Navigation sensors, battery, Jetson, wiring, and survey payload must be explicitly reconciled with the final assembly's mass/COM and mount transforms. A correct-looking model is insufficient evidence for these quantities.

Self-collision is disabled in the current runbook. Standing cannot establish leg-leg or linkage clearance throughout a learned gait. Perform swept-clearance and representative dynamic collision checks; add suitable collision constraints/geometry where needed. The importer also introduces a small tibia-frame angular discrepancy of roughly 0.0029 degrees, much smaller than the inertia defect, which should be evaluated against a chosen geometric tolerance.

The assembly README still lists superseded femur/tibia limits. Its RS05 peak effort is not a continuous-duty budget. Repair documentation and record continuous, transient, temperature, voltage, and slew assumptions explicitly.

## What James has been developing

The Git history identifies `palerdr` and James Cenawood with the same author email. This describes committed work, not a claim about every contributor or private work in progress. Review baseline: [main at 81d7c6f](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/tree/81d7c6f).

| Area | Present on main | Remaining |
|---|---|---|
| Architecture/tooling | Six packages for core, environment, training, evaluation, runtime, navigation; uv/Python pinning; CI; experiment supervision and evidence conventions | Exercise deployment supervision on the real Spark environment; unify mirror and release provenance |
| Contracts/runtime | 66-element proprioceptive observation contract, 18-action pipeline, command validation, clipping/slew logic, parity tests | v1 runtime defaults, inference backend, hardware state estimator, CAN driver and watchdog integration |
| RL/rewards | Velocity/yaw tracking; deck height/orientation/motion; slip/contact/support; torque/duty/slew/limits and diagnostics | Correct-asset omnidirectional training, rough terrain, real sensor observations, calibrated actuator model and broader randomization |
| Sensors | Mid-360 mount/self-occlusion study; simulated lidar, IMU and D455 prototypes with noise/age/dropout controls | New-CAD moving-leg occlusion study, physical calibration, reliable near-ground observations, operational LIO/elevation pipeline |
| Navigation | Small waypoint follower and command-producer seam | Polygon coverage, obstacle/traversability planning, localization uncertainty handling, coverage accounting, map UI |
| Requirements | Consolidated `dar.md`, `docs/PLAN.md`, `docs/TRAINING.md`, `STATUS.md`; no-RTK direction | Ratify owners, performance targets, interfaces and release gates using current user requirements |

Selected commits: `15fba81` architecture/contracts; `25a12f5` sensor study; `9a3cf96` workspace/CI; `1c32a92` separate CAD task; `352cbb8`/`ae039e6` documentation; `168b76c`/`7a12af1` requirements and no RTK.

Existing operational tasks are flat-ground tasks. Their randomization includes friction/restitution and limited body-mass variation; it is not an all-terrain sim-to-real suite. Stage2G explicitly rewards an alternating tripod phase. Keep that archived experiment separate from the requested learned, phase-unconstrained gait.

Main's recorded best checkpoint belongs to the mock. It misses its moving yaw gates and the highest-command deck gate and is explicitly not hardware-ready. Stage2G's improvement claim has no admitted artifact bundle. Conversely, main's assertion that nobody has trained the CAD asset is now stale: Spark has scratch CAD runs, but they are not admitted, correctly versioned evidence for the new program. Neither statement should be compressed to “the robot already walks reliably.”

The sensor mount result is also preliminary: the zero-self-hit mast study used the mock and a particular pose. Its roughly 2.47 m nearest-ground intersection cannot establish near-foot terrain visibility. [Livox specifies](https://www.livoxtech.com/mid-360/specs) a -7° to +52° vertical field of view, so a high upright mounting position needs a deliberate near-ground sensing strategy.

## Before the next new training campaign

1. Preserve the archived mock evidence and create a reproducible baseline from main, reconciling local CAD changes and Spark's modified configs.
2. Repair/import/test inertia, then reconcile reset geometry, measured or explicitly provisional dynamics, frames and v1 runtime mapping.
3. Freeze a new task/observation/action/asset manifest. Keep the actor's deployment observations available on hardware.
4. Pass short Isaac import, standing and directional checks with resolved config and hashes. Then begin the new curriculum under supervised launches.
5. Put the acceptance ladder, interface owners, independent ground truth and failure tests in place before a long training run.

See [the roadmap snapshot](ROADMAP.md) and [the living plan](../../docs/PLAN.md) for the multi-team sequence. The accompanying documentation revision updates the current plan and preserves its predecessor in `docs/archive/`. This audit remains a dated evidence snapshot; proposed targets are not silently ratified. The user's final instruction pauses new runs; the sequence above is preparatory, not an active queue.
