# MKII step 1: corrected serial asset and mechanism inspection

4 September 2026. The requested offline implementation is complete. **This is a
verified conversion of the serial URDF, not admission of a physically exact
robot or a training policy.** No SimulationApp, physics stepping or training
was launched. The linkage inspection exposed an additional model-fidelity
issue described below; G0 remains open.

## What changed

The original imported USD inverted each principal-axis orientation. A new
portable bundle preserves the imported layers and authors a stronger layer
whose inertia tensors and angular limits come from the serial URDF. It is
stored at:

`robot/hexapod_mkii_assy/usd/hexapod_mkii_serial_v2/hexapod_mkii_serial_v2.usda`

All 19 link tensors now reconstruct correctly in their link frames. Maximum
absolute component error is **1.621046e-9 kg m²**. The numerical gate also checks
mass/COM, units, scale, floating-base topology, all 18 joint anchors/axes/limits,
and all 171 collision primitives. Source mesh bytes remain unchanged. It does
not independently establish fidelity of every visual surface to the CAD.

The geometry comparison allows 0.1 mm displacement, body orientation error of
1e-4 rad, and collider orientation error of 2e-4 rad. The retained import's
maximum combined world collision-surface displacement bound is approximately
**11.72 µm**. Small converter rotation rounding remains within these stated
tolerances; this is not a claim of bit-exact transforms.

The authoring converts NumPy's column eigenbasis into Gf's row convention and
checks the result through independently transformed basis vectors. Both
tensors are at the center of mass; moving the COM does not require a
parallel-axis shift in this comparison. See the official
[MassAPI definition](https://openusd.org/release/api/class_usd_physics_mass_a_p_i.html)
and [Gf matrix convention](https://openusd.org/release/api/class_gf_matrix3d.html).

The nominal reset keeps the original joint angles and raises the plate from
0.130 m to **0.142964 m**. Exact primitive bounds give at least **5.000 mm** foot
clearance and **30.235 mm** non-foot clearance at those nominal angles. The
geometric contact height is 0.137964 m; it is not a measured equilibrium height.
The environment inherits joint reset jitter, so the nominal 5 mm claim must
not be applied to randomized resets. The stance evidence separately reports
the sampled jitter envelope; live startup still requires validation.

A new opt-in `mkii_v2` task uses anatomical forward = body −Y, left = +X,
up = +Z, and a schema-2 manifest tied to the serial URDF hash. Runtime targets
are mapped by joint name and use matching defaults, soft limits, action scale,
standing attenuation and slew limits. Measured starting joint positions are
required by the new runtime factory. Existing mock/CAD-v1 tasks, checkpoint
lineages and the 112-file archived pipeline manifest remain unchanged.

## Inspect the mechanism

Serve the repository root:

```sh
python3 -m http.server 8322 --bind 127.0.0.1
```

Open `http://127.0.0.1:8322/robot/hexapod_mkii_assy/preview/inspection_v2.html`.
The page uses local, pinned Three.js and URDFLoader files and all 77 local STL
files, so it does not need a CDN. Default **Linkage visual** uses the existing
31-link URDF, with 18 independent joints and 12 mimic joints. **Play** sweeps a
knee; choose another joint/leg, all six legs, or the stance-to-zero pose cycle.
Pause, speed, manual sliders, view buttons, collision display and part picking
support inspection. Motion is kinematic and can cross the ground; it is not a
walking controller. Model switches preserve the camera and joint pose.

The apparent pushrod regression came from showing the serial training model:
it merges the push lever and pushrod into the femur. The tibia then moves while
the rod stays still. The linkage view restores lever = +knee and rod = −knee
motion from the existing URDF, without adding a second parent to its tree.
The serial model remains selectable so this approximation is visible.

## Linkage fidelity is still an open gate

The documented cut-point locations are approximately **0.5 mm apart**, mostly
along the pin axis. The existing assembly report already notes a 0.5 mm axial
knee shift relative to the leg export. This is distinct from the much larger
serial-model separation: approximately **16.3 mm at the reset stance** when the
rod is held with the femur. The audit separates point, axial, transverse and
axis-angle residuals rather than silently snapping meshes together.

An axial offset between reference points on a pin is not alone proof of a
mechanical gap. Conversely, mimic animation is not evidence of a physical
constraint. The 0.1 mm cut-point coincidence diagnostic fails on the current
reference points. Reconcile the assembly and leg-export pin frames before
authoring physical loop constraints. The historical exported coxa angles for
left rear and right front also exceed the present software limits; this is
recorded evidence, not a reason to expand unmeasured hardware limits.

The serial asset includes the merged parts' mass but omits their relative
motion and constraint reactions. Its inertia round-trip pass means that USD
matches this approximation; it cannot establish correct linkage dynamics.
For a physical reference model, retain the moving bodies, reconcile the pin
frames and actuator/transmission coordinates, and author the cut joint as an
external constraint excluded from the reduced-coordinate articulation.
NVIDIA documents this supported approach and its solver error tradeoff in
[Rig Closed-Loop Structures](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/rig_closed_loop_structures.html).
No such physics model was authored or certified in this step. Compare it
against the serial approximation and the leg stand before choosing the
training representation or claiming 1:1 behavior.

## Reproduce the offline checks

From the repository root with its locked development environment:

```sh
uv sync --dev
uv run python -m unittest discover -s isaaclab/tests
uv run python tools/assets/audit_mkii_stance.py
uv run python tools/assets/audit_mkii_linkage.py
uv run python tools/assets/prepare_mkii_usd.py --check \
  robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf \
  robot/hexapod_mkii_assy/usd/hexapod_mkii_serial_v2/hexapod_mkii_serial_v2.usda
```

The linkage audit reports the measured discrepancy; add
`--require-point-coincidence` to turn that particular diagnostic into a failing
CLI gate. Tests deliberately require the known discrepancy to be reported,
not mislabeled as a pass.

To prepare another fresh, absent output directory from a raw self-contained
import, use `tools/assets/import_urdf_to_usd.py URDF NEW_OUTPUT --source-usd RAW_USD`.
This CPU path copies the source, authors contact reports, then performs the
complete gate before publishing. Existing/nonempty outputs are rejected.
**Do not run the contact-report helper on a prepared bundle:** its immutable
manifest already includes the contact authoring. The fresh SDK import path
without `--source-usd` is retained but was not exercised on this Isaac build.

The corrected bundle was copied to the matching new versioned path on Spark.
It reopened and passed the same CPU check using the installed image's OpenUSD
26.8, with no GPU devices and a read-only project mount. The old deployed USD
and source mirror were preserved. This does not validate PhysX parsing,
contacts, torque, articulation order or stability.

Evidence is in [the step-1 bundle](../artifacts/mkii_step1_2026-09-04/README.md)
and the asset's `asset_validation.json` / `SHA256SUMS`. The simulation manifest
is not a hardware release manifest or checkpoint admission gate.

## Next acceptance work

Resolve the linkage representation above, then build a versioned simulator
validator for the new task. Record observed named joint order, startup and
settled contact/torque behavior, reset jitter, individual joint sweeps,
anatomical direction/yaw tests and runtime parity. Existing `validate.py`
targets the historical task and cannot admit v2. `train_mkii_v2.py --dry-run`
only explains future task wiring; it is not an acceptance check or permission
to train. Its wrapper currently rejects checkpoint resume pending a separate
compatibility check. No new gait schedule or trained walking animation was
introduced.
