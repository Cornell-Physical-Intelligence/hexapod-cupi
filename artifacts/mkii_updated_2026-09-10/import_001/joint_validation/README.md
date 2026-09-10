# Independent articulated-model validation

The user confirmed that this revision replaces the old external four-bar with
three direct-drive joints per leg. This audit independently checks the new
19-body/18-joint inspection candidate. **Geometry and bounded visualization
checks pass after narrowing the knee range; the asset is not qualified for
Isaac dynamics or sim-to-real deployment.**

## Verified geometry

- All 1,753 original part instances occur exactly once in a connected, acyclic
  19-link/18-revolute-joint graph. Every original STL has a byte-identical
  ASCII-named output copy; all 1,753 URDF mesh references map correctly.
- At supplied CAD coordinates, 10,888,953 unique mesh vertices across all part
  instances match the original exported XML/STLs within **0.738 micrometers**.
  The source XML is rounded; the builder uses the higher-precision part poses.
- The serialized URDF's RPY transforms independently reconstruct the model
  within **6.23e-14** maximum matrix-entry error, including near-gimbal-lock
  parts. No part exceeds the 10-micrometer source-precision audit threshold.
- All 19 link masses are positive; all inertia tensors are symmetric positive
  definite and satisfy the principal-inertia triangle inequality. Their raw
  CAD mass totals **5.147603654203134 kg**. This checks mathematical validity,
  not the completeness of modeled motor/payload mass.

The raw-CAD, nominal-motor-mass-corrected and inspection variants share geometry
and joint bounds; this audit's collision result applies to that shared geometry.
It does not validate the corrected actuator inertia approximation or dynamics.
The exact geometry-and-bounds digest is recorded in `validation_report.json`;
it excludes model commentary and link-inertia metadata.

## Bounds found by checking the actual solids

Offsets are relative to radial yaw, femur +20 degrees above horizontal and knee
-110 degrees relative to femur (tibia downward at neutral).

| Joint | Inspection interval | Meaning |
| --- | --- | --- |
| Yaw | -25 to +25 degrees | Around radial yaw |
| Femur | -20 to +20 degrees | Absolute femur elevation 0 to +40 degrees |
| Knee | **-3 to +20 degrees** | Absolute relative knee -113 to -90 degrees |

The initial knee -20-degree endpoint was rejected: tibia intersected the femur
and attachment plate, and in some poses the coxa spacer/bottom plate. The
initial report is preserved under `initial_unsafe_envelope/`. It also records
a transient XML serialization defect subsequently fixed by the builder.

A follow-up scan tested all six legs at 41 knee angles in one-degree increments
from -20 to +20 degrees, at each of three femur offsets (-20, 0, +20). The -8
endpoint still intersects; -7 has only 0.036 mm clearance. The retained -3-degree
endpoint has **1.027 mm minimum primary-structure separation** in those checks.
No structure intersections occur between -3 and +20 on that sampled grid.

The final independent check used **133 poses**: neutral; each joint at both
endpoints; all eight coupled corners for each leg; 16 synchronized/alternating
full-robot corner poses; 32 fixed-seed 18-dimensional Sobol samples. It used
original triangle surfaces with 59 shared FCL BVHs and 1,753 independently
positioned parts. There are **zero new cross-body contact pairs** and **zero
primary-structure intersections** at all 133 poses.

These finite checks do not certify continuous swept volumes, every joint
combination, mechanical hard stops, cables, loaded deflection or stable support.
The bounds are an inspection envelope, not measured hardware travel limits.

## Existing geometry needs review

Every final sampled pose retains 126 cross-body surface-contact pairs: 54 in
motor/bearing components, and **72 tibia-to-M3-screw pairs** (12 per leg).
These are retained and reported, not removed from the model to pass a check.

Four representative RM screw/tibia pairs have 0.286–0.299 mm screw-vertex
inside-depth in the **original supplied CAD pose**, and about 0.299 mm at the
inspection neutral. This is measurable overlap with a watertight, consistently
oriented tibia mesh, not merely floating-point contact. It therefore predates
the derived neutral stance.

The affected M3 screws follow the femur's transforms across the six leg copies,
whereas each tibia has its independently varied angle. The working body map
assigns those screws to femur. Because the export omitted mates, review both
that attachment-side inference and the physical clearance rather than assuming
all these overlaps are intended. `overlap_review_parts.json` gives all exact
zero-based source IDs for the viewer. Representative pairs are tibia ID 6
(`rm_tibia`) with screws 9, 235, 243 and 244 (`rm_femur`).

The inspection URDF's detailed per-part triangle collision elements are not an
Isaac collision strategy. Exact motor/fastener rendering can stay while an
explicit reviewed physical contact model is developed separately.

## Ground and animation

At neutral, raising the body reference by **76.611091 mm** places the lowest
foot on the floor; the six feet differ in height by less than 0.704 micrometers.
The bounded joint sweeps can move a foot **25.139 mm below** that fixed support
plane, although non-tibia geometry stays at least 71.611 mm above it. A motion
inspection should lift the robot about 31 mm farther than neutral floor-contact
height (or adjust the floor) to keep the full visual sweep visibly clear.
Independent joint motion is not a planted-foot or stable-standing test.

## Reproduce

Install the scratch-only FCL wheel if needed (it is not a repository dependency):

```sh
uv pip install --python /path/to/project/.venv/bin/python \
  --target ./deps --no-deps python-fcl==0.7.0.11
/path/to/project/.venv/bin/python validate_candidate.py \
  --model-dir /tmp/hexapod-urdf-build-20260910 \
  --source /tmp/hexapod-urdf-intake-20260910 --out ./audit_output
```

`refine_knee_bounds.py` records the narrower-range evidence.
`inspect_neutral_contacts.py` and `inspect_source_cad_contacts.py` evaluate the
representative overlap depths. These bounded follow-up scripts record their
concrete input directory in the source. `deps/` is a local installed wheel and
should not be copied into the repository evidence bundle.
