# Updated Onshape geometry recovery

This scratch audit recovers the geometry of the **user-confirmed direct-drive serial mechanism** from
`HexapodLegUpdatedV2.zip`, SHA-256
`31f33044285c3de2dbd85180e7c4886c31d80da4270dfd6b94bbd50d150a4cf7`.
The raw source remains a single rigid link, no joints, 1,753 part instances and
59 unique STL meshes. No exported mate graph, signed mate Z frames, limits or
closure pairs exist in the URDF. Geometry-based recovery cannot certify those
missing semantics or physical travel limits.

## Reproduce

```sh
/Users/andreboufama/Documents/CUPI/HEXAPOD/.venv/bin/python recover_geometry.py \
  --source /tmp/hexapod-urdf-intake-20260910 --out /tmp/hexapod-urdf-joint-recovery-20260910
```

Only NumPy, SciPy and trimesh are required. The script reads XML/STLs, never
unpickles the supplied source or uses historical joint axes. It writes bounded
JSON files. `legacy_import_diagnostic.log` records the old importer failing
against the changed structural mesh families; that failure is preserved.

## Geometric result

Each leg has three independently corroborated pivot lines. The explicit fitted
mesh hypotheses are holder local Z, insert local Z, flange local Y and motor
housing local Z. Each outer-cylinder ring is least-squares fitted; SVD of outer
wall normals independently checks the axial direction. Across the 18 assembled
sets the maximum independent line offset is **0.874 micrometers**, and maximum
unsigned axis-direction mismatch is **0.001744 degrees**. The motor housing's
faceted/chamfered wall determines the larger angle residual. These numbers are
internal CAD tessellation/export consistency, not mechanical manufacturing or
calibration accuracy.

Geometric measurements are 49.000 mm yaw-axis to shoulder, 73.502 mm shoulder to
knee, and a maximum tibia local +X extent of 130.000 mm from the knee. The last
number is an extreme surface coordinate, **not a spherical foot center**. The
collision audit confirms the foot end is not exactly spherical; use its detailed
shape and directional support geometry for reset height/contact.

Exactly six `femur_first_stage` and `tibia_attatchment_plate` pairs share rigid
poses. Each of six `tibia` parts varies independently relative to its femur.
The source includes 18 motor/holder/insert/flange sets and no external pushlever
or pushrod structural mesh. This supports 19 main rigid bodies and 18 active
revolute joints; it does not support reusing the previous physical four-bar
mechanism. The user subsequently confirmed in this task on 10 September: “yup, no more four bar, makes it way easier”. The direct-drive serial topology is confirmed; individual attachment ownership remains inferred.

## Ownership and output schema

`provisional_body_ownership.json` provides every source visual index exactly
once. Body naming follows the repo convention +X left, -Y forward: `lf lm lr rf
rm rr`.

| Main rigid body | Parts |
| --- | ---: |
| Chassis (`body`) | 199 |
| Each coxa (six total) | 141 |
| Each femur (six total) | 97 |
| Each tibia (six total) | 21 |
| Total | 1,753 |

1,674 assignments reproduce the full relative part pose across the six leg
copies with no more than 3 micrometers translation error and 0.002 degrees RMS
rotation error. This includes 120 repeated chassis parts. The remaining 79 are
an explicit, bounded chassis set: top/bottom enclosure and bottom plate; six
standoff plates; 21 M4 standoffs; 24 M4 PEM nuts; 24 M4 countersunk screws; one M3
low-profile screw. Each center lies inside the current enclosure bounds. These
79 are marked engineering inference requiring CAD ownership confirmation; the
code fails on an unreviewed remainder instead of silently putting it in body.

The `tibia_attatchment_plate` is on the **femur**, despite its name. The yaw motor
housing, holder and their attached fasteners follow the **coxa**. The yaw output
flange, insert and matching output-carrier group follow the **chassis**. At
shoulder and knee, housing/holder follow the proximal body and output groups
follow the distal body. These are supported by independently differing
assembly poses across the six leg copies, not a keyword-only assignment.

`joint_geometry_recovery.json` supplies, for each named hinge:

- Current CAD-world pivot point and unit direction; the line point is chosen in
  the structural middle plane. Yaw height comes from that leg's shoulder height.
- Independent supporting part IDs and their line/direction residuals.
- Child rigid-body frame in CAD world, its transform relative to its parent's
  rigid frame, and axis in the child/joint frame.
- The anchor-part transform in world for each body.

The proposed zero convention is **all joint coordinates zero at the provided
CAD pose**. Those poses vary across legs. This is distinct from a later
canonical stance, radial-yaw zero and hardware encoder zero. No physical pivot
origin is moved to change the starting stance.

## Remaining accuracy limits

Full-pose repetition is strong attachment evidence for asymmetric structure and
fasteners. Rotationally symmetric bearing rings, motor internal parts and some
flanges can have equivalent geometry under different rotations. Their displayed
Onshape transforms do not prove which race/rotor rotates physically. The JSON
mapping is therefore a reproducible main-body hypothesis, not a CAD-exported
mate graph. Whole motor components are lumped to their inferred attachment
side; geared rotor reflected inertia and individual bearing-race dynamics need
measured actuator identification.

There are no geometric hard-stop sweeps, cable clearances, real encoder signs,
measured stops or Isaac simulation results in this audit. Do not promote its
axes or provisional ownership to a qualified sim-to-real model, reuse old
four-bar joint bounds, or present CAD RGBA colors as physical materials.
