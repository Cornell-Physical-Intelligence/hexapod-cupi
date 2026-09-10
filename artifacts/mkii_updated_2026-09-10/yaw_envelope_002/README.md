# Intrinsic coxa yaw envelope from the updated CAD

All six proposed yaw intervals below have a **continuous 0.5 mm minimum gap to
the six chassis standoff plates** in the current geometry model. Each endpoint
was rounded inward to 0.01 degree before publication. This calculation includes
all 141 parts assigned to each coxa, including fasteners and motor housings.
Femur and tibia rigid groups are excluded, as the user requested for intrinsic
yaw limits. Their expanded travel can still cause coupled collisions.

Absolute azimuth is the coxa-to-shoulder direction in the body frame: +X = 0°,
+Y = 90°, and forward (−Y) = −90°. Positive rotation is counterclockwise viewed
from above. Intervals are unwrapped around each leg's current stance; headings
such as −204° or +226° are valid continuous azimuth coordinates.

| Leg | Absolute lower ° | Recommended zero ° | Absolute upper ° | Shift from old radial zero ° | Rebased joint interval ° |
| --- | ---: | ---: | ---: | ---: | ---: |
| LF | -125.01 | -50.715 | 23.58 | +17.4836 | ±74.295 |
| LM | -47.23 | -0.515 | 46.20 | -0.5150 | ±46.715 |
| LR | -24.61 | 50.140 | 124.89 | -18.0585 | ±74.750 |
| RF | -204.61 | -129.860 | -55.11 | -18.0586 | ±74.750 |
| RM | 132.77 | 179.485 | 226.20 | -0.5150 | ±46.715 |
| RR | 54.99 | 129.285 | 203.58 | +17.4836 | ±74.295 |

The corner neutral directions move approximately 17.5–18.1 degrees away from
the former body-center radial reference. The recommended neutral is the
angular midpoint of the actual opening between standoff stops. This addresses
the odd-looking corner stance by centering usable yaw travel; it does not move
any CAD bearing, pivot or part. Middle-leg changes are only about half a degree.

## Boundary evidence

The full revolution was searched at 2-degree spacing. The connected free
interval containing the prior stance was selected. Each transition was refined
by exact triangle-distance queries until the angular collision bracket was no
wider than **0.00001 degree**. The raw limits below are the just-free sides of
those near-zero-gap brackets; JSON retains both free and intersecting samples.
A 0.5 mm offset was then solved, followed by inward 0.01-degree rounding.

Pairs are **coxa source part ID / chassis standoff source part ID**, both
zero-based indices from the supplied export. Mesh names and nearest points are
recorded in the JSON.

| Leg | Raw lower transition ° | Raw upper transition ° | Lower limiting pair | Upper limiting pair |
| --- | ---: | ---: | --- | --- |
| LF | -125.814435 | 24.341571 | 613 / 32 | 607 / 41 |
| LM | -47.976821 | 46.961160 | 362 / 41 | 364 / 36 |
| LR | -25.357407 | 125.651008 | 686 / 36 | 688 / 27 |
| RF | -205.357058 | -54.348597 | 524 / 19 | 526 / 32 |
| RM | 132.022916 | 226.961385 | 261 / 128 | 263 / 19 |
| RR | 54.185361 | 204.341313 | 451 / 27 | 445 / 128 |

## Continuous check

For each part vertex, its distance from the fixed yaw axis is at most R, where
R was measured over every vertex of the complete coxa group (about 84 mm).
For an angular interval [a,b], evaluate the exact distance from the whole coxa
to all six standoffs at its midpoint. Rotating to any angle within the interval
can move any vertex by at most:

`2 R sin((b − a) / 4)`

Subtracting this motion bound from the midpoint's minimum triangle distance
therefore lower-bounds separation over the entire interval. Adaptive subdivision
produced 31–39 accepted intervals per leg. Every accepted interval exceeds
0.5 mm, with a 1e-10 m numerical acceptance guard. The weakest resulting lower
bound is **0.500128 mm**. The leaves completely tile the published intervals;
endpoint identity, leaf coverage and every bound formula were independently
rechecked. `continuous_clearance_certificate.json` preserves every accepted
subinterval and its actual distance query.

This is a one-dimensional certificate for the current triangle geometry and
inferred rigid grouping. It is not an all-joint configuration certificate or
measured hardware stop. Manufacturing error, cables, deflection and hardware
encoder calibration are outside this calculation.

## Other body geometry

The top enclosure, bottom enclosure and bottom plate were checked separately
from the six standoffs. Motor-output and bearing interfaces were excluded from
that structural crosscheck. No earlier surface intersection was found over the
full sampled revolution. However, an existing coxa M3 screw lies only about
**0.150 mm** above the bottom enclosure. This small gap persists at the yaw
endpoints and midpoint, so **0.5 mm is not a global body-clearance claim**.
The detailed nearest part IDs are recorded per leg.

A midpoint component-containment check also tests both directions: coxa
components inside standoff solids and standoff components inside coxa solids.
It uses an actual surface vertex from every face-connected component, bounding
box exclusion and ray containment where needed. See
`neutral_component_containment.json` for results and any non-watertight mesh
limitations; retain the source-surface scope of the continuous certificate.

## Files and reproduction

- `yaw_published_joint_bounds.json`: stable builder input. Bounds and zero shift
  are in the previous viewer coordinates. Rebase the yaw origin by zero_shift,
  subtract it from CAD coordinates, and subtract it from both limits. Physical
  geometry must remain unchanged.
- `yaw_geometry_limits.json`: full source IDs, raw/refined/published endpoints,
  absolute azimuths, prior-zero deviations, distances and certificate summaries.
- `raw_threshold_yaw_limits.json`: the unrounded result, retained unchanged.
- `full_rotation_scan.json`: full-revolution coarse samples.
- `continuous_clearance_certificate.json`: adaptive proof intervals.
- `neutral_component_containment.json`: connected-component containment check.

With NumPy, SciPy, trimesh and python-fcl 0.7.0.11 available:

```sh
python derive_yaw_envelope.py --model-dir /path/to/original/model/bundle --out /path/to/audit
python derive_yaw_envelope.py --model-dir /path/to/original/model/bundle --out /path/to/audit --certify-existing
python check_neutral_containment.py
```

The containment helper records its concrete scratch input paths. The main
script accepts input/output paths and never modifies source meshes or the
robot model. Raw source part ownership and old evidence remain unchanged.

## Independent check of the integrated model

`verify_integrated_certificate.py` was run against the current integrated
`robot/hexapod_mkii_updated_v1/model.json`, whose exact SHA-256 is recorded in
`integrated_model_verification.json`. It independently:

- Measured each absolute lower endpoint, new zero and upper endpoint by forward
  kinematics, including unwrapped azimuths; checked previous-zero mapping and
  symmetric rebased intervals, with all yaw defaults still zero.
- Recomputed the maximum yaw radius from unprocessed STL vertices, rather than
  trusting the radius stored in the original certificate.
- Verified that proof leaves exactly partition every published interval and
  recalculated every motion-bound formula.
- Re-ran all **216** midpoint distance queries using the integrated, rebased
  joints and unprocessed original triangle vertices. The weakest independently
  verified continuous lower bound remains **0.500128 mm**.
- Rechecked supplied-CAD-pose geometry for all **846 coxa part instances** against
  the original source URDF in the archived ZIP. Maximum vertex error stays below
  0.738 micrometers, within the original XML rounding precision.

Results are in `integrated_model_verification.json`,
`independent_midpoint_distance_checks.json` and `verification.log`.
`supporting_input_provenance.json` explains the exact builder-input precursor:
`yaw_geometry_limits_used_by_builder.json` matches the builder-recorded SHA-256;
the final report adds only the later component-containment summary. Published
endpoint numbers and model/URDF bytes were unchanged by that metadata addition.

Re-run the independent verifier with normal python-fcl availability, or supply
`--fcl-path /path/to/scratch/deps` for the isolated wheel installation. Its
repository-relative defaults resolve the integrated model and archived source.
`file_manifest.json` and `SHA256SUMS` cover the contents of this new bundle;
predecessor bundles were preserved.
