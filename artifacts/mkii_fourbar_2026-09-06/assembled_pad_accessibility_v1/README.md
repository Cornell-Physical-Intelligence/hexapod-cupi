# Assembled LF pad occupancy audit

The largest cavity-overfill error in the proposed convex pad is **not filled
by the assembled tibia**. Its witness lies outside the bounding boxes of all
45 non-silicone CAD visual instances attached to `lf_tibia`, with a minimum
distance of **15.951 mm to those boxes**. Therefore every corresponding CAD
solid is at least that far away. This particular finding does not depend on
winding numbers or watertightness assumptions.

The single convex pad cannot be described as exact external collision geometry
at that location. Some other filled regions do coincide with the tibia, so the
audit also distinguishes useful internal filling from empty volume.

## Scope and evidence

This read-only CPU audit uses the frozen
[pad candidate v1](../pad_collision_candidate_v1/README.md), the original
silicone STL, and all other `lf_tibia` visuals from the exact source linkage
URDF. Each visual's local origin and rotation is applied before testing
occupancy; mesh scale is explicitly checked to remain one. There are **45
non-silicone instances using 25 unique meshes**, including the machined tibia,
inserts, bearing components, screws, spacers and washers.

The candidate's existing barycentric surface grid has 3,970 points. This audit
first selects points farther than 0.1 mm from the actual silicone surface,
then uses the closed silicone mesh to distinguish inside material from outside
material. The 91 selected points inside silicone represent small inward
approximation errors, not cavity overfill. The resulting 352 outside points
are tested against every other visual solid.

| Result | Sample points |
| --- | ---: |
| Candidate surface samples | 3,970 |
| More than 0.1 mm from the silicone surface | 443 |
| Outside the silicone solid | 352 |
| Of those, occupied by other assembled CAD | 158 |
| Of those, empty after all other CAD checks | 194 |
| Ambiguous occupancy classifications | 0 |
| Outside every non-silicone mesh bounding box | 155 |

All 158 occupied points are inside `machined_tibia.stl` (visual index 2).
Its bounding box also contains 39 points that its actual solid does not fill.
The remaining 155 empty points lie outside every other visual's bounding box.
These are sample counts, not percentages of surface area or volume, and not
probabilities of terrain contact.

## Worst witness

The previous proposal's largest sampled overfill lies **7.807 mm from the
nearest silicone surface**. Its coordinates are preserved for inspection:

| Frame | x (mm) | y (mm) | z (mm) |
| --- | ---: | ---: | ---: |
| Original silicone STL | 235.951 | −12.750 | −2.967 |
| `lf_tibia` joint/link frame | −114.889 | 206.102 | −5.512 |

The point has zero generalized winding in the silicone mesh to numerical
precision. More importantly, none of the other 45 transformed visual bounding
boxes contains it. An independent verifier recomputes this exclusion directly
from the original URDF transforms and STL vertices, without calling the
occupancy classifier.

## Robustness and limits

Every referenced non-silicone mesh passed closed-edge, consistent-winding and
nondegenerate-triangle checks. The audit uses oriented solid angles for points
inside a visual's bounding box. A near-integer nonzero winding denotes occupied
material; near-zero winding denotes outside. Near-surface or noninteger cases
are classified uncertain rather than silently rounded. Bounding boxes are
expanded by 10 nm, and winding is compared to integers within 1e-6. All
tolerances and per-mesh results are recorded in [report.json](report.json).

Edge-manifold checks alone do not rule out self-intersections. Accordingly,
winding-based classifications retain that geometric caveat. The worst witness
and 155 box-excluded samples remain outside regardless of that limitation.
Four analytic tests cover known cube inside/outside winding, reversed
orientation, an open surface and one inconsistently wound face.

This audit establishes occupancy of the LF tibia rigid assembly only. It does
not prove a rock's approach path, evaluate other moving links or absent wiring,
or measure the real silicone's shape under load. It also does not establish
that all six legs have identical assembly occupancy. No assumption is made
that a neighboring moving part permanently fills the empty space.

For accurate terrain contact, review the complete exterior profile and use
multiple convex pieces where empty concavities can be reached. Increasing
vertices in a single convex hull cannot preserve those concavities. The current
candidate remains useful as an outer-envelope comparison, with the limitations
shown in its [orthographic figure](../pad_collision_candidate_v1/pad_comparison.png).
No new decomposition or runtime asset was created by this audit.

## Reproduce

Run in a disposable copy to preserve published evidence. The dependency
environment is the frozen candidate's isolated requirement file:

```sh
uv run --no-project --with-requirements artifacts/mkii_fourbar_2026-09-06/pad_collision_candidate_v1/requirements.txt python artifacts/mkii_fourbar_2026-09-06/assembled_pad_accessibility_v1/audit_assembly.py
uv run --no-project --with-requirements artifacts/mkii_fourbar_2026-09-06/pad_collision_candidate_v1/requirements.txt python artifacts/mkii_fourbar_2026-09-06/assembled_pad_accessibility_v1/test_geometry.py
uv run --no-project --with-requirements artifacts/mkii_fourbar_2026-09-06/pad_collision_candidate_v1/requirements.txt python artifacts/mkii_fourbar_2026-09-06/assembled_pad_accessibility_v1/verify.py
```

[report.json](report.json) records exact inputs, all transforms and topologies,
classification totals, and the 20 largest sampled silicone gaps.
[sampled_overfill_classification.csv](sampled_overfill_classification.csv)
contains all 352 classified overfill samples.
[verification.json](verification.json) records source hashing and the
independent worst-witness exclusion. Neither the frozen pad proposal nor any
runtime URDF/USD was modified.
