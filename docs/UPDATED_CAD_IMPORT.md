# Approved CAD geometry and import provenance

The program lead approved the detailed direct-drive model and its nominal motor correction
for training. [`robot/active_model.json`](../robot/active_model.json) selects its
exact identities: 1,753 source part instances, 59 mesh byte streams, 19 bodies
and 18 joints. Use the [model guide](../robot/hexapod_mkii_updated_v1/README.md)
for simulation inputs and viewer controls.

The supplied ZIP contains one rigid link and no exported joints. We recovered
bearing axes and inferred rigid grouping from the six posed leg copies. The program lead
confirmed the serial topology. Mechanical review of fastener, bearing and rotor
ownership remains open. Preserve the original rounded-square toe and mesh bytes.

## Coordinates and travel

Use body +X left, −Y forward and +Z up. Positive joint rotation follows local
+Z. Preserve the viewer pitch zero: femur elevation +20° and knee angle −110°.
The program lead specified femur travel −120°…+80° and tibia travel −5°…+180° about those
zeros. Combined poses can intersect the body, adjacent legs or the ground.

Each coxa zero is the midpoint of its source standoff-clearance interval, using
a 0.5 mm plate buffer. Absolute azimuth runs counterclockwise from +X; forward
is −90°. Keep intervals unwrapped across ±180°.

| Leg | Absolute lower / ° | New absolute zero / ° | Absolute upper / ° | Viewer yaw range / ° |
|---|---:|---:|---:|---:|
| LF | −125.01 | −50.715 | +23.58 | ±74.295 |
| LM | −47.23 | −0.515 | +46.20 | ±46.715 |
| LR | −24.61 | +50.140 | +124.89 | ±74.750 |
| RF | −204.61 | −129.860 | −55.11 | ±74.750 |
| RM | +132.77 | +179.485 | +226.20 | ±46.715 |
| RR | +54.99 | +129.285 | +203.58 | ±74.295 |

The [yaw audit](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/mkii_updated_2026-09-10/yaw_envelope_002/README.md)
checks 141 coxa parts per leg against six standoffs. Its 216 midpoint queries
support a weakest continuous gap of 0.500128 mm. This claim covers standoff
plates; a coxa fastener-to-enclosure gap is about 0.150 mm. The earlier 133-pose
collision test covers the retired narrow inspection envelope.

Use [`joint_review.json`](../robot/hexapod_mkii_updated_v1/joint_review.json)
and the [frozen coordinate report](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/3ccfd4a12aa7b95347c0884ac3ca466cad97b8aa/robot/hexapod_mkii_updated_v1/BUILD_REPORT.md)
for rebase formulas and invariance checks. CAD joint sweeps establish travel;
walking qualification needs native policy recordings and physical gates.

## Mass and source geometry

The CAD sum is 5.147603654203 kg. The motor correction adds 128.804699 g to each
of 18 motor sets to reach the nominal 191 g per motor and 7.466088235226 kg total.
The added housing inertia uses a provisional Ø46 mm ×35.1 mm solid cylinder.
Read [MASS_INERTIA](../robot/hexapod_mkii_updated_v1/MASS_INERTIA.md) for link COMs
and full tensors, and [RS05_SPEC_REVIEW](RS05_SPEC_REVIEW.md) for rating scope.
Rotor inertia and the complete payload/electronics inventory still need measurement.

The import audit reconstructs source vertices within 0.738 µm. Fitted bearing
axes agree within 0.874 µm line separation and 0.001744° unsigned direction.
The [intake bundle](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/tree/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/mkii_updated_2026-09-10/import_001)
preserves source bytes and the numerical audits.

Mechanical review must resolve 72 source tibia/M3-screw overlap pairs and 54
cross-body motor/bearing pairs. Four inspected screw penetrations measure
0.286–0.299 mm in the original CAD pose. The viewer's overlap filter identifies
the tibias and screws; see
[`overlap_review_parts.json`](../robot/hexapod_mkii_updated_v1/overlap_review_parts.json).

## Native and hardware scope

The approved USD keeps visual geometry and inertias, with 153 exposed-structure
SDF colliders. The [USD audit](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/mkii_updated_2026-09-10/usd_002/README.md)
checks serialized frames and inertia against the URDF. Omitted internal fasteners
remain in mass and rendering; the simulator does not model their snagging.

Researchers completed native standing and walking experiments on named source
packs; [STATUS](https://cornell-physical-intelligence.github.io/hexapod-cupi/#findings) records their outcomes. Obtain fresh matching
standing admission for a new source allocation. Before hardware deployment,
measure actuator behavior and encoder mapping, resolve clearance/ownership
findings, and add the measured sensor/payload inventory. An 18-action historical
checkpoint has no implied compatibility with this robot.

For a new articulated CAD export, retain a version-pinned assembly, named
`dof_<joint_name>` revolute mates and `link_<name>` frames. Verify signed axes,
dynamics retrieval and limits against the [original import procedure](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/3ccfd4a12aa7b95347c0884ac3ca466cad97b8aa/docs/UPDATED_CAD_IMPORT.md#onshape-export-conventions).
