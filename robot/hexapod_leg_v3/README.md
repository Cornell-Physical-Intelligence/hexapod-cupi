# Hexapod Leg v3 (hyper-accurate, from onshape-to-robot export)

Built by `robot/tools/import_onshape_leg.py` from the 2026-08-26 onshape-to-robot
export (single merged link) by registering the proven v2 body split onto the new
geometry (per-body Kabsch on surface features, <=0.02 mm RMS; hinge axes verified
from both adjacent bodies, worst 0.21 deg / 0.08 mm; motor-internals bodies solved
via the 1-DOF hinge constraint).

- `urdf/leg_v3_linkage.urdf` — 6 bodies, 5 joints; the tibia four-bar is a
  parallelogram (lever 30.2 / rod 78.2 / crank 30.5 / ground 77.5 mm) closed
  exactly by `<mimic>` joints; the cut pivot's anchors are documented in-file.
- `urdf/leg_v3_serial.urdf` — 3-DOF pipeline-compatible variant (linkage welded).
- `part_overrides.json` — user-verified rigid attachments the bore geometry
  cannot decide (knee mount rings + their four +y M3 screws -> femur, RS05
  output hub -> push lever). Survives re-exports.
- `leg_parts.json` — the complete per-body record (frames, inertials, every
  part instance in the export frame) consumed by `robot/tools/import_onshape_hexapod.py`
  to build the six-leg assembly (`robot/hexapod_mkii_assy`).
- Mass 0.7888 kg to Onshape's exact fused value; assembled COM within 0.4 mm.
- Preview: `robot/hexapod_leg_v2/preview/index.html` (serves both models,
  CAD colors, body-tint debug toggle, click-to-identify).

Upgrade path: naming the Onshape mates `dof_coxa_yaw`, `dof_femur_pitch`,
`dof_tibia_pitch`, `dof_tibia_lever`, `dof_tibia_rod`, `closing_tibia_linkage`
makes onshape-to-robot emit true per-link inertials and the loop natively,
removing the registration step entirely.
