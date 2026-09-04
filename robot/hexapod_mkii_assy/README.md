# Hexapod MKII full assembly (from the onshape-to-robot export)

Built by `robot/tools/import_onshape_hexapod.py` from the 2026-09-03
`hexapod-mkii-assy` export (a single merged link: 1927 part instances, exact
per-part Onshape mass properties, per-part CAD colours, no mate structure) by
registering the leg v3 record (`robot/hexapod_leg_v3/leg_parts.json`) onto
every leg, body by body. `assembly_report.md` is the full audit of that
import; regenerate everything with

```sh
python3 robot/tools/import_onshape_hexapod.py \
  --source "/Users/andreboufama/Downloads/export (1)" --yaw-zero radial
```

## Files

- `urdf/hexapod_mkii_linkage.urdf` — 31 links / 30 joints: `body` plus, per
  leg, `coxa`, `femur`, `tibia`, `tibia_push_lever`, `tibia_pushrod`. The
  tibia four-bar is closed by `<mimic>` joints (lever = +tibia, rod = -tibia);
  the cut pivot of each loop is documented in a comment.
- `urdf/hexapod_mkii_serial.urdf` — 19 links / 18 revolute joints (lever and
  pushrod welded into the femur): the training-pipeline contract.
- `meshes/` — the 77 export STLs with ASCII-safe names (vendor motor parts
  renamed `motor_*`; the map is in `assembly_report.json`).
- `part_overrides.json` — user-verified attachments the leg record cannot
  know (the screw head caps are hard-attached to the first joint bottom plate).
  The yaw motor split (output flange + hub fixed to the frame, everything else
  rotating with the coxa) is the importer's `--yaw-output-side` rule.
- `preview/standalone_template.html` + `robot/tools/pack_urdf_viewer.py` —
  build the single-file viewer (meshes embedded, CAD feature edges
  precomputed from the 22 deg dihedral threshold) that is published as the
  shareable web preview: joint sliders, zero / exported / gait poses, tint by
  link, fasteners on/off, collision primitives, joint axes, edges,
  click-to-identify.
- `preview/index.html` — three.js viewer: 18 sliders, zero / CAD / stand
  poses, tripod-gait animation, CAD colours or body tints, collision shapes,
  click-to-identify. Serve the repo root (`.claude/launch.json` →
  `hexapod-preview`, port 8321) and open
  `/robot/hexapod_mkii_assy/preview/index.html`.

## Conventions

- Body frame = the export frame: z up, origin on the bottom plate, x/y as in
  Onshape. Forward is `-y` (the pipeline's convention for this layout: the
  body is long along y, middle legs on ±x).
- Leg names `lf lm lr rf rm rr` = left/right × front/middle/rear, left = +x.
  Links `<leg>_coxa` …, joints `<leg>_coxa_yaw`, `<leg>_femur_pitch`,
  `<leg>_tibia_pitch` (+ `<leg>_tibia_lever_pivot`, `<leg>_tibia_rod_pivot`
  mimics in the linkage model).
- `coxa_yaw` positive = counter-clockwise about body +z for every leg, zero =
  leg pointing straight out from the body centre (the yaw mates are locked at
  arbitrary angles in the CAD; `--yaw-zero cad` reproduces the CAD placement
  instead). Limits ±0.872665 rad.
- `femur_pitch` / `tibia_pitch` zero = the CAD pose. Positive `femur_pitch`
  raises the femur (46 deg above horizontal at zero, horizontal at -0.804);
  positive `tibia_pitch` opens the knee (74 deg interior at zero, closed at
  -1.29, straight at +1.85). Limits femur -1.745329..0.55, tibia -0.95..1.75
  (`joint_limits.json`; derived from the CAD kinematics, to be replaced by
  measured hardware stops). Motor 3 drives the push lever and the
  parallelogram carries its delta 1:1 to the knee. The leg-internal joint
  transforms were measured from this assembly (mean of the six legs,
  leg-to-leg spread 0.03 mm); the knee sits 0.5 mm further along its axis
  than in the leg export.
- Reset/validation stance (`stance.json`): femur -0.25, tibia -0.55, bottom
  plate 0.124 m up (reset 0.130); static hip and knee torque 0.9 N m each.
- Inertials: parallel-axis sums of Onshape's per-part mass, centroid and
  inertia (from the export's `robot.pkl`), with one hard override: every
  RS05 actuator totals 191 g (its vendor CAD is a 62 g hollow shell; the
  difference sits on the link carrying the housing as a solid cylinder the
  size of the housing). Everything else keeps its CAD mass, so the robot
  weighs 8.261 kg against Onshape's fused 5.942 kg; `--motor-mass 0`
  reproduces the CAD exactly. Only the
  yaw motor's output flange and hub (bolted through the frame) belong to the
  `body` link; the motor body, its top plate and the screws on them rotate
  with the coxa.
- Collisions: one primitive per structural part (parts of 2 cm3 or more,
  no fasteners or motor internals): oriented boxes for plates, covers and
  the tibia, one cylinder per motor and per round part, two spheres per
  silicone foot pad.
- Materials: the 14 CAD colours are shared `<material>` definitions.

## Isaac Sim / Isaac Lab

`docs/OPERATIONS.md` §10 is the import, contact-report, and validation
procedure on the Spark. `packages/hexapod_env/hexapod_env/assets/spec.py`
(`MKII_V1_ASSET`) mirrors the names, limits, stance, and foot-pad geometry
from this directory; change them here and there together, and
`isaaclab/tests/test_mkii_v1_asset_contract.py` binds the two. The linkage
model needs mimic-joint support (Isaac Sim >= 4.5) and is for visual and
kinematic checks only. With 1927 visual meshes the model is heavy for
thousands of cloned environments; fastener and motor-internal STLs can be
dropped from the visuals if rendering becomes the bottleneck.

## Regenerating

Never hand-edit the URDFs; change the inputs and re-run the importer above.
Keep `part_overrides.json` and the `--yaw-output-side` rule when re-exporting.
Read `assembly_report.md`: every leg body must register at sub-0.1 mm, no leg
part may be missing, and the mass totals must match. Mesh names stay ASCII.
`python3 robot/tools/pack_urdf_viewer.py --out tmp/hexapod_mkii_viewer.html`
builds the single-file web preview (`tmp/` is git-ignored).
