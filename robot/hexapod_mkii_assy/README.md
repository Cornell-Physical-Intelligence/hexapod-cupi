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
- `preview/hexapod_core.js` — the shared engine (mesh unpacking with CAD
  feature edges, kinematic tree, styles, explode, section cut, labels,
  shadows, capture) inlined into both pages by `robot/tools/pack_urdf_viewer.py`.
- `preview/standalone_template.html` — the interactive viewer (published as
  the shareable web preview). Styles: CAD colours, tint by link, tint by leg,
  white + edges, white shaded, hidden line, ghost (x-ray), blueprint; face,
  edge and background colour pickers, background "none" for transparent PNG
  export. Two-level explode (assemblies away from the body, parts away from
  their link; fasteners travel along their own axis, screws toward the head)
  with an animated showcase, section cut on any axis, isolate by part, link,
  leg, class or name search, in-canvas labels, ground shadows, centre of mass
  marker, two-point measure, joint sliders and poses (zero, stance, as
  exported, tripod gait), snapshot lightbox (opaque or transparent).
  Headless renders: `?capture=<name>&post=http://localhost:8322/save&w=2400&h=1600`
  plus `style=`, `view=iso|top|front|side` or `theta=&phi=`, `explode=`,
  `parts=`, `isolate=lf,body`, `hide=fastener`, `labels=1`, `shadows=1`,
  `section=y&sectionAt=0`, `pose=stance`, `bg=hex|none`, `face=hex`,
  `edge=hex`, `grid=0`, `transparent=1`; the page posts a PNG to the `post`
  URL (a 20-line `http.server` that decodes base64 into a file).
- `preview/showcase_template.html` — a clean auto-playing tour for a website
  (no controls): reveal, leg fly-through, exploded views, blueprint, x-ray,
  section sweep, tripod gait, plan view, looping with fades and captions.
  Query options: `bg=hex`, `face=hex`, `edge=hex`, `captions=0`, `speed=`,
  `shots=reveal,explode,...`; click pauses. Pack with
  `python3 robot/tools/pack_urdf_viewer.py --template robot/hexapod_mkii_assy/preview/showcase_template.html --out tmp/hexapod_mkii_showcase.html`
  and drop the single HTML file into the site (it loads three.js from cdnjs
  and fonts from Google Fonts).
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
- `femur_pitch` / `tibia_pitch` keep the leg reference convention (motor
  positions at the shared CAD zero, limits 0…1.74533 and 0…2.53073 rad, RS05
  5.5 N·m / 50.27 rad/s). The leg-internal joint transforms were measured
  from this assembly (mean of the six legs, leg-to-leg spread 0.03 mm); the
  knee sits 0.5 mm further along its axis than in the leg export.
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

Import `urdf/hexapod_mkii_serial.urdf` as a floating-base USD with
`tools/import_urdf_to_usd.py` (Isaac Sim python; URDF inertials kept, no
convex decomposition, self-collision off), then author contact reports with
`tools/enable_nested_contact_reports.py`. `package://hexapod_mkii_assy/...`
resolves from this directory, exactly as the mock package was imported.
`isaaclab/hexapod_rl/asset_cfg.py` points at
`usd/hexapod_mkii_serial/hexapod_mkii_serial.usda` (override with
`HEXAPOD_USD_PATH`), names the joints and links, carries the joint limits and
the reset stance from `joint_limits.json` / `stance.json`, and
`isaaclab/validate.py` runs the standing gate: 18 joints, 19 rigid bodies,
six foot sensors, finite observations, no falls, no non-foot ground contact,
and computed torque under the 1.6 N*m rating.

The linkage model needs mimic-joint support (Isaac Sim >= 4.5) and is meant
for visual/kinematic checks. With 1927 visual meshes the model is heavy for
thousands of cloned environments; the STLs of fasteners and motor internals
can be dropped from the visuals if rendering becomes the bottleneck. The
Phase 1-2 curricula in `isaaclab/hexapod_rl/phase2_cfg.py` still carry the
mock's stance values and belong to the archived mock checkpoints.
