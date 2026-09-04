# Hexapod MKII full assembly (from the onshape-to-robot export)

`robot/tools/import_onshape_hexapod.py` built this package from the
2026-09-03 `hexapod-mkii-assy` export (a single merged link: 1927 part
instances, exact per-part Onshape mass properties, per-part CAD colours, no
mate structure). It registered the leg v3 record
(`robot/hexapod_leg_v3/leg_parts.json`) onto each leg, body by body.
`assembly_report.md` is the full audit of that import. Regenerate the package
with

```sh
python3 robot/tools/import_onshape_hexapod.py \
  --source "/Users/andreboufama/Downloads/export (1)" --yaw-zero radial
```

## Files

- `urdf/hexapod_mkii_linkage.urdf`: 31 links / 30 joints: `body` plus, per
  leg, `coxa`, `femur`, `tibia`, `tibia_push_lever`, `tibia_pushrod`. The
  tibia four-bar motion is approximated by `<mimic>` joints (lever = +tibia,
  rod = -tibia); the cut pivot of each loop is documented in a comment. The
  documented pin points differ by about 0.5 mm, mainly axially. Mimic motion
  is not a physical loop constraint; see [the step-1 audit](../../docs/MKII_STEP1.md).
- `urdf/hexapod_mkii_serial.urdf`: 19 links / 18 revolute joints (lever and
  pushrod welded into the femur): the training-pipeline contract.
- `meshes/`: the 77 export STLs with ASCII-safe names (vendor motor parts
  renamed `motor_*`; the map is in `assembly_report.json`).
- `part_overrides.json`: user-verified attachments the leg record cannot
  know (the screw head caps are hard-attached to the first joint bottom plate).
  The yaw motor split (output flange + hub fixed to the frame, everything else
  rotating with the coxa) is the importer's `--yaw-output-side` rule.
- `preview/standalone_template.html` + `robot/tools/pack_urdf_viewer.py`:
  build the single-file viewer (meshes embedded, CAD feature edges
  precomputed from the 22 deg dihedral threshold) that is published as the
  shareable web preview: joint sliders, zero / exported / gait poses, tint by
  link, fasteners on/off, collision primitives, joint axes, edges,
  click-to-identify.
- `preview/index.html`: three.js viewer: 18 sliders, zero / CAD / stand
  poses, tripod-gait animation, CAD colours or body tints, collision shapes,
  click-to-identify. Serve the repo root (`.claude/launch.json`,
  `hexapod-preview`, port 8321) and open
  `/robot/hexapod_mkii_assy/preview/index.html`.
- `preview/inspection_v2.html`: current offline inspection with local
  dependencies, animated linkage by default, serial-model comparison,
  knee/joint sweeps, pose cycling, play/pause and speed controls. These are
  kinematic inspection motions, not a learned gait or physics simulation.

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
  (`joint_limits.json`, derived from the CAD kinematics and valid until the
  team measures the hardware stops). Motor 3 drives the push lever and the
  parallelogram carries its delta 1:1 to the knee. The importer measured the
  leg-internal joint transforms from this assembly (mean of the six legs,
  leg-to-leg spread 0.03 mm). The knee sits 0.5 mm further along its axis
  than in the leg export.
- Current simulation stance (`stance_v2.json`): coxa 0, femur -0.25, tibia
  -0.55; geometric contact height 0.137964 m, reset height 0.142964 m. These
  heights are derived from the actual collision primitives and give every pad
  at least 5 mm clearance at the nominal pose before reset jitter. They are
  not a measured settling height or
  a torque validation. Historical `stance.json` stays unchanged for v1
  reproducibility; its 0.130 m reset penetrates the ground by up to 7.964 mm.
- Inertials: parallel-axis sums of Onshape's per-part mass, centroid and
  inertia (from the export's `robot.pkl`), with one hard override: every
  RS05 actuator totals 191 g (its vendor CAD is a 62 g hollow shell; the
  difference sits on the link carrying the housing as a solid cylinder the
  size of the housing). Everything else keeps its CAD mass, so the robot
  weighs 8.261 kg against Onshape's fused 5.942 kg; `--motor-mass 0`
  reproduces the CAD masses. Only the
  yaw motor's output flange and hub (bolted through the frame) belong to the
  `body` link; the motor body, its top plate and the screws on them rotate
  with the coxa.
- Collisions: one primitive per structural part (parts of 2 cm3 or more,
  no fasteners or motor internals): oriented boxes for plates, covers and
  the tibia, one cylinder per motor and per round part, two spheres per
  silicone foot pad.
- Materials: the 14 CAD colours are shared `<material>` definitions.

## Isaac Sim / Isaac Lab

`docs/OPERATIONS.md` §10 records the historical import, contact-report, and
validation procedure. Follow [MKII_STEP1.md](../../docs/MKII_STEP1.md) for the
current immutable bundle workflow; contact reports are now authored by the
importer before hashing, not added after preparation. The CAD-v1 reset/import is archived: the original USD
has incorrect inertia orientations and cannot be admitted for new training.
The corrective importer/integrity gate and
`packages/hexapod_env/hexapod_env/assets/mkii_v2.py` (`MKII_V2_ASSET`) define
the corrected simulation lineage of this same mechanical URDF. Its USD has
a separate path under `usd/hexapod_mkii_serial_v2/` and uses
`HEXAPOD_MKII_V2_USD_PATH` for an explicit override.

Run the CPU geometry gate after every model/stance change:

```sh
python3 tools/audit_mkii_stance.py
python3 -m unittest discover -s isaaclab/tests -p test_mkii_v2_stance_and_frames.py
```

The gate evaluates all 171 collision primitives with full joint transforms,
checks the source URDF hash and requires the configured heights to match the
measured geometry. The nominal pose has at least 5 mm foot clearance, but the
inherited environment then adds independent uniform ±0.03 rad offsets to all
18 joints. The audit therefore also samples five offsets per joint on each
independent three-joint leg branch (125 samples per leg, 750 leg poses total),
including the endpoints and interior. The sampled minimum is **1.265 mm** for
a foot and **26.499 mm** for a non-foot collider. The complete jitter intervals
remain inside the articulation's soft joint limits. Thus the 5 mm nominal
clearance is not a claim about the randomized reset poses.

This grid is evidence at the sampled poses, **not a proof over the continuous
jitter range**. The report distinguishes that limitation from the exact
interval check for joint limits. A pass does not certify inter-leg collision,
contact dynamics, actuator/linkage dynamics or post-import standing. A new
asset must still pass the live simulator acceptance gate.
Measured hardware limits must still replace the current CAD-derived ranges.

`HexapodMkiiV2FlatEnvCfg` uses anatomical navigation commands: positive
forward is body -Y, positive left is body +X, positive yaw remains body +Z.
Linear velocity, angular velocity and gravity use that same proper rotation.
The new config uses the CAD runtime manifest for actions and an explicit
0.040 rad / 20 ms target slew limit; it is not a tuned omnidirectional policy.
Its joint order is checked by name against the imported articulation before
use. The next Isaac acceptance run must record the corrected import
articulation order; no GPU acceptance is implied by the CPU geometry gate.

Registration is explicit: call
`hexapod_env.tasks.mkii_v2.register.register_mkii_v2()` before resolving
`Isaac-Velocity-Flat-Hexapod-MKII-V2-Direct-v0`. Existing launchers do not
automatically select or register this ID. Historical `spec.py`
(`MKII_V1_ASSET`), `stance.json`, task IDs and their tests remain unchanged.
New revisions require a newly hashed asset/manifest and reviewed stance. The linkage
model serves visual and kinematic checks only; physical loop authoring and
actuator/transmission validation remain open. With 1927 visual meshes the model is heavy for
thousands of cloned environments. If rendering becomes the bottleneck, drop
the fastener and motor-internal STLs from the visuals.

## Regenerating

Do not hand-edit the URDFs. Change the inputs and re-run the importer above.
Keep `part_overrides.json` and the `--yaw-output-side` rule when you
re-export. Read `assembly_report.md`: each leg body must register below
0.1 mm, each leg part must be present, and the mass totals must match. Mesh
names stay ASCII.
`python3 robot/tools/pack_urdf_viewer.py --out tmp/hexapod_mkii_viewer.html`
builds the single-file web preview (`tmp/` is git-ignored).
