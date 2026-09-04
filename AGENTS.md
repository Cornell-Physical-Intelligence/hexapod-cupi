# Agent guide: Hexapod MKII (hexapod-cupi)

Read this before touching the robot model, the Isaac Lab task, or the Spark.
`HANDOFF.md` is the training-run history; its section 7 asset paths and joint
table describe the archived mock and are superseded by this file.

## The robot model (state on 2026-09-03)

- Training asset: `robot/hexapod_mkii_assy/`, the CAD assembly built from the
  Onshape export. Import `urdf/hexapod_mkii_serial.urdf` (19 links, 18
  revolute joints, no fixed joints). `urdf/hexapod_mkii_linkage.urdf` closes
  the tibia four-bar with mimic joints and is for kinematic/visual checks only.
- `robot/hexapod_mkii_mock_assy/` is archived. The Phase 1-2 checkpoints, the
  `revolute_*` joint names, the 6.3 kg mass budget and the stance values inside
  `isaaclab/hexapod_rl/phase2_cfg.py` belong to it. Do not point new work at it.
- Conventions (the code, the URDF and `assembly_report.md` all follow these):
  - body frame = URDF export frame: z up, forward = -y, left = +x; root link
    `body` (its origin is the bottom plate, so "root height" is plate height)
  - legs `lf lm lr rf rm rr` (left/right x front/middle/rear); links
    `<leg>_coxa`, `<leg>_femur`, `<leg>_tibia`; joints `<leg>_coxa_yaw`,
    `<leg>_femur_pitch`, `<leg>_tibia_pitch`
  - zero pose = the CAD pose for femur/tibia; `coxa_yaw` zero points the leg
    straight out of the body, positive = counter-clockwise about body +z
  - positive `femur_pitch` raises the femur (46 deg above horizontal at zero,
    horizontal at -0.804); positive `tibia_pitch` opens the knee (74 deg
    interior at zero, closed at -1.29, straight at +1.85)
  - limits: coxa +-0.872665, femur -1.745329..0.55, tibia -0.95..1.75
    (`robot/hexapod_mkii_assy/joint_limits.json`; femur/tibia are derived from
    the CAD kinematics and must be replaced by measured hardware stops)
  - reset/validation stance (`robot/hexapod_mkii_assy/stance.json`): femur
    -0.25, tibia -0.55, plate 0.124 m up (reset 0.130); static hip and knee
    torque 0.9 N*m each
  - mass 8.261 kg: Onshape per-part properties plus a hard override of 191 g
    per RS05 (the vendor CAD is a 62 g shell); the body link is 1.843 kg
- `isaaclab/hexapod_rl/asset_cfg.py` mirrors those numbers (names, limits,
  stance, USD path). Change them there and in the JSON files together.
- Runtime joint/action order is whatever the imported articulation reports
  (`robot.joint_names`); the code looks joints up by name. Never hardcode an
  index order; record the observed order in `HANDOFF.md` after the first
  import.

## Isaac Sim / Isaac Lab runbook (Spark `spark-e26c`)

Environment (from `HANDOFF.md`): Isaac Sim 6.0.1, Isaac Lab 3.0.0, Docker
Compose image `isaac-lab-base` bind-mounting `/home/orionh/HEXAPOD` at
`/workspace/hexapod`, launcher `/workspace/isaaclab/_isaac_sim/python.sh`,
`PYTHONPATH=/workspace/hexapod/isaaclab`. Before launching anything, check for
unrelated GPU workloads (`nvidia-smi`, `docker ps`) and never compete with
them; use `isaaclab/deploy/hexapod-rl` for training runs.

1. Sync: `git pull` on branch `codex/isaaclab-training-artifacts` in
   `/home/orionh/HEXAPOD`.
2. Generate the USD (inside the container, from `/workspace/hexapod`):
   ```sh
   python tools/import_urdf_to_usd.py \
     robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf \
     robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda
   ```
   Expected output: `rigid_bodies=19 revolute_joints=18`, the `.usda` plus a
   `payloads/` folder next to it, and the nested hierarchy
   `Robot/Geometry/body/lf_coxa/lf_femur/lf_tibia` (check
   `payloads/base.usda`). If the importer API differs on this Isaac Sim build,
   import from the GUI with: floating base, import inertia tensor ON, density
   0 (keep URDF masses), merge fixed joints ON, self-collision OFF, convex
   decomposition OFF, collision from visuals OFF, position drives; save to the
   same path. Meshes resolve through `package://hexapod_mkii_assy/...` from
   the package directory.
3. Contact reports (same container):
   ```sh
   python tools/enable_nested_contact_reports.py \
     robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda
   ```
   Expected: `CONTACT_REPORTS_ENABLED bodies=19`.
4. Validate (the acceptance gate; nothing trains before it passes):
   ```sh
   python isaaclab/validate.py --num_envs 32 --steps 1000
   ```
   It requires 18 joints, six feet, finite observations, no terminations or
   truncations, no coxa/femur/tibia-shaft ground contact after settling, and
   computed torque under the RS05 1.6 N*m rating (saturation fraction below
   0.5 %). Expect `post_settle_mean_base_height_m` near 0.12 and
   `max_abs_computed_torque_nm` well under 1.6. `isaaclab/deploy/validate-stance-sweep`
   runs three candidate stances for comparison.
5. Commit the generated `usd/` folder if it is meant to be shared, then update
   `isaaclab/hexapod_rl/asset_cfg.py`'s `USD_PATH` default only if the path
   changed (`HEXAPOD_USD_PATH` overrides it per run).
6. Before any training run on this asset, re-seed the curriculum stances in
   `isaaclab/hexapod_rl/phase2_cfg.py` from `stance.json`; they still hold the
   mock's angles and would fold the legs above the body. The archived stage-2
   hash manifest (`isaaclab/deploy/stage2_pipeline.sha256`) will flag
   `asset_cfg.py`; that is expected and applies to the mock pipeline only.

Troubleshooting:

| symptom | likely cause |
|---|---|
| `Unexpected contact-body layout` from `env.py` | USD hierarchy is not `Geometry/body/<coxa>/<femur>/<tibia>`; check the payload and the `ROBOT_GEOMETRY_ROOT` regexes in `env_cfg.py` |
| `Expected 18 joints` / `Expected 6 feet` | wrong URDF imported (linkage variant, or the mock), or fixed joints not merged |
| torque saturation on standing | stance moved away from `stance.json`, or masses changed; re-derive with the stance search in the importer notes |
| foot contacts counted as shaft contacts | `distal_foot_min_y_m` (0.155, tibia frame +Y) no longer matches the pad; re-check the pad spheres in the URDF |
| meshes missing after import | `package://` not resolvable: import from a checkout where `robot/hexapod_mkii_assy/meshes/` sits next to `urdf/` |

## Regenerating the model

- From a new Onshape export (`onshape-to-robot`, single merged link):
  `python3 robot/tools/import_onshape_hexapod.py --source "<export dir>"`
  (numpy + scipy; reads the leg record `robot/hexapod_leg_v3/leg_parts.json`,
  `part_overrides.json`, `joint_limits.json`; writes both URDFs and
  `assembly_report.md`). Read the report: every leg body must register at
  sub-0.1 mm, no leg part may be missing, and the mass totals must match.
- User-verified attachments live in `part_overrides.json` (screw caps on the
  coxa, pushrod cover on the push lever) and in the importer's
  `--yaw-output-side` rule (only the yaw motor's flange and hub stay on the
  frame). Keep them when re-exporting.
- Web preview: `python3 robot/tools/pack_urdf_viewer.py --out tmp/hexapod_mkii_viewer.html`
  builds the single-file viewer (published as a Claude artifact); `tmp/` is
  git-ignored. Local viewer: `.claude/launch.json` -> `hexapod-preview` (port
  8321), page `robot/hexapod_mkii_assy/preview/index.html`.
- Never hand-edit the URDFs; change the inputs and regenerate. Keep mesh
  names ASCII (the packer and importer rename vendor parts to `motor_*`).

## Checks that must stay green

- `cd isaaclab && python3 -m unittest discover -s tests -p 'test_*.py'`
  (288 tests, no Isaac Sim needed).
- `python3 -c "import xml.etree.ElementTree as ET; ET.parse('robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf')"`
  plus the structural checks in `robot/hexapod_mkii_assy/README.md`
  (unique names, all meshes present, positive-definite inertia, lower < upper).
