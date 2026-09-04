# Agent guide: Hexapod MKII (hexapod-cupi)

Read this before touching the robot model, the Isaac Lab task, or the Spark.
`CLAUDE.md` holds the repository invariants and `docs/TRAINING.md` §1 the asset
block for both robot models; this file is the hands-on runbook for asset v1
(the CAD assembly) and its Isaac Sim import. The Phase-0 mock's paths and its
`revolute_*` joint table (`docs/archive/HANDOFF-2026-08-26.md` §7,
`hexapod_core/joints.py`) stay valid for the mock's own task IDs.

## The robot model (state on 2026-09-03)

- Training asset: `robot/hexapod_mkii_assy/`, the CAD assembly built from the
  Onshape export. Import `urdf/hexapod_mkii_serial.urdf` (19 links, 18
  revolute joints, no fixed joints). `urdf/hexapod_mkii_linkage.urdf` closes
  the tibia four-bar with mimic joints and is for kinematic/visual checks only.
- `robot/hexapod_mkii_mock_assy/` is the Phase-0 lineage. The Phase 1-2
  checkpoints, the `revolute_*` joint names, the 6.3 kg mass budget and the
  stance values inside `phase2_cfg.py` belong to it, and every existing gym
  task ID keeps loading it (task IDs are frozen; see `CLAUDE.md`). New work
  uses asset v1 under its own task ID,
  `Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0` (`HexapodMkiiV1FlatEnvCfg`).
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
- `packages/hexapod_env/hexapod_env/assets/spec.py` (`MKII_V1_ASSET`) mirrors
  those numbers (names, limits, stance, USD path, foot-pad geometry). Change
  them there and in the JSON files together; `test_mkii_v1_asset_contract.py`
  binds the spec to the URDF and the JSON. The USD path defaults to the
  container path and is overridden per run by `HEXAPOD_MKII_V1_USD_PATH`.
- Runtime joint/action order is the imported articulation's own order
  (`robot.joint_names`), and the action vector is positional, so it is a
  contract: `packages/hexapod_core/hexapod_core/joints_v2.py` records the
  predicted order (breadth-first from `body`: six coxa_yaw, six femur_pitch,
  six tibia_pitch, legs lf lm lr rf rm rr) and is marked **provisional**.
  `HexapodEnv` compares it against the articulation at construction and
  refuses to run on a mismatch. After the first import, take the
  `joint_names=` line from `validate.py`'s output: if it matches, flip
  `RUNTIME_ORDER_STATUS` to `"confirmed"` in `joints_v2.py` and in the spec in
  the same commit as that transcript; if it differs, correct both tuples
  together. Nothing trains against a provisional order.

## Isaac Sim / Isaac Lab runbook (Spark `spark-e26c`)

Environment (from `HANDOFF.md`): Isaac Sim 6.0.1, Isaac Lab 3.0.0, Docker
Compose image `isaac-lab-base` bind-mounting `/home/orionh/HEXAPOD` at
`/workspace/hexapod`, launcher `/workspace/isaaclab/_isaac_sim/python.sh`,
`PYTHONPATH=/workspace/hexapod/isaaclab`. Before launching anything, check for
unrelated GPU workloads (`nvidia-smi`, `docker ps`) and never compete with
them; use `isaaclab/deploy/hexapod-rl` for training runs.

1. Sync: `/home/orionh/HEXAPOD` is a bind-mounted mirror, not a git checkout.
   Sync the source there as `docs/OPERATIONS.md` describes and verify it with
   `sha256sum -c isaaclab/deploy/stage2_pipeline.sha256` (the manifest covers
   the asset-v1 modules too).
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
   python isaaclab/validate.py --asset mkii_v1 --num_envs 32 --steps 1000
   ```
   (`--asset mock` re-validates the Phase-0 lineage.) It requires 18 joints,
   six feet, finite observations, no terminations or truncations, no
   coxa/femur/tibia-shaft ground contact after settling, and computed torque
   under the RS05 1.6 N*m rating (saturation fraction below 0.5 %). Expect
   `post_settle_mean_base_height_m` near 0.12 and `max_abs_computed_torque_nm`
   well under 1.6. It prints `joint_names=[...]`: keep that transcript, it is
   the evidence for the runtime joint order (see above). If the environment
   raises "Articulation joint order differs from the asset's runtime joint
   contract", the prediction in `joints_v2.py` was wrong; correct it and the
   spec from the printed order, do not bypass the check.
   `isaaclab/deploy/validate-stance-sweep` runs three candidate stances for
   comparison.
5. Commit the generated `usd/` folder if it is meant to be shared, then update
   `MKII_V1_ASSET.usd_path_container` in `assets/spec.py` only if the path
   changed (`HEXAPOD_MKII_V1_USD_PATH` overrides it per run).
6. Curriculum stages for this asset derive from `HexapodMkiiV1FlatEnvCfg`
   under new task IDs, with stances re-derived from `stance.json` and a fresh
   stance sweep for the 8.26 kg mass distribution (ADR-0001). The
   `phase2*_cfg.py` classes stay on the mock; do not re-seed them. Any change
   to a manifest-listed file means regenerating
   `isaaclab/deploy/stage2_pipeline.sha256`; it must pass locally and on the
   Spark.

Troubleshooting:

| symptom | likely cause |
|---|---|
| `Unexpected contact-body layout` from `env.py` | USD hierarchy is not `Geometry/body/<coxa>/<femur>/<tibia>`; check the payload and the `MKII_V1_GEOMETRY_ROOT` sensor paths in `env_cfg.py` |
| `Articulation joint order differs from the asset's runtime joint contract` | PhysX ordered the joints differently from the `joints_v2.py` prediction; correct `joints_v2.RUNTIME_JOINT_NAMES` and `MKII_V1_ASSET.runtime_joint_names` from the printed `joint_names` |
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

- `python3 -m unittest discover -s isaaclab/tests` from the repo root (no
  Isaac Sim needed; without `torch`/`numpy` the dependent modules surface as
  import errors and the rest runs; read the total off the run, it grows with
  every test). `test_mkii_v1_asset_contract.py` covers this asset.
- `python3 -c "import xml.etree.ElementTree as ET; ET.parse('robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf')"`
  plus the structural checks in `robot/hexapod_mkii_assy/README.md`
  (unique names, all meshes present, positive-definite inertia, lower < upper).
