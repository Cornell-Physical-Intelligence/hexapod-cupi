# Hexapod MKII — autonomy and RL walking

Software for the Hexapod MKII, whose mission is to **survey a bounded area
drawn on the fly**: an operator draws a zone on a map, places the robot near
it, and the robot steadily traverses the zone as a steady platform for data
collection ([`dar.md`](dar.md), the team lead's requirements). The repository
holds the Isaac Sim 6.0.1 / Isaac Lab `DirectRLEnv` + RSL-RL PPO walking
pipeline, the robot models, the frozen contracts between locomotion,
perception, and navigation, and the evidence trail of every experiment. Two
robot models live in `robot/`: the Onshape mock in
`robot/hexapod_mkii_mock_assy/` that every Phase-0 checkpoint was trained on,
and the CAD assembly in `robot/hexapod_mkii_assy/` (asset v1, ADR-0001) that
new training targets. Each has its own gym task IDs; see "Robot models" below.

**Start with [`dar.md`](dar.md)** for the mission and the steps to it,
[`STATUS.md`](STATUS.md) for the current best checkpoint, the current target,
open contradictions, and next actions, and [`docs/ROADMAP.md`](docs/ROADMAP.md)
for milestones and gates. This README is a map, not a status page.

## Repository map

```text
dar.md / DAR.png                The mission (requirements slide) and the steps from here to it
STATUS.md                       Current state. Rewritten in place, never appended
CLAUDE.md                       Agent/contributor guide and project invariants
HANDOFF.md                      Stub pointing at the split documentation
docs/ROADMAP.md                 Program direction: workstreams, milestones, gates
docs/ARCHITECTURE.md            Autonomy layers and contracts (locomotion / perception / navigation)
docs/decisions/                 Architecture decision records
docs/ONBOARDING.md              New-member read order and setup
docs/TRAINING.md                Task and training design, curriculum, acceptance gates
docs/OPERATIONS.md              Spark runbook, GPU protocol, launchers, screens
docs/incidents/                 Frozen forensic records of specific failures
docs/archive/                   Superseded documents, kept verbatim
robot/hexapod_mkii_assy/        Asset v1: CAD assembly URDF package (urdf/, meshes/, assembly_report.md, preview/)
robot/hexapod_mkii_mock_assy/   Phase-0 mock URDF package (urdf/, meshes/, launch/)
robot/hexapod_leg_v3/           Single-leg reference the assembly import registers onto
robot/sensors/                  Livox Mid-360 datasheet values and mount placement study
robot/tools/                    onshape-to-robot importers (leg, full assembly), viewer packer
packages/hexapod_core/          Frozen interface contracts: observation, action, command, joints
packages/hexapod_env/           Direct RL task: env, rewards/, assets/, stage configs, registration
packages/hexapod_train/         Run composition, run contract, startup supervisor
packages/hexapod_eval/          Acceptance gates (gates.py) and the evaluation entry point
packages/hexapod_runtime/       Deployable policy runtime: observation builder, action pipeline
packages/hexapod_nav/           Command-producer seam and an example waypoint follower
configs/                        Named experiment baselines and intervention deltas
isaaclab/hexapod_rl/            Compatibility shims re-exporting packages/hexapod_env
isaaclab/deploy/                Launchers, services, and the pipeline SHA-256 manifest
ops/                            hexctl operator CLI; attic/ holds retired deploy scripts
isaaclab/tests/                 Unit and contract tests
tools/                          URDF-to-USD import and physics preparation scripts
viewer/                         Vite + React + urdf-loader web viewer
artifacts/                      Evidence: checkpoints, evaluations, videos, probe ledger
```

## Quickstart

Set up the environment and run the test suite (no simulator required). The
repository is a uv workspace; `uv sync` installs the pinned interpreter, the
six `packages/` editable, and the test dependencies from `uv.lock`:

```sh
uv sync
uv run python -m unittest discover -s isaaclab/tests
```

With a bare `python3` and no dependencies installed, the torch- and
numpy-dependent modules surface as import errors and only the dependency-free
tests run. The suite total grows with every added test, so read the count off
the run rather than from this page. uv manages the local and CI environment
only; training runs in the Spark container as before.

Run the interactive viewer:

```sh
cd viewer
npm install
npm run dev   # http://localhost:5173
```

Regenerate asset v1 from source:

```sh
# URDF package from the onshape-to-robot export (macOS/Linux, numpy + scipy)
python3 robot/tools/import_onshape_hexapod.py --source "<export dir>" --yaw-zero radial
# USD for Isaac Sim (Isaac Sim python, headless), then contact reports:
python tools/import_urdf_to_usd.py \
  robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf \
  robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda
python tools/enable_nested_contact_reports.py \
  robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda
# Standing validation gate (Isaac Lab), asset v1 by default:
python isaaclab/validate.py --asset mkii_v1 --num_envs 32 --steps 1000
```

The Phase-0 mock is regenerated with `python3 tools/generate_robstride_urdf.py`
followed by the same USD import and contact-report steps against
`robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda`.

Training and evaluation run on the DGX Spark host, not locally. See
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## Robot models

Every robot model is described once, in
`packages/hexapod_env/hexapod_env/assets/spec.py`: USD path, root link, joint
and link names, joint limits, reset stance, foot-pad geometry, and the expected
runtime joint order. Environment configs read the spec; adding a robot means
adding a spec and one env config under a **new** gym task ID. Existing task
IDs keep loading the model they were trained on.

### Asset v1 — CAD assembly (`robot/hexapod_mkii_assy/`)

Built from the Onshape export by `robot/tools/import_onshape_hexapod.py`;
`robot/hexapod_mkii_assy/assembly_report.md` audits the import. Task ID
`Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0`.

- `urdf/hexapod_mkii_serial.urdf`: 19 links, 18 actuated revolute joints
  (`<leg>_coxa_yaw`, `<leg>_femur_pitch`, `<leg>_tibia_pitch` for legs
  `lf lm lr rf rm rr`), root link `body`
- Body frame: z up, forward = -y, left = +x; coxa_yaw zero points the leg
  straight out, femur/tibia zero is the CAD pose (see
  `robot/hexapod_mkii_assy/joint_limits.json` for the sign conventions)
- Joint limits: coxa ±0.87 rad, femur -1.745 to +0.55 rad, tibia -0.95 to
  +1.75 rad (femur/tibia derived from the CAD kinematics; replace with
  measured hardware stops)
- Mass 8.261 kg from Onshape's per-part properties with every RS05 hard-set
  to its published 191 g; per-part collision primitives; two sphere pads per
  foot; meshes: binary STL, meters, z up. No sensor payload yet.
- Reset/validation stance (`robot/hexapod_mkii_assy/stance.json`): femur
  -0.25 rad, tibia -0.55 rad, bottom plate 0.124 m above the ground, static
  hip and knee torque 0.9 N·m each
- Runtime joint order: `packages/hexapod_core/hexapod_core/joints_v2.py`,
  provisional until read back from the imported articulation; the environment
  refuses to run if the articulation disagrees
- RS05 policy limit: 1.6 N·m continuous, 5.5 N·m published short-duration peak,
  50.27 rad/s nominal no-load speed, 0.0007 kg·m² output armature

### Phase-0 mock (`robot/hexapod_mkii_mock_assy/`)

The model every checkpoint under `artifacts/` was trained on. Its task IDs,
joint names (`revolute_*`), stance values, and the v1 runtime joint order in
`packages/hexapod_core/hexapod_core/joints.py` are frozen with those
checkpoints.

- 26 links, 25 joints (18 actuated revolute: coxa / femur / tibia × 6 legs)
- Joint limits: coxa ±0.87 rad, femur 0 – 1.75 rad, tibia 0 – 2.53 rad
- Physics mass target: 6.3 kg total — a 1.5 kg body plus six 0.8 kg complete
  legs (user-specified targets, not measurements); estimated inertias, no foot
  pads, collision geometry from visual meshes

RS05 housing inertia, joint friction, latency, and thermal parameters are not
published for either model. The remaining hardware prerequisites are listed
under "Sim-to-real gates still required" in
[`docs/TRAINING.md`](docs/TRAINING.md#sim-to-real-gates-still-required);
no checkpoint here is hardware-ready.

Varied-terrain sensor-fusion training is a separate approval-gated phase. Do
not start a long terrain or sensor-fusion run without explicit user approval.
