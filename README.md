# Hexapod MKII — autonomy and RL walking

Software for the Hexapod MKII, an 18-joint RobStride RS05 hexapod whose mission
is to survey a bounded area an operator draws on a map, steadily, as a stable
platform for data collection. `dar.md` is the mission; `docs/PLAN.md` is the
architecture, milestones, and decisions; `STATUS.md` is what is true right now.

## Read order

1. `dar.md`: the requirements and the demo ladder.
2. `CLAUDE.md`: the invariants. They bind people as much as agents.
3. `docs/PLAN.md`: layers, contracts, workstreams, gates, decisions.
4. `STATUS.md`: current best checkpoint, target, open contradictions.
5. `docs/TRAINING.md` and `docs/OPERATIONS.md` when you touch the task or the Spark.
6. The `CLAUDE.md` inside each `packages/hexapod_*/` you touch.

## Repository map

```text
dar.md / DAR.png                Mission: requirements slide, numeric blanks, steps, demo ladder
STATUS.md                       Current state; rewritten in place, never appended
CLAUDE.md (= AGENTS.md)         Invariants, repository map, test command
docs/PLAN.md                    Architecture and contracts, workstreams, milestones and gates, decisions
docs/TRAINING.md                Task and training design, curriculum, acceptance gates
docs/OPERATIONS.md              Spark runbook: environment, GPU protocol, launchers, screens, asset import
docs/incidents/                 Frozen forensic records; append-only
docs/archive/                   Superseded documents, kept verbatim
packages/hexapod_core/          Frozen contracts: observation, action, command, joints, actuator
packages/hexapod_env/           Env, rewards/, assets/ (one spec per robot model), stage configs, registration
packages/hexapod_train/         Run composition, run contract, startup supervisor
packages/hexapod_eval/          gates.py acceptance thresholds; evaluation entry point
packages/hexapod_runtime/       Deployable runtime: observation builder, action pipeline
packages/hexapod_nav/           Command-producer protocol and an example waypoint follower
configs/                        Named experiment baselines and intervention deltas
isaaclab/hexapod_rl/            Compatibility shims re-exporting hexapod_env
isaaclab/deploy/                Launchers, systemd units, stage2_pipeline.sha256
isaaclab/tests/                 Unit and contract tests
ops/                            hexctl operator CLI; attic/ holds retired deploy scripts
robot/hexapod_mkii_assy/        Asset v1: CAD assembly URDF package and its README (conventions, regeneration)
robot/hexapod_mkii_mock_assy/   Phase-0 mock every existing checkpoint was trained on
robot/hexapod_leg_v3/           Single-leg reference the assembly import registers onto
robot/sensors/                  Mid-360 datasheet values and mount placement study
robot/tools/                    onshape-to-robot importers, viewer packer
tools/                          URDF-to-USD import and physics preparation
viewer/                         Vite + React + urdf-loader web viewer
artifacts/                      Evidence: checkpoints, evaluations, videos, probe ledger
```

## Quickstart

The repository is a uv workspace. `uv sync` installs the pinned interpreter,
the six `packages/` editable, and the test dependencies from `uv.lock`. No
simulator is needed.

```sh
git clone https://github.com/Cornell-Physical-Intelligence/hexapod-cupi.git
cd hexapod-cupi
uv sync
uv run python -m unittest discover -s isaaclab/tests
cd viewer && npm install && npm run dev   # http://localhost:5173
```

Training and evaluation run on the DGX Spark over Tailscale, never locally;
ask the project lead for an invite and an account, then follow
`docs/OPERATIONS.md` §2 before anything else. uv manages laptops and CI only.

## Robot models

Every robot model is described once, in
`packages/hexapod_env/hexapod_env/assets/spec.py`: USD path, root link, joint
and link names, limits, reset stance, foot-pad geometry, expected runtime
joint order. Env configs read the spec. Adding a robot is a new spec plus one
env config under a new gym task ID; existing IDs keep loading the model they
were trained on.

- Asset v1, `robot/hexapod_mkii_assy/`: the CAD assembly, 8.26 kg, task ID
  `Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0`. Conventions, limits,
  stance, and regeneration are in its README; the Spark import and validation
  procedure is `docs/OPERATIONS.md` §10. Its runtime joint order
  (`hexapod_core/joints_v2.py`) is provisional until confirmed there.
- Phase-0 mock, `robot/hexapod_mkii_mock_assy/`: the model every checkpoint
  under `artifacts/` was trained on. Its task IDs, `revolute_*` joint names,
  stance values, and `hexapod_core/joints.py` order are frozen with those
  checkpoints.

No checkpoint in this repository is hardware-ready (`docs/TRAINING.md` §7).

## Working rules

- Branch per task; PR to `main`; CI green; one reviewer from another
  workstream for anything that touches a contract.
- New numbers come with an artifact; a result without a checkpoint hash,
  evaluation payload, and screen config is a note.
- A decision that changes a contract or the plan gets a dated entry in
  `docs/PLAN.md` §7.
- Nobody launches an attempt on the Spark that is not in the shared queue
  (`docs/OPERATIONS.md` §3).
