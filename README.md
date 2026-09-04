# Hexapod MKII: autonomy and RL walking

Software for the Hexapod MKII, an 18-joint RobStride RS05 hexapod. Its mission
is to survey a bounded area an operator draws on a map and to hold a steady
deck for data collection while it does so. `dar.md` states the mission.
`docs/PLAN.md` holds the architecture, milestones, and decisions. `STATUS.md`
records the current state.

## Read order

1. `dar.md`: the requirements and the demo ladder.
2. `CLAUDE.md`: the invariants. They bind people and agents alike.
3. `docs/PLAN.md`: layers, contracts, workstreams, gates, decisions.
4. `STATUS.md`: current best checkpoint, target, open contradictions.
5. `docs/TRAINING.md` and `docs/OPERATIONS.md` before you touch the task or
   the Spark.
6. The `CLAUDE.md` inside each `packages/hexapod_*/` you touch.

## Repository map

```text
dar.md / DAR.png                Mission: requirements slide, numeric blanks, steps, demo ladder
STATUS.md                       Current state; rewritten in place
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
robot/hexapod_mkii_mock_assy/   Phase-0 mock; each existing checkpoint trained on it
robot/hexapod_leg_v3/           Single-leg reference the assembly import registers onto
robot/sensors/                  Mid-360 datasheet values and mount placement study
robot/tools/                    onshape-to-robot importers, viewer packer
tools/                          URDF-to-USD import and physics preparation
viewer/                         Vite + React + urdf-loader web viewer
artifacts/                      Evidence: checkpoints, evaluations, videos, probe ledger
```

## Quickstart

The repository is a uv workspace. `uv sync` installs the pinned interpreter,
the six `packages/` as editable installs, and the test dependencies from
`uv.lock`. You do not need a simulator.

```sh
git clone https://github.com/Cornell-Physical-Intelligence/hexapod-cupi.git
cd hexapod-cupi
uv sync
uv run python -m unittest discover -s isaaclab/tests
cd viewer && npm install && npm run dev   # http://localhost:5173
```

Training and evaluation run on the DGX Spark over Tailscale. Ask the project
lead for an invite and an account, then follow `docs/OPERATIONS.md` §2 before
you launch anything. uv manages laptops and CI only.

## Robot models

`packages/hexapod_env/hexapod_env/assets/spec.py` describes each robot model
once: USD path, root link, joint and link names, limits, reset stance,
foot-pad geometry, and expected runtime joint order. Env configs read the
spec. To add a robot you write a new spec and one env config under a new gym
task ID. Existing IDs keep loading the model they trained on.

- Asset v1, `robot/hexapod_mkii_assy/`: the CAD assembly, 8.26 kg, task ID
  `Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0`. Its README holds the
  conventions, limits, stance, and regeneration steps. `docs/OPERATIONS.md`
  §10 holds the Spark import and validation procedure. Its runtime joint order
  (`hexapod_core/joints_v2.py`) stays provisional until that procedure
  confirms it.
- Phase-0 mock, `robot/hexapod_mkii_mock_assy/`: each checkpoint under
  `artifacts/` trained on this model. Its task IDs, `revolute_*` joint names,
  stance values, and `hexapod_core/joints.py` order are frozen with those
  checkpoints.

No checkpoint in this repository is hardware-ready (`docs/TRAINING.md` §7).

## Working rules

- One branch per task, a PR to `main`, and green CI. A change to a contract
  needs one reviewer from another workstream.
- A new number comes with an artifact. A result without a checkpoint hash, an
  evaluation payload, and a screen config counts as a note.
- A decision that changes a contract or the plan gets a dated entry in
  `docs/PLAN.md` §7.
- You launch an attempt on the Spark only from the shared queue
  (`docs/OPERATIONS.md` §3).
