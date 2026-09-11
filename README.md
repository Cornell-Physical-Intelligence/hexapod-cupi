# Hexapod MKII: autonomy and RL walking

**Research is paused for handoff to James (11 September 2026 UTC). Start with [James's handoff guide](docs/JAMES_HANDOFF.md).** Current code, results and unpublished preparation are preserved; no research/training continuation is queued. External Spark jobs remain blocked.

Software for the Hexapod MKII, an 18-joint RobStride RS05 hexapod. Its mission
is to survey a bounded area an operator draws on a map and to hold a steady
deck for data collection while it does so. `dar.md` states the mission.
`docs/PLAN.md` holds the architecture, milestones, and decisions. `STATUS.md`
records the current state.

For the canonical detailed-model preparation, historical study results, verified job state
and next actions, start with [STATUS.md](STATUS.md). The living [plan](docs/PLAN.md)
and dated [training history](HANDOFF.md) provide context;
[PROJECT_STATUS.md](PROJECT_STATUS.md) redirects to the canonical status. Dated
reports are evidence, not live process status.

**Stage 2 remains incomplete.** Success requires smooth translation through all
bearings, both yaw signs, combined translation/turning and paths, reversals and
stops, and quiet zero-command standing without persistent stepping or body
oscillation. Tracking, contact, stability and motor-load gates must pass together;
a training budget finishing or an appealing video does not close the stage.

## Read order

1. `dar.md`: the requirements and the demo ladder.
2. `CLAUDE.md`: the invariants. They bind people and agents alike.
3. `docs/PLAN.md`: layers, contracts, workstreams, gates, decisions.
4. [STATUS.md](STATUS.md): current experiment coordination, production-model state,
   target and open contradictions.
5. `docs/TRAINING.md` and `docs/OPERATIONS.md` before you touch the task or
   the Spark.
6. The `CLAUDE.md` inside each `packages/hexapod_*/` you touch.

## Repository map

```text
dar.md / DAR.png                Mission: requirements slide, numeric blanks, steps, demo ladder
STATUS.md                       Current state; rewritten in place
PROJECT_STATUS.md               Compatibility pointer to canonical STATUS.md
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
robot/hexapod_mkii_updated_v1/ Canonical detailed direct-drive training model, motor weights corrected
robot/active_model.json        Canonical URDF/model/USD identities
robot/hexapod_mkii_assy/        Historical CAD and four-bar asset lineages
robot/hexapod_mkii_mock_assy/   Archived Phase-0 mock; new geometry studies are explicit
robot/hexapod_leg_v3/           Single-leg reference the assembly import registers onto
robot/sensors/                  Mid-360 datasheet values and mount placement study
robot/tools/                    onshape-to-robot importers, viewer packer
tools/                          URDF-to-USD import and physics preparation
viewer/                         Default entry point to the detailed CAD inspector
artifacts/                      Evidence: checkpoints, evaluations, videos, probe ledger
```

## Canonical training robot

The user-approved [detailed direct-drive URDF](robot/hexapod_mkii_updated_v1/urdf/hexapod_updated_rs05_mass_corrected.urdf) is the ground-truth model for **all future training**, with the motor weight overrides: **7.466088235 kg**, 19 bodies, 18 joints, and all 1,753 original CAD parts. [robot/active_model.json](robot/active_model.json) pins the selected URDF/model/USD hashes. The repository viewer now opens its detailed part and full-range joint inspector.

See [joint limits and import checks](docs/UPDATED_CAD_IMPORT.md), [all link weights/COMs/inertias](robot/hexapod_mkii_updated_v1/MASS_INERTIA.md), and [Isaac preparation](artifacts/mkii_updated_2026-09-10/usd_002/README.md). The original CAD mass is preserved separately; the added motor inertia distribution is a labeled estimate. Native SDF cooking and a new asset-bound runtime/admission are required before training. Do not start new simplified-model or four-bar training. Historical task IDs/checkpoints keep their original assets and remain available for reproduction.

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

## Historical robot models

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
- Physical four-bar candidate: the newer [assembly documentation](robot/hexapod_mkii_assy/README.md)
  and [physical training contract](docs/MKII_FOURBAR_TRAINING.md) describe 31
  bodies, 30 articulation coordinates and 18 motor actions. The 19-link serial
  reduction is historical and cannot establish physical linkage qualification.
- Phase-0 mock, `robot/hexapod_mkii_mock_assy/`: the historical mock checkpoint
  fleet trained on this model. Its task IDs, `revolute_*` joint names,
  stance values, and `hexapod_core/joints.py` order are frozen with those
  checkpoints.
- User-selected [C geometry study](artifacts/length_study_2026-09-09/README.md):
  72.5 mm femur, 126 mm tibia, unchanged coxa, with current CAD mass/inertia and
  RS05 constraints transferred to scaled mock geometry (approximately 8.2608 kg).
  This isolated study predates the now-canonical updated direct-drive model. Detailed C
  motor/linkage fit, payload and hardware geometry remain unverified.

The exact [C forward Benchmark 1](artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300/README.md)
is an immutable user-selected PPO demonstration. Preserve its checkpoint, video,
configuration and checksums; new evaluations belong elsewhere. It has unresolved
torque-demand limits and does not establish all-direction or hardware readiness.

No checkpoint in this repository is hardware-ready (`docs/TRAINING.md` §7).

## Parallel terrain and perception preparation

The user has requested Stage 2 completion with terrain/perception work proceeding
in parallel. PPO controls the 18 joint targets; a separate motion producer requests
forward/left velocity and yaw rate, allowing arcs and independent path/body heading.
See the [omnidirectional study plan](artifacts/omni_flat_2026-09-09/RESEARCH_AND_PLAN.md)
and [terrain/sensing plan](artifacts/project_review_2026-09-04/TERRAIN_AND_SENSING_PLAN_2026-09-09.md).
Consult STATUS for the current checkpoint and scheduler state.

- [CPU terrain readiness](artifacts/terrain_readiness_2026-09-09/readiness.json)
  and [mild curriculum](artifacts/terrain_readiness_2026-09-09/mild_curriculum_001/curriculum.json)
  prepare procedural assets and evaluation inputs; runtime admission is separate.
- [Camera-mount study](artifacts/sensor_mount_study_2026-09-09/README.md) records
  visibility on its hashed serial-CAD snapshot. It does not qualify sensor placement on the canonical updated CAD.
- [CPU perception replay](artifacts/perception_readiness_2026-09-09/README.md)
  exercises timestamped depth/point-cloud transforms and an age/uncertainty-aware
  map with synthetic data. It claims no real-sensor calibration, ROS integration,
  trained terrain policy or hardware qualification.

The user has a Mid-360 and D455; additional sensors may be evaluated. Preserve
the project's procurement and physical-test decisions. Unknown or stale support
stays unknown, and adding sensors does not bypass actuator or model gates.

## Working rules

- One branch per task, a PR to `main`, and green CI. A change to a contract
  needs one reviewer from another workstream.
- A new number comes with an artifact. A result without a checkpoint hash, an
  evaluation payload, and a screen config counts as a note.
- A decision that changes a contract or the plan gets a dated entry in
  `docs/PLAN.md` §7.
- You launch an attempt on the Spark only from the shared queue
  (`docs/OPERATIONS.md` §3).
