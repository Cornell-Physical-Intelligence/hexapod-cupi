# Agent and contributor guide

This repository trains walking policies for the Hexapod MKII, a 6.3 kg
six-legged robot with 18 RobStride RS05 joints, using Isaac Sim 6.0.1 with
Isaac Lab `DirectRLEnv` and RSL-RL PPO. Training runs on a shared DGX Spark
host through hardened launchers; the repository holds the task source, the
deployment scripts, the tests, and a curated evidence trail of every experiment
that has been run. The work is organized as a staged curriculum, and the
current research target is Stage2C: stable anatomical-forward walking with a
low, steady deck that stays inside the RS05 torque envelope. Progress is gated
by explicit numeric acceptance criteria, not by how a gait looks.

## Repository map

```text
STATUS.md                       Current state; rewritten in place, never appended
docs/ROADMAP.md                 Program direction: workstreams, milestones, gates, Spark sharing
docs/ARCHITECTURE.md            Autonomy layers and the contracts between them (C1-C4, O1, A1)
docs/decisions/                 Architecture decision records; numbered, never edited once accepted
docs/ONBOARDING.md              Read order, local setup, Spark access, working rules
docs/TRAINING.md                Task/training design, curriculum, acceptance gates
docs/OPERATIONS.md              Spark runbook, GPU protocol, launchers, screens
docs/incidents/                 Frozen forensic records; append-only
docs/archive/                   Superseded documents, kept verbatim
packages/hexapod_core/          Frozen contracts: observation, action, command, joints, actuator
packages/hexapod_env/           Env, rewards/, per-stage configs, gym registration
packages/hexapod_train/         Run composition, run contract, startup supervisor
packages/hexapod_eval/          gates.py acceptance thresholds; evaluation entry point
packages/hexapod_runtime/       Deployable runtime: observation builder, action pipeline
packages/hexapod_nav/           Command-producer protocol and an example waypoint follower
configs/                        Named experiment baselines and intervention deltas
isaaclab/hexapod_rl/            Compatibility shims re-exporting hexapod_env
isaaclab/deploy/                Launchers, systemd units, stage2_pipeline.sha256
ops/                            hexctl operator CLI; attic/ holds retired deploy scripts
isaaclab/tests/                 Unit and launcher/analyzer contract tests
robot/                          URDF package, meshes, generated USD
tools/                          URDF/USD generation and physics preparation
viewer/                         Vite + React viewer
artifacts/                      Checkpoints, evaluations, videos, probe ledger
```

Where things live:

- **Current state** — `STATUS.md`. One page, rewritten in place. If a fact is
  about right now, it goes there and nowhere else.
- **Direction** — `docs/ROADMAP.md` and `docs/decisions/`. Where the program
  is going and why; a plan change is a roadmap rewrite plus an ADR.
- **Design** — `docs/`. Durable contracts and procedure that outlive any run.
  `docs/ARCHITECTURE.md` defines the locomotion / perception / navigation
  layers and the contracts between them.
- **Evidence** — `artifacts/`. Checkpoints, evaluation JSON, resolved configs,
  hashes, videos, and the probe ledger.

## Invariants

These hold regardless of what a task prompt asks for. If an instruction
conflicts with one of them, stop and raise the conflict.

- **Gym task IDs are frozen.** The IDs in
  `packages/hexapod_env/hexapod_env/register.py` (re-exported by the
  `isaaclab/hexapod_rl` shim) are referenced by string from checkpoints,
  launchers, evaluation payloads, and manifests. Add new IDs; never rename or repurpose an existing one.
- **Checkpoint files and SHA-256 manifests are immutable.** A file under
  `artifacts/.../checkpoints/` and its recorded hash never change. A new result
  is a new file with a new hash and a new manifest entry.
- **Failed-attempt evidence and the probe ledger are append-only.** Never
  delete, rewrite, or reuse the label of a failed attempt. The ledger at
  `artifacts/phase2_recovery_stage2c_stable_forward/probes/README.md` and the
  records under `docs/incidents/` grow by appending. Infrastructure failures are
  preserved as carefully as results, because a lost failure looks like a
  negative result later.
- **Never alter acceptance gates or tests to make a screen pass.** The Stage2C
  gates in `docs/TRAINING.md` §6 and the contract tests exist to reject
  candidates. Changing a threshold to admit a checkpoint destroys the value of
  every earlier comparison. If a gate is genuinely wrong, say so and leave it
  in place pending an explicit decision. `packages/hexapod_eval/hexapod_eval/gates.py`
  is the machine-readable copy of those thresholds, bound to the enforcing
  grader by `isaaclab/tests/test_eval_gates_contract.py`.
- **Comparisons require a common playback limiter.** Every formal measurement
  on record uses `0.040 rad / 20 ms`. Numbers taken at another limiter are not
  comparable and must not be placed in the same table.
- **The Spark mirror is not a git repository.** `/home/orionh/HEXAPOD` is a
  bind-mounted directory. Path expectations matter: source is synced there and
  verified with `sha256sum -c isaaclab/deploy/stage2_pipeline.sha256`, and git
  commands run against it will not do what they do here.
- **Do not compete with unrelated workloads on the shared GPU.** Hold
  `/tmp/hexapod-isaac-gpu.lock`, gate on producer scripts and their
  descendants, and never signal a process this project did not create. See
  `docs/OPERATIONS.md` §3. `ops/hexctl` is the operational entry point; it
  composes and supervises the hardened launchers rather than replacing them.
- **Claims need artifacts.** A result is real when it has a checkpoint, a
  SHA-256, an evaluation payload, and a recorded screen configuration. A commit
  message alone is not evidence.
- **No secrets in files, logs, or chat.** That includes the contents of
  `/home/orionh/IsaacLab/docker/.env.base`.

## Tests

Canonical command:

```sh
python3 -m unittest discover -s isaaclab/tests
```

The full suite requires `torch`, `numpy`, `pyyaml`, and
`opencv-python-headless`. Without them, the ten torch-dependent modules surface
as 10 import errors and the rest of the suite still runs; that is the expected
result on a machine with no simulator dependencies, and documentation changes
must not alter it. Do not hardcode a suite total here — it moves with every
added test; read it off the run instead. Isaac Sim itself is not required for
the test suite.

Run the focused tests for whatever you touched. Reward math, reset batching,
launcher contracts, analyzer contracts, and shard merging all have dedicated
test modules under `isaaclab/tests/`.
