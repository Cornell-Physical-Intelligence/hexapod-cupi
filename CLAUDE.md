# Agent and contributor guide

Autonomy software for the Hexapod MKII, a six-legged robot with 18 RobStride
RS05 joints. Mission: survey a bounded area an operator draws on a map and
hold a steady deck for data collection while doing so (`dar.md`). The team
trains walking policies with Isaac Sim 6.0.1, Isaac Lab `DirectRLEnv`, and
RSL-RL PPO on a shared DGX Spark through hardened launchers. Explicit numeric
acceptance criteria gate progress. For the current C-study Stage 2, the user requires visual smoothness matching the accepted forward clip as well as quantitative gates. The
current program is no-RTK bounded-area coverage with a learned
omnidirectional gait on the real CAD assembly. The historical Stage2C is archived mock research. The user approved the detailed direct-drive CAD on 10 September and designated its motor-weight-corrected URDF as ground truth for all future training. The C-length study is now historical; native admission of the new model precedes further training. `STATUS.md` records execution
state; `docs/PLAN.md` holds the living plan and `docs/NEXT_RUNS.md` prepares
the next campaign without launching it.

## Current execution priority and publication

The user approved `robot/active_model.json` as the canonical model selection for **all future training**, using `robot/hexapod_mkii_updated_v1/urdf/hexapod_updated_rs05_mass_corrected.urdf` with the motor weight overrides (7.466088235 kg). Do not start new training on the simplified C-study/mock or historical four-bar models. Prepare a new model-bound runtime and native Isaac admission for the detailed 19-body/18-joint direct-drive robot first. Preserve old task IDs, checkpoints and evidence unchanged for historical reproduction; do not silently make an old task load this new asset. Stage 2 still requires all-direction/path smoothness and quiet standing, with terrain/perception prepared in parallel. Read `STATUS.md` and `docs/UPDATED_CAD_IMPORT.md`. Every meaningful verified step must be committed and pushed with the relevant Markdown context; this is explicit user authorization. Preserve teammates' commits, use a current main integration, run relevant checks, and verify the remote SHA. Historical frozen evidence is not rewritten to update status.

**Weather compute priority, 10 September 2026:** the user explicitly authorizes overriding all weather activity to prioritize HEXAPOD PPO. Identified weather services and timers may be deferred when they block training; preserve their outputs and record exact ownership and restoration/defer decisions. This authorization does not cover unrelated non-weather workloads. Keep the hardened launchers, shared GPU locks, exact container cleanup and bounded recovery controls. Do not delay a ready HEXAPOD allocation merely to favor a weather job.

## Mandatory live research poster framework

**Hard user rule: every agent and human contributor changing this repository must follow [docs/PROJECT_SITE.md](docs/PROJECT_SITE.md).** Read it before edits and include this rule in every delegated task. The public research poster is driven by `site/project.json`, verified evidence references and `site/updates/` records. Every change must include a new update record covering its changed paths, with a plain-English explanation, affected workstreams, evidence and next action. Update the central presentation registry when implementation, goals, gates, geometry, results or media change. For changes with no project-facing impact, record the specific reason in a checked no-impact entry; silence is not an exemption. Never rewrite frozen research evidence to satisfy this rule.

Run `python3 tools/project_site.py check --base <base-commit>` and `python3 tools/project_site.py build` before publication. CI rejects uncovered repository changes and broken evidence/media references; each push to main rebuilds and deploys GitHub Pages. STATUS remains the authoritative execution snapshot; the site reads it and links immutable results. Latest checkpoint, latest video and qualified benchmark are separate records. Do not relabel old media or mark a stage passed without its unchanged numeric and visual evidence. Root retains responsibility for integration and verifying deployment.

## Architecture partnership and parallel work

The user requests Claude Code with Fable 5.1 (`claude-fable-5-1`) with maximum reasoning (`--effort max`) as a partner for major architecture decisions and substantial parallel implementation/review work. Consult it on controller/action architecture, learning objectives, terrain/perception interfaces and consequential experiment changes. Give workers bounded work areas and concrete deliverables; keep one owner for Spark execution and integration. Preserve the actual model/session, inputs, findings, disagreements and the resulting engineering decision. Independently review proposed code and run relevant checks before adoption. Report unavailable authentication/model access honestly; do not silently substitute a model or claim a consultation that failed. Routine verified repairs and ongoing authorized jobs can continue while reviews run.

## Updated physical design and historical four-bar campaign

The user confirmed on 10 September 2026 that the new physical design removes the
external four-bar and has 18 direct-drive joints/19 main rigid bodies. New CAD
intake and offline preparation use `robot/hexapod_mkii_updated_v1/` and
`docs/UPDATED_CAD_IMPORT.md`. The user visually approved grouping and full joint travel, then selected the
motor-weight-corrected URDF as ground truth for training. The CAD mate graph was
absent: inferred attachment ownership, source screw overlaps, measured hardware
stops and motor inertia distribution remain explicit uncertainties. Native Isaac
physics remains an admission task. It is not admitted by either existing runtime.
Do not apply the four-bar motor adapter or its runner to this updated asset.

The following contract is retained **only for the historical four-bar lineage**:

The versioned physical task and guarded runner are documented in
`docs/MKII_FOURBAR_TRAINING.md`. Use its 31-body/30-coordinate asset and explicit
18-motor adapter for work on that exact four-bar model; keep serial-v2 characterization and
archived mock task contracts intact. The third active coordinate is the pushlever
motor, with per-leg CAD offsets. Never substitute serial knee defaults.

Use `isaaclab/deploy/run-mkii-fourbar` for bounded physical validation/training.
Only a matching physical admission report can admit scratch PPO; resume additionally
requires checkpoint bytes and source/asset/motor lineage to match. The shared
coordination file controls yielding compute. Hardware calibration and terrain
qualification remain explicit follow-on gates.

## Repository map

```text
dar.md / DAR.png                Mission: requirements slide, numeric blanks, steps, demo ladder
STATUS.md                       Current state; rewritten in place
CLAUDE.md (= AGENTS.md)         This file; AGENTS.md is a symlink to it for other agents
docs/PLAN.md                    Architecture and contracts, workstreams, milestones and gates, decisions and evolving context
docs/TRAINING.md                Task/training design, curriculum, acceptance gates
docs/OPERATIONS.md              Spark runbook: environment, GPU protocol, launchers, screens, asset import (§10)
docs/incidents/                 Frozen forensic records; append-only
docs/archive/                   Superseded documents, kept verbatim
packages/hexapod_core/          Frozen contracts: observation, action, command, joints, actuator
packages/hexapod_env/           Env, rewards/, assets/ (one spec per robot model), per-stage configs, gym registration
packages/hexapod_train/         Run composition, run contract, startup supervisor
packages/hexapod_eval/          gates.py acceptance thresholds; evaluation entry point
packages/hexapod_runtime/       Deployable runtime: observation builder, action pipeline
packages/hexapod_nav/           Command-producer protocol and an example waypoint follower
configs/                        Named experiment baselines and intervention deltas
isaaclab/hexapod_rl/            Compatibility shims re-exporting hexapod_env
isaaclab/deploy/                Launchers, systemd units, stage2_pipeline.sha256
isaaclab/tests/                 Unit and launcher/analyzer contract tests
ops/                            hexctl operator CLI; attic/ holds retired deploy scripts
robot/                          hexapod_mkii_assy (asset v1, README = conventions), hexapod_mkii_mock_assy (Phase-0), leg v3, sensors, importers
tools/                          URDF/USD generation and physics preparation
viewer/                         Vite + React viewer
artifacts/                      Checkpoints, evaluations, videos, probe ledger
```

Locations by kind of content:

- **Current state**: `STATUS.md`. A fact about right now goes there and
  nowhere else.
- **Direction and design**: `docs/PLAN.md` (layers, contracts, milestones,
  decisions), `docs/TRAINING.md` (the task), `docs/OPERATIONS.md` (the Spark).
  A plan change edits `PLAN.md` and updates the dated decision/context record in its §9.
- **Evidence**: `artifacts/`. Checkpoints, evaluation JSON, resolved configs,
  hashes, videos, and the probe ledger.
- **Package rules**: each `packages/hexapod_*/CLAUDE.md` holds that package's
  ownership and prohibitions. The robot READMEs under `robot/` hold each
  model's conventions.

## Invariants

These protect reproducibility and the meaning of historical evidence. Explicit
user instructions take precedence over this guide. Apply these conventions to
their documented asset/experiment lineage; do not treat mock-specific targets
as requirements for a new CAD policy.

- **Gym task IDs are frozen.** Checkpoints, launchers, evaluation payloads,
  and manifests reference the IDs in
  `packages/hexapod_env/hexapod_env/register.py` (re-exported by the
  `isaaclab/hexapod_rl` shim) by string. Add new IDs. Do not rename or
  repurpose an existing one.
- **Checkpoint files and SHA-256 manifests are immutable.** A file under
  `artifacts/.../checkpoints/` and its recorded hash do not change. A new
  result is a new file with a new hash and a new manifest entry.
- **Failed-attempt evidence and the probe ledger are append-only.** Do not
  delete, rewrite, or reuse the label of a failed attempt. The ledger at
  `artifacts/phase2_recovery_stage2c_stable_forward/probes/README.md` and the
  records under `docs/incidents/` grow by appending. Preserve infrastructure
  failures with the same care as results, because a lost failure looks like a
  negative result later.
- **Do not alter acceptance gates or tests to make a screen pass.** The
  Stage2C gates in `docs/TRAINING.md` §6 and the contract tests exist to
  reject candidates. A threshold changed to admit a checkpoint destroys the
  value of each earlier comparison. If you believe a gate is wrong, say so and
  leave it in place until the team makes an explicit decision.
  `packages/hexapod_eval/hexapod_eval/gates.py` is the machine-readable copy
  of those thresholds, and `isaaclab/tests/test_eval_gates_contract.py` binds
  it to the enforcing grader.
- **Comparisons require a common playback limiter.** Each formal measurement
  on record uses `0.040 rad / 20 ms`. Numbers taken at another limiter are not
  comparable. Do not place them in the same table.
- **The Spark mirror is not a git repository.** `/home/orionh/HEXAPOD` is a
  bind-mounted directory. You sync source there and verify it with
  `sha256sum -c isaaclab/deploy/stage2_pipeline.sha256`. Git commands run
  against it do not behave as they do here.
- **Do not compete with unrelated workloads on the shared GPU.** Hold
  `/tmp/hexapod-isaac-gpu.lock`, gate on producer scripts and their
  descendants, and do not signal a process this project did not create. See
  `docs/OPERATIONS.md` §3. `ops/hexctl` is the operational entry point. It
  composes and supervises the hardened launchers and does not replace them.
- **Claims need artifacts.** A result is real when it has a checkpoint, a
  SHA-256, an evaluation payload, and a recorded screen configuration. A
  commit message alone is no evidence.
- **No secrets in files, logs, or chat.** That includes the contents of
  `/home/orionh/IsaacLab/docker/.env.base`.

## Tests

Canonical command (the repository is a uv workspace; run `uv sync` once,
then):

```sh
uv run python -m unittest discover -s isaaclab/tests
```

`uv.lock` pins the test dependencies (`torch`, `numpy`, `pyyaml`,
`opencv-python-headless`). `uv sync` installs them with the six `packages/`
as editable installs. A bare `python3 -m unittest discover -s isaaclab/tests`
still works. Without those dependencies the torch- and numpy-dependent modules
surface as import errors and the rest of the suite still runs. That is the
expected result on a machine with no dependencies, and documentation changes
must not alter it. Do not hardcode a suite total here, because it moves with
each added test. Read it off the run instead. The test suite does not need
Isaac Sim, and uv does not touch the Spark container. Dependency changes go
through `uv add` and you commit them with the updated lockfile.

Run the focused tests for whatever you touched. Reward math, reset batching,
launcher contracts, analyzer contracts, and shard merging have dedicated test
modules under `isaaclab/tests/`.
