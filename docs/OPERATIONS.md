# Spark operations runbook

> **9 September 2026 update:** the user prioritizes the selected C-study Stage 2 controller, requiring smooth walking/pathing in every direction and quiet standing, with terrain/perception implementation in parallel. [STATUS](../STATUS.md) is current; [the C-study contract](../experiments/c_length_study/README.md) separates its pinned runtime and 1.6 N·m study cap from the physical four-bar program. Earlier pause, plan and actuator statements below apply to their dated physical lineage. Commit/push each verified step with relevant Markdown and preserve other work on main.


> Current procedure: the project lead has authorized short live Spark validation of the corrected serial CAD-v2 model. Use the isolated, bounded launcher in [§11](#11-corrected-serial-cad-v2-live-validation), which preserves the original mirror and still requires exclusive GPU admission. This is standing characterization, not full G0 or permission to adopt an archived policy. [STATUS.md](../STATUS.md) records current execution; [live evidence](../artifacts/mkii_step2_2026-09-04/README.md) records the results. The earlier [asset audit](../artifacts/project_review_2026-09-04/URDF_VALIDATION.md) remains the record of the original import defect.

Durable operating procedure for hexapod training and evaluation on the DGX
Spark host. It covers the environment, the audit that precedes each launch,
the GPU lock protocol, launcher and formal-screen usage, the specification and
status of the attempt-aware startup supervisor, and the asset import
procedure.

No password, token, key, or secret-file content belongs in this file or in any
log or chat message.

## 1. Environment

```text
host: spark-e26c / 100.82.166.9 over Tailscale SSH
user: orionh
project mirror: /home/orionh/HEXAPOD (not a Git repository)
Isaac Lab source: /home/orionh/IsaacLab
OS: Ubuntu 24.04.3 LTS, aarch64, Linux 6.17.0-1031-nvidia
GPU: NVIDIA GB10; driver 580.173.02; driver CUDA 13.0
Isaac Sim: 6.0.1-rc.7+release.42383.32955d8d.gl3.0.0
Isaac Sim commit: 045ca8b59622b99a408092124377c66346e8d9c2
Isaac Lab: repository VERSION 3.0.0, commit ffff603eafc6b74264a5261cc0183d6a65390d78
Python: 3.12.13 inside the Isaac container
PyTorch: 2.10.0+cu130; CUDA available; device NVIDIA GB10
RSL-RL: rsl-rl-lib 5.0.1
Gymnasium: 1.2.1
```

`/home/orionh/HEXAPOD` is a bind-mounted mirror. It is not a Git repository.
Git does not version changes made there, and a `git` command run against it
does not behave as it does in this checkout. Sync source to it on purpose and
verify with the applicable manifest. Current CAD-v2 validation uses a separate
source snapshot under `/home/orionh/HEXAPOD_runs/mkii_v2_step2_a0f0b39/source`,
mounted read-only at the same container path. Its output goes to the sibling
`runs/` directory; see §11.

No Conda, venv, or uv is used for training. The Docker Compose image
`isaac-lab-base` bind-mounts `/home/orionh/HEXAPOD` at `/workspace/hexapod` and
launches through `/workspace/isaaclab/_isaac_sim/python.sh`. The repository's
uv workspace (`pyproject.toml`, `uv.lock`) manages laptop and CI environments
only; do not run `uv sync` inside the mirror, and do not sync a `.venv/` to it.

Required non-secret environment inside the container:

```text
PYTHONPATH=/workspace/hexapod/isaaclab
PYTHONDONTWRITEBYTECODE=1
ISAACSIM_VERSION=6.0.1
ISAACSIM_ROOT_PATH=/isaac-sim
ISAACLAB_PATH=/workspace/isaaclab
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all
```

`/home/orionh/IsaacLab/docker/.env.base` is required for the compose stack. Do
not copy its contents into logs, chat, or documentation.

A PyTorch warning reports that GB10 compute capability 12.1 is newer than the
build's advertised 12.0 ceiling. The same image has trained many successful
jobs, so this warning is not a confirmed failure cause.

## 2. Audit before touching anything

Inspect host processes, GPU use, containers, services and the shared lock
before a GPU launch. The current CAD-v2 launcher performs and records those
checks itself, including checks after container creation and during the run.
Read `/home/orionh/SPARK_COMPUTE_COORDINATION.md` before every new launch,
and at checkpoint boundaries during future long runs, for sharing requests
from another agent. The [repository copy](SPARK_COMPUTE_COORDINATION.md)
records the coordination procedure.

The following sequence additionally verifies the **archived Stage2 release and
its checkpoint**, for reproducing that lineage only. Its manifest and parent
checkpoint hash are not prerequisites for new CAD-v2 standing validation:

```bash
ssh orionh@100.82.166.9
cd /home/orionh/HEXAPOD
date -u
ps -eo pid,ppid,lstart,etime,%cpu,%mem,args --sort=-%cpu | head -80
pgrep -af 'nsva_dl|validate_nsva|score_clip|validate_b51|nowcast_run|torch.*compile|train_model_only_resume|evaluate_checkpoint' || true
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
systemctl is-active hexapod-rl-training.service || true
nvidia-smi
flock -n /tmp/hexapod-isaac-gpu.lock true
sha256sum -c isaaclab/deploy/stage2_pipeline.sha256
sha256sum isaaclab/logs/rsl_rl/hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/2026-08-25_23-44-14_accel_scale4_yaw160_trackguard_20260825T234500Z_seed86/model_2.pt
```

For an archived Stage2 reproduction,
`sha256sum -c isaaclab/deploy/stage2_pipeline.sha256` verifies every frozen entry
and must pass in the matching local and Spark source snapshots. That lineage's
parent checkpoint hash must equal
`a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a`.

Also check whether an automated watcher has already acted before launching
anything by hand.

## 3. GPU lock and shared-workload protocol

The current short validation launcher remains **exclusive**. The Spark is
shared with unrelated workloads, so the following admission and cleanup rules
still apply to these runs:

- Hold `/tmp/hexapod-isaac-gpu.lock` for the entire logical run, from before
  container creation to cleanup.
- Never signal, stop, modify, or compete with an unrelated workload. Observed
  producers include `/root/nsva_dl.sh` and its GPU children such as
  `validate_nsva.py`, plus `score_clip.py`, `validate_b51.py`, and
  torch-compile / clip-scoring workers. `/root/cvnba` is not ours; do not touch
  it.
- A free GPU is not sufficient evidence that a producer is finished. Wait until
  the producer script and all of its descendants are gone, then recheck the
  GPU, active Docker containers, the systemd service, and the shared lock.
- Producers spawn GPU children after their own start, so a clean prelaunch
  check can turn stale seconds later. Recheck contention as soon as the
  container exists and again before PPO starts. The startup supervisor in §6
  exists to close this post-launch race.
- Cleanup acts only on the exact immutable container ID owned by this run. It
  must never use `docker rm -f` on a broad match and must never touch a process
  it did not create.

The latest instruction is to **use full available Spark compute now**, until
another agent requests sharing through `/home/orionh/SPARK_COMPUTE_COORDINATION.md`.
Current exclusive validation already permits full available compute; no quota
is enabled. Read that shared file before each new launch and at checkpoints
during future long runs. Its [repository copy](SPARK_COMPUTE_COORDINATION.md)
holds the handoff procedure. If sharing is requested, the former **60% hexapod /
40% other work** split is the starting preference for coordination. MPS and a
shared long-training launcher have not been configured or validated by this
work. Coordinate and measure a paired pilot before relying on concurrent GPU
training, and leave unrelated jobs untouched.

Historical Stage2 team workflow (new CAD-v2 validation uses §11):

- One shared queue (a document or issue label) lists the next attempts in
  order with experiment file, intervention file, seed, and label. You launch
  only attempts on that list, and you run `ops/hexctl doctor` first.
- Perception and navigation work do not use the Spark GPU. They run on
  laptops, the bench computer, and the companion computer. You record sim
  integration bags in scheduled windows.
- A run longer than an hour goes through a systemd unit or a `screen` session
  with the label in its name, so the next person can see what runs and who
  owns it.
- Each member has their own Tailscale identity and Unix account, or the team
  shares one account and keeps the label discipline above. Unlabeled runs on
  a shared account make a lost failure look like a phantom result.

The operational objective is minimum idle GPU time without colliding with
unrelated work and without blind retry storms: detect pre-AppReady stalls within
45-90 seconds, preserve diagnostics, prove exact-container cleanup, and perform
at most one bounded identical retry while holding the GPU lock for the logical
run.

## 4. Archived Stage2 launchers

These helpers and the following Stage2 launchers reproduce the archived mock
lineage. **Do not use `hexapod-rl start` for CAD v2**: its service selects a
legacy checkpoint/task and is not the corrected-model validator. The bounded
CAD-v2 launcher is documented in §11.

Persistent Stage2 training service helpers:

```sh
hexapod-rl status       # persistent training service
hexapod-rl logs         # live PPO metrics
hexapod-rl latest       # newest checkpoint
hexapod-rl stop         # cleanly stop the run
hexapod-rl start        # start a fresh configured run
```

Single-probe launch from the immutable parent. Use a new unique label every
time; never reuse or overwrite a previous attempt label, including a failed
one:

```bash
cd /home/orionh/HEXAPOD
label=accel_probe21_bilateral_long2_ref1800_seed99_slew040_retry2_YYYYMMDDTHHMMSSZ
out=/home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c/${label}.launcher.log
test ! -e "$out"
nohup ./isaaclab/deploy/probe-stage2c-single-current-best \
  "$label" 99 \
  env.inactive_bilateral_longitudinal_contact_moment_reward_scale=-2.0 \
  env.inactive_bilateral_longitudinal_contact_moment_reference_nm=1.8 \
  >"$out" 2>&1 </dev/null &
```

The launcher pins every other setting, verifies the parent hash, holds the
shared GPU flock, gates on Docker and the systemd service, writes atomic
artifacts, takes a dense checkpoint inventory, and cleans up by exact container
ID. Do not append unreviewed Hydra overrides; only allowlisted interventions
belong on the command line.

Relevant deploy paths:

```text
isaaclab/deploy/probe-stage2c-single-current-best
isaaclab/deploy/calibrate-stage2c-bilateral-current-best
isaaclab/deploy/screen-stage2c-probe-sharded
isaaclab/train_model_only_resume.py
isaaclab/deploy/hexapod-rl-training-stage2c-stable-forward.service
isaaclab/deploy/stage2_pipeline.sha256
```

## 5. Formal sharded screen

After a probe produces an exact 12-checkpoint inventory, resolve
`child.run_path` and run the formal sharded screen. Arguments are: batch label,
child run basename, parent run directory, parent checkpoint file, parent
SHA-256, screen name, then the checkpoint indices.

```bash
cd /home/orionh/HEXAPOD
./isaaclab/deploy/screen-stage2c-probe-sharded \
  <successful_probe_batch_label> \
  <timestamped_child_run_basename> \
  2026-08-25_23-44-14_accel_scale4_yaw160_trackguard_20260825T234500Z_seed86 \
  model_2.pt \
  a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a \
  formal_seed60_slew040_sharded_v1 \
  0 1 2 3 4 5 6 7 8 9 10 11
```

Rules that make the screen admissible:

- Screen all 12 children before interpreting the arm. Do not read a partial
  batch.
- The merger is fail-closed. It requires exact parsed-JSON equality of the
  parent reports and wrapper action-processing records each shard repeats,
  and it enforces exact `model_0.pt` through `model_11.pt` membership and
  ordering.
- Launch shard B only after shard A has reached AppReady.
- Harden shard cleanup to immutable container IDs before you trust the
  sharded screen unattended.
- A causal gain does not promote a checkpoint by itself. The absolute Stage2C
  gates in `docs/TRAINING.md` §6 decide admission.

## 6. Specification: attempt-aware startup supervisor

**Specification. `packages/hexapod_train` implements it. In-vivo validation
is pending.** The text below describes current launcher behavior only where it
repeats a fix that already shipped (§7). Do not cite this section as evidence
that a run ran under supervision.

**Implementation status (2026-08-27).** `packages/hexapod_train` implements
the supervisor, `ops/hexctl` exposes it (`compose`, `doctor`, `probe`,
`screen`), and unit tests under `isaaclab/tests/` cover it. Nobody has run it
against the Spark, so no in-vivo validation exists and no attempt has run
under supervision. The bash launchers remain the executors. `hexctl` composes
their argv and supervises one as a child process. It does not replace their
flock, Docker and service gates, atomic artifacts, or exact-container-ID
cleanup. Three items are partial by design. Item 1: each launcher takes the
shared lock itself, so the lock holds per attempt and drops at the retry
boundary. Item 8: the pinned 64-environment LR=0 warmup does not exist yet.
Item 10: the sharded launcher starts both shards itself, so `hexctl screen`
verifies the ordering from the shard logs and refuses to certify a violation.
It does not enforce the order at launch.

Implement in the hardened launcher:

1. Hold `/tmp/hexapod-isaac-gpu.lock` across the entire logical run.
2. Gate not only current GPU, Docker, and service use, but producer scripts and
   their descendants: `nsva_dl.sh`, `validate_nsva.py`, `score_clip.py`,
   `validate_b51.py`, and torch-compile / clip-scoring workers.
3. Recheck contention as soon as the container exists and again before PPO
   starts.
4. Set `PYTHONUNBUFFERED=1`.
5. Require `Loading user config` by 45 seconds and the AppLauncher completion
   marker by 90 seconds. Keep 420 seconds as the overall cap for 12 updates.
6. At 30 seconds without progress, atomically capture exact container inspect,
   container logs, `docker top`, `nvidia-smi`, owned `/proc/*/{status,wchan}`,
   log size and last line, run-directory inventory, OOM state, and the parent
   hash.
7. Allow one identical retry under a new immutable attempt label only for a
   proven pre-AppReady stall with no run or checkpoint, an unchanged parent, no
   traceback and no OOM, and proven cleanup. Never retry user signals,
   post-AppReady failures, Docker errors, or unverifiable cleanup.
8. Before that one full retry, a pinned 64-environment, one-update, LR=0
   runtime warmup may be used, but it must use the hardened exact-ID path and be
   marked diagnostic / non-policy. If the warmup fails, stop. Do not use the
   legacy smoke script unchanged: it calls `docker rm -f` and lacks the shared
   lock contract.
9. Preserve `attempts/00`, `attempts/01`, `retry.decision`, hashes, and a root
   outcome that states whether a retry occurred. Overall success must not hide a
   retry.
10. For sharded evaluation, launch shard B only after shard A reaches AppReady,
    and harden shard cleanup to immutable IDs before trusting it unattended.

Diagnostic sequencing for the underlying failure class: add flushed wrapper
milestone timestamps and opt-in `faulthandler` / SIGUSR1 dumps around the
deferred import and `runpy` boundaries before changing caches, IPC, or image
layout. Shared writable Kit/OV caches are a plausible lead without proof. Do
not auto-delete caches or Docker volumes.

Contract tests that must move with the implementation:

```text
isaaclab/tests/test_probe_stage2c_single_launcher_contract.py
isaaclab/tests/test_calibrate_stage2c_bilateral_launcher_contract.py
```

## 7. Already fixed

The exact-container cleanup in both hardened launchers uses `|` delimiters
in the Docker Go template in place of a literal `\t`, and Docker's current
`--timeout 30` option. The change is on the Spark, passed the test suite, and
Probe21 attempt 1 validated it in vivo. Recorded hashes at the time of the
fix:

```text
probe launcher SHA:       5f69625d7638e538f2c6f48c2907c9b42c61ac0329afa340362f1cfc1c5cc506
calibration launcher SHA: 810a4ed436f935f280043a8a54a0f2ae6e6af8a72e647af0dff632ca9b8c5ece
manifest SHA:             c104eb22024077f8c875b2d8a632d437ff47bb89c2c7da8999570de5bdec99c2
```

Background on the parser bug and the stall class it exposed:
`docs/incidents/2026-08-26-preappready-stall.md`.

## 8. Evidence handling

- Every attempt, including every failed attempt, keeps its artifact directory.
  Never delete or overwrite failed-attempt evidence.
- Mirror remote probe artifacts into
  `artifacts/phase2_recovery_stage2c_stable_forward/probes/<label>/` and record
  the outcome in that directory's `README.md` ledger, which is append-only.
- Checkpoints copied into `artifacts/` are immutable, and their SHA-256 entries
  in `artifacts/README.md` are immutable.
- Global `*.log` ignore rules mean logs inside artifact directories do not
  enter a normal commit unless force-added or the ignore policy is refined.

## 9. Working without Spark access

A session that cannot reach the Spark or the Tailscale network can review
source, run the CPU-safe test suite, and prepare patches. It cannot claim to
have resumed, monitored, evaluated, or recorded live training. In that case,
return the exact patch plus the exact remote commands, working directory,
expected output, and required return evidence, and let a session with Spark
access execute them. Do not request or accept credentials in chat.

## 10. Historical import procedure (asset v1)

This section is retained for interpreting and reproducing asset-v1 evidence.
It is superseded by §11 for new work. A successful old standing check does not
repair the original USD's inertia orientation defect, and its reset/height
expectations do not apply to the corrected v2 configuration.

Asset v1 is `robot/hexapod_mkii_assy/`. Its README holds the conventions,
limits, stance, and regeneration steps. The spec is `MKII_V1_ASSET` in
`packages/hexapod_env/hexapod_env/assets/spec.py` and the task ID is
`Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0`. Do not train on it before
step 4 passes and step 5 confirms the runtime joint order.

1. Sync the source to `/home/orionh/HEXAPOD` and verify
   `sha256sum -c isaaclab/deploy/stage2_pipeline.sha256` (the manifest covers
   the asset modules).
2. Generate the USD inside the container from `/workspace/hexapod`:
   ```sh
   python tools/import_urdf_to_usd.py \
     robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf \
     robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda
   ```
   Expected: `rigid_bodies=19 revolute_joints=18`, the `.usda` plus a
   `payloads/` folder beside it, and the nested hierarchy
   `Robot/Geometry/body/lf_coxa/lf_femur/lf_tibia` in `payloads/base.usda`.
   If the importer API differs on this build, import from the GUI with these
   settings: floating base, import inertia tensor on, density 0 (keep URDF
   masses), merge fixed joints on, self-collision off, convex decomposition
   off, collision from visuals off, position drives. Save to the same path.
   `package://hexapod_mkii_assy/...` resolves from the package directory, so
   import from a checkout where `meshes/` sits beside `urdf/`.
3. Contact reports, same container:
   ```sh
   python tools/enable_nested_contact_reports.py \
     robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda
   ```
   Expected: `CONTACT_REPORTS_ENABLED bodies=19`.
4. Validate (`--asset mock` re-validates the Phase-0 lineage):
   ```sh
   python isaaclab/validate.py --asset mkii_v1 --num_envs 32 --steps 1000
   ```
   The validator requires 18 joints, six feet, finite observations, no
   terminations or truncations, no coxa/femur/tibia-shaft ground contact after
   settling, and a post-settling computed-torque saturation fraction below
   0.5 % against this historical baseline's 1.6 N m threshold. `max_abs_computed_torque_nm` includes
   startup; the historical gate does not require this raw maximum to stay
   below 1.6. Keep startup and settled results separate. Old height/torque
   expectations are not acceptance criteria for a new asset.
   `isaaclab/deploy/validate-stance-sweep` runs three candidate stances.
5. Confirm the runtime joint order. The action vector is positional, so the
   articulation's own `joint_names` order is a contract.
   `hexapod_core/joints_v2.py` predicts it (breadth-first from `body`: six
   coxa_yaw, six femur_pitch, six tibia_pitch, legs lf lm lr rf rm rr) and
   marks it provisional. `HexapodEnv` refuses to construct on a mismatch. Keep
   the `joint_names=[...]` line from step 4 as evidence. If it matches, set
   `RUNTIME_ORDER_STATUS` to `"confirmed"` in `joints_v2.py` and in the spec,
   in the same commit as that transcript. If it differs, correct both tuples
   together. Do not bypass the check.
6. Commit the generated `usd/` folder if the team shares it. The USD path
   defaults to the container path in the spec, and `HEXAPOD_MKII_V1_USD_PATH`
   overrides it per run. Curriculum stages for this asset derive from
   `HexapodMkiiV1FlatEnvCfg` under new task IDs, with a fresh stance sweep for
   the 8.26 kg mass distribution. The `phase2*_cfg.py` classes stay on the
   mock. The archived `stage2_pipeline.sha256` is now frozen; new revisions
   need a separate manifest and must preserve the historical source/evidence.

| symptom | likely cause |
|---|---|
| `Unexpected contact-body layout` from `env.py` | The USD hierarchy differs from `Geometry/body/<coxa>/<femur>/<tibia>`. Check the payload and the `MKII_V1_GEOMETRY_ROOT` sensor paths in `env_cfg.py` |
| `Articulation joint order differs from the asset's runtime joint contract` | PhysX ordered the joints in a different order from the `joints_v2.py` prediction. Correct `joints_v2.RUNTIME_JOINT_NAMES` and `MKII_V1_ASSET.runtime_joint_names` from the printed `joint_names` |
| `Expected 18 joints` / `Expected 6 feet` | You imported the wrong URDF (linkage variant, or the mock), or the importer did not merge fixed joints |
| torque saturation on standing | The stance moved away from `stance.json`, or masses changed. Re-derive with the stance search in the importer |
| foot contacts counted as shaft contacts | `distal_foot_min_y_m` (0.155, tibia frame +Y) no longer matches the pad. Re-check the pad spheres in the URDF |
| meshes missing after import | `package://` did not resolve. Import from a checkout with `meshes/` beside `urdf/` |


## 11. Corrected serial CAD-v2 live validation

The current source snapshot is
`/home/orionh/HEXAPOD_runs/mkii_v2_step2_a0f0b39/source/`. Its model is the same
19-link, 18-joint serial CAD URDF, with a separately generated corrected USD
at `robot/hexapod_mkii_assy/usd/hexapod_mkii_serial_v2/hexapod_mkii_serial_v2.usda`.
The config is `hexapod_env.tasks.mkii_v2.config.HexapodMkiiV2FlatEnvCfg`, registered
as `Isaac-Velocity-Flat-Hexapod-MKII-V2-Direct-v0`. It uses anatomical navigation
axes and the v2 source/action contract. The configured reset plate height is
0.142964 m, with 5 mm geometric foot clearance **before** the inherited
independent ±0.03 rad joint jitter. CPU sampling found a 1.265 mm minimum after
jitter; the live validator measures the actual randomized reset positions.

Prepare and verify the corrected USD and source snapshot before launching.
The old `/home/orionh/HEXAPOD` mirror, old task IDs and frozen Stage2 manifests
remain untouched. `a0f0b39` names the source baseline; the launcher writes
`source.SHA256SUMS` for the exact files actually supplied to each run, including
subsequent fixes. No parent policy/checkpoint is loaded for standing validation.

Run from the Spark host using the isolated source's new launcher:

```sh
/home/orionh/HEXAPOD_runs/mkii_v2_step2_a0f0b39/source/isaaclab/deploy/validate-mkii-v2 \
  --source-dir /home/orionh/HEXAPOD_runs/mkii_v2_step2_a0f0b39/source \
  --source-commit uncommitted \
  --output-root /home/orionh/HEXAPOD_runs/mkii_v2_step2_a0f0b39/runs \
  --num-envs 32 --steps 1000 --timeout-seconds 900
```

Add `--dry-run` to inspect the composed invocation without launching anything.
Replace `uncommitted` with a published commit only when the supplied source has
been verified against it; the directory's baseline name alone is insufficient.
A first startup probe can use `--num-envs 1 --steps 100`; its report is explicitly
`run_kind: short_probe`. At least 32 environments and 1,000 completed steps
produce `run_kind: acceptance`, which refers only to this standing gate.
Use a new generated run directory for every attempt, including failures.

The supervisor holds host FD9 on `/tmp/hexapod-isaac-gpu.lock` from before
container creation through cleanup. It starts a detached container behind a
CPU-only admission barrier, checks unrelated producers/GPU use/services before
creation and twice afterward, and keeps checking during execution. It refuses
unrelated GPU-capable containers, while allowing CPU-only containers to remain.
It mounts source read-only and the run directory writable. It uses the existing
Compose SDK environment without printing the secret env file, applies the
combined telemetry startup mitigation, and passes `--viz none --device cuda:0`.
The execution timeout is at most 900 seconds, followed by bounded cleanup.
There is no automatic restart or retry. Only the run's nonce-labelled, exact
immutable container ID can be stopped or removed; other jobs are never stopped.

Each run directory contains:

- `source.SHA256SUMS`: the exact supplied source-file hashes.
- `supervisor.json`: admission/runtime checks, process status and exact-ID cleanup.
- `container.log`: preserved startup, validation and shutdown output.
- `report.json`: validator measurements and graded outcome, when the validator
  reaches report persistence. Absence is a failure, even if Kit exits zero.

The validator persists its report before Kit shutdown can terminate the Python
process. A zero container exit alone is insufficient: the supervisor requires
strict JSON with `pass` and `standing_gate_pass` both literally true, no errors,
the correct task and run kind, and the requested number of completed steps and
environments. A graded failure stays failed even if SDK shutdown returns zero.
Read the report and supervisor together; directory presence does not mean a job
is running or that it passed. Curated results are recorded in the
[live validation evidence](../artifacts/mkii_step2_2026-09-04/README.md).

This is a **partial standing gate, not full G0 or full physical-model approval**.
It checks the corrected asset in the native runtime, joint identity, measured
reset clearance, finite states, held-zero commands, standing contacts,
terminations and torque demand. Report raw startup computed torque separately
from applied torque and settled saturation: initial contact/jitter transients
must not be presented as continuous capability, or hidden by a settled average.
The physical four-bar dynamics and driven joint/direction checks remain pending.
The serial approximation still fixes the push lever and rod to the femur;
a visually connected mimic-linkage preview does not repair that physics.
A passing report grants no general-terrain or hardware-transfer claim and does
not admit an old policy for training or deployment.

The [RS05 specification review](RS05_SPEC_REVIEW.md) corrects the motor
interpretation: these are short tests with a 1.6 N·m applied cap, although
5.5 N·m peak is already present in the URDF/config. The 2.364431 N·m startup
raw demand is below that peak; the historical `startup_raw_rating_exceeded`
flag refers to the 1.6 baseline. Vendor continuous stall is 1.2 N·m, and rotating
ratings depend on cooling. Neither this scalar cap nor a standing pass qualifies
the unmodeled torque-speed/thermal envelope or endurance.

Before training a new physical task, version its motor parameters and resolved
configuration identity in the simulation/runtime manifest, and qualify
torque-speed/voltage, thermal, bounded burst/recovery and phase-current limits.
The current manifest omits motor parameters; hardware CAN/current/temperature
protection is not implemented. Do not simply raise `effort_limit` to 5.5 N·m
indefinitely or reinterpret the preserved baseline reports as that new model.
