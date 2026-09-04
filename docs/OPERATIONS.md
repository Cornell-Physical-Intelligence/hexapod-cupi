# Spark operations runbook

> Execution update, 2026-09-04: [current status](../STATUS.md) is planning-only, with hexapod jobs stopped and no restart queued. The [asset audit](../artifacts/project_review_2026-09-04/URDF_VALIDATION.md) found an unresolved all-link USD inertia conversion defect. The procedures below do not override the pause or replace the new campaign's [prerequisites](NEXT_RUNS.md).

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
verify with the manifest.

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

Run this read-only sequence first, every time, and change nothing until it is
clean:

```bash
ssh orionh@100.82.166.9
cd /home/orionh/HEXAPOD
date -u
ps -eo pid,ppid,lstart,etime,%cpu,%mem,args --sort=-%cpu | head -80
pgrep -af 'nsva_dl|validate_nsva|score_clip|validate_b51|torch.*compile|train_model_only_resume|evaluate_checkpoint' || true
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
systemctl is-active hexapod-rl-training.service || true
nvidia-smi
flock -n /tmp/hexapod-isaac-gpu.lock true
sha256sum -c isaaclab/deploy/stage2_pipeline.sha256
sha256sum isaaclab/logs/rsl_rl/hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/2026-08-25_23-44-14_accel_scale4_yaw160_trackguard_20260825T234500Z_seed86/model_2.pt
```

`sha256sum -c isaaclab/deploy/stage2_pipeline.sha256` verifies every entry in
the manifest and must pass both locally and on the Spark. The parent checkpoint
hash must equal
`a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a`.

Also check whether an automated watcher has already acted before launching
anything by hand.

## 3. GPU lock and shared-workload protocol

The Spark is shared with unrelated workloads. The rules are:

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

Sharing the Spark across the team:

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

## 4. Launchers

Persistent training service helpers:

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

## 10. Importing and validating a robot model (asset v1)

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
   settling, and computed torque under the RS05 1.6 N m rating (saturation
   fraction below 0.5 %). Expect `post_settle_mean_base_height_m` near 0.12
   and `max_abs_computed_torque_nm` below 1.6 with margin.
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
   mock. After you change a manifest-listed file, regenerate
   `stage2_pipeline.sha256`.

| symptom | likely cause |
|---|---|
| `Unexpected contact-body layout` from `env.py` | The USD hierarchy differs from `Geometry/body/<coxa>/<femur>/<tibia>`. Check the payload and the `MKII_V1_GEOMETRY_ROOT` sensor paths in `env_cfg.py` |
| `Articulation joint order differs from the asset's runtime joint contract` | PhysX ordered the joints in a different order from the `joints_v2.py` prediction. Correct `joints_v2.RUNTIME_JOINT_NAMES` and `MKII_V1_ASSET.runtime_joint_names` from the printed `joint_names` |
| `Expected 18 joints` / `Expected 6 feet` | You imported the wrong URDF (linkage variant, or the mock), or the importer did not merge fixed joints |
| torque saturation on standing | The stance moved away from `stance.json`, or masses changed. Re-derive with the stance search in the importer |
| foot contacts counted as shaft contacts | `distal_foot_min_y_m` (0.155, tibia frame +Y) no longer matches the pad. Re-check the pad spheres in the URDF |
| meshes missing after import | `package://` did not resolve. Import from a checkout with `meshes/` beside `urdf/` |
