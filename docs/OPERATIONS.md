# Spark operations runbook

Research is paused for James. Native validation, training, recording and
unattended continuation require his explicit resume. The exclusive reservation
and external automation blocks remain in force. Repository checks and publication
can continue. [Compute coordination](SPARK_COMPUTE_COORDINATION.md) owns that policy.

The sections below retain verified host conventions and recovery controls.
Recheck host state before an authorized allocation. [Archived procedures](archive/README.md)
contain the old Stage2/four-bar launch recipes and supervisor specification;
use their original source revision only for explicitly scoped reproduction.
The approved model is selected by [robot/active_model.json](../robot/active_model.json).

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
verify with the applicable manifest. Canonical standing and unlaunched PPO
source snapshots live under `/home/orionh/HEXAPOD_runs/canonical_direct_20260910`;
the [pause archive](../artifacts/handoff_2026-09-11/james_001/README.md) identifies
the exact source directories and retained data. Choose an explicitly admitted
source for any resumed allocation. Historical CAD-v2 paths remain in the archive.

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

## 2. Before an authorized allocation

Read the host and repository coordination records. Inspect actual processes,
GPU use, containers, services, shared locks and continuation automation. Bind
source, model, configuration and the launcher's admission report to the same
release. Recheck ownership after container creation and before learning begins.
An old manifest or historical checkpoint cannot admit a new model. Preserve all
prelaunch and post-exit receipts with the attempt's evidence.

## 3. GPU lock and shared-workload protocol

The latest user instruction reserves the Spark for HEXAPOD user compute at all
times, including between runs. This supersedes earlier weather-only priority
and sharing-on-request preferences. Read the current header of
`/home/orionh/SPARK_COMPUTE_COORDINATION.md` before dispatch. Identified competing
producers must remain deferred until a later user instruction releases or
changes the reservation. Preserve their work and exact recovery state, SSH,
operating-system services and host health. These admission and cleanup rules
still apply:

- Hold `/tmp/hexapod-isaac-gpu.lock` for the entire logical run, from before
  container creation to cleanup.
- Defer competing user compute through source-verified producer controls,
  retaining exact identity, outputs and restart state. Never use broad process
  matching to kill work or modify another project's model/data. A stopped or
  frozen CUDA process may retain its context and memory; keep resource checks
  truthful and do not treat suspension as a free GPU.
- A free GPU is not sufficient evidence that a producer is finished. Wait until
  the producer script and all of its descendants are gone, then recheck the
  GPU, active Docker containers, the systemd service, and the shared lock.
- Producers spawn GPU children after their own start, so a clean prelaunch
  check can turn stale seconds later. Recheck contention as soon as the
  container exists and again before PPO starts. Use the admitted launcher’s startup supervisor to close this post-launch race.
- Cleanup acts only on the exact immutable container ID owned by this run. It
  must never use `docker rm -f` on a broad match. Deferral of another producer
  is a separate, recorded reservation operation.

Persistent scheduler blocking must survive per-job cleanup and reconnects.
Snapshot only the timers active when that job starts; restoring that snapshot
does not release the broader reservation. Keep the reservation marker and
owned masks/queue lock until explicit user release, then restore only recorded
prior state after identity checks. For a historical condition-only unit, verify
loaded drop-in paths and no pending daemon reload; a skipped start may return
success without activation. For the newer masked units, require the exact
`/dev/null` symlink, loaded and file state `masked`, and inactive state. Services
must report PID0; timers normally expose no MainPID. Retain reload metadata:
the old condition drop-ins can leave `NeedDaemonReload=yes` even after reload
on an otherwise loaded mask. Direct-start and re-enable rejection demonstrate
the block; do not report that flag as `no`. The
[coordination policy](SPARK_COMPUTE_COORDINATION.md) names the current reservation. Another workload's sharing request alone cannot override the
current user instruction. No MPS quota or hardware GPU partition is implied;
manual or privileged launches still require truthful detection and deferral.

### 3.1. Complete external automation block

The 11 September user instruction requires external Spark automation to remain
stopped, including any upstream triggers, while HEXAPOD retains priority.
The [block and recovery record](../artifacts/operations_2026-09-11/spark_automation_block_001/README.md)
defines the exact controls. Preserve its original unit files, enable states,
worker source backup and queue outputs for a later user-authorized release.

Mask the base `stormscope-`, `stormscope-private-` and `stormscope-private-v2-`
timer/service families and Ollama. The private scout's restart policy means
disabling timers alone is insufficient. Keep the four system `wx-forecast`
timer/service units masked. A repository installer can try to enable timers;
validate that the masks reject both direct start and `enable --now`.

The reconstruction worker must reject startup while the reservation marker
exists. Keep `hexapod-exclusive-reconstruction-queue.service` enabled and verify
that its exact helper owns `queue-worker.lock` in `/proc/locks`. Bind its current
PID to the acquisition receipt and command line. That receipt names the full
`ACTIVE` marker path; per-job restoration receipts instead name the reservation
directory. These are different fields and must not be compared interchangeably.
Do not erase request files, completed reconstructions or partial outputs.

Check actual GitHub workflow inventories before assuming Git is a scheduler.
Disable identified competing triggers when present; a repository with no
workflows has nothing to cancel. Preserve HEXAPOD's own jobs, repository CI,
Pages deployment. The later James handoff pauses continuation automation and all new research allocations; retain the reservation helper. Every native successor must
recheck these operational controls without changing its admitted robot source,
servo, physics or quality gates.

**Handoff pause, 11 September 2026:** `/home/orionh/HEXAPOD_runs/canonical_direct_20260910/james_handoff_pause_001/PAUSED.json` records the user pause. The Codex continuation is PAUSED, and the old `hexapod-restore-forecasting-20260909.timer` is stopped with its transient units absent. No research job is queued. James or the user must explicitly resume project work; external scheduler release remains a separate user decision. Read [JAMES_HANDOFF](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/62fd7448264c5ebe051ba2de1d9a72844d6b4c3a/docs/JAMES_HANDOFF.md) before using historical launchers.

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
