# Spark operations runbook

## Forward-example pilot allocations, 17 September 2026

The successor identified two orphan reconstruction producers, PIDs 3163647 and
3167591, that could launch COLMAP despite the retained scheduler masks. Under
James's compute authorization, the successor preserved their source and process
identities, then paused both with SIGSTOP. The successor preserved two active
containers as recovery images before stopping them. The
[recovery receipt](../artifacts/restart_2026-09-14/spark_ownership_007/RECEIPT.json)
binds these actions and the output mounts. Raw process arguments and source
copies remain in the private Spark recovery directory. Restore a producer after
James releases the reservation and its PID, start time and command hash match.
An [earlier observed container](../artifacts/restart_2026-09-14/spark_ownership_006/RECEIPT.json)
exited before intervention; that receipt claims no stop or recovery image.

The first pilot check failed before container creation because its input mount
used an unsupported destination. The second reached native initialization but
failed at the update-zero checkpoint save. Fresh source packs correct both
interfaces and retain the failed attempts. The
[paired native check](../artifacts/forward_example_ppo_20260917/NATIVE_SMOKE_001.json)
records completed two-update runs and exact cleanup for both arms. Source freeze
`516cb2cb26c6b5abfb655a36408f022e684cfeb5621a3b0fe496e72d615253f8`
binds the successor packs. The coordination hash and reservation remain unchanged.
These receipts describe their recorded allocations; recheck live resources
before another dispatch.

The scratch arm then completed all 1,200 updates and exact cleanup. A new
reconstruction producer started container
`638167108e5963302b2cda60c10e1f36173466656fc8169f28082a0b99e573db`
after that allocation ended. The example-arm preflight rejected the occupied
host before creating a container. The successor preserved the new producer's
source and process identity, paused PID 3210795, preserved the container as a
recovery image and stopped it. The [incident receipt](../artifacts/restart_2026-09-14/spark_ownership_008/RECEIPT.json)
records empty resources and three paused reconstruction producers. The script
on disk differs from incident 007; retain each incident's source copy for its
own recovery. A fresh example-arm preflight passed with the same coordination
hash and retained reservation controls.

The example arm completed all 1,200 updates and exact cleanup. Before the
scheduled policy evaluations, the successor found another COLMAP container,
`edfe87124776a849581201443d252f0e2c5658dd47b130286c996f64339a3d70`,
that started after training ended. The successor preserved its recovery image
and the new producer's source, then paused PID 3304563 and stopped that exact
container. The [recovery receipt](../artifacts/restart_2026-09-14/spark_ownership_009/RECEIPT.json)
records the result. The first policy-evaluation preflight passes with empty
resources, all 32 user masks and four system masks. Earlier process states remain
timestamped evidence; verify process identity before any later restoration.

The guard interrupted example-arm evaluation at update 600 after an unrelated
reconstruction CUDA process appeared. The [incident readback](../artifacts/restart_2026-09-14/spark_ownership_010/READBACK.json)
records that the process exited before inspection. The two other inspected
processes run static HTTP servers; the [classification](../artifacts/restart_2026-09-14/spark_ownership_010/CLASSIFICATION.json)
records no stop action and clarifies the private recovery contents. We preserved
all 65 files from the interrupted allocation, including the unfinished stop
capture. The [retry receipt](../artifacts/forward_example_ppo_20260917/EVALUATION_RETRY_001.json)
binds a fresh allocation to the same checkpoint, seed and runtime. We repeat
all scheduled cases under identity `example_evaluate_update000600_002`; the
original service failure and partial capture retain their identities.

## PPO comparison allocations, 17 September 2026

The successor identified and stopped the COLMAP container
`b58dba529de41d0dd4e8d176712ceaa706abec2de3e9e04bef576fd741a337cc`
under James's full-compute authorization. The container used automatic removal.
Before stopping it, the successor preserved a recovery image and the mounted
output paths. The [verified recovery receipt](../artifacts/restart_2026-09-14/spark_ownership_004/RECOVERY_VERIFIED.json)
contains the image identity and restore command. The original command receipt
includes a Docker warning before the image ID; preserve it and use the verified
receipt for restoration after James releases the reservation.

The first direct-task admission ended at the existing AppReady deadline and
captured no physics samples. Its [cleanup receipt](../artifacts/ppo_reference_comparison_20260917/admission_pack_001/run/cleanup.json)
confirms removal of the exact owned container. A second source pack remains
unlaunched. The standard PPO comparison uses the existing one- and 128-replica
admission. It leaves the direct-task port's native admission pending.

The improved reference replay completes its native forward screen and exact
container cleanup. The [audit](../artifacts/ppo_reference_comparison_20260917/review_smooth_001/RESULT.json)
records the result. Source packs and bindings retain the coordination SHA-256
`c89c99ebdd16941e3f8e7f23ef3525aeb360c78361aa49aa04e766b381a4727c`.
Recheck live ownership before another allocation; these receipts retain their
own terminal states and do not describe current host availability.

The standard PPO baseline completes 1,200 updates and 3,686,400 transitions.
Its native process exits with code 0. A new COLMAP container,
`b136f6089ba776b17c117779af212e10bbd5a427143b62f1b65fb64cad9d5008`,
starts during training and blocks the wrapper's final resource check. The
successor preserves a recovery image and the mounted reconstruction outputs,
then stops that container under James's existing authorization. The
[recovery receipt](../artifacts/restart_2026-09-14/spark_ownership_005/RECEIPT.json)
and [cleanup command receipt](../artifacts/ppo_reference_comparison_20260917/vanilla_train_001/RECOVERY_CLEANUP.json)
record the recovery. The original service exit remains 1; the final checkpoint
and native results retain their bytes. Use the separate native exit and cleanup
evidence when assessing this attempt. Concurrent reconstruction invalidates an
isolated throughput claim for the training interval.

## Successor allocations, 16 September 2026

The successor restored the ten base Stormscope masks after preserving their
unit bytes and prior states on Spark. The
[mask receipt](../artifacts/restart_2026-09-14/spark_ownership_002/REMASK.json)
verifies all 32 user masks, four system masks and the retained queue lock.
The coordination SHA-256 remains `c89c99ebdd16941e3f8e7f23ef3525aeb360c78361aa49aa04e766b381a4727c`.
Public copies contain selected states and hashes; foreign unit bytes remain
in the remote recovery directory.

The uniform refit completed its 1,000 controls, but a manual reconstruction
container started eight seconds after dispatch and blocked the final resource
check. The service retains its failed exit. The successor stopped the exact
container `26deae115bc8ff946464f0ff404379a650b6cf2d01b4ae9b1d6e1a71b9d0ebc5`,
preserved it and its mounted outputs, then ran the frozen cleanup-only command.
The [incident receipt](../artifacts/restart_2026-09-14/spark_ownership_003/RECEIPT.json)
and original failure journal preserve that distinction. Its restart policy is
`no`; restore it with the recorded command after an explicit reservation release.

The onset20 trial then completed with a successful service exit and cleanup.
The [final readback](../artifacts/restart_2026-09-14/paper_walk_execution_001/POST_REFIT_RESERVATION_001.json)
records PID 0 for both services, no GPU compute apps or active containers, and
both GPU locks available. The reservation remains retained. Recheck those live
states before another allocation; this receipt records its own timestamp.
Both native results fail forward tracking and qualify no Stage 2 behavior.

**Previous dispatcher paused for handoff on 15 September 2026 UTC.** James said:
“take a pause for now and let someone else continue work, push all non commited
changes”. Publish the preserved work, then relinquish execution. This supersedes
the current dispatcher's continuation instruction; another designated lead may
continue the existing authorized goal without another user permission. No new
lead or handoff task queue is created. The old heartbeat remains PAUSED.

The [pause receipt](../artifacts/restart_2026-09-14/pause_20260915_001/RECEIPT.json)
records all four latest native units inactive/PID0, no containers and no GPU
compute apps at 01:58:15 UTC. The reservation, scheduler masks and reconstruction
queue lock remain retained. Remote pause receipts are under
`/home/orionh/HEXAPOD_runs/restart_20260914/pause_20260915_001`.
The shared coordination file now hashes to
`c89c99ebdd16941e3f8e7f23ef3525aeb360c78361aa49aa04e766b381a4727c`.
A successor must verify live resources/ownership and create a fresh guard and
launch binding to these bytes, preserving all historical identities. A pause
snapshot is not continuing live telemetry.

James authorized a fresh canonical qualification sequence on 14 September 2026
after visually confirming the mass-corrected URDF: standing, walking/stopping,
terrain, then survey. The selected URDF SHA-256 is
`9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78`.
That authorization superseded the earlier research pause for that sequence. Each stage still needs its
own admission and evidence; authorization does not establish a running job or a
passing result. The exclusive reservation and external-automation blocking
requirements remain policy; verify the live controls before allocating compute.
[Compute coordination](SPARK_COMPUTE_COORDINATION.md) owns that policy.

On the same date James explicitly authorized Codex to take full ownership of
Spark compute and stop competing processes. The lead dispatcher may defer
competing CPU/GPU workloads and their restart triggers without another approval.
Use the identified service or exact process identity, preserve its source,
outputs and restart state, and keep SSH, networking and essential host services
available. This authorization alone does not establish that compute is free.

The active user goal is the existing Stage 2 milestone on the confirmed model,
including paper-informed PPO training and an actual Isaac Sim policy video.
The successor's remaining scope extends beyond diagnostic acquisition through same-source native admission,
learning, held-out evaluation and the existing numerical and human visual
acceptance requirements. The maintained prototype is documented in
[experiments/paper_walk](../experiments/paper_walk/README.md). A native smoke,
checkpoint or recording is evidence of its own scope; none alone completes
Stage 2. Preserve full standing/contact checks and the unchanged direction,
transition, quiet-stop, torque and smoothness requirements. Recorded failed
attempts keep their original identities and do not become admissions.

The 14 September paper-walk source002 configuration passes the original standing
gates at one and all 32 replicas with neutral `[0, -0.30, 0.40]` radians per leg.
[Recomputed standing evidence](../artifacts/restart_2026-09-14/paper_walk_execution_001/admission_001/verification.json)
binds those scoped passes. The subsequent
[native replay](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_replay_001/standing/native_replay/report.json)
retains accepted demonstrations for all 21 commands, with one duplicate excluded
from 32 demonstrations. The first
[PPO attempt](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_001/standing/state.json)
then completed 200 updates and 153,600 transitions without behavior cloning.
[Cleanup](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_001/cleanup.json)
confirms that attempt's container is absent and the reservation remains retained.
These facts establish neither walking/stopping qualification nor current host
availability. Earlier inspection-zero failures remain unchanged. Read the
current registry and verify live host state before dispatch.

Later [train005](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_005/standing/state.json)
completed the lower-exploration 200-update PPO contrast. Its
[evaluation008](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_008/standing/evaluation/summary.json)
was deliberately stopped with four complete failed direction trials and nine
missing probes. A separate [CPU BC fit](../artifacts/paper_bc_fit_001/REPORT.json)
has zero PPO updates; its complete native forward, quiet and stop trials fail.
[Training evidence](TRAINING.md) records their exact scope. Ordered probe subsets
preserve every omitted case as missing, and neither acquisition completion nor
cleanup advances Stage 2. Preserve the interrupted allocation's final receipts
as well as the earlier stop-decision snapshot; they record different times.

The subsequent [evaluation011](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_011/standing/evaluation/summary.json)
completes its three selected forward, quiet and stopping probes; all fail,
with ten omitted probes and all 96 Stage 2 cases still missing. Acquisition and
exact-container cleanup pass. The separate
[fit003 verification](../artifacts/paper_bc_fit_003/VERIFICATION.json) records CPU
velocity calibration and strict restoration only, with zero PPO updates and no
native training. The [128-replica raw audit](../artifacts/restart_2026-09-14/paper_walk_execution_001/verification_batch128_001/audit.json)
passes every replica. The lead separately adopted
[admission002](../artifacts/restart_2026-09-14/paper_walk_execution_001/admission_002/verification.json)
for the exact one- and 128-replica simulation layouts; admission001 remains
unchanged. Require the intended allocation's exact source, model and replica
configuration. The [completed evaluation012](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_012/standing/evaluation/summary.json)
records three failed forward, quiet and stopping probes for the velocity-supervised
BC candidate, including a recorded native joint-velocity bound failure in the
stop. Full acquisition and cleanup do not turn these failures into acceptance.
[CPU fit004](../artifacts/paper_bc_fit_004/REPORT.json) separately prepares fresh
128-replica learning identity, with zero PPO updates and no native actions.
Neither record authorizes bypassing matching admission or strict checkpoint
configuration checks.

The subsequent [train006](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_006/standing/state.json)
completes 200 PPO updates and 614,400 native transitions on the admitted 128
replicas. Its checkpoint and exact-container cleanup are verified. Training
records 142 joint-limit terminations and weak commanded-direction movement;
it does not qualify walking. The separate
[completed evaluation013](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_evaluate_013/standing/evaluation/summary.json)
fails all three actual PPO forward, quiet and stop probes. The
[independent full-trace audit](../artifacts/restart_2026-09-14/paper_walk_execution_001/verification_evaluate_013/README.md)
and [actual forward-video review](../artifacts/restart_2026-09-14/paper_walk_execution_001/visual_review_006/REVIEW.json)
confirm the failed scope. Quiet lacks six-foot support throughout its fixed
window; three remaining support losses still fail the stop. Exact-container
cleanup passes and establishes only that allocation's terminal state.

The [adopted stability decision](../artifacts/restart_2026-09-14/paper_ppo_stability_review_004/DECISION_001.json)
selects bounded model/Adam rollback and retries plus frozen loaded actor
statistics, preserving physics, reward and physical gates. The separately
[verified CPU migration](../artifacts/restart_2026-09-14/paper_ppo_migration_001/run_001/VERIFICATION.json)
retains the learned checkpoint's state under an explicit v3 identity. Neither
CPU inference parity nor preserved RNG bytes constitutes native or PhysX resume.
Source018 is frozen; the [train007 dispatch](../artifacts/restart_2026-09-14/paper_walk_execution_001/DISPATCH_train_007.json)
records the lead-owned launch at 00:50:01 UTC. The separately verified
[completed train007](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_007/standing/state.json)
adds all 20 updates and 61,440 transitions; its checkpoint and exact-container
cleanup pass. The bounded update behavior improves as designed on its collected
rollouts, but 14 joint-limit terminations remain and walking is unqualified.
The [completed evaluation014 audit](../artifacts/restart_2026-09-14/paper_walk_execution_001/verification_evaluate_014/README.md)
verifies forward tracking failure and original quiet/recovery-window passes,
with complete native capture and owned-container cleanup. Since forward walking
did not succeed, recovery is not a demonstrated stop from successful walking.
The [completed train008](../artifacts/restart_2026-09-14/paper_walk_execution_001/results_train_008/standing/state.json)
adds 100 updates and 307,200 transitions to checkpoint320. Strict restoration,
all 20 transferred files and exact-container cleanup verify; walking remains
unqualified.
The [completed evaluation015 binding review](../artifacts/restart_2026-09-14/paper_walk_execution_001/verification_evaluate_015/BINDING_REVIEW.json)
verifies matching physics/checkpoint, complete capture and exact-container
cleanup, with all three policy probes failed. Forward movement remains near
zero; LF, LR and RM lack contact throughout the quiet/recovery windows while
LM, RF and RR stay supported. The [root recording review](../artifacts/restart_2026-09-14/paper_walk_execution_001/visual_review_008/REVIEW.json)
finds no sustained walking. Further continuation from checkpoint320 is paused.

Root adopts the [separate source019 startup protocol](../artifacts/restart_2026-09-14/paper_walk_execution_001/ROOT_REVIEW_source_019.json)
and [explicit BCfit004 migration002](../artifacts/restart_2026-09-14/paper_bc_migration_002/ROOT_ADOPTION_001.json).
Each arm uses its own fresh native app and original reset: cold20 is all BC;
neutral4-plus-policy20 begins with 200 actual neutral controls and then 1,000 BC
controls without another reset. Complete video, 400 Hz contacts and substeps,
initial/handoff readbacks and action-source labels remain part of the receipt.
The scripted prefix is physically screened separately and cannot qualify
learned standing or hide a failure behind the policy-window slice.
The [completed paired audit](../artifacts/restart_2026-09-14/paper_walk_execution_001/startup_pair_analysis_001/REVIEW.md)
verifies both complete attempts, all 78 transferred files, 17,600 physics steps
and 394,457 contact patches, with exact-container cleanup. Both use migrated
BCfit004 checkpoint `dfb6d3ecc6ff5fae30c77bbac9056875eec6f19c9cb129917f97542d3b68d52b`,
with zero PPO updates. Both fail only the original planar tracking screen:
0.050607534 m/s cold and 0.059982400 m/s neutral4 against 0.025 m/s.
The warm prefix closely matches the recorded teacher onset but does not produce
sustained walking. Its 200 scripted controls remain separately labeled and
screened; neither their quietness nor recording completion qualifies the policy.
Six plots preserve the complete intervals. The maintained generic analyzer
currently rejects the Unicode `action_source` channel; its failure is preserved,
and the audit uses artifact-only typed loaders with all numeric/hash checks.
Do not strip that channel from the immutable raw record.

The refit helper is integrated as `experiments/paper_walk/refit_bc.py` and
executed on CPU under [protocol 002](../artifacts/restart_2026-09-14/paper_bc_refit_protocol_002/README.md);
the [paired result](../artifacts/paper_bc_refit_001/REVIEW.md) holds both
candidates. The maintained `reservation.py` now binds the pause coordination
hash `c89c99eb...`; a successor still verifies live ownership, resources and
both GPU locks on the host, then freezes `source_020` and fresh bindings before
any native candidate evaluation. Candidate fitting and native acceptance remain
separate.
Recheck live reservation, exact source/configuration and matching admission for
every allocation; a saved dispatch receipt is not live host telemetry or evidence
of completion. The existing 13 learning probes, 96 full Stage 2 cases and
cold-start qualification requirements remain unchanged.

A [local transport cleanup](../artifacts/restart_2026-09-14/paper_walk_execution_001/local_transport_cleanup_001.json)
removes only four verified redundant agent-created transfer archives. All
extracted raw evidence, checkpoints, inventories and complete remote archives
remain preserved; no unique result or failed attempt is discarded.

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

The 14 September full-ownership instruction reserves Spark user compute for
HEXAPOD at all times, including between runs, and authorizes stopping competing
workloads. This supersedes earlier weather-only priority and sharing-on-request
preferences. One lead dispatcher owns these operations. Read the current header of
`/home/orionh/SPARK_COMPUTE_COORDINATION.md` before dispatch. Identified competing
producers must remain deferred until a later user instruction releases or
changes the reservation. Preserve their work and exact recovery state, SSH,
operating-system services and host health. These admission and cleanup rules
still apply:

- Hold both `/opt/wx/gpu.lock` and `/tmp/hexapod-isaac-gpu.lock` for the entire
  logical run, from before container creation to cleanup.
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

Use the [later directory-entry block](../artifacts/operations_2026-09-11/spark_automation_block_002/README.md)
when checking reconstruction protection. The earlier regular-file gate was
overwritten. Its successor is the real directory
`/home/orionh/ithaca-reconstruction/queue-worker.py`, with an unconditional
`__main__.py` blocker, independently backed by the persistent queue lock.
Verify its identity and live lock holder; do not restore the superseded file
gate. Preserve the exact original backups and bind any new controls into a new
source/guard identity before allocation.

Check actual GitHub workflow inventories before assuming Git is a scheduler.
Disable identified competing triggers when present; a repository with no
workflows has nothing to cancel. Preserve HEXAPOD's own jobs, repository CI,
Pages deployment and the reservation helper. The 14 September restart superseded
the 11 September research pause; the current dispatcher is now paused as stated
above, and external scheduler release remains a separate
user decision. Every native successor must
recheck these operational controls without changing its admitted robot source,
servo, physics or quality gates.

**Historical handoff pause, 11 September 2026:**
`/home/orionh/HEXAPOD_runs/canonical_direct_20260910/james_handoff_pause_001/PAUSED.json`
records the user pause. At that handoff, Codex continuation was PAUSED, the old
`hexapod-restore-forecasting-20260909.timer` was stopped with its transient units
absent, and no research job was queued. Preserve those receipts and recheck the
current continuation, restore timers and producer state before a fresh run.
Read [JAMES_HANDOFF](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/62fd7448264c5ebe051ba2de1d9a72844d6b4c3a/docs/JAMES_HANDOFF.md)
before using historical launchers.

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

## 10. Completed optimizer replay, 16 September 2026

The successor ran `hexapod-trajectory-replay-001-20260916.service` with a fresh
[source and launch binding](../artifacts/trajectory_optimizer_20260917/native_pack_002/PACK.json).
The native replay completed 1,000 controls. The
[terminal readback](../artifacts/trajectory_optimizer_20260917/native_pack_002/REMOTE_INVENTORY.json)
records PID 0, exit code 0 and no active containers or GPU compute apps.
The [cleanup receipt](../artifacts/trajectory_optimizer_20260917/replay_001/cleanup.json)
confirms removal of the exact container and retention of the reservation,
including 32 user masks and four system masks. This receipt records the end of
that allocation; it does not release the Spark reservation or describe live host
state after its timestamp.
