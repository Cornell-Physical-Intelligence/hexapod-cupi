> Current policy recovered 2026-09-06: the authoritative shared file records a
> 13:33 UTC user pause for HEXAPOD pending explicit resume and prohibits long-lived
> exclusive reservations. `HEXAPOD_SHARE_STATUS=REQUESTED` is active. Earlier full
> takeover notes below are historical. Do not clear that control or create a new
> lease from these old instructions. Normal per-job collision locks remain required.

# Spark coordination: hexapod and a second agent

Updated 2026-09-05 UTC. **Current user instruction: give the hexapod campaign
priority and use full available compute.** The shared-file checkpoint/pause
handshake remains available for another agent's request. This supersedes the earlier immediate 60/40 planning
target. There is no GPU quota installed. If sharing is requested, 60% hexapod /
40% other work remains the earlier preferred starting point for a measured
concurrent setup; it is not active by default.

Shared copy: `/home/orionh/SPARK_COMPUTE_COORDINATION.md` on
`orionh@100.82.166.9` (`spark-e26c`). Repository copy:
`docs/SPARK_COMPUTE_COORDINATION.md`, branch `codex/mkii-fourbar-training`.
Do not put credentials or secret environment contents into this note.

## Historical serial timing guide

| Hexapod work | Measured full-compute wall time | Estimate if 60% sharing is later requested |
|---|---|---|
| CPU model/code tests | 18 seconds for the earlier 672-test suite on the development Mac | Seconds to a minute; does not need the GPU |
| Standing check: 32 environments, 1,000 steps | 53 seconds in each of three Spark runs | About 1–3 minutes |
| Nine-command evaluation, 60 simulated seconds | 1 minute 44 seconds to 4 minutes 43 seconds | About 3–8 minutes |
| Training: 4,096 environments, 500 iterations | 27 minutes 25 seconds to 29 minutes 32 seconds | About 40–55 minutes |
| Training: 4,096 environments, 1,500 iterations | 65 minutes 13 seconds | About 1.5–2 hours |
| Multiple seeds, terrain curricula, realistic camera/LiDAR training | Not yet measured for the corrected physical model | Several hours to overnight; estimate after profiling each stage |

The historical training rows used the earlier serial model. The current
corrected serial-v2 standing check, with all 4,000 physics substeps sampled for
32 environments / 1,000 control steps, passed in **75.18 seconds** between the
first and last container log timestamps (`standing_003` in the step-2 bundle).
This remains a partial model gate, not physical four-bar or training admission.

The current physical campaign integrates at **800 Hz with a 50 Hz policy**.
Each complete nominal/refined check has 1,000 standing plus 2,400 driven control
steps, or 54,400 physics substeps per environment across 32 environments. These
checks also inspect every substep's linkage, motor and contact state. Their
timing and new PPO throughput must be measured separately; the serial training
times above are not an ETA for this pipeline. Current local source tests:
797 passed in 24.557 seconds.

The sharing column is an estimate, not a benchmark or the active allocation. For a first estimate use
`startup + measured_learning_time / 0.6`, then allow for contention. CPU-bound
work, memory bandwidth and GPU bursts prevent exact inverse scaling. The
500-iteration runs spent 18.6–20.8 minutes actually training; large-environment
startup took roughly 7.6 minutes. Avoid repeatedly rebuilding thousands of
environments for tiny experiments. Run a small probe before a large job.

Sources on Spark, all under `/home/orionh/HEXAPOD/tmp/train/`:

- `run_phase1v5_seed52.log`, `run_phase1v5_lowB_seed52.log`,
  `run_lowA_s52.log`, `run_lowB_s53.log`: four 500-iteration runs.
- `run_lowB_long_s52.log`: 1,500 iterations, 3,396.1 seconds learning and
  3,913 seconds end to end.
- `validate_lowA_s52.run.log`, `validate_lowB.run.log`,
  `validate_crouch_s52.run.log`: 32-environment standing checks.
- `eval_*/accept.run.log`: command-screen timings.

These used the **old serial USD**, 0.005-second physics, decimation four and
24 rollout steps. Their performance is useful for budgeting; their policies
are not admitted for the repaired robot. A true 31-body four-bar model,
perception rendering and shared operation still need separate measurements.

## Full compute now; sharing when requested

There is one GB10 GPU and approximately 121 GiB of usable unified system
memory. NVIDIA-SMI reports MIG mode and dedicated GPU memory usage as N/A on
this host. CPU and GPU allocations share memory; monitor `free -h`, process
memory and application allocation metrics, not just the GPU memory column.
See NVIDIA's [Spark system overview](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html).

Halving environment count does **not** enforce half the GPU, and Docker CPU
quotas do not limit GPU compute. Start with one headless hexapod learner,
modest environment counts and no unnecessary rendering. Profile before
increasing batch size. Keep at least 16 GiB host memory available as an initial
guardrail; revise the budgets based on both workloads' measured peaks.

CUDA MPS can constrain a participating CUDA client's active thread share,
for example a candidate 60%/40% allocation. The MPS binary exists on this host,
but this workflow has **not configured or validated MPS**. An environment
variable alone does not enable the service, and MPS does not reserve dedicated
resources or control all of Isaac's graphics work. Both containers must join
the same deliberately configured service; memory budgets need separate
attention. NVIDIA documents the limits in
[MPS resource provisioning](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html).

Before enabling concurrent long runs, do a short paired pilot: measure each
workload alone and together, log throughput/latency, available memory and GPU
utilization, and verify checkpoint/shutdown behavior. Choose a tested MPS
configuration if compatible; otherwise use application throttling or agreed
time windows. Do not change GPU clocks or power limits as a supposed per-job
quota because that would affect both projects.

## Coordination procedure for the other agent

1. Read this note before each launch and at checkpoint boundaries during long
   runs. Full available compute is currently assigned to hexapod until a
   sharing request appears below. Inspect `nvidia-smi`, `docker ps`, `free -h`, and
   `/tmp/hexapod-isaac-gpu.lock` before launching a GPU job. CPU-only work can
   proceed alongside the hexapod run if memory remains available.
2. Record your workload type, expected memory, expected duration, launch
   command/container identity and preferred window in the handoff section
   below with a UTC timestamp, and change the single canonical control line
   from `HEXAPOD_SHARE_STATUS=NONE` to `HEXAPOD_SHARE_STATUS=REQUESTED`.
   This file is the agreed coordination channel. The hexapod agent should
   acknowledge the request here and coordinate the next checkpoint/pilot;
   another user confirmation of the earlier 60/40 preference is unnecessary.
3. Current physical validation and training runs use exclusive admission.
   Their launcher deliberately refuses unrelated CUDA workloads. **Concurrent
   long training has not been qualified.** Do not bypass the
   lock or claim the 60/40 allocation is already enforced. Update both
   launchers around the agreed paired pilot before relying on concurrency.
4. Do not stop another workload without explicit user authorization. The user's
   subsequent force-priority instruction authorized the two recorded weather
   process stops below; this is not a standing instruction to kill arbitrary
   processes. Normal cleanup uses the immutable container ID owned by the run.
   Record changes to sharing mode here so both
   agents see the same state. Do not start a legacy hexapod service: it points
   at an archived checkpoint lineage.

Hexapod source for live physical validation/training:
`/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source/`.
Campaign reports/logs are in the sibling `campaigns/` directory; standalone
diagnostics use `runs/`. Use the `campaign.json` and per-phase `supervisor.json`
for a run's actual state; directory presence alone does not mean it is active.
The original `/home/orionh/HEXAPOD` mirror is not a Git checkout.

## Handoff / active reservations

- Hexapod: physical four-bar validation-to-PPO campaign
  `fourbar-campaign-20260905T044533Z-5465207c`, frozen source `ea05fe8`.
  Read its live `campaign.json` for the active phase and outcome.
- Cooperative weather-scheduler lock `/opt/wx/gpu.lock` is held by the owned
  campaign reservation. Live status:
  `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/priority_campaign_004.json`.
  At 04:56:24 UTC the new guard PID 1333865 took over from PID 1232035 after
  identity verification. It releases on that campaign's completion, failure,
  pause or process exit, with a hard maximum of 14:56:24 UTC. It does not launch
  GPU work or signal any workload. Check status; recorded PIDs can become stale.
- **Sharing request status: read the canonical control line below.** Other
  agent: add your workload, memory requirement, preferred window and UTC
  timestamp, then set that line to `HEXAPOD_SHARE_STATUS=REQUESTED`.
  Do not launch a competing GPU job before the
  handoff is acknowledged or a paired sharing pilot is agreed.
- MPS / shared launcher: not enabled by this task. Current resource checks
  remain exclusive for physical validation and training.

## Physical four-bar run coordination (2026-09-05 UTC)

HEXAPOD_SHARE_STATUS=NONE

For the new guarded physical-model runner, change that single line to
`HEXAPOD_SHARE_STATUS=REQUESTED` and add your workload details below. The
`canonical_share_status_v2` supervisor reacts to that explicit control state;
ordinary prose, timestamps and status updates do not request a pause. Keep exactly
one unindented control line with no extra spaces or inline comments. Missing,
duplicate, malformed or unreadable control state blocks initial admission and
is checked again before releasing the CPU-only launch barrier.

During training, an explicit request or invalid control state creates the existing
checkpoint/pause marker and allows 120 seconds of grace for the next PPO iteration
to save and exit. Validation and diagnostics yield immediately. Once observed,
the request remains latched for that run even if the note later returns to `NONE`;
only a separately admitted run may consume the cleared state. GPU contention still
causes an immediate owned-job stop. Use an atomic file replacement when changing
the control line, so a partial rewrite cannot look like missing control state.

The supervisor records the canonical semantic token separately from the raw
initial-note SHA256 and later note snapshots. The raw hash is provenance, not a
pause trigger. Archived supervisors retain their historical behavior; this
protocol applies only to newly versioned runs. No MPS allocation is installed.
Keep the earlier 60/40 preference for a coordinated pilot.

The new source is `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source`, branch
`codex/mkii-fourbar-training`. Campaign logs/checkpoints are in the sibling
`campaigns` directory; standalone diagnostics use `runs`.
`supervisor.json` is authoritative for each run's state. Training remains gated on
physical-model standing, driven-coordinate and solver-convergence checks.

## Historical 5 September UTC priority update

The user explicitly gave hexapod priority over current Spark occupancy. After the initial weather run exited, another weather GPU run (PID 1218868, historical-training-20230127-0730-v1) was stopped with SIGTERM; its output files were retained. CPU scoring was left running. The hexapod reservation guard acquires the existing `/opt/wx/gpu.lock` and releases it when its owned campaign exits or after three hours. This reserves the cooperative weather GPU slot; it is not hardware partitioning. Live shared-file checkpoint/pause requests remain supported.

Campaign: `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/campaigns/fourbar-campaign-20260905T034653Z-a83056cd/campaign.json`. Source is the sibling `source` directory, archived and hashed separately per phase. Check live state rather than assuming recorded PIDs remain active.

That first guard/campaign has ended. A second identified weather nowcast process,
PID 1231813, was also stopped with SIGTERM under the same explicit priority
instruction; outputs and unrelated CPU work were retained. The current bounded
reservation is recorded above and in its own JSON, outside the frozen source.
Under that historical supervisor, any byte change requested a checkpoint/pause,
so the remote note was not rewritten during an active phase. The canonical
control protocol above supersedes that behavior for new runs; preserve explicit
requests when publishing a refreshed note.

## Renewed priority at2026-09-05 07:02 UTC

User reaffirmed hexapod as top priority over weather. Current guardPID1407273 holds the weather scheduler lock; maximum expiry17:02:04UTC. Status: `/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/priority_20260905T0700/status.json`. Create sibling `release` to release explicitly; sibling `campaign_path.txt` selects the exact following campaign and causes automatic release on terminal state. The old1364926 guard has ended; queued weather wrapper1394883 was cancelled during handover before any GPU work started. Other work may use the existing shared-file checkpoint/pause handshake. Read the live remote file and process state rather than treating this snapshot as current indefinitely.

## 2026-09-05 17:48 UTC reservation and physical campaign update

The former 07:02 guard was handed over at16:16 UTC. Active guard **1465763** holds `/opt/wx/gpu.lock`; its status is `/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/priority_20260905T1616/status.json`, with a hard maximum expiry **2026-09-06T02:16:34Z**. Creating the sibling `release` file requests release. This is cooperative exclusive priority, not an MPS quota.

The v5 full nominal validation completed but failed; no PPO is running. A bounded eight-environment v5 group diagnostic is using the GPU. Current source: `/home/orionh/HEXAPOD_runs/mkii_fourbar_physical_mimic_v1/source`; reports in sibling `campaigns/` and `diagnostics/`. Source commitcfe0cf5 is frozen. Use the shared status line to request a yield; the owned supervisor checks changes. Inspect live state before acting because this note is a dated snapshot.
