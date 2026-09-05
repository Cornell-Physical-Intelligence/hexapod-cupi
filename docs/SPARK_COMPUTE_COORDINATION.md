# Spark coordination: hexapod and a second agent

Updated 2026-09-05 UTC (4 September locally). **Current user instruction:
hexapod may use full available compute until the other agent requests sharing
through this file.** This supersedes the earlier immediate 60/40 planning
target. There is no GPU quota installed. If sharing is requested, 60% hexapod /
40% other work remains the earlier preferred starting point for a measured
concurrent setup; it is not active by default.

Shared copy: `/home/orionh/SPARK_COMPUTE_COORDINATION.md` on
`orionh@100.82.166.9` (`spark-e26c`). Repository copy:
`docs/SPARK_COMPUTE_COORDINATION.md`, branch `codex/mkii-simulation-integrity`.
Do not put credentials or secret environment contents into this note.

## Timing guide

| Hexapod work | Measured full-compute wall time | Estimate if 60% sharing is later requested |
|---|---|---|
| CPU model/code tests | 18 seconds for the current 672-test suite on the development Mac | Seconds to a minute; does not need the GPU |
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
   below and explicitly write **SHARING REQUESTED**, with a UTC timestamp.
   This file is the agreed coordination channel. The hexapod agent should
   acknowledge the request here and coordinate the next checkpoint/pilot;
   another user confirmation of the earlier 60/40 preference is unnecessary.
3. Current short hexapod acceptance runs still use the exclusive shared lock.
   Their launcher deliberately refuses unrelated CUDA workloads. **A shared
   long-training launcher has not yet been implemented.** Do not bypass the
   lock or claim the 60/40 allocation is already enforced. Update both
   launchers around the agreed paired pilot before relying on concurrency.
4. Never stop the other agent's job. Cleanup must use the immutable container
   ID owned by the current run. Record changes to sharing mode here so both
   agents see the same state. Do not start a legacy hexapod service: it points
   at an archived checkpoint lineage.

Hexapod source for live validation:
`/home/orionh/HEXAPOD_runs/mkii_v2_step2_a0f0b39/source/`.
Run reports/logs are in the sibling `runs/` directory. Use the `supervisor.json`
for a run's actual state; directory presence alone does not mean it is active.
The original `/home/orionh/HEXAPOD` mirror is not a Git checkout.

## Handoff / active reservations

- Hexapod: short corrected-serial standing validation and linkage-model
  preparation. No long training campaign or persistent GPU reservation has
  been started by this work. Full available compute is authorized until a
  sharing request is recorded below.
- **Sharing request status: NONE RECORDED.** Other agent: add your workload,
  memory requirement, preferred overlap window, UTC timestamp and
  **SHARING REQUESTED** here. Do not launch a competing GPU job before the
  handoff is acknowledged or a paired sharing pilot is agreed.
- MPS / shared launcher: not enabled by this task. Current resource checks
  remain exclusive for short acceptance jobs.

## Physical four-bar run coordination (2026-09-05 UTC)

HEXAPOD_SHARE_STATUS=NONE

For the new guarded physical-model runner, change that single line to
`HEXAPOD_SHARE_STATUS=REQUESTED` and add your workload details below. Any change
to the shared file during training requests a checkpoint and pause at the next
PPO iteration; the supervisor allows at most 120 seconds before stopping only its
owned container. GPU contention causes an immediate owned-job stop. No MPS
allocation is installed. Keep the earlier 60/40 preference for a coordinated pilot.

The new source is `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/source`, branch
`codex/mkii-fourbar-training`. Logs/checkpoints are in the sibling `runs` directory.
`supervisor.json` is authoritative for each run's state. Training remains gated on
physical-model standing, driven-coordinate and solver-convergence checks.

## 5 September UTC priority update

The user explicitly gave hexapod priority over current Spark occupancy. After the initial weather run exited, another weather GPU run (PID 1218868, historical-training-20230127-0730-v1) was stopped with SIGTERM; its output files were retained. CPU scoring was left running. The hexapod reservation guard acquires the existing `/opt/wx/gpu.lock` and releases it when its owned campaign exits or after three hours. This reserves the cooperative weather GPU slot; it is not hardware partitioning. Live shared-file checkpoint/pause requests remain supported.

Campaign: `/home/orionh/HEXAPOD_runs/mkii_fourbar_v1/campaigns/fourbar-campaign-20260905T034653Z-a83056cd/campaign.json`. Source is the sibling `source` directory, archived and hashed separately per phase. Check live state rather than assuming recorded PIDs remain active.
