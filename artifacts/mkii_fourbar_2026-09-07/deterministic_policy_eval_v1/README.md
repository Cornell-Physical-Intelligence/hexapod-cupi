# Deterministic flat-policy evaluation, prepared only

This external harness measures what a real trained policy does under a fixed command schedule. No GPU evaluation has been launched, no checkpoint has been created, and this directory contains no learned-policy result. Runtime source, reward, physics, action limiter, model and existing admission gates remain unchanged.

The evaluator requires functional source identity `c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc` and reuses the immutable v2 capture input checks by loading `capture_common.py` only after verifying SHA-256 `ff60b5227ea8fbc2f3fc3ffe5091f0691ccff13cc1dab8e85063bb48d24df993`. The original helper verifies the physical admission, completed unpaused training report, checkpoint bytes/sidecar, source/runtime lineage and exact trained-state digests. The evaluator verifies the actual loaded runtime and restores the exact actor, critic, optimizer, normalization and adaptive learning-rate state; only deterministic inference follows. No checkpoint is saved.

## Cases and timing

Twelve independent worlds use the same admitted flat model and named 18-motor action mapping. Each receives 50 zero-command settle controls (1 s), 150 motion controls (3 s), 100 zero-command stop controls (2 s), then 150 controls with the negative motion command (3 s): 450 controls, 9 simulated seconds, below the task's 20-second time limit.

| World | Motion command (forward m/s, left m/s, yaw rad/s) |
| --- | --- |
| stand | 0, 0, 0 |
| forward / backward | ±0.15, 0, 0 |
| left / right | 0, ±0.15, 0 |
| four diagonals | ±0.10, ±0.10, 0, covering every sign pair |
| yaw left / right | 0, 0, ±0.30 |
| combined | 0.10, 0.05, 0.20 |

Every value is within the existing training command range. The settle/stop phases still use the learned deterministic policy with a zero command; no scripted stand or gait is imposed. The original sampler is temporarily replaced on this environment instance, including its reset callback. Before every action, the scheduled command is written and the frozen wrapper refreshes observations; the exact command must equal the observation's command slice. This prevents using a stale observation or letting a reset silently substitute a random command. All temporary hooks are restored on exit.

## What is measured

The frozen `PhysicalTrainingGuard` still measures every 1600 Hz physics substep before any reset: 14,400 global substeps for a completed evaluation. An additional external hook calls the original `_get_dones`, then copies final physical state before Isaac Lab performs automatic reset. Raw rows distinguish `pre_` (before action), `end_` (final physics state before reset) and `post_` (after transition/reset). Repeated observations retain the source's previous-clipped-action semantics.

`states.npz` stores the actual policy observation/action, scheduled command, plate position/XYZW orientation, root COM linear velocity and body angular velocity in anatomical coordinates, active joint position/velocity/acceleration, applied/raw torque, clipping, burst headroom, pad/shaft support and slip speed, reward, done/termination/timeout flags, source termination reasons, and post-reset pose/target. Thus the measured velocity uses the same actual COM source as the training reward, not a plate-position finite difference. COM and plate motion are both retained. A failure after completed rows preserves those rows as `states_partial.npz`, with its hash in the failed report.

`metrics.json` gives each case and each phase its commanded/achieved velocity, tracking RMSE/MAE/95th-percentile error, valid-transition plate displacement/path distance, support/slip, torque, clipping, acceleration and minimum budget. Done transitions are excluded from these tracking/path summaries and explicitly counted; their true terminal data remain in the raw archive. Later transitions after a reset are included and labeled by the fixed global schedule. Read reset counts together with tracking: repeated falls must not be mistaken for successful traversal. The 0.01 m/s near-stationary statistic is descriptive; stand and pure-yaw cases intentionally may have zero translation. Stop distance is the sum over valid transitions during the two-second stop window; it is not an independently qualified stopping-distance guarantee.

`report.json` and `supervisor.json` use `pass` only for execution and integrity. `policy_skill_pass` remains null, and hardware/navigation/terrain admission remains false. No numerical walking-success threshold is invented. High reward cannot replace command-conditioned motion, survival and contact evidence. One short deterministic schedule and seed cannot establish terrain robustness, sustained thermal capability or survey accuracy.

## Host ownership and lifecycle

`run_evaluation.py` is a headless, non-rendering supervisor adapted from the reviewed external capture supervisor. It holds `/opt/wx/gpu.lock` through an O_RDONLY descriptor and the project's `/tmp/hexapod-isaac-gpu.lock` on FD 9 for this actual job only. It verifies the source's coordination/resource gates, read-only source/checkpoint/helper/tool mounts, CPU admission barrier, immutable container ID and ownership label, no-restart policy and final artifact hashes. Cleanup targets only its exact container, and both locks remain held until cleanup completes. The bounded timeout is 60–1800 seconds; default 1800. Coordination requests stop this evaluation rather than affecting another workload.

The full original argv, selected source/helper/tool hashes, request, report, source manifest and container log are retained. All NPZ/JSON writes and the final report occur inside the simulation context before `env.close`, because native Kit shutdown may exit the process. Host verification happens after exact-container cleanup. `--dry-run` validates the real immutable inputs and emits only metadata; it does not acquire locks, create an output directory, start a container or reserve compute.

Deployment and GPU execution were not part of preparing this artifact. When authorized for dispatch, use the immutable source, an actual completed checkpoint/admission/training-report trio, and the exact frozen v2 helper directory:

```sh
python3 run_evaluation.py \
  --source-dir /home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/source \
  --capture-tools-dir /path/to/frozen/policy_capture_tools_v2 \
  --checkpoint /path/to/completed/checkpoint.pt \
  --admission /path/to/admission.json \
  --training-report /path/to/completed/report.json \
  --output-root /path/to/new/evaluation_outputs \
  --timeout-seconds 1800 --dry-run
```

Remove `--dry-run` only for the separately dispatched actual job. A matching completed last segment of an explicitly verified training continuation can provide the input report; this evaluator itself does not prove how many earlier training segments ran.

## CPU verification

From the repository root:

```sh
.venv/bin/python -m unittest discover \
  -s artifacts/mkii_fourbar_2026-09-07/deterministic_policy_eval_v1 \
  -p test_evaluation.py -v
```

Fourteen tests pass: phase boundaries/ranges/signs; anatomical frame conversion; command refresh and reset callback; pre-reset hook ordering; terminal exclusion and stationary-policy detection; stale observation/schedule rejection; frozen-helper/source/input/report checks; artifact corruption rejection; headless read-only argv; metadata-only dry run; mocked successful and failed-report lifecycle; exact-container cleanup; coordination stop and both lock releases. These tests use temporary, clearly labeled CPU fixtures and do not create a policy result. An independent agent reviewed command/observation/reset hooks against the captured installed SDK and found no concrete timing defect. Native CUDA execution remains untested for this new harness.
