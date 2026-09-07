# Bounded full PPO continuation v1

This is a separately versioned host coordinator for the exact c2af43c 1600 Hz campaign. It changes no simulation, training, admission, host timeout or acceptance gate. The code and its 24 focused CPU tests passed independent review; `VALIDATION.json` records this author's hash-bound run, and `../full_ppo_continuation_review_v1/` contains independent evidence. These tests exercise mocked process/evidence transitions, not real PPO learning or policy quality. Deployment and any later outcomes must be recorded separately.

## Exact binding and operation

`deployment_config.json` binds source commit `c2af43ca0f384a4c2c7ab8f1d627f309dc78a683`, functional SHA `c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc`, and the original manifest SHA `8d6ad0b053e2b2a73e6610a51871443efdeb1234d7c44165dbf27a67498e66fb`. It also binds the campaign, wrapper, original video follower, external tool manifests, and exact process identities. The imported physics timestep must be 0.000625 seconds, decimation32, policy timestep0.02 seconds. Source and dependency hashes are verified with stdlib before importing their code.

The coordinator waits without a GPU reservation. If the original campaign completes its qualified 512-world, 1,000-update full phase, it verifies final campaign/supervisor/checkpoint evidence and exits without dispatching a continuation or capture; the existing follower owns that capture.

If completed-update timings demonstrate the original full phase cannot fit its unchanged 21,600-second host deadline, the coordinator can create only that exact owned container's `stop_requested` file. It first verifies supervisor PID/start ticks/boot/argv and the supervisor-recorded container ID/name/owner. A request is allowed only inside a margin of 300 seconds plus twice the measured update duration, while measured completion ETA exceeds remaining job time. The ETA includes the final100-control inference probe. Insufficient timing or an ETA that fits causes no action.

The conservative job clock starts at `/proc` start ticks on the same Linux boot, before the frozen host's actual timeout clock. Overall execution is capped at18hours from coordinator startup, including waiting for physical qualification, training, cleanup and capture. It uses persisted CLOCK_BOOTTIME, rejects a different boot, and does not extend bounds when wall-clock time changes.

After an owned pause, it requires the successful paused report, exact container cleanup, the campaign's matching paused terminal record, and the existing video follower's `no_video` state and exit without a capture reservation. Then it can start at most two further training segments: three full-training segments total, counting the original. Each uses the unchanged frozen host with the original admission bytes and last exact checkpoint. The requested chunk size is bounded from measured throughput, preserving startup/checkpoint/inference/capture reserves and a15percent throughput margin. Each actual job uses `/usr/bin/flock --nonblock --no-fork /opt/wx/gpu.lock`; busy admission fails closed, and the frozen host retains its own coordination and ownership checks.

Every segment retains its real requested/completed counts, report/checkpoint/sidecar/supervisor hashes, start/next iterations, and policy/algorithm-state digests. A partial400-of1000 original plus a complete600-of600 resume means1,000 full updates, not1,600. The three scratch updates are excluded: final next_iteration must be1003. Optimizer, actor, critic, normalizer and adaptive-learning-rate state continue through the frozen trainer's guarded checkpoint loading. Environment episodes and RNG trajectories restart between processes; this is learning continuation, not bitwise uninterrupted simulation.

Only an exactly1,000-update chain with a final unpaused finite100-control inference probe can create `chain_attestation.json` and start one guarded15-second1280×720 policy capture. Original partial reports are never rewritten as full completion. The capture host retains its1800-second render timeout, while the coordinator bounds the supervisor to2160seconds and verifies final video evidence. A successful training/capture artifact is not terrain or hardware qualification.

## Failure and restart behavior

Source/admission/control changes, physical failures, report or checkpoint mismatches, missing exact cleanup, a live original video follower, duplicate capture reservations, count overshoot, exhausted segment/time budget, or unrelated/reused PID/container identities reject automatically. Code/config/binding changes cannot reuse the same state directory. A launch reservation is durable before Popen; the spawned PID is saved immediately before waiting for its post-exec identity. An ambiguous reservation is terminal and never automatically retried. A fully bound active training/capture process can be monitored after coordinator restart without launching another.

A pause request can race with the last requested update. If the original reaches update1000 but produces a paused report without final inference, this version fails closed: it neither adds update1001 nor fabricates an inference pass. That rare case requires a separately reviewed inference-completion procedure. Abrupt process/container failure can also leave only a periodic checkpoint and is not automatically recovered by this coordinator. An interruption forwards a signal only to an exact coordinator-owned child; unrelated processes and the original campaign are not killed.

## Deployment procedure

Stage the manifest-listed files into the new external directory `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/continuation_tools_v1`. Do not place them inside the frozen source. Verify `SHA256SUMS` and the expected manifest hash from the review record before executing. Existing tool directories are `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/followup_tools_v2` and `.../policy_capture_tools_v2`; the config pins their bytes.

Exact Linux CPU-only preflight (no state directory creation, no GPU job, no stop request):

```sh
/usr/bin/python3 -B /home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/continuation_tools_v1/continuation.py --config /home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/continuation_tools_v1/deployment_config.json --state-dir /home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/continuation_001 --preflight-only
```

After explicit deployment authorization and successful unchanged-source preflight, use the identical command without `--preflight-only` as one detached CPU process, with stdin closed and stdout/stderr directed to `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/continuation_001.launcher.log`. Record PID/start ticks/boot/argv, bound initial state, and confirm its own descriptors hold no GPU lock. Do not relaunch into a new state directory after ambiguity. No app automation, fixed5-minute progress event or persistent GPU reservation is used.

Local focused validation command from the repository worktree:

```sh
.venv/bin/python -m unittest discover -s artifacts/mkii_fourbar_2026-09-06/full_ppo_continuation_v1 -p test_continuation.py -v
```

The tests import the reviewed sibling `policy_capture_tools_v2` and `campaign_capture_followup_v2` artifacts. The prepared remote folder names differ; do not infer standalone remote test portability from the successful local suite. The production CLI imports only its explicitly configured source/tools and requires Linux `/proc`, CLOCK_BOOTTIME, pidfd and flock support.
