# Incident 2026-08-26: pre-AppReady Isaac/Kit startup stalls

**Status: frozen.** This is a forensic record of what was observed on
2026-08-26 and earlier. It is append-only: correct it by appending a dated
follow-up section, never by rewriting the record above. Current state belongs
in `STATUS.md`; the runbook and the unimplemented supervisor specification
belong in `docs/OPERATIONS.md`.

Source material: `docs/archive/HANDOFF-2026-08-26.md` sections 4 and 5, and
`artifacts/phase2_recovery_stage2c_stable_forward/probes/README.md`.

## 1. Failure class

A container starts, `train.py` emits its stock deprecation warning, and then
nothing further happens. The run never prints `Loading user config`, never
prints the Kit clock, and never prints
`[ISAACLAB] AppLauncher initialization complete`.

The stall is therefore before AppLauncher, before user config, before scene
allocation, before checkpoint load, before reward evaluation, and before CUDA
context creation.

Signature observed on the stalled attempts:

- Python asleep in `futex_do_wait`.
- CPU near zero.
- No GPU context for the process.
- No NVRM XID, no OOM, and no segfault found.
- Truncated launcher log; for Probe21 attempt 0 the log was 706 bytes and ended
  after container creation and the deprecation warning.

Successful startups for comparison reach `Loading user config` at about `+2 s`
and complete AppLauncher at about `+12 s`.

## 2. Affected attempts

Four attempts reached this same boundary:

| Attempt | Label | Outcome |
| --- | --- | --- |
| Probe19 | `accel_probe19_yaw_slew4_ref040_slew040_20260826T132100Z` | Pre-AppReady timeout; no run, no checkpoint; the yaw-rate-slew arm remains untested |
| Probe20 attempt 0 | `accel_probe20_matched_zero_seed99_slew040_20260826T152600Z` | Timed out before AppReady; container inspection, empty-GPU evidence, launcher metadata, and logs preserved separately from the successful retry; produced no candidate |
| Probe21 full attempt 0 | `accel_probe21_bilateral_long2_ref1800_seed99_slew040_20260826T172645Z` | Pre-AppReady timeout; cleanup parser refused with `identity-mismatch`; container stopped by hand after its artifact was preserved |
| Probe21 full attempt 1 | `accel_probe21_bilateral_long2_ref1800_seed99_slew040_retry1_20260826T174200Z` | Pre-AppReady stall; signaled after an unrelated GPU workload appeared; fixed exact-ID cleanup succeeded |

### Probe21 attempt 0 detail

```text
label: accel_probe21_bilateral_long2_ref1800_seed99_slew040_20260826T172645Z
remote artifact: /home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c/<label>
local mirror: artifacts/phase2_recovery_stage2c_stable_forward/probes/<label>
started:  2026-08-26T17:26:58Z
finished: 2026-08-26T17:33:58Z
overall/docker/train exit codes: 124 / 124 / 124
checkpoint/run directory: none
```

The 420-second watchdog fired as designed.

### Probe21 attempt 1 detail

```text
label: accel_probe21_bilateral_long2_ref1800_seed99_slew040_retry1_20260826T174200Z
started:          2026-08-26T17:42:55Z
signal requested: 2026-08-26T17:45:02Z
cleanup finished: 2026-08-26T17:45:32Z
overall/docker/train exit codes: 143 / 130 / 130
cleanup: stopped, stop_status=0, postcheck=absent
checkpoint/run directory: none
```

Only the hexapod launcher was signaled. The fixed cleanup validated the
immutable container ID, stopped the container, proved it absent, and touched no
unrelated process.

Both Probe21 attempts are infrastructure failures, not negative policy results.
The bilateral longitudinal contact-moment arm is still scientifically untested.

## 3. What the successful identical retry rules out

An identical Probe20 seed and configuration was retried and succeeded:

```text
batch: /home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c/
       accel_probe20_matched_zero_seed99_slew040_retry1_20260826T160100Z
```

It reached `Loading user config` at about `+2 s`, completed AppLauncher at about
`+12 s`, and finished all 12 updates and `3,538,944` samples in `45.67 s`.

Because the retry differed from the stalled attempt in nothing but time, the
12,288-environment count and the reward configuration are not credible direct
causes of the startup stall. The stall is intermittent and environmental.

Successful logs also contain OmniHub reconnect messages and GB10/PyTorch
capability warnings. Those warnings appear in runs that train normally, so they
are not diagnostic on their own.

The likely failure class is an intermittent Carbonite/Kit/import-entry deadlock
or a runtime/cache/IPC race. Shared writable Kit/OV caches are a plausible
lead, not proof. No cache or Docker volume was deleted, and none should be
deleted on this hypothesis alone.

The old Docker healthcheck searches a questionable Kit-log tree and cannot be
trusted as an AppReady signal until it is validated inside a successful live
container.

## 4. Cleanup parser bug and fix

The cleanup path in the hardened launchers built a Docker Go template that
contained a literal `\t`. Docker emitted literal backslash-t sequences rather
than tab characters, Bash did not split the identity fields, and the launcher
could not match the container it had created. Behaving fail-closed, it reported
`identity-mismatch` and refused to act. That refusal was correct: it prevented
cleanup from acting on an identity it could not verify. The consequence was
that the exact known stalled container from Probe21 attempt 0 had to be stopped
by hand, after its artifact was preserved.

Fix: the exact-container cleanup in both hardened launchers now uses `|`
delimiters instead of `\t`, and Docker's current `--timeout 30` option. The
change was synced to the Spark, passed the test suite, and was validated in vivo
during Probe21 attempt 1, where cleanup stopped the container and proved it
absent.

Hashes recorded at the time of the fix:

```text
probe launcher SHA:       5f69625d7638e538f2c6f48c2907c9b42c61ac0329afa340362f1cfc1c5cc506
calibration launcher SHA: 810a4ed436f935f280043a8a54a0f2ae6e6af8a72e647af0dff632ca9b8c5ece
manifest SHA:             c104eb22024077f8c875b2d8a632d437ff47bb89c2c7da8999570de5bdec99c2
```

## 5. NSVA GPU-contention race

Probe21 attempt 1 was interrupted by a second, separate problem. The prelaunch
resource check was clean. After the container had been created, an unrelated
root-owned workload claimed the GPU:

```text
PID 217667  /root/cvnba/venv/bin/python sq/validate_nsva.py data/nsva \
  --out results/nsva_validation.json
ancestor: PID 215354 bash /root/nsva_dl.sh
GPU at the audit: about 76-82%, 1366 MiB
```

The producer script `/root/nsva_dl.sh` existed before the preflight and spawned
its GPU child afterward. A one-shot prelaunch gate cannot catch that ordering.
The race is closed only by gating on producer scripts and their descendants and
by rechecking contention after container creation and again before PPO starts,
which is item 2 and item 3 of the supervisor specification in
`docs/OPERATIONS.md` §6.

Nothing belonging to the unrelated workload was signaled, stopped, or modified.

## 6. Related but distinct failure, same date

The first Probe21 p50-telemetry calibration attempt
(`accel_probe21_bilateral_cal_ref050_telemetry_seed99_20260826T170241Z`) failed
for an unrelated reason: a reset-batch indexing shape mismatch in
`isaaclab/hexapod_rl/env.py`. That is a code fault with a traceback, not a
pre-AppReady stall, and it is explicitly outside this incident's class. The
helper was corrected to consume already-selected aligned tensors, the test suite
passed afterward, and the failed attempt's traceback was preserved. The
corrected retry and the calibrated-reference collection both completed.

## 7. Evidence preservation

Every attempt directory listed here is preserved, locally and remotely, and is
append-only. The probe ledger at
`artifacts/phase2_recovery_stage2c_stable_forward/probes/README.md` records each
attempt's disposition. Do not delete, rewrite, or reuse a failed attempt's
label.
