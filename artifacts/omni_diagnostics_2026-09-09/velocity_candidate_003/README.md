# C-study candidate003: standing and real PPO runner admitted

The target-velocity candidate passed fresh 32 × 1000-control flat standing,
32-replica zero-mean calibration, 32-replica sampled exploration at standard
deviation 0.005 for 20 seconds, and two actual stand-only PPO updates with exact
checkpoint action readback. This admits the bounded next pilot; it does not
establish learned walking or Stage 2 completion.

The minimal correction initializes the installed RSL logger's absent local
writer sentinel before an early checkpoint save. The other five candidate
executables, physical setup, exploration, rewards and gates remain unchanged.
The real installed RSL 5.0.1 CPU regression first reproduced the prior exception,
then passed actual learning, save/reload, optimizer readback and wrong-source/hash
rejection. Its deliberately corrupted checkpoint and old early-save checkpoint
are preserved as negative-test evidence, not candidate policies.

- [Original calibration and all replicas](results/run/probe/calibration.json), [actual PPO smoke](results/run/probe/runner_smoke.json), [fresh flat admission](results/run/flat/admission.json).
- [Real-RSL CPU report](results/real_rsl_cpu/attempt_001/report.json), [exact regression source](results/real_rsl_cpu/input/real_rsl_cpu_regression.py), [31-test CPU log](candidate/test_output.txt).
- [Six candidate files and original preparation freeze](candidate/FREEZE_SHA256.json), [minimal patch](candidate/candidate003_minimal.patch), [deployed source map](results/campaign_source_hashes.json).
- [Full logs and campaign](results/run/campaign.json), [verified summary](verified_summary.json), [checkpoint byte/embedded-contract audit](checkpoint_audit.json), [original 48-file result map](full_result_SHA256SUMS.json).
- [Post-run integrity and exact-container cleanup](results/postrun_verification.json), [pause021 restoration](results/forecast_pause_021/restored.json).

All 909 source files and 550 admitted asset files remained unchanged; all fetched
raw results and CPU regression files matched remote hashes. Both exact owned
containers were absent after exit and pause021 restored its forecasting timers.
The next pilot was already using the Spark during this audit; no later lock or
GPU-idle claim is made.

Deployed source manifest SHA-256:
`00d4e07e0e31776e3d2faa0136a2b1386f79fd4940bf81d2f4a484155f81bb25`.
Executable source identity:
`fb5e472bb710daa50d1d7f16a6f3812d2ed845e2400fcc6769859b3fb0b9519e`.
The controller is `formal_004`, 0.04 rad per 20 ms and 8 rad/s², with 495/498
actor/critic inputs. Frozen shared telemetry predates the shared quaternion
repair; this candidate's executed capture converts actual SDK XYZW into WXYZ
directly and retains its raw quaternion. Do not reuse old shared quiet scoring.

The original flat admission file retains a pre-final-save `status: running`
field, while its physical gate is passed. The completed flat state and campaign
provide the terminal result; no raw status field was rewritten. Candidate003's
preparation README is also preserved as a historical pre-dispatch record.

Replay the frozen CPU prototype tests from a checkout with project dependencies:

```sh
uv run python -B artifacts/omni_diagnostics_2026-09-09/velocity_candidate_003/replay_cpu.py
```

The [separate 50-update pilot](../velocity_pilot_003/README.md) measures what
happened after this admission. Its outcome does not rewrite the probe result.
