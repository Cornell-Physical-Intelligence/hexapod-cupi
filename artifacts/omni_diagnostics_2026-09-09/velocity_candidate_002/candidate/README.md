# Candidate 002: smaller exploration, full-episode calibration

This is a fresh bounded correction to the rejected candidate 001. It has not run on the GPU. Stage 2 remains incomplete. Root owns review, dispatch and forecasting restoration.

Candidate 001 proved that the new action path can hold still: all 32 zero-mean replicas passed. Its actual Gaussian std 0.05 trial failed 31 of 32 replicas, with one termination. Requested torque followed accumulated target-position error, rather than current action: correlation 0.9994 versus 0.0459. Target drift reached approximately 0.15 rad despite ordinary same-episode target increments never exceeding 0.00710 rad. Joint-limit and rate bounds cannot prevent random integrated target drift while planted feet resist motion.

Only `candidate_runner.py` changes among the six executable candidate files:

- Initial Gaussian action std becomes **0.005**.
- The sampled standing calibration becomes **1,000 control steps / 20 seconds**, matching the walking episode duration.
- The immutable plan must declare those exact values as `omni.initial_exploration_std` and `omni.sampled_calibration_control_steps`.

The actor mean still starts at zero. The formal 0.04 rad/20 ms target step, 8 rad/s² acceleration, 1.6 N·m torque cap, PD gains, sensor noise, filters, reward weights, contact/failure gates and per-replica scoring remain unchanged. This is a measured smaller-noise admission attempt, not a claim that the new setting is already safe or sufficient for gait exploration. The 0.03 diagnostic profile is not substituted or pooled into the result.

`candidate002_minimal.patch` shows the complete behavioral change from candidate 001. The other five candidate files are byte-identical. Existing 001 sources, rejected results, checkpoints and benchmarks are preserved.

Thirty tests pass, including the actual candidate methods, named action/state controller, checkpoint/source incompatibility, wrong exploration declaration rejection, and four bounded host-launch guards. Run:

```sh
.venv/bin/python -B -m unittest discover -s tmp/omni_velocity_candidate_002 -p 'test_*.py' -v
```

`build_source.py` uses the same reviewed `bf7a5e8` executable base and the same verified 550-file full C asset/40°–120° stance package as candidate 001. It overlays these six files and the bounded probe launcher, writes the two explicit new plan values, and produces a fresh source manifest. It never copies newer unrelated production/runtime changes into this comparison.

```sh
.venv/bin/python -B tmp/omni_velocity_candidate_002/build_source.py --candidate tmp/omni_velocity_candidate_002 --manifest tmp/omni_velocity_candidate_002/FREEZE_SHA256.json --output tmp/omni_velocity_candidate_002/source_002
```

Root can dispatch `launch_velocity_probe_spark.py --source <new source> --output <new run>` only through the established job-scoped locks, unrelated-workload checks, identified forecasting pause/restoration and bounded systemd wrapper. The launcher runs a fresh full 32 × 1,000 standing admission, then zero-mean and std-0.005 calibration. Only if both pass does it attempt the actual two-update stand-only runner learn/save/reload smoke. There is no std-0.01 bracket, walking extension or automatic continuation.

The next walking step, if admitted, is a separate reviewed 50-update scratch pilot with matched initial/final per-direction and quiet/stop evaluations. The prototype launcher under `tmp/omni_velocity_train_001` is intentionally bound to rejected candidate 001 and cannot be reused unchanged for this new identity/std. It needs an explicit reviewed update after this probe passes. An early progress video must identify the checkpoint, commands and incomplete qualification; no walking-video promise follows from CPU or standing success alone.
