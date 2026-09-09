# MKII live standing validation and physical linkage evidence

This bundle follows the immutable [offline step-1 evidence](../mkii_step1_2026-09-04/README.md). The user subsequently authorized Spark execution. Runs use an isolated source directory, the corrected serial-v2 USD, a bounded supervisor and unique output directories. No learner, training queue or historical checkpoint was started.

**Scope:** standing characterization of the 19-body serial approximation. This does not certify the real four-bar mechanism, dynamic command directions, hardware transfer, terrain capability or complete G0. The [physical reference directory](physical_fourbar_reference/README.md) contains recovered current-CAD pin frames and the next implementation plan; it is not a solver-tested physical asset.

**Final result:** `standing_003/report.json` passes the hardened schema-v3 standing gate with 32 environments, 1,000 control steps and all 4,000 physical substeps observed at 200 Hz. The run's validator hash matches the published source. The complete CPU suite passes **672 tests** in 17.976 seconds; all **112 archived pipeline hashes** still match.

## Run history

| Directory | Work | Outcome and interpretation |
|---|---|---|
| `probe_001` | 1 environment, requested 100 steps | Native startup failure. Standalone OpenUSD 26.8 was loaded before Kit's different 25.11 ABI. No physical acceptance report; preserve the crash evidence. |
| `probe_002` | 1 environment, 100 completed steps | Physics ran, then Kit exited zero before Python could save its report. The supervisor returned failure because the report was missing. Its old `cleanup: FAILED` field conflated missing-report rejection with cleanup; the owned container was subsequently confirmed absent. |
| `probe_003` | 1 environment, 100 steps | Report persisted before Kit shutdown; short standing probe passed under the initial policy-step sampling criteria. Nominal imported masses and deterministic materials were used. |
| `standing_001` | 32 environments, 1,000 steps | Initial standing criteria passed, with 54.84 seconds between first and last container-log timestamps. This is a diagnostic result: measurements were sampled at 50 Hz and the initial global saturation/contact checks had limitations found in review. It does not supersede the hardened validator's later outcome. |
| `standing_002` | 32 environments, requested 1,000 steps | Hardened validator ran through at least logged step 700. The resource supervisor detected unrecognized GPU PID 1178517 and stopped only its owned container; no final report was produced. This attempt is not a physics pass or fail. The PID was absent at follow-up; no identity or workload attribution is established. |
| `standing_003` | 32 environments, 1,000 steps / 4,000 physics substeps | Hardened standing gate passed. Container-log wall time 75.18 seconds, supervisor exit zero, exact owned container removed. No learner or persistent job was started. |

Each run preserves `source.SHA256SUMS`, `supervisor.json`, `container.log` and, when produced, `report.json`. The `admitted` file is a CPU preflight barrier release, not robot/model acceptance. Source provenance is explicitly `uncommitted` with exact per-file hashes; the final repository commit records the published implementation. Source changes between attempts are intentional and traceable. A process exit of zero alone is insufficient for success.

The startup integration fixes are versioned: use the actual config instance's simulation field, isolate standalone USD checking in a subprocess, run another asset gate after Kit loads its own USD library, include the reviewed telemetry mitigation once, and persist both failure and success reports before native cleanup can exit Python. The host supervisor checks the report's explicit pass, task, environment/step counts and run kind, independently of exit status.

## Model and runtime identity

- Task: `Isaac-Velocity-Flat-Hexapod-MKII-V2-Direct-v0`.
- URDF SHA-256: `6109956e9e3a7a648bc4afa7310904c2fd3f7cb790e157d83212339a73b7fbc0`.
- USD root SHA-256: `033d31d58f18cc2ec83afcdf7962aef014d03afdcc45e37f7fa23332b14e886b`; referenced layers are also checked by the asset manifest.
- 19 bodies, 18 joints, six named tibia contact sensors. Both standalone USD 26.8 and Kit-native USD 25.11 preserve the repaired asset within the step-1 tolerances.
- Observed joint order: `lf/lm/lr/rf/rm/rr_coxa_yaw`, then the same six-leg order for `_femur_pitch`, then `_tibia_pitch`. The complete names appear in each report. The versioned manifest matches this order; mapping is checked before stepping.
- Physics timestep 0.005 s; policy step 0.020 s. Nominal reset plate height 0.142964 m, with independent ±0.03 rad reset jitter and seed 0.
- Base-mass randomization is disabled for nominal validation. Robot material is fixed at static/dynamic friction 1.0 and restitution 0.0. These are explicit simulation assumptions, not measured terrain coefficients.

## Initial full-run measurements

`standing_001/report.json` records 1.722 mm minimum URDF collision clearance at live reset state readback, 0.135687 m mean settled plate height, 0.879947 N·m peak settled computed torque, no terminations/truncations, and no detected non-foot contact at its sampling rate. Startup computed torque reached 2.364431 N·m while applied torque was capped at approximately 1.6 N·m. This exceeds the imposed cap and the website's rotating rated value, **not the published 5.5 N·m peak**. The inherited actuator already records 5.5 N·m but clips delivered effort at 1.6 N·m. The user correctly requested reconciliation with manufacturer specifications; the [RS05 review](../../docs/RS05_SPEC_REVIEW.md) establishes a **1.2 N·m continuous-stall rating**, with cooling-dependent 1.6/1.8 N·m rotating ratings. The 1.6 cap is not universally conservative. `startup_raw_rating_exceeded` in the immutable report means above the 1.6 N·m baseline, not above hardware peak capability.

These early torque peaks and contact findings are **50 Hz policy-endpoint samples**, not a continuous or 200 Hz physical-substep bound. Review also found that a global saturation average could hide one overloaded motor and that invalid loaded contact positions could evade shaft classification. The final validator fixes these paths; the original reports remain unchanged.

## Hardened full-run measurements

| Metric in `standing_003/report.json` | Measured result |
|---|---|
| Reset minimum collision clearance from live state FK | 1.722 mm |
| Settled plate height: mean / minimum | 0.135687 / 0.135602 m |
| Settled computed and applied torque peak | 0.880025 N·m |
| Settled saturation: global / worst environment-motor | 0 / 0 |
| Minimum loaded foot-pad count across all settled environments/substeps | 6 |
| Detected non-foot ground contacts | 0 |
| Terminations / truncations | 0 / 0 |
| Settled tilt RMS | 0.014086 degrees |
| Startup computed peak / applied peak | 2.364431 / approximately 1.6 N·m |
| Startup worst environment-motor saturation duty | 0.625% over the four-second startup window |

The validator samples after each actual scene update and requires exactly four 0.005-second updates per control step; it rejects a backend that hides decimation. It reads current contact forces, checks finite loaded contact positions, records per-environment support, and accumulates saturation separately for every motor/environment. For this deliberately capped baseline, the settled peak must be at most 1.60001 N·m and worst motor duty below 0.5%; these are not the manufacturer's peak-torque envelope. CPU regressions inject first-substep spikes, shaft contacts and invalid loaded positions that disappear by the policy endpoint, and mutate simulator buffers between captures. Reports are persisted before Kit shutdown and the supervisor independently requires a valid passing report.

Even with every physical substep sampled, each tibia's contact sensor reports a contact aggregate. Simultaneous pad and shaft contact can be hidden by its centroid; per-collider contact qualification remains necessary for complete G0. Static anatomical vector wiring is checked, but physical driven sign, command-direction and yaw tests remain unrun. This bundle grants no training or hardware admission.

## Reproduction and compute

Use the command in [Operations §11](../../docs/OPERATIONS.md#11-corrected-serial-cad-v2-live-validation). The source mount is read-only and outputs are separate under `/home/orionh/HEXAPOD_runs/mkii_v2_step2_a0f0b39/runs/`. The supervisor holds `/tmp/hexapod-isaac-gpu.lock`, checks unrelated work before and during execution, imposes a 900-second execution timeout, and cleans up only its immutable owned container ID. Wi-Fi interruption does not justify launching a duplicate: inspect the existing supervisor report and container first.

The latest user instruction is **full available compute for hexapod until another agent requests sharing through `/home/orionh/SPARK_COMPUTE_COORDINATION.md`**. The [repository copy](../../docs/SPARK_COMPUTE_COORDINATION.md) contains historical timings, the request procedure and the earlier 60/40 preference to use if sharing is requested. No MPS partition or shared long-training launcher has been installed. Short acceptance remains exclusive.
