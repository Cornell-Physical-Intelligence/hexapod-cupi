# Actual device001: independent short bridge review

Both actual CUDA phases pass the exact host002 raw-result checks and this independent review. The source009 controller, device adapter001 and observation005 interface work together for the declared zero-command hold at 1 and 32 replicas. This is a **short device bridge pass**, not a fresh full quiet-standing, walking, PPO or velocity-fidelity admission.

| Actual phase | Controls / new hold | Actor / critic | Minimum final distal supports | Postsettle 400 Hz requested/applied peak (N·m) | Final raw SDK rate max (rad/s) | Final 20 ms angle-derived rate max (rad/s) |
|---|---|---|---:|---:|---:|---:|
| N1 | 264 / 64 (1.28 s) | 846 / 849 | 6 | 1.136380 | 0.011030 | 0.0000954 |
| N32 | 264 / 64 (1.28 s) | 846 / 849 | 6 | 1.483903 | 0.025998 | 0.0020838 |

Each allocation first runs the original 200-control canonical startup/settling. The new device reference then holds zero command for 64 controls. It makes no planned or confirmed liftoffs and loads no actor. Final measurement validity, reference readiness, distal contacts and contact validity all pass; no final terminal or forbidden-contact flag is set. The pinned runtime additionally enforces these checks during the run, and the original 400 Hz observer completes all 2,113 samples without an error. The 1.28 s hold is too short to replace the existing ten-second quiet requirement.

## Actual clock and packet checks

All 264 rows for all 14 sensors in every replica have valid recorded timestamps, matching last-update times and zero observed cache age. I independently rebuilt every subsequent row using eight float32 additions of the float32-rounded 2.5 ms increment. The actual first/last timestamps are 0.0199999976903 and 5.28009700775 s; the latter is the expected repeated-addition result, not a silently rounded 5.28 s clock. These values describe the verified synchronous simulator sensor cache, not future camera or hardware latency.

Both phases have the identical schema hash `c2ab56b61c8ca56412090cd42cf109c90576bd5e78602a17eb29ada6d0559845`. This hash reflects the actual named runtime joint order; it need not equal a differently ordered CPU fixture's hash. The actor equals the first 846 critic entries exactly; the remaining three critic entries equal the explicitly unverified simulator-reported body twist. All five reported-joint-rate history slots exactly match their corresponding physical substep endpoints and intended scale. Final joint angles/rates and raw XYZW root pose equal the actual 400 Hz endpoint. The separate interval-rate output exactly equals the difference between the last two actual 20 ms joint-angle endpoints; its actor slice and validity mask agree.

The original in-process source009 reader comparison reports 31 matched channels. This review independently checks the stored raw channels and packet sections available in the exported evidence; it does not pretend to reconstruct every previous device sensor packet from unavailable full history. The two replica counts share the same schema and semantics, but their actual physical state values are not expected to match across different replicas/origins.

## The rate discrepancy remains visible

Over the actual 4.00–5.28 s hold, the worst integrated reported-rate minus actual angle change is:

| Phase | Replica / joint | Actual angle change (rad) | Reported-rate trapezoid integral (rad) | Difference (rad) |
|---|---|---:|---:|---:|
| N1 | 0 / `revolute_2_6` | -0.0000190735 | +0.0139411828 | +0.0139602563 |
| N32 | 6 / `revolute_2_1` | -0.0000550747 | +0.0330920205 | +0.0331470952 |

The bridge preserves this discrepancy rather than replacing reported rates with angle differences. No native cause or velocity-fidelity pass is claimed, and no gate or policy schema is changed by this review.

## Measured instrumented timing

| Median per control (ms) | N1 | N32 |
|---|---:|---:|
| Reference computation and target setter | 15.97 | 16.16 |
| Environment step including capture | 67.86 | 68.81 |
| Nested device capture, already inside environment step | 13.04 | 12.76 |
| Observation/history encoding | 6.66 | 6.35 |

Full percentiles and maxima are in [report.json](report.json). These measurements include retained legacy CPU telemetry copies, synchronizations, the 400 Hz observer and nested capture. Do not add nested capture twice or extrapolate this short instrumented hold into PPO training throughput.

## Identities, retrieval and restoration

The read-only remote audit independently verifies complete inventories:

- Source009: 926 files, manifest `04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e`.
- Device adapter001: 18 files, freeze `be4870a8f7ca0857ba50ffd3a0c92ac5925b9a816767d66d826af7b59957aa0e`.
- Observation005: 160 files, freeze `22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63`.
- Host002: eight files, freeze `adbb60b3d77b8d6b268dcf3060232fcf3c493361259a0997582e694d62a35658`.
- Admitted source009 study assets: all 550 files equal receipt `99fd81ee260bd517e6590fe56b8bb46715827fef1a295553f154524a1978ce00`.

All 38 fetched run/pause039 payloads match the remote hashes in [remote_audit.json](remote_audit.json). Both exact owned container IDs and their names are absent. The unit reports success, exit status 0 and inactive/dead. Pause039 restoration is recorded at Unix time 1789027997.2217932. This is a terminal audit at the recorded time; subsequent root-owned jobs can legitimately change current workload state.

The untouched fetched run is in [raw/reference_device_smoke_001](raw/reference_device_smoke_001), with its campaign, both accepted receipts, observation packets, raw physical data and logs. Pause/restoration evidence is in [raw/forecast_pause_039](raw/forecast_pause_039). [audit_remote.py](audit_remote.py) performed only read-only hashing, Docker inspection and systemd status reads over the configured Tailscale SSH alias. This reviewer launched no GPU job and changed no source, asset, gate or main-branch file.

[review.py](review.py) performs local identity, host-gate, clock, packet and independent rate-integration checks; every assertion passes. Its output is [review.log](review.log). To reproduce locally, use a copy to preserve frozen outputs:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tmp/reference_device_smoke_actual001_review/review.py
```
