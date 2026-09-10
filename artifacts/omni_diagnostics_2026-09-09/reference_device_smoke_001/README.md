# Actual 1/32-replica device bridge

Both actual CUDA allocations pass the reviewed short device-interface checks. Each runs 200 controls of canonical startup/settling followed by 64 controls (1.28 s) of zero-command reference hold. The exact 846/849 observation schema, actual sensor clocks, target/reference state and raw physics telemetry remain consistent. No actor was loaded or PPO started; this short hold is not full quiet-standing, walking or velocity-fidelity admission.

| Run | Actor / critic | Recorded controls | Valid actual sensor-clock rows | Postsettle 400 Hz requested/applied peak |
|---|---|---:|---:|---:|
| 1 robot | 846 / 849 | 264 | 264 | 1.136380 N·m |
| 32 robots | 846 / 849 | 264 | 264 | 1.483903 N·m |

The independent review verifies every clock row against eight float32 physics-timestamp increments, all five raw joint-rate history slots against actual endpoints, actor/critic shared-packet equality, final contact/reset flags, and the separate 20 ms angle-derived rate channel. The reported-rate versus actual-angle discrepancy remains visible: the largest integrated difference over the 1.28 s hold is 0.01396 rad for N1 and 0.03315 rad for N32. No native cause or velocity-fidelity pass is claimed.

Instrumented median reference/setter time is about 16 ms per control; environment step is 68–69 ms and observation/history encoding 6.4–6.7 ms. Environment-step time already includes about 13 ms of nested capture. These timings include legacy CPU copies, synchronizations and the 400 Hz observer; they are not a PPO throughput projection.

[The detailed actual review](actual_review/README.md) contains the compact packet/rate/timing tables. [The original campaign](actual_review/raw/reference_device_smoke_001/campaign.json), [N1 acceptance](actual_review/raw/reference_device_smoke_001/replicas_1_accepted.json), [N32 acceptance](actual_review/raw/reference_device_smoke_001/replicas_32_accepted.json), and all raw arrays/logs are retained unchanged. [Pause039 evidence](actual_review/raw/forecast_pause_039) records restoration at Unix time 1789027997.2217932. The independent remote audit verified exact owned-container absence, successful inactive unit exit, complete source/adapter/observation/host inventories, 550 unchanged admitted assets and all 38 fetched raw payloads.

## What is preserved

- [Frozen device adapter001](owner/device_adapter): 18 files and its original freeze, including the installed sensor API receipts and CPU tests.
- [Original host001](owner/host001) and [corrected executed host002](owner/host002): both immutable. Host002 adds independent actual clock progression validation; the [independent host review](independent_review/host/README.md) retains the 10/11 test results and exact tiny-delta review.
- [Independent device preparation review](independent_review/device/README.md), including its ten CPU tests.
- [Exact dispatch guard001](owner/guard001), with its [eight independently passing mocked restoration tests](validation/guard_tests.log) and [read-only remote identity check](validation/guard_remote_identity.json).
- [Original preflight receipts](preflight), including the exact source/asset/observation bindings checked before either allocation.
- [Frozen actual review](actual_review), whose original 44-file manifest includes all 38 raw payloads exactly once.

The already published [source009 physical/reference package](../reference_physics_009/README.md) and [observation005 package](../reference_policy_observation_005_001/README.md) are referenced rather than duplicated. Exact source926, observation160 and admitted-asset550 maps are included under [references](references). [RECONSTRUCTION.json](RECONSTRUCTION.json) lists original/copied bundle identities and repository locations. The device owner/source additions are retained, but copied historical test commands may require the original sibling dependency layout; the portable terminal verifier below does not.

## Portable read-only verification

```sh
python3 verify_payload.py
```

[verify_payload.py](verify_payload.py) needs only the Python standard library. It verifies the whole publication inventory, every copied original freeze, referenced maps and all raw remote hashes. It then re-runs the exact executed host002 numeric-result gates for N1 and N32 against their original accepted receipts. It imports neither Isaac nor Torch, writes no outputs and launches nothing. It does not need the omitted full source009/observation trees or an active Spark connection.

This is a terminal evidence bundle. No geometry, source009 physical gate, benchmark payload or production default was changed. Subsequent runtime priorities and admissions belong in the project's current status document.
