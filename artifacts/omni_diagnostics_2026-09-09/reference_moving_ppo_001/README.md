# Moving PPO001 terminal: support outcome followed by reset API failure

Fresh standing and both 32-replica calibrations passed. The untrained profile then stopped after **224 controls**, before any PPO update. Replica 6 lost retained RR support during the planned LF swing; the attempted reset raised an inference-tensor mutation error. The failed profile, incomplete reset ledger and full traceback are preserved. This is not a trained policy's rejected evaluation or a completed moving-PPO pilot.

| Result | Actual evidence |
|---|---|
| Standing | 32 × 1000; original physical and quiet gates pass |
| Calibration | Zero-mean and sampled std 0.02 trials; all 32 quiet in each |
| Profile | 200 startup + 24 active controls; planned 512 active controls incomplete |
| Trigger | Replica 6 RR force 0.597675 N; only four retained supports excluding planned LF |
| API failure | `reset_buf.copy_` outside inference mode; physical reset completion not recorded |
| PPO | 0 updates; no training or later decision screens |
| Cleanup | Three owned names and three IDs absent; pause049 restored |

[Independent review](independent_review/README.md) reproduces calibration metrics exactly, checks all recorded 400 Hz endpoints and contact clocks, and distinguishes the verified 223-control pre-event prefix from the incomplete reset at control 224. It preserves the full audit primitive's `KeyError('reset_kind')` failure. No post-startup 400 Hz torque request exceeds 1.6 N m in these phases; initial startup excursions and the continuing SDK-rate/angle discrepancy remain explicit.

A reset-only API repair cannot by itself satisfy the unchanged no-event profile gate: the preceding reference-support outcome still needs review. The immutable initial checkpoint and actual strict reload receipt are retained, but no optimizer update or improved policy is inferred. Actor/history/bootstrap GPU tensors were not exported; the raw audit does not claim to replay them.

## Complete immutable evidence

All **56 original remote files** are preserved with exact whole-file SHA-256 hashes. Two calibration NPZs exceed 100 MiB: `physics_substeps.npz` (170,428,908 bytes) and `trace.npz` (121,419,205 bytes). They are stored as ordered **64 MiB-or-smaller lossless chunks** with per-chunk and original whole-file hashes. All remaining files are copied unchanged under [evidence](evidence/). Original full files remain remote and in the local read-only results directory. Nothing was compressed, re-encoded or truncated for publication.

The terminal audit verifies unchanged **926 source009**, **35 consumer**, **3 host**, **2 guard**, **18 bridge**, **160 observation** and **550 asset** files. Source maps are included under [references](references/); the full exact consumer/host/guard preparation is already published in [moving preparation001](../reference_moving_ppo_preparation_001/README.md). This terminal bundle does not duplicate or adopt a successor consumer. [RECONSTRUCTION.json](RECONSTRUCTION.json) records every copied/referenced lineage.

The ten-file independent review, seven-file frozen audit primitives and the frozen chunk utility are copied byte-for-byte. The remote preflight, three actual jobs, logs, accepted/immutable phase maps, partial profile files, forecasting pause and restoration receipt all remain raw evidence. The terminal user unit failed/exit 1 with invocation `53fa4c8afc7940e0931a77c18d310557`; restoration time is Unix `1789046082.2900512`.

Run the portable read-only verifier with standard Python 3:

```sh
python3 verify_payload.py
```

Reconstruct all originals into a new external directory for NumPy replay:

```sh
python3 reconstruct_raw.py --output /path/to/new/raw-directory
```

The numerical review accepts that raw directory, the separately reconstructed exact source009 tree and the included frozen audit-primitives directory. It requires NumPy; integrity verification and reconstruction use only the standard library. Neither utility launches Isaac, contacts Spark, signals services or edits an immutable source.
