# Moving PPO: nine updates, then rejected recovery

This bundle preserves the complete terminal evidence from the separately reviewed ten-update allocation on 2026-09-10. **Nine PPO updates completed. The tenth collection failed at physical control 2,600**, when row 8's RM tibia still requested more than 1.6 N·m at the end of its 200-control timeout recovery. No decision-010 checkpoint, matched cold forward/stop evaluation, or final quiet policy screen was produced. This result does not establish walking improvement or complete Stage 2.

The original 74-file, zero-update infrastructure admission remains a separate immutable [admission bundle](../reference_learning_ppo_admission_001/README.md). Its exact bundle and initial checkpoint hashes are recorded in [RECONSTRUCTION.json](RECONSTRUCTION.json). The reviewed ten-update decision is copied [verbatim](reviewed_decision001.json); the allocation does not become successful because its learner executed some updates.

| Evidence | Terminal result |
|---|---|
| Learner state and update ledger | Nine completed updates; nine finite-loss rows retained |
| Host campaign | Failed; completed-allocation update field remains 0, with separate reported-training count 9 |
| Failure | Control 2,600, row 8, RM tibia `revolute_2_1`; last eight 400 Hz samples peak at 1.62764847 N·m requested |
| Contact state | All 32 rows have six distal contacts; no nonfoot contact or native termination among rows completing timeout recovery |
| Applied torque | Clipped to the actual float32 representation of 1.6 N·m; this does not make excessive requested demand pass |
| Checkpoint/evaluation | No decision-010 or cold/quiet policy evaluation produced |
| Exact owned training container | Recorded name and recorded ID both absent in read-only terminal inspection |
| Forecasting | Pause 052 restored at Unix time 1789049775.7754648 |
| Independent remote integrity | Complete source/input maps and both prior admission and training raw payload sets unchanged |

The [frozen failure review](failure_review/README.md) is included byte-for-byte. It shows a persistent loaded equilibrium over the last two seconds, with the canonical target and zero recovery residual. It proposes a separately versioned reset comparison. That hypothesis is not proof of a native solver cause, a changed gate, or authorization to rerun training. The raw SDK velocity and angle-derived evidence remain available without substituting one for the other.

## Integrity and ownership

The [initial read-only terminal audit](terminal_audit/remote_audit.json) verifies the failed invocation `804a727558374268b185325305148b1a`, raw hashes, the exact training container name/ID absence, and pause restoration. Its full-source recheck was initially pending; that historical field is preserved. The later [independent full audit](terminal_audit/full_integrity/result.json) closes that gap and verifies:

- Physical source 926 files, consumer 51, host 3, original admission guard 11, bridge 18, observation bundle 160, and continuation guard 7.
- All 550 admitted asset files, plus the host's read-only validation of standing/calibration/checkpoint and decision bindings.
- All 74 initial admission raw files and all 46 training raw files, unchanged against their original snapshots.

All six full-audit input/result files are copied verbatim with an additional local copy manifest. Portable source maps are in [references/](references/); complete prior source preparation remains referenced by exact hash in [the preparation bundle](../reference_learning_ppo_preparation_001/README.md). Historical continuation-guard CPU-readiness fields describe preparation before final binding; its actual frozen bytes and remote map are authoritative here.

Only the single training container was allocated during this continuation. The restorer's historical `owned_cleanup_checked` list is empty; the explicit read-only Docker name/ID absence records provide the separate cleanup evidence. No additional phase or checkpoint is inferred from the host's planned schedule.

## Lossless raw storage and reconstruction

All **46 original files, totaling 460,583,084 bytes**, are represented in [RAW_STORAGE.json](RAW_STORAGE.json). The 210,040,770-byte physics-substep NPZ and 226,863,443-byte ordinary trace NPZ exceed 100 MiB, so they are stored as ordered chunks of at most 64 MiB. Every chunk and whole original has a SHA-256 and byte count. Other raw files are copied unchanged. There is no compression rewrite, truncation, LFS migration, or NPZ regeneration.

Run the standard-library portable verifier from any location:

```sh
python3 /path/to/reference_learning_ppo_train_001/verify_payload.py
```

Reconstruct every original file into a new directory outside this immutable bundle:

```sh
python3 /path/to/reference_learning_ppo_train_001/reconstruct_raw.py \
  --output /path/to/new_training_raw
```

The reconstruction first verifies the full bundle and failure receipts, then streams every chunk and checks all original sizes/hashes. The raw analyzer can subsequently be rerun using NumPy and the reconstructed tree; write its report outside the frozen bundle. Integrity verification itself does not rerun physics or independently reproduce the optimizer's gradient calculations.

No GPU, process signal, hardware change, or production adoption occurs through these publication utilities. The later decision on reset correction or another training allocation is outside this terminal bundle.
