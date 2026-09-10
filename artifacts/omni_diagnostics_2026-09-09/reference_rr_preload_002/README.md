# RR preload002: local support improved; trial rejected

Fresh standing passed for all **32 replicas**, then the left-strafe diagnostic rejected after **599 of 2400 controls**. The one-shot 0.5 mm RR preload increased RR force, but LM later fell below the unchanged 1 N support threshold while RF unloaded. The full 400 Hz record also contains **two requested torques above 1.6 N m**. This is a rejected physical experiment, with no completed stop or quiet interval, PPO training, or Stage 2 admission.

| Evidence | Actual result |
|---|---|
| Original failing instant, 10.10 s | RR force increased from 0.945591 N to 3.939688 N |
| Pre-intervention prefix | First 420 physical rows byte-equal to the original trial |
| Correction | One RR landing, 8.40–8.70 s, C2 endpoint at −0.5 mm |
| Completed landings | LF, RR, LM; three total |
| Terminal support loss | LM 0.896118 N; RF airborne; four admitted supports |
| Requested torque at 50 Hz | Maximum 1.534293 N m |
| Requested torque at 400 Hz | RF tibia 1.626537 and 1.624497 N m; applied torque clipped at approximately 1.6 N m |
| Cleanup | Both names and both recorded IDs absent; pause048 restored |

![Actual support and torque comparison](review/actual_comparison.png)

The trial extended by 94 controls (1.88 s) and retained RR support after the correction. Those observations do not establish acceptable whole-robot load distribution or justify applying the correction to other feet. In particular, the separate arc failure involved RM, not RR. No next controller change is adopted here.

[Independent review](review/README.md) documents the original-gate replay, force-free intervals, emitted target alignment, high-rate torque events and the remaining angle-versus-reported-rate discrepancy. All standing physical metrics and quiet verdicts reproduce; three tiny derived heading values differ by about 6.83 × 10⁻⁶ degrees and are retained explicitly. No gate was changed. Contact forces were recorded only at 50 Hz, so this bundle makes no claim of 400 Hz force or measured penetration data.

## Preserved evidence and reconstruction

[Evidence](evidence/) contains all **33 exact remote payloads**, including the two actual jobs, full standing and strafe NPZs, logs, preflight and pause/restoration receipt. [Remote audit](evidence/remote_audit.json) binds **932 unchanged source files**, **550 unchanged assets**, all four current name/ID absence checks and the failed user-unit invocation. The predecessor diagonal checks in the preflight remain separate historical evidence. The largest payload is 75,735,087 bytes, below the Git 100 MiB limit; no compression, truncation or chunking was needed.

The frozen ten-file [independent review](review/FREEZE_SHA256.json) and two-file [guard](guard/FREEZE_SHA256.json) are copied byte-for-byte. Source and preparation are referenced through exact manifests in [REFERENCED_INPUTS.json](REFERENCED_INPUTS.json), rather than duplicating 932 source files. Published [preparation002](../reference_rr_preload_preparation_002/README.md) supplies the stdlib-only import correction and source reconstruction chain; the original comparison comes from [directional002](../reference_directional_002/README.md). These preparation receipts are not additional physical results.

Run the portable, read-only integrity verifier with any standard Python 3:

```sh
python3 verify_payload.py
```

Optional `--source`, `--assets` and `--original-trace` arguments verify separately reconstructed source, admitted assets and original comparison trace against their full maps. The numerical replay accepts explicit paths and a fresh output directory:

```sh
python3 review/review_actual.py --raw evidence --source /path/to/reconstructed/source \
  --original-trace /path/to/directional002/trace.npz --output /path/to/fresh/review
```

Numerical replay requires NumPy and the exact source dependencies. The portable verifier requires only Python's standard library. Neither command launches Isaac or alters the preserved evidence.
