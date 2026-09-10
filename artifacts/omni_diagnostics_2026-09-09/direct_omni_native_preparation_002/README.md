# Native retry preparation002: strict reload correction

This bundle preserves the exact **native003, host002 and guard003 preparation** for a fresh direct315/318 CAPS smoke. It contains source, CPU tests, source-build and preflight evidence only. It makes no statement about the retry's running or terminal result, policy quality, quiet standing, completed pilots or Stage 2 acceptance. Actual smoke results belong in separate terminal bundles.

The preceding [smoke001](../direct_omni_train_smoke_001/README.md) passed standing and completed two PPO updates, then failed strict checkpoint reload. Its normalizer `_std` buffer had been created inside RSL's inference context. That failed smoke remains unadmitted; its weights are not the starting point of this retry.

## Exact correction and unchanged behavior

The only changed native runtime function is [`direct_training.verify_reload`](native003/direct_training.py). Before the existing strict reload, it replaces only registered actor/critic buffers marked as inference tensors with value-preserving ordinary-tensor clones. Values, device, dtype, model parameters, optimizer objects, checkpoint bytes and RNG state remain preserved. Strict actor/critic/normalizer/Adam and deterministic-action comparisons still run. This is a reload into the existing runner; it is not a new-runner or physical qualification claim.

The complete 598-file source inventories have identical paths. Only `tools/direct_training.py` and its `source_origin.json` provenance differ from source001. The entrypoint, original dynamics, geometry, mass, actuators, observations, action processing, rewards, curriculum, CAPS loss, allocations, command sampling and constant/stop criteria are unchanged. The plan remains `9fb7d0b01480523972b4f1c17e717756d7dcf1c54b9caa436245671937de538c`. The original warm start remains `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`.

Host002 changes only its source and native-contract pins. Its entry adapter, all launch functions, phase order, read-only mounts, checkpoint rules, cleanup and per-phase deadlines remain unchanged. Guard003 binds this successor and the exact previous failed-job cleanup receipts. It retains the owned-container checks, actual AppReady deadline, GPU locks, active-timer restoration and bounded fallback. None of these preparation files grants permission to resume a failed checkpoint or launches another allocation automatically.

| Frozen component | Payloads | Manifest SHA256 |
|---|---:|---|
| [native003](native003/README.md) | 26 | `20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb` |
| [host002](host002/README.md) | 10 | `19545653a00e635cc14ed9415f9799978da34674e84f6cad76eb42bd2a961ac4` |
| [guard003](guard003/README.md) | 16 | `123c34a1830be1051fe17f35e728bfcf795b0d497fe5d4b043dd78e484906128` |
| [independent guard review](independent_guard_review/review.json) | 2 | `f544018d92b7a0b1a2b1a294ceec52a78dfb1445f5ff6f141f2652f59431acbd` |

Every original payload and its original freeze manifest are copied verbatim. Historical draft reports, parent README sections and earlier test logs inside those freezes are retained as history, not rewritten or counted as new tests.

## Evidence and allocation boundary

The [root build receipt](root_checks/source_build002.json) binds the exact read-only fetched [source002 map](inputs/source002_sha256.json): `64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e`, 598 files. [SOURCE_DELTA.json](SOURCE_DELTA.json) compares it with source001. Full source/asset trees are referenced by their complete hash inventories rather than copied again.

The [host preflight](root_checks/host_preflight002.json) records same-source standing/train/evaluation admission checks without a GPU allocation. Its empty stderr is retained. Root reran [16 native tests](root_checks/native_root_tests002.log), including three focused real-RSL normalization/reload tests, and [29 guard tests](root_checks/guard_root_tests003.log). Host002's own 21 CPU tests and the independent guard's separate 29-test review are also preserved. These are separate recorded executions, not additive counts of unique tests. The new native tests perform no PPO updates or physics steps; they reproduce the old inference-buffer exception and test the correction, unchanged values/RNG/checkpoint/Adam, and corrupted-optimizer rejection. Earlier CPU learning regressions remain historical evidence.

The allocated smoke is fresh standing32×1,000, CAPS train32×24×2, then final constant and final moving-to-stop diagnostics. The guard targets `direct_omni_train_smoke_002` with pause055, 2,520 seconds outer runtime, 180 seconds stop and a 45-minute fallback. Each later 1,024×24×50 pilot remains an independent root-owned allocation starting from the original checkpoint, requiring a fully completed matching smoke. Preparation, completed acquisition and physical acceptance are distinct.

## Read-only verification

```sh
python3 -B verify_payload.py
```

The portable standard-library verifier checks the entire bundle and all four original freezes, exact build/preflight identities, complete source-map differences, native AST equality except `verify_reload`, and host AST equality except the two pins. It does not import Torch/Isaac, load a checkpoint, query Spark, execute the guard or modify files. The full source map was fetched read-only; no GPU, Git or existing-document edits were performed while assembling this bundle.
