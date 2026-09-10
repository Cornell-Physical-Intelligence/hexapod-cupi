# Extended direct PPO preparation and smoke launch snapshot

Source004 was built once from the exact cold589 source and verified against its complete **599-file** inventory. Native005 adds an explicit 500-update allocation with milestone diagnostics; it preserves the original checkpoint, dynamics, curriculum and objective mathematics. The normal-CAPS 500-update comparison remains a separately dispatched experiment. This bundle contains preparation and a historical smoke launch snapshot, not its terminal result or a quality gain.

Root recorded **16 native, 47 host and 37 guard CPU tests passing**, verified transferred inputs, and completed the actual read-only host preflight. The native root rerun is recorded in the root readiness report; the root host and guard transcripts are included. Owner test receipts remain distinct and unchanged. Packaging only rechecked hashes; it did not repeat the tests or access Spark.

## Exact scope at capture

The smoke unit is `hexapod-direct-omni-train-smoke-004-20260910.service`, invocation `5d7736253ce84660baefda6a672bf010`, output `direct_omni_train_smoke_004`, with `forecast_pause_061`. At the captured snapshot it was active, the campaign was preparing, no phase had been accepted, and no training updates were recorded yet. This is not a statement of current live state. The 500-update allocation had not been dispatched at this snapshot and is not launched by guard005.

This smoke uses 32 replicas × 24 controls × 2 quiet-priority updates, preceded by fresh standing and followed by final constant-command and stop diagnostics. A later same-source normal-CAPS extended run requires completed smoke admission, explicit root selection and its own guarded allocation. It must start from the original checkpoint; no automatic continuation is authorized.

The original checkpoint SHA is `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`. Source004 manifest SHA is `aa70659d33243f1e28e9c71b1a7f9d33341642db770fd7356192573a26fe904e`, with plan `b2286828148c89fb34624d38beaf84b389584f7bda4d2648512e017da4b50acb` and origin `850aa811e702dcfd0284862c0c27f475e82a5335aab06b96e0ea21ea51e6f8b7`.

## Preserved inputs

| Directory | Original payload count | Scope |
|---|---:|---|
| [native005](inputs/native005/README.md) | 28 | Native allocation, receipts and regression evidence |
| [host004](inputs/host004/README.md) | 37 | Explicit smoke/pilot/extended dispatch and versioned extended deadline |
| [guard005](inputs/guard005/README.md) | 28 | Bounded smoke, exact prior-owner cleanup, dual locks and timer restoration |
| [root readiness](inputs/root_readiness/README.md) | 14 | Source build/audit, preflight, root test transcripts and launch snapshot |

All 107 original payloads and all four original freeze manifests are copied byte for byte. Old draft and readiness wording inside those frozen bundles describes its own historical stage. The root readiness evidence supplies the later build and launch facts without rewriting it. Prior quiet-priority pilot completion is cleanup ancestry; its **0/48 quiet result remains a failure**. Stage 2 is incomplete.

The full 599-file source hash inventory is preserved in [the root source audit](inputs/root_readiness/remote_source_audit.json). The large complete source asset tree is not duplicated here. Rebuilding or rechecking those source bytes needs the original cold source/assets and builder inputs; verification of this publication's exact preserved evidence is standalone.

## Verify

Run `python3 -B verify_bundle.py` from any directory. The verifier resolves paths relative to itself, checks the complete outer file inventory and every hash, then independently checks each unchanged nested freeze manifest and its full payload inventory. It rejects added files, missing files, changed bytes, path escapes and symbolic links. It does not import the launchers or perform remote actions.

The intended publication path is `artifacts/omni_diagnostics_2026-09-09/direct_omni_extended_preparation_001`. Root owns repository adoption, the required central project-site update and all GPU execution. The companion update proposal is outside this immutable wrapper.
