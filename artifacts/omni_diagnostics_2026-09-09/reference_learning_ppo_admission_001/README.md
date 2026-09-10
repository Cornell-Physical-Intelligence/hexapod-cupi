# Actual learning recovery admission001 — zero PPO updates

The three-phase admission allocation completed: fresh standing, calibration, and **712 controls of the new recovery infrastructure**. The independent numerical review passed the reset/history/critic/clock evidence. **No PPO update occurred, no gait was physically admitted, and no automatic training continuation was authorized.** Any later root decision or training run is outside this historical bundle.

The raw experiment contains **six finite reference-failure transitions across five event controls**. Two rows completed the full 200-control recovery before the bounded run ended; four other rows were still recovering. There were **15,443 active transitions and 941 excluded recovery transitions** after initial startup. The zero-residual controller's physical failures remain in the evidence; recovery success does not turn them into successful walking episodes.

The independent review replays all 5,697 recorded physics samples, all 14 sensor clocks and reset epochs, the actual final packet / selected history reset, and zero bootstrap for the true terminal events. No GPU timeout event occurred; timeout bootstrap has CPU test evidence only. No normalization, rollout storage, optimizer or learning-iteration update is claimed during this infrastructure phase. Measured timing is instrumented recovery timing, not PPO training throughput.

Active-row requested torque peaked at **1.590736 N·m**, with no active sample above 1.6. Initial startup and excluded recovery requested-torque excursions remain recorded separately: maxima **3.961889** and **3.462666 N·m** respectively, with actuator clipping retained. This is one reason the infrastructure result must not be presented as a physical locomotion qualification.

[Independent numerical review](independent_numerical_review/README.md) and [report](independent_numerical_review/report.json) retain per-row progress, exact events, source bindings, corruption tests and limited memory evidence. Its owner freeze is **8a9cecf0cd137821a9cf690bc4d60422aeecb0fd205f8c52082c219e804d429a** (13 payloads), copied unchanged.

## Identity and restoration

Both terminal audits verify unchanged source009 (926 files), consumer003 (51), host001 (3), guard001 (11), observation005 (160), bridge001 (18) and 550 admitted asset files. All **three exact owned names and container IDs are absent**. Forecasting timer restoration is recorded at Unix **1789048842.716838**. The terminal unit is inactive with exit code zero.

Systemd collected the transient unit's live InvocationID after termination. The first strict audit rejected that empty field, and its empty output/error are preserved under `terminal_audit/audit_attempt001`. The separately frozen audit002 requires the exact historical start and terminal accounting journal records for invocation **d161f77036f346d2b1abe244d95b2dbe**. It does not silently accept an unknown or reused invocation. Both before/after raw inventories and every source/asset/ownership/restoration check match across the completed fetch.

The full 137-payload preparation is [referenced separately](../reference_learning_ppo_preparation_001/README.md), manifest **46796a7939968f363a2cc39799bce50c9d97c2bcbd8bb7f7cef1481d09119fd0**. Its source/host/guard payloads are not duplicated here. Exact input manifest copies are retained in `references/`; reconstruction provenance is in [RECONSTRUCTION.json](RECONSTRUCTION.json).

## Portable raw evidence

All **74 original raw files, 540,347,369 bytes**, are preserved. Two calibration NPZs exceed 100 MiB and are stored as lossless chunks of at most 64 MiB. No compression, reencoding, truncation or original-file mutation is used. Every chunk and reconstructed whole file has a SHA256 binding. The original remote and local files remain intact.

```sh
python verify_payload.py
python reconstruct_raw.py --output /path/to/fresh_external_directory
```

Both commands use only the standard library and run without Isaac, Torch, SSH or GPU access. The verifier checks every frozen payload and chunk stream, original raw hashes, completed phase receipts, immutable phase maps, input inventories, historical invocation, cleanup/restoration, and the independent numerical result's stated scope. Reconstruction verifies all 74 output hashes again. This is an immutable terminal evidence bundle, not a runnable training allocation.
