# Canonical-model native inspection host

This prepares one native Isaac inspection of the user-selected motor-mass-corrected detailed direct-drive robot. It is not a task, standing test, gain search, controller, training run or physical admission. The exact canonical selector in `inputs/active_model.json` identifies the 7.466088235 kg, 19-body/18-joint asset and its URDF, model and USD. The mass distribution remains the documented nominal motor correction, not a measured identification.

The host accepts only phase `inspection`. Its native source is the separately frozen `updated_native_inspection_001` producer owned by the native inspection agent. That producer verifies the exact nine-file asset, performs its declared zero-gravity initialization and eight explicit 2.5 ms steps, and checks native body/joint/inertia/limit/SDF readback. SDK warm-up is recorded separately. No target, gains, actuation or old checkpoint is supplied by this host. The native source is bound to freeze `c59ebd53bb221fd17b46d9fbc31e3df7240271bfd13d6f7269d7307876c7bfaa` (21 payloads), and the exact canonical asset identities are bound. The pending-pin negative path remains tested; root must review and perform actual Spark preflight before allocation.

The container mounts only native inspection source at `/inspection:ro`, the canonical asset at `/asset:ro`, and a fresh writable output. It runs `/inspection/run_inspection.py --asset /asset --output /output/inspection --device cuda:0 --headless`. Source009's Python supervisor is reused only on the host for exact ownership, locks, contact-overflow scanning, AppReady/liveness and cleanup. Its `run_owned` and `owned_container` functions are not rewritten or recompiled. No historical C-study source, asset or admission is mounted in the native container.

The inherited runtime-field name is retained for compatibility, but its value is the new canonical inspection source freeze, scoped `canonical_native_inspection_only`. It is never the historical C-runtime hash. The host replaces only the command callback, source verifier, expected runtime identity and truthful inspection metadata. Native result validation and an actual post-exit original-input readback precede completed campaign status. The host then inventories every inspection payload, including native shutdown logs/error files excluded from the producer's pre-close seal, into `inspection_immutable.sha256.json` and rechecks that inventory during finalization. Failure, interrupted startup/cooking and final integrity errors remain failed evidence; a successful inspection still records training and physical admission as false.

The unchanged supervisor permits 600 seconds total and requires the actual `REFERENCE_SCREEN_APP_READY` marker within 90 seconds. Native code emits that marker immediately after real AppLauncher construction, before asset loading/SDF cooking. Cooking consumes the remaining 600-second envelope. This is a bounded first attempt with unknown native cooking duration, not an assurance of timely completion. There is no automatic extension or retry.

```sh
python3 -B launch_inspection_spark.py --preflight-only \
  --source /home/orionh/HEXAPOD_runs/canonical_direct_20260910/inspection_source_001 \
  --asset /home/orionh/HEXAPOD_runs/canonical_direct_20260910/asset_001 \
  --supervisor-source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009 \
  --output /home/orionh/HEXAPOD_runs/canonical_direct_20260910/native_inspection_001
```

Root alone freezes final input bindings, transfers, performs real preflight and dispatches through the separate `canonical_native_inspection_guard_001`. The prior completed C-study preview is only that guard's cleanup ancestry. No old physical admission carries forward.

Twelve focused CPU host tests pass, covering pending pins, standard-library import, exact canonical asset identities, source-only native runtime binding, read-only mounts, single-phase scope, original ownership functions, no-checkpoint CLI, failure-preserving finalization and post-exit integrity. Guard tests are separate. Draft test logs are kept with their exact scope; no native simulation was executed by these tests.

Any adoption follows docs/PROJECT_SITE.md: root adds the bounded central update and context, retaining the distinction between offline preparation, native inspection, physical admission and later training. No tracked file, existing evidence or remote state was changed here.
