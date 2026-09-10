# Native005: fixed extended direct-position PPO preparation

CPU preparation only. Root has not built this source or admitted/launched the new allocation. The schema is `direct315_extended_native_v4`. Existing native004 and all measured failures remain immutable.

`extended` is one explicitly selected **1024 × 24 × 500** allocation from original checkpoint `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`. Normal CAPS and quiet-priority CAPS remain separately selectable; this preparation queues neither and does not authorize both. The existing curriculum branch also remains available without changing its behavior. There is no resume path and no automatic continuation.

Smoke remains 32 × 24 × 2, with quiet-priority CAPS, fresh matching standing and both final evaluation acquisitions. Both pilot and extended require a **completed same-source smoke**, including its terminal campaign/source integrity and exact four accepted phase receipts. A previous source's smoke cannot admit native005. The original 50-update pilot and smoke have identical runner configuration, objective, evaluation gates, physical configuration, observation/action semantics, checkpoint calls and diagnostic cadence to native004; only the version identity changes.

For extended only:

- 500 updates, 12,000 control steps per continuously running replica, 12,288,000 aggregate transitions and 10,000 optimizer minibatches. The PPO storage remains 24 transitions × 1024 replicas and is checked before collection.
- Gradient measurements at relative updates **1 / 10 / 25 / 50 / 100 / 250 / 500**, minibatches 1 and 20: 14 sparse actor-gradient rows. Existing quiet/moving validity masks and diagnostic math are unchanged.
- Native `save_interval=1` and the actual RSL save implementation remain unchanged. Starting at native iteration 1847 gives ordinary files `model_1847.pt` through `model_2346.pt` (500 distinct files, plus RSL's existing duplicate final save call). Relative decision copies are `decision_001.pt`, `010`, `025`, `050`, `100`, `250`, `500`; explicit `final.pt` and strict actor/critic/normalizer/Adam reload remain mandatory.
- The wrapper validates actual runner configuration and storage before collecting. After successful collection/reload it verifies all 500 nonempty ordinary file names, seven decision source hashes/relative iterations, final hash and declared cadence. The host result validator independently recomputes the readback and verifies all raw hashes and completion counts. A model ZIP saved under a different filename can have a different file hash; no incorrect byte-equality assumption is made between the final file and the last native autosave.
- Explicit decision copies are made after successful learning, as in the parent. If learning fails earlier, the complete prefix of native autosaves remains preserved and unadmitted; no synthetic final or decision success is produced. The failure receipt retains the actual completed update count.
- Declared host training bound is **1800 seconds**, other phases 600 seconds, AppReady remains 90 seconds. The separate host owner must implement and bind the actual deadline change in both supervisor seams; merely passing an argument cannot alter the frozen supervisor's two 600-second literals. No host/source/guard is built here.

The measured native004 50-update learner took 100.344715 seconds. A tenfold **1003.447-second linear extrapolation is sizing context, not an ETA or success prediction**. Root reported all 48 quiet trials failed; the largest 1.870 m stop-window displacement included a reset, so it is not continuous physical drift. This preparation does not claim longer training will correct the failure. The branch decision remains with root after the physical-result review.

All 500 native autosaves may remain on Spark (about 2.19 GB using current file sizes). Publication should inventory/hash every remote raw file and explicitly omit ordinary autosaves from local copies while preserving selected decision/final checkpoints and analysis-complete raw data. There is no new save filter or callback. Raw audit tensors remain retained as in native004 and grow with collected controls; fixed PPO storage does not imply fixed total memory. Existing 50-update CUDA figures do not establish full 500-update peak memory or throughput.

## Small reviewed code delta

`direct_config.py` adds the allocation/protocol, shared milestone selectors and an extended-only algorithm argument. `caps_ppo.py` reads that validated milestone tuple; its PPO/loss/RNG operations are unchanged. `direct_training.py` adds extended-only configuration/storage and terminal checkpoint readback, with the original save method untouched. `direct_contract.py` extends same-source smoke admission and validates the declared larger result. `build_source.py` adds only the new allocation choice. Loss, curriculum, gradient arithmetic, stop scorer and quiet gates are byte-identical to the frozen parent.

The plan's historical `training_iterations=50` remains the original pilot default; the mandatory `direct_recovery_training` protocol and explicit selection bind extended to 500, and the entrypoint resolves `direct_selection['updates']` before training. No implicit command-line iteration override is accepted.

## Focused CPU evidence

`tests_focused_001.log`: 15 passing tests, including 8 successor tests plus inherited parser/physical-AST/smoke-boundary/stdlib-host and optimizer-evidence checks. `tests_runtime_rejection_001.log`: one additional passing test covering five invalid runtime storage/config/cadence states before collection. The initial `tests_extended_001.log` preserves a test-only syntax failure, corrected before the passing runs.

The actual one-update CPU RSL oracle comparison uses the unchanged old50 configuration and compares frozen native004 versus native005: exact losses, actor/critic/normalizers/Adam, global and CAPS-private RNG, history, learning rate and diagnostics. The 500-update save/receipt regression uses the **actual wrapper with synthetic learning**, not a simulator or 500 PPO updates; it tests RSL's pinned update/save ordering from native index 1847. The synthetic test saves tiny checkpoint stand-ins. No real physical or CUDA admission follows from this result.

Reproduce all 16 distinct focused tests:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=tmp/reference_residual_ppo_001/_deps:tmp/direct_omni_recovery_001/native_005 \
.venv/bin/python3 -m unittest test_extended \
  test_native.NativeTests.test_entry_only_declared_physics_allocation_delta \
  test_native.NativeTests.test_stdio_host_import_has_no_torch_or_numpy \
  test_native.NativeTests.test_completed_smoke_campaign_boundary_not_just_phase_states \
  test_native.NativeTests.test_actual_late_app_flags_parse_before_launch \
  test_optimizer_contract -v
```

The existing `build_source.py --source <exact cold589> --output <fresh successor>` interface is preserved for root's later authorized build. It has not been executed in this preparation. `DEPENDENCIES.json`, `PARENT_NATIVE004_SHA256.json`, `CPU_READINESS.json` and `DELTA.patch` bind the local CPU evidence and inherited bytes. Eventual publication must follow `docs/PROJECT_SITE.md`; the bounded update proposal is provided separately.
