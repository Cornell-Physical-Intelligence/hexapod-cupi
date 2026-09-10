# Pause059 guard for quiet-priority smoke003 — frozen preparation

This is a bound successor of frozen smoke guard003. It prepares one explicit native004 `quiet_priority` smoke: standing32×1000, training32×24×2, final constant48×12 s and stop48×32 s. **No launch, pause, remote mutation or GPU work has been performed by this preparation.** Root alone binds final reviewed hashes and dispatches. Exact final source599/native43/host20 pins are now recorded in `FINAL_BINDINGS.json`; historical draft receipts below remain unchanged.

Reserved source/preparation/host names are `direct_omni_train_source_003`, `direct_omni_train_preparation_003`, and `direct_omni_train_host_003` under `/home/orionh/HEXAPOD_runs/mock_length_study_20260909`. Output is `direct_omni_train_smoke_003`; pause directory is `forecast_pause_059`; unit is `hexapod-direct-omni-train-smoke-003-20260910.service`. The host branch selector is explicitly `quiet_priority`, schema `direct315_quiet_priority_native_v3`, with temporal/spatial weights0.1 and additional quiet temporal weight1.0. There is no automatic pilot.

## Previous owner is cleanup ancestry only

The actual previous unit is `hexapod-direct-omni-preview-001-20260910.service`, invocation `7121f177c458460e81de0ef7f4a967f3`. Its host campaign failed because the expected final-integrity receipt was absent; the native recording job completed with exit0 and cleanup true. Source integrity and restoration were independently recorded. The guard does not reinterpret that host failure or claim policy qualification.

`previous_owner/` contains byte copies of four actual receipts from the already audited preview publication. The four hashes are bound in `PRIOR_PINS`: campaign, recording job, pause058 record and restoration. `previous_owner_remote_audit.json` preserves the earlier read-only audit, including matching invocation and both exact name/ID absence results; it is historical evidence, not a live GPU-state claim.

Before pause, the runtime rechecks the previous unit is inactive/failed, rejects a conflicting nonempty invocation, verifies all four receipt hashes, verifies the failed recording-only campaign scope and original source/host identity, requires completed recording exit0/cleanup true, and verifies **only that recording job's exact container name and ID are absent**. A collected unit with an empty live invocation is allowed only with the pinned historical receipts. Missing receipts, active/replaced units, existing containers and unknown Docker errors fail closed.

## Preserved safeguards

The embedded restorer is byte-identical to frozen guard003. It restores only previously active forecasting timers and performs identity-checked cleanup of this new owner's containers. The fallback is armed before timer/service mutation and expires after45 minutes. The transient owner has2520 s runtime plus180 s stop allowance. Original per-job source009 supervision and dual GPU locks remain.

New host/source/native manifests and the original checkpoint are checked before pause and again before dispatch. The coordination file remains pinned to `22c615d67c6c4dc14ab370b291f8eddd72f8475ad34118b91a4237a34ee05fab`. Foreign CUDA workloads are rejected before any pause; the independently started halo replay is not silently included in timer ownership. Root handles any separate authorized weather deferral outside this guard. No historical source hash is reused as a new source pin.

## CPU tests and final binding

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tmp/direct_omni_train_smoke_guard_004 -p 'test_*.py' -v
```

`test_output_draft_002.txt`: **31 tests pass**. The suite executes actual guard/restorer code against fake services/processes and temporary files. It covers original valid cleanup, missing/changed/malformed receipts, incorrect recording scope, unit replacement/collection, unknown inspection, exact two-container-identifier absence, failed source/selection checks, fallback-before-pause order, revalidation-before-dispatch, exact-owner cleanup after uncertain dispatch, and foreign CUDA rejection without mutation.

The original binding checklist was: before freezing, root supplies the final native004 contract SHA, measured source003 manifest SHA, frozen host003 manifest SHA and exact launcher SHA. Fill only those pending constants, verify the real remote inputs, and preserve the current previous-owner pins unless a different job actually becomes the immediately preceding owner. The final bound runtime now passes `require_final_bindings()`; this does not replace live remote preflight or grant a GPU slot.

Any repository publication must follow `docs/PROJECT_SITE.md`: corresponding evidence/progress/roadmap updates or checked no-impact declaration, a new bounded `site/updates/` record, and required site checks/build. Parent evidence under `parent_evidence/` remains historical; frozen originals are untouched. Operational admission is distinct from locomotion acceptance and does not advance Stage2 by itself.

Final validation: `test_output_final.txt` records31 focused passing tests after binding, including all8 unchanged embedded-restorer tests. Host20/native43 inventories were verified locally; root retains source599 remote verification and dispatch ownership.
