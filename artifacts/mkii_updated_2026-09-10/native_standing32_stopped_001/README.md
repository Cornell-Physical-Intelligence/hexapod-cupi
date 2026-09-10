# Canonical32 standing: operational throughput stop

Root stopped the exact owned32-robot standing allocation after measured throughput projected beyond its unchanged600-second phase budget. This is an **operational stop, not a physical rejection and not an accepted32-robot standing result**. No policy ran. The original source002 and physical/contact/quiet gates remained unchanged throughout the allocation.

At the recorded checkpoint, the contact stream had reached explicit counter1,761 of the planned8,000 at194.46s since dispatch; the latest complete row had1,638 patches. Logs had reached100 and200 controls. Root's `STOP_DECISION.json` and exact-owner `STOP_REQUEST.json` are retained. Subsequent data written before shutdown is also retained; this checkpoint is not presented as the final number of simulated steps.

The host campaign and job say `stopped` with `InterruptedError('Stop requested')`. The user service ended failed/exit1, MainPID0. The native `state.json` remains its **stale initial snapshot**, `status=running`, `explicit_steps_completed=0`, `checks={}`. That does not mean no simulation occurred, and it has not been rewritten as a terminal receipt. There is no final `session.json`, full-row-clock proof or standing report. Empty retained error lists cannot establish physical acceptance. The terminal auditor preserved this incomplete acquisition and verified exact owned name/ID absence, source/asset/input invariance and pause008 restoration.

## Complete lossless evidence

All **27 original raw files /1,836,888,673 bytes** are accessible, with no selection or omission. The original1,742,688,165-byte contact JSONL is stored through the exact acquired230,306,397-byte gzip stream. Those compressed bytes are split, without recompression, into five ordered parts of at most48MiB. All remaining raw files retain their exact bytes, including three flushed native substep NPZ chunks, the original native snapshot, logs, stop request and pause receipts. The original acquisition's324,506,905 stored bytes are represented losslessly; no full1.8GB expansion or temporary decoded copy was created.

`terminal/RAW_SHA256.json` binds original raw names, sizes and hashes. `terminal/RAW_ENCODING.json` preserves the original fetch encoding receipt, whose gzip path is now represented by parts. **`RAW_STORAGE.json` is the publication's authoritative storage mapping**: it records each part's hash and size, exact concatenated compressed hash and size, and the original uncompressed hash and size. Each part is below Git's100MB limit. The native NPZ payloads are immutable hardlinks locally; Git content is ordinary file bytes.

Run `python3 -B -S verify_bundle.py`. It checks every published payload and original frozen component, the actual stop/cleanup identities, all compressed bytes and all27 decoded raw hashes in bounded1MiB reads. It never expands a decoded file to disk, starts Isaac or changes an admission. Three small stream tests cover split gzip headers/payloads, missing/reordered/corrupt parts and unsafe paths.

To access a raw file, `python3 -B -S stream_raw.py run/standing/contacts.jsonl` writes its exact original bytes to standard output. Pipe it to a streaming reader; redirect to a file only if deliberately allocating the full decoded size. Identity-encoded raw files are directly available under `terminal/` as well.

## Runtime and performance provenance

- `source/`: exact51-payload standing002 source, freeze`acb58970…`, canonical19-body/18-joint detailed direct-drive model and provisional actuator assumptions.
- `host/`: exact22-payload host, freeze`c533454d…`; unchanged600s/90s ownership supervision.
- `guard/`: exact44-payload32-admission guard, freeze`f7c23291…`, requiring the same-source completed one-robot result.
- `auditor/`: exact7-payload terminal auditor. `terminal/audit.json` verifies source51, host22, guard44, nine-file canonical asset, prior admission and ownership supervisor926 before/after the run.
- `root_checks/`: real CPU setup, guard tests, dispatch, progress, measured throughput and operational stop decision/request.
- `performance_review/`: the separate frozen29-payload CPU investigation, freeze`304f9db2…`. All8,000 completed one-robot rows and93 replicated cases have exact candidate-versus-parent output parity on the review CPU; seven tests pass. The draft reduces local weighted32 classifier time from316 to166s while retaining all patches/gates/order. It did **not** run in this stopped allocation and is not a native speed or admission result.

Invocation`ce49acca09a14d77b081fc5cb6c41622` and its stopped owner are preserved exactly. Pause008 restored at Unix1789078963.631424. This is a historical terminal receipt, not live Spark status.

The next step is a separately versioned, explicitly reviewed optimization and actual supported-standing measurement. Runtime pins and admission must follow that source; CPU parity does not rewrite a previous native state hash. Root owns current execution, living documents and central poster publication under `docs/PROJECT_SITE.md`. This evidence stays immutable.
