# Spark automation block: verified operations snapshot

The user directed that Spark's external automated jobs, including external Git-triggered jobs, stop so HEXAPOD can retain compute priority. At the saved observations on **11 September 2026, 00:31–00:37 UTC**, root stopped the exact reconstruction worker, established its persistent queue block, and verified **31 user unit masks:30 forecast timer/service names plus Ollama**. Four system `wx-forecast` timer/service names were already masked. No new robot training or retry launch is claimed by this bundle.

## What was actually verified

- Every selected user unit had both `LoadState=masked` and `UnitFileState=masked`, inactive state, and service `MainPID=0`. Timer units do not expose `MainPID`. Exact names and mask paths are in [the successful mask receipt](actual/automation_block_verified.json).
- Four direct starts and one `enable --now` attempt were rejected as masked. These are five specific probes; they are not tests of every possible external launcher. Masking the identified installed names also blocks those same names being re-enabled by the observed deployment path. It does not prevent someone replacing a mask file or directly starting another program.
- Eleven masked units retain old reservation-condition drop-ins and report `NeedDaemonReload=yes` even after the operator's explicit reload. The raw fields are preserved. Loaded state, file state and the five rejection probes verified the masks; this bundle does **not** claim every reload flag became `no`.
- The exact reconstruction worker was PID3803709, start tick140218694, in `session-c11072.scope`. Root bound the PID with a pidfd, verified identity again and sent SIGTERM. Its exit and the replacement queue-lock holder were verified. The [receipt](actual/reconstruction_block_receipt.json) binds the source's early reservation check, original/changed source hashes, the enabled `hexapod-exclusive-reconstruction-queue.service`, lock-holder PID3814169 and unchanged reservation marker hash.
- CUDA process queries were empty at the two saved successful verifications. This is a historical observation, not a live claim or a general root/manual-CUDA partition. Original per-job locks, workload checks and exact cleanup remain necessary. Only a later user instruction releases the persistent reservation.

## Original state and restoration evidence

The [original unit snapshot](actual/exclusive_automation_block_001/before.json) contains31 exact unit states and original file identities/modes. At capture,6 unit files were enabled (the five base forecast timers and Ollama),10 were disabled and15 static. One scout service was failed and the private scout service was in auto-restart/activating state with PID0. These are original states, not instructions to start every unit on release.

The [remote backup inventory](actual/backup_inventory.json) records32 original files totaling16032bytes:31 unit files under `/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_automation_block_001/original_units/` and the original reconstruction worker under `exclusive_reconstruction_001/queue-worker.original.py`. The original snapshots' hashes match the successful receipts; each unit backup's hash/mode matches its original metadata. Backup **contents remain on Spark**. This public package intentionally contains no foreign unit-source files, project-source contents, queued requests or reconstruction outputs.

A future user-authorized release must first compare current installed masks/gate/holder identities with these recorded identities, verify the remote original bytes, restore only the recorded originals and appropriate prior enable state, and verify the resulting state. It must preserve the queue, outputs and unrelated services. No release/restoration is performed or automatically scheduled here. The worker has no claimed mid-inference checkpoint resume. The receipt records an empty pending queue before/after and no queue mutation; it does not claim that an interrupted inference completed or that every remote output was independently inventoried.

## Failure and discovery records stay separate

The first unit installer completed the masks and then failed its verification lookup with `KeyError: 'MainPID'`, because timers lack that property. The [actual stderr](failures/automation_block.stderr) and exact original operator script are retained. Its empty stdout placeholder is hash-inventoried in `SELECTION.json` and **excluded as a result**, not called successful JSON. The later separate verifier produced the successful receipt. The scheduler inventory's initial failed scan also retains its stderr; its bounded successor inventory succeeded. Nothing rewrites these failures.

The read-only inventory found disabled private timers were insufficient: a private scout service could still auto-restart; the reconstruction worker was a daemonized queue worker outside those timers. The source maps and selected operational state preserve those findings without copying unrelated source excerpts or listings. The upstream SSH/HTTP/Git producer of reconstruction remained unidentified; root-crontab access was unavailable. These limits are not assertions that such producers do not exist.

## GitHub evidence and public selection

The accessible `CornellGeoData/Forecasting-Pipeline` repository returned **zero workflows and zero runs**, with no workflow paths in the recorded untruncated main-tree listing. The exact GET responses are in [github/](github/). No GitHub workflow was disabled, run canceled or repository changed. Pull-only visibility cannot rule out private webhooks, inaccessible automation or manual SSH producers. A source `git rev-parse` use recorded dependency provenance; it was not proof of a Git launch trigger. The historic Git report's proposals are separate from the later actual mask/queue receipts.

At assembly, all14 scheduler and24 Git-inventory payloads matched their original frozen maps. Only selected metadata and receipts are public. `SELECTION.json` records every source file's hash/size and its copied path or explicit exclusion; `COPY_VERIFICATION.json` records byte equality and original complete local checks. `operator_scripts/` contains the exact HEXAPOD-authored operations used; these scripts are **evidence, not commands to rerun**. The portable verifier never imports them or contacts Spark. Private Codex automation details are not included.

```sh
python3 -S /absolute/path/to/spark_automation_block_001/verify_bundle.py
```

The verifier checks public inventory and byte hashes, selected-original bindings, both saved receipt semantics, unit/backup mappings, exact stopped-worker identity, five refused probes,11 reload flags and read-only GitHub evidence. It cannot replay omitted original sources, fetch remote backup contents or establish current machine state. This package contains no GPU or remote mutation by the packaging agent. Root owns shared documentation, the new central `site/updates` record, checks and publication under `docs/PROJECT_SITE.md`.
