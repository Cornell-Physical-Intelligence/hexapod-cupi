# Optimized canonical standing: interrupted by unrelated CUDA

The one-robot standing run started on the detailed direct-drive model with the reviewed contact-classifier optimization. Its original ownership monitor detected an unrelated CUDA process and yielded the owned job. **This is an operational interruption. No completed standing pass, physical rejection, walking result or training start is claimed.**

The host recorded `RuntimeError('Unrelated CUDA process appeared; yielding this owned job')`. The competing process was PID3758066, `/home/orionh/ithaca-reconstruction/env/bin/python`, in `session-c10825.scope`. The host's own container name and exact ID were absent at the terminal audit, all pinned input trees remained unchanged, and pause009 restoration completed. This bundle preserves that historical audit; it is not a current Spark status probe.

## What actually ran and what remains incomplete

- Source003 planned1replica×1000controls,8native steps per20ms control (8000steps at400Hz). No policy was loaded. Source003 uses the reviewed exact-output contact-classifier optimization; scene, geometry, provisional servo and gates were retained.
- Unit: `hexapod-canonical-native-standing-003-20260910.service`; invocation: `ff270c50458f47e5940dd93c4b3aa845`. The host job lasted approximately101.83s before interruption. Terminal unit status was failed/exit1 with MainPID0.
- The raw log reached the printed `control=800` milestone and eight NPZ chunks were persisted. This is a lower-bound progress observation, not an exact final control count or completed score. The native `state.json` remains its original `running` snapshot with `explicit_steps_completed=0`, empty checks and no errors. That field was not finalized and must not be interpreted as no physics having occurred.
- No final `session.json`, `control_trace.npz` or standing report exists in the exact31-file raw inventory. No missing result was reconstructed or scorer run on an invented complete trace. The raw `native_errors.json` is empty; its incomplete lifecycle cannot establish a final no-error admission.
- The terminal auditor accepted the failure/cleanup evidence with `errors=[]`, `standing_completed=false`, `raw_acquisition_completed=false` and `native_validation.attempted=false`. Those flags describe the audit scope; they do not accept standing.

Source003 still needs fresh matching1and32standing admissions. The separate405/408 PPO implementation is CPU preparation only. The provisional software servo/48V envelope remains uncalibrated hardware behavior; no voltage or motor qualification follows from this interruption.

## Preserved payloads

| Folder | Exact input | Payloads before its manifest |
|---|---|---:|
|`source/`|standing003 `e922ff13…`|87|
|`host/`|host003 `968caeb4…`|28|
|`guard/`|guard004 `5f13bfd6…`|33|
|`auditor/`|auditor003 `1628f4e5…`|7|
|`root_checks/`|source/host/guard transfer, actual setup, tests, dispatch and progress snapshot|10|
|`terminal/`|all31logical raw payloads plus original audit/encoding/inventory metadata|31raw +4metadata|

Every frozen component and root snapshot is byte-verified by `PROVENANCE.json`. No source package or final response was selectively rewritten. There are no Claude thinking/event streams in this selection. The canonical asset itself is not duplicated; its exact9-file map and before/after verification remain bound in the native/source and terminal evidence. Old source002 and the separate32-replica throughput-stop evidence remain unchanged outside this package.

All **31original raw files totaling144,818,179bytes** are present as **23,788,742stored bytes**. Thirty are stored without transformation. The138,049,419-byte contact JSONL uses the exact gzip bytes fetched from Spark; it was neither recompressed nor truncated during publication. `RAW_STORAGE.json` binds both stored and decompressed SHA256/size, while the original `terminal/RAW_ENCODING.json` and `terminal/RAW_SHA256.json` remain unchanged. No large temporary decompressed file is needed. All public payloads stay below48MiB.

## Portable verification

From a clean checkout, this command checks every public byte, all component manifests, exact failure/cleanup/source identity and all31decompressed raw hashes using bounded1MiB reads:

```sh
python3 -S artifacts/mkii_updated_2026-09-10/native_standing_interrupted_003/verify_bundle.py
```

It uses only Python's standard library and never imports the native inspector, Isaac, Torch, a host launcher or a GPU tool. `stream_raw.py` can emit one logical raw file to stdout without creating a decoded copy. Its3focused tests cover exact identity/gzip reads including a partial final line, mutated stored bytes, truncated gzip and path escape. `STREAM_VERIFICATION.json` records the local complete raw replay.

## Publication ownership

Only this new artifact directory was written for this subtask. Root owns the relevant Markdown/STATUS update, new bounded `site/updates` record, site validation/build and commit/push under `docs/PROJECT_SITE.md`. `PUBLICATION_PROPOSAL.json` suggests the factual update without modifying shared files. The poster should retain the mission milestones: first walking benchmark reached historically, omnidirectional motion in progress, terrain/perception prepared in parallel, autonomous surveying as the final objective.
