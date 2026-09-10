# Contingent CAPS500 progress recording

This is CPU preparation for one actual 38-second native direct-PPO video. It does not allocate a GPU or claim that the currently selected training run has completed. The final checkpoint, completed campaign, six jobs and pause063 restoration hashes remain `PENDING` in `SELECTION.example.json`. Root must supply a separate exact selection receipt after actual completion and explicitly choose the final checkpoint. The example refuses before any process call or pause.

The reserved owner is `hexapod-direct-omni-extended-preview-001-20260910.service`, output `BASE/direct_omni_extended_preview_001`, pause064. BASE is `/home/orionh/HEXAPOD_runs/mock_length_study_20260909`. Root is the sole dispatcher and must confirm the reserved paths are still unused.

Only `direct_omni_train_extended_caps_002`, invocation `ca9455cefb554c7691e7f8b2ed3ac939`, is eligible. Its campaign must actually complete all six phases and 500 updates using source004, native005 and training host005. The host revalidates every accepted phase and immutable phase tree, standing admission, original-policy lineage and selected final checkpoint. The guard additionally requires all six completed jobs, twelve distinct container names/IDs proven absent, terminal inactive/exit-zero state, and exact restoration of the timers active before pause063. These checks run before pausing weather and again before recording dispatch. A collected transient InvocationID is accepted only with the exact statically bound invocation and pinned historical/terminal receipt lineage; a different nonempty invocation rejects. The included active snapshot is historical provenance only, never terminal evidence.

The selected checkpoint hash must agree across the root selection, actual final checkpoint bytes and accepted training receipt. The selection campaign hash must equal its prior-owner campaign pin. Smoke, 50-update pilots, quiet-priority branches, partial/failed campaigns, a missing final stop phase, changed source/schema/host/checkpoint, duplicate IDs and unknown Docker state reject. Quiet performance is not a new admission gate here: a completed but unqualified CAPS500 result can be selected for an honestly labeled progress video. No automatic selection or Stage2 promotion occurs.

The frozen preview adapter004 and host003 remain unchanged. They retain 1900 physical controls, 950 frames at 25 fps, 1280×720 full-robot capture and 1× playback, ground command arrows/labels, actual trajectory, first-terminal failure capture, strict actor/critic/normalizer readback and no ideal-pose feedback. The pre-shutdown seal executes before native application close; the host separately verifies the original inputs and sealed output after process exit. There is no rerender or fabricated motion in this preparation.

The existing source009 recording supervisor remains at 600 seconds with a 90-second AppReady limit. This guard preserves the original preview owner's 720-second bound, 180-second stop allowance and 15-minute conditional restoration fallback. The embedded restorer is byte-identical to preview guard001. Both GPU locks, forecast-service ancestry checks, rejection of unknown CUDA workloads, graceful authorized weather stopping and restoration of only previously active timers are preserved. Coordination is bound to `35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3`; that authorization does not permit stopping arbitrary non-weather processes.

## Root's eventual CLI

After independently verifying actual terminal results, copy `SELECTION.example.json` to a separate selected receipt and replace every pending hash using actual bytes. Do not modify this frozen bundle. The receipt requires the completed campaign hash, selected final checkpoint hash and exactly nine prior hashes: campaign, six jobs, pause063/pause.json and pause063/restored.json.

```sh
/usr/bin/python3 BASE/direct_omni_extended_preview_guard_001/launch_guarded_remote.py \
  --selection BASE/direct_omni_extended_preview_selection_001.json \
  --selection-sha256 ACTUAL_ROOT_SELECTION_SHA256
```

`BASE` above is a placeholder for the absolute run directory, not a shell environment variable supplied by this bundle. The invoked frozen host uses explicit `--source`, `--contract`, `--supervisor-source`, `--adapter`, `--pilot`, `--checkpoint`, `--checkpoint-sha256`, `--campaign-sha256` and fresh `--output`; `--pilot` is the existing API field for the selected extended campaign. The host mounts source, native contract, adapter, full selected run, standing and checkpoint read-only, with only the fresh recording output writable.

## CPU verification and scope

All 29 focused tests pass in `tests_final.log`, including actual guard ordering with fake effects, unchanged embedded cleanup, pending/partial/wrong-source/checkpoint rejection, twelve absence checks, actual native005 extended selection, the 599-file source004 entry binding, actual host command mounts, seal/readback linkage and a standard-library-only import. `tests_draft_002.log` preserves an initial test fixture key error (`run_omni_flat.py` instead of the actual `tools/train_length_study.py`); only that test expectation was corrected. No GPU or remote operation occurred. `INPUT_VERIFICATION.json` records the original dependency maps verified unchanged.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tmp/direct_omni_extended_preview_guard_001 -p 'test_*.py' -v
python3 -B tmp/direct_omni_extended_preview_guard_001/verify_bundle.py
```

The central framework in `docs/PROJECT_SITE.md` applies to eventual adoption: root must add a bounded site update, update the presented execution state/media only from actual evidence and run the required site checks. `PROJECT_SITE_UPDATE_PROPOSAL.json` is a preparation-only proposal, not a tracked update or execution claim. Latest checkpoint, latest recording and accepted benchmark remain separate.
