# Guarded native direct-PPO preview

This frozen behavior allocates exactly one fresh 38-second recording, after root selects a completed native003 curriculum or CAPS pilot. No GPU job has been dispatched by this preparation. The selected checkpoint is not chosen by the guard, and a complete progress video is not policy qualification.

- Output: `BASE/direct_omni_preview_001`.
- Owner: `hexapod-direct-omni-preview-001-20260910.service`.
- Pause: `BASE/forecast_pause_058`; both pause and output must be unused.
- Host deadline: unchanged 600 seconds, AppReady within 90 seconds.
- External owner: 720 seconds, 180-second stop timeout, 15-minute fallback restoration.
- Coordination: exact `22c615d67c6c4dc14ab370b291f8eddd72f8475ad34118b91a4237a34ee05fab`.

`BASE` is `/home/orionh/HEXAPOD_runs/mock_length_study_20260909`. Source002, native003, the20-file preview002 adapter and the7-file preview host are pinned in code. Rendering code remains byte-identical to the already tested preview adapter; this guard does not construct an actor or modify physics.

## Explicit late selection

Copy `SELECTION.example.json` to a separate root-owned receipt outside this frozen bundle. Root fills the selected completed pilot name (`direct_omni_train_pilot_curriculum_001` or `direct_omni_train_pilot_caps_001`), exact campaign SHA and final checkpoint SHA. Root also binds the terminal preceding CAPS invocation and each exact attempted job/campaign/pause057/restoration receipt. Attempted phases must be a prefix of the six planned phases; the previous CAPS job may have completed or failed, but it must be terminal with unchanged inputs and proven cleanup. Selection of a recording checkpoint still requires a fully completed six-phase pilot independently verified by the frozen adapter. Thus prior owner cleanup is separate from selected-policy acquisition completion and quality.

```sh
python3 launch_guarded_remote.py \
  --selection /absolute/path/root_selected_preview.json \
  --selection-sha256 EXACT_ROOT_SELECTED_RECEIPT_SHA256
```

The example's placeholders are intentionally rejected before service inspection, pause creation or process launch. The selected pilot/final path is derived from the two permitted names; arbitrary paths, smoke and unbound hashes are not accepted. Selection bytes are checked again before dispatch. Root must pin this guard's freeze and runtime externally before invoking it. Only root launches or signals Spark work.

Before any pause, the guard verifies source/input/selection bytes, the selected complete pilot, the prior CAPS terminal receipt map, every exact old container name **and** ID absence, and the exact previously active timer restoration. It rejects an active or conflicting prior invocation and unknown Docker inspection failures. Collected transient InvocationID is allowed only with the explicitly bound terminal receipts. GPU workloads must belong to the user-authorized forecasting services; unrelated processes reject before pause.

The existing guarded sequence is retained: arm fallback first, gracefully stop previously active forecasting timers/services, acquire both `/opt/wx/gpu.lock` and `/tmp/hexapod-isaac-gpu.lock`, run the original preflight, release guard locks, reverify inputs, dispatch the one owned host. The source009 host reacquires both locks during its allocation. The exact reviewed embedded restorer stops only the owned identity, refuses unknown cleanup, and restores only timers that were active before the pause. An input failure after pause restores directly without stopping an unlaunched owner; an uncertain dispatch failure stops only this new owner before restoration.

## Verification and publication

Eighteen focused CPU tests pass: seven selection/ownership tests, three executions of the actual guard main with effects mocked, and eight executions of the unchanged embedded restorer. They cover smoke/unbound selection, reused output, exact12 name/ID absence checks for a six-job predecessor, wrong invocation, missing restoration, unknown Docker state, lock/deadline constants, recovery ordering and identity-safe stopping. The host's eight tests and adapter's23 tests cover source/selected-checkpoint/partial-pilot/partial-recording admission separately. No local test proves actual future camera availability or policy quality.

This is ignored `tmp/` preparation. Under `docs/PROJECT_SITE.md`, root adds an append-only evidence/update record when publishing this bundle. After an actual verified recording, root updates the media registry with its real controller/checkpoint, SHA, 1× playback and qualification limits. No site, tracked source or frozen evidence was edited here.
