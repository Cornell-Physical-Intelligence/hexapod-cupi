# Single native-PPO preview host

CPU-only preparation for one fresh, unqualified 38-second full-robot recording. The exact adapter002 remains frozen: 1,900 controls, 950 frames at 25 fps, 1× playback, ground command arrows and actual motion trail, quiet/forward/stop/strafe/stop/arc/stop sequence. There is no training, pose forcing or replacement actor. A first terminal event stops and preserves a partial failed video.

Inputs pin native003/source002 (598 files), its original 16-file legacy inference package, source009's 926-file supervisor tree, the native26-file contract and the reviewed20-file preview adapter. The source009 `run_owned` and `owned_container` functions are imported without body edits: both GPU locks, 600-second allocation, 90-second AppReady, liveness/resource/contact-overflow checks and identity-checked cleanup remain unchanged. Only command, native source verification and truthful recording metadata callbacks are supplied. Source009 is not the physical environment.

The selected same-source curriculum or CAPS **50-update pilot** must be completed with all six immutable phase receipts, exact standing admission and the exact final checkpoint. Smoke/partial campaigns are rejected by the frozen preview contract. The selected campaign and checkpoint hashes are explicit CLI inputs; no checkpoint is preselected here. Completed acquisition does not make the policy qualified.

```sh
python3 launch_preview_spark.py \
  --source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_train_source_002 \
  --contract /home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_train_preparation_002 \
  --supervisor-source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009 \
  --adapter /home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_preview_adapter_002 \
  --pilot SELECTED_COMPLETED_PILOT_ROOT \
  --checkpoint SELECTED_COMPLETED_PILOT_ROOT/train/policy/final.pt \
  --checkpoint-sha256 SELECTED_FINAL_SHA256 --campaign-sha256 SELECTED_CAMPAIGN_SHA256 \
  --output /home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_preview_001 \
  --preflight-only
```

`--preflight-only` verifies read-only inputs without creating output or constructing the simulator. It rejects reused/overlapping outputs. Actual allocation is root-owned through the separate guard, never by an agent running this example.

The container mounts source, contract, adapter, pilot, standing admission and checkpoint read-only. Only the fresh `/output` is writable. The output contains the native `recording/` state, video, trace, configuration and provenance, plus host job/contact audit/log/campaign receipts. Completion requires exact source/checkpoint identity, actor/critic normalization readback, 1,900 controls/950 frames, no early event, verified video/trace hashes, successful App close and unchanged inputs. Rendering completion is not a Stage 2, quiet-stop, torque or terrain acceptance gate. Root must independently decode the actual video after acquisition.

Eight focused CPU tests passed, including actual imported ownership-function identity, native source callback, RO mounts, exact selected campaign/checkpoint forwarding, reused output rejection, partial video and failed-state preservation. Smoke/partial-pilot and wrong checkpoint rejection remain the independently tested frozen adapter contract (23 tests), reused rather than reimplemented here. No GPU execution occurred.

This bundle was prepared only in ignored `tmp/`. Under `docs/PROJECT_SITE.md`, publishing it requires a new append-only site update, and any eventual video requires updating the latest-recording entry with the actual checkpoint, SHA, playback rate and limits. No existing site/docs/frozen evidence was changed.
