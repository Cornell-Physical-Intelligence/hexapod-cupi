# Native005 CAPS500 diagnostic preview

This is a prepared external recording adapter for an explicitly selected, completed source004 CAPS500 run accepted by training host005. No future campaign or checkpoint is guessed, and no video has been rendered by this adapter. A failed or partial retry is not eligible. Stage 2 remains incomplete.

The adapter requires six completed phases: standing, original constant, original stop, train500, final constant and final stop. It checks the exact native005 contract, source004 plan/dynamics, host005 freeze, accepted phase receipts, full phase trees, standing alias and actual final checkpoint bytes. The final model SHA is mandatory on the CLI. The legacy `--pilot` argument and `pilot` provenance field now point to the completed **extended** run; they do not mean a 50-update pilot. Smoke, 50-update, curriculum and quiet-priority campaigns are rejected. Completion is diagnostic acquisition and never a quiet/quality pass.

## Unchanged recording and shutdown

Rendering, control loop, source entry instrumentation, strict actor/critic/normalizer readback, first-terminal handling and shutdown seal remain byte-identical to repaired adapter003. The exact selected model acts deterministically through the native direct315 evaluator. No prescribed body trajectory or ideal-pose feedback is added.

The full robot is recorded at 1280×720, 25 fps, 1× real-time playback: 1900 actual 20 ms controls and 950 frames over 38 seconds. The schedule remains quiet4s, forward6s, stop4s, left-strafe6s, stop4s, forward-left-arc6s, stop8s. Ground command arrows, labels and the actual trajectory trail remain. An early terminal event preserves the pre-reset trace and last valid image, then fails the complete-video contract.

The pre-shutdown integrity seal is written before the exact installed SimulationApp.close behavior. The host separately requires a successful native process exit, owned cleanup and actual original-input readback after exit. No required evidence relies on code executing after a process-exiting close. The original failed preview and its absent after-close finalizer remain untouched.

Only explicit rendering changes are one environment, RGB camera rendering and output dimensions; the saved native 48-environment evaluation config and render delta remain separate. Target limiter, PD, solver, masses, joint order and actor inputs are unchanged.

## Source and invocation

Source004 has 599 files, manifest `aa70659d33243f1e28e9c71b1a7f9d33341642db770fd7356192573a26fe904e`. Native005 freeze is `0bdd2abf3811e68ac59e4fbc325eb5d2bb96afd2183eb84c4d37faeb86ff9e9f`; eligible training host005 freeze is `2db7e44ec40543753a6d1325f6d5dc9ed81be2948931b2ca670deffda68ce959`. The entry bytes were reconstructed by the two unchanged frozen builders and exactly match the root-audited source004 entry SHA `1f0b7d04ba539da146d08756be474ade4dba8fbb3f542daaf63a3529c66eff70`. This is a small source fixture reconstruction, not a new source/asset build.

```sh
python3 record_direct_preview.py --preflight-only \
  --source-root /absolute/direct_omni_train_source_004 \
  --native-contract /absolute/direct_omni_train_preparation_004 \
  --pilot /absolute/completed_extended_caps_retry \
  --checkpoint /absolute/completed_extended_caps_retry/train/policy/final.pt \
  --checkpoint-sha256 ACTUAL_FINAL_CHECKPOINT_SHA256 \
  --admission /absolute/completed_extended_caps_retry/standing/admission.json \
  --output /absolute/fresh_recording_output
```

Root must supply actual terminal selection and use the prepared preview host003/owned guard for execution. The CLI example is not a launch request. All source, contract, selected run, checkpoint and admission mounts stay read-only; only the fresh recording output is writable.

## Verification and framework handoff

The final CPU suite checks source004's exact entry seam and AST round trip; unchanged evaluation construction; six-phase/500-update selection; wrong source/native/schema/checkpoint and missing phases; pending-host refusal; native camera timing; unchanged complete/terminal control loops; strict model readback; and the process-exiting shutdown seal. Synthetic fixtures are not physical recordings. `verify_bundle.py` checks this entire frozen inventory without simulator imports.

No remote actions, GPU launches, existing tracked-file changes or frozen-input edits occurred. Root owns eventual adoption, the append-only central `site/updates/` record required by `docs/PROJECT_SITE.md`, actual preview selection and media publication. Future media must label the model actually recorded and preserve the distinct latest checkpoint, video and accepted benchmark.
