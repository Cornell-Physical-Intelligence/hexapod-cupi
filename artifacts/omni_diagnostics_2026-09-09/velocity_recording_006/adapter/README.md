# Recorder004: admitted-package correction for candidate003 — 2026-09-10

This is a prepared recording adapter, **not a recording or a qualification result**. It uses the complete C-study robot and the new 495/498 target-velocity PPO actor. It preserves the exact training source, checkpoint sidecar and embedded controller contract, and uses the actual frozen strict checkpoint loader. No files inside the training source are changed.

This remains bound to root's source003 snapshot at `tmp/omni_velocity_launch_003/source_003` (remote `omni_velocity_source_003`). Recorder004 corrects only the admitted-package binding: the pilot used its copied post-validation `inputs/study` package, so recording must use that identical package rather than the original source-tree asset. The [source003 recorder](../omni_velocity_recording_003/README.md) and [source002 recorder](../omni_velocity_recording_001/README.md) remain frozen. The completed pilot did not establish useful walking and regressed sustained quiet/stop evaluations; this recording will expose its behavior, retain the first-terminal-event stop, and never label it qualified.

The revised required arguments are `--package /pilot/inputs/study --probe-campaign /probe/campaign.json --study-tree-receipt /probe/inputs/study_after_flat.sha256.json --pilot-inputs-receipt /pilot/inputs_before.sha256.json`. The accepted probe campaign SHA is `ca436a5439526cfd9b742143650fa1a6b9f132e812fa02f38415bf0324327155`. Every one of the 550 package files must match both the post-validation probe tree and the pilot receipt's `study/` subset; the full pilot input tree and copied admission/calibration/smoke are also verified. Root's host mounts `/probe`, `/pilot`, `/recording` and executable source read-only. Source identity still comes only from the unchanged source003 executable tree. The final pilot checkpoint is `88f8e60a7b1536229f51524cdc646f06d53e4f33cabb74aa89ebde73d06f7247`.

## What is implemented

`record_candidate_video.py` renders one continuous 34-second episode, at 50 Hz physics-control sampling and 25 fps playback. Its fixed sequence is 3 s quiet start; forward, left strafe, reverse, right strafe, left turn and right turn for 3 s each; two 4 s arcs; and 5 s quiet end. Translation is 0.10 m/s, pure turns ±0.20 rad/s, and arc yaw ±0.15 rad/s. Every overlay distinguishes requested command, ramped command seen by the actor and measured motion. Observation noise and the existing command ramp are retained. These short quiet windows do not replace the separate sustained quiet/stop qualification.

The robot receives only `runner.get_inference_policy(...)` outputs through the unchanged candidate action pipeline. There is no reference gait, waypoint follower, body pose override, joint teleport or segment reset. The camera follows the actual robot. Reused `GroundDrawing` meshes show blue command-integrated travel, gold heading and orange actual trajectory, plus a live requested travel arrow, yaw arc and numeric forward/left/yaw labels on the ground. Those meshes have no physics or collision behavior. Reference calculations never move the articulation.

Termination and truncation stay enabled. The recorder retains the pre-reset diagnostic snapshot, stops at the first event and labels the last valid render as a terminal event. It never continues into the automatically reset episode or reports a shortened clip as complete. Runtime exceptions leave `failure.json`; they do not mark qualification passed.

`recording_env.py` mirrors the frozen entrypoint's physical configuration and runner construction. A CPU AST comparison verifies every shared `cfg.*` assignment except the explicit rendering/layout fields. It writes both the matched evaluation configuration and the actual recording configuration. Logged differences are one environment instead of 48, RGB rendering at 1280×720, camera enablement, an explicit recording seed and command schedule. Reward weights, action semantics, target-velocity/acceleration controller, gains, 1.6 N·m torque cap, contact/failure conditions, asset, stance and actor/critic widths are preserved. The episode horizon remains the evaluator's 90 seconds.

Quaternion handling explicitly uses installed Isaac Lab **XYZW**. The pre-reset candidate trace retains both `quaternion_world_xyzw` and the explicitly reordered `quaternion_world_wxyz`; the recorder checks their agreement. Heading is calculated from the XYZW rotation of body forward `(0,-1,0)`, with SciPy comparison tests. This requirement also applies to any future physics adapter for the separately frozen support-reference prototype: use an explicit XYZW quaternion or a declared 3×3 transform, never an assumed order.

## Identity and required inputs

The source003 binding is:

```text
source_sha256  fb5e472bb710daa50d1d7f16a6f3812d2ed845e2400fcc6769859b3fb0b9519e
manifest_sha   00d4e07e0e31776e3d2faa0136a2b1386f79fd4940bf81d2f4a484155f81bb25
plan_sha256    356b72f0e82b207ba6105cb42057643d21a5b911d0517661465dcfe57745d00f
urdf_sha256    e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c
variant        f050_t060
stance_index   0
actor/critic   495 / 498
profile        formal_004, illustrative acceleration8rad/s²
```

The recorder requires a passed standing admission, passed matching std0.005 calibration and runner smoke, and the completed exact-source 50-update scratch pilot state. A final recording requires that state's `checkpoint_sha256` to match the explicitly provided final checkpoint bytes. The optional initial-policy comparison also requires completion of the same pilot and uses `initial.pt`, labeled `scratch_initial`. A stand-only schema-smoke checkpoint cannot be relabeled as a walking pilot.

Before AppLauncher, the unchanged `candidate_runner.preflight_candidate` runs in its supported evaluation role and verifies source/asset/plan/admission/checkpoint metadata. This does not change its mode whitelist or make its unsupported video branch work. After the runner is constructed, its unchanged `load_candidate` checks the embedded contract, 495/498 tensor schema, finite weights and strict tensor readback. Inference-buffer repair remains the candidate loader's reviewed behavior. No checkpoint is converted to another action or observation scheme.

`provenance.json` separates training-source identity, checkpoint SHA, admission/calibration/smoke/pilot-state hashes and external recording-code hashes. All relevant identities are checked again after the rollout. `video.json` always keeps `stage2_complete=false` and `qualification_performed=false`. Render success cannot promote a controller or bypass the initial/final directional and quiet/stop evaluations.

## Launch preparation

Root must schedule rendering through the existing Spark job controls after the calibration/smoke/pilot completes. This adapter neither launches remote jobs nor pauses forecasting. The source argument names the frozen executable snapshot, and the package names the pilot's separately admitted read-only asset copy. Use the existing Isaac launcher and external recorder directory:

```sh
/workspace/isaaclab/_isaac_sim/python.sh "$RECORD_ADAPTER/record_candidate_video.py" \
  --source-root "$RECORD_SOURCE" \
  --package "$RECORD_PILOT/inputs/study" \
  --checkpoint "$RECORD_PILOT/train/policy/final.pt" \
  --checkpoint-sha256 "$RECORD_CHECKPOINT_SHA256" \
  --checkpoint-label scratch_50_update_final \
  --admission "$RECORD_PROBE/flat/admission.json" \
  --calibration "$RECORD_PROBE/probe/calibration.json" \
  --runner-smoke "$RECORD_PROBE/probe/runner_smoke.json" \
  --pilot-state "$RECORD_PILOT/train/state.json" \
  --probe-campaign "$RECORD_PROBE/campaign.json" \
  --study-tree-receipt "$RECORD_PROBE/inputs/study_after_flat.sha256.json" \
  --pilot-inputs-receipt "$RECORD_PILOT/inputs_before.sha256.json" \
  --output "$RECORD_OUTPUT" \
  --headless --enable_cameras --device cuda:0
```

Resolve those variables to the actual probe and pilot output layout; the adapter does not infer an available checkpoint or use a `latest` path. `--output` must be fresh and outside the frozen source. `--preflight-only` verifies source/metadata/admissions without importing AppLauncher or launching a simulator; it does not perform the later embedded tensor readback. Run with `PYTHONDONTWRITEBYTECODE=1` or `python -B` to avoid cache files in the frozen source.

Expected artifacts are `rollout.mp4`, first/last frame PNGs, the pre-reset `trace.npz`, `video.json`, provenance, progress and resolved environment/agent YAMLs. In a terminal case, the final endpoint frame and planned sequence completion are intentionally absent. Inspect actual frames and video before sharing; CPU contracts cannot prove the installed renderer, camera framing or codec works.

## CPU verification and remaining runtime risk

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tmp/omni_velocity_recording_004 -p 'test_*.py' -v
```

Fifteen tests cover exact source digest agreement with the actual candidate implementation; checkpoint byte/sidecar/source rejection; matching calibration/smoke/pilot evidence; full 550-file admitted-package and pilot-input receipt verification; changed, missing or unlisted file rejection; XYZW/SciPy rotation agreement; the continuous command schedule and existing one-step ramp timing; actor/critic shapes and finiteness; shared physical-config assignments; a complete 1,700-step fake-API rollout; and terminal-event preservation without post-reset rendering. The fake renderer output is kept only in temporary test directories and is never presented as a walking video.

The GPU path is unexecuted. Remaining runtime checks are actual AppLauncher/RGB availability, one full robot in frame, strict checkpoint tensor load, deterministic policy rollout, frame quality and codec output. A weak pilot may stand, drift or fall; the recording is intended to expose that outcome. It must not substitute a scripted reference or disable feedback, resets or torque gates to manufacture a walking clip.


## Recorder005: actual RGB initialization correction

Recorder004 loaded the exact checkpoint and constructed the full robot, then failed at the first RGB call with `Missing/blank initial Isaac RGB frame`. Its source, failure and pause023 restoration remain immutable. Installed DirectRLEnv lazily creates the RGB render product/annotator at first access, which did not yet contain image data. This version performs at most24 render-only warmup attempts within30seconds before physics/control begins and records attempt shapes/variance. It does not step physics, reset after the sequence begins, change the actor or accept blank frames. Two independent tests exercise initially blank then valid output and bounded persistent-blank failure. Recording rollout and all contracts remain unchanged.


## Recorder006: traced startup after stalled005

Recorder005 did not reach renderer startup: after preflight it remained blocked in AppLauncher, with all Python threads waiting on futexes, no CUDA context and0%GPUutilization. Root preserved process/resource snapshots and two denied debugger-attach attempts, then stopped only its owning unit rather than consume the full600-second allocation. This successor adds45-second repeated Python traceback dumps around preflight/AppLauncher and explicit startup markers, canceled when the application is ready. It retains the005 render-only correction and all source/checkpoint/physics/recording contracts. The native blocking cause is not yet established.
