# Native direct-PPO progress recording adapter

This external entrypoint runs the exact native003 direct315/318 evaluator's construction and checkpoint load, then records one continuous **38-second** command sequence. It is ready for a bounded native rendering test once the root owner selects a completed training campaign and its exact final checkpoint. No GPU execution, new video, locomotion acceptance or Stage 2 completion is claimed by this CPU preparation.

The source is the 598-file native tree `64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e`, plan `9fb7d0b01480523972b4f1c17e717756d7dcf1c54b9caa436245671937de538c`, native003 contract freeze `20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb`. The selected weights must be the exact final checkpoint accepted by the completed native host `19545653a00e635cc14ed9415f9799978da34674e84f6cad76eb42bd2a961ac4`. Only a completed 50-update curriculum or CAPS pilot is eligible. Smoke campaigns are rejected. All six original/final diagnostic phases must be present, and the selected checkpoint is required at runtime from the root-selected completed campaign. These are acquisition-completion checks; poor quiet or torque measurements remain poor measurements.

## Behavior and visible evidence

| Simulation time | Requested forward / left / yaw |
|---|---|
| 0–4 s | 0 / 0 / 0 |
| 4–10 s | +0.10 m/s / 0 / 0 |
| 10–14 s | 0 / 0 / 0 |
| 14–20 s | 0 / +0.10 m/s / 0 |
| 20–24 s | 0 / 0 / 0 |
| 24–30 s | +0.10 m/s / 0 / +0.20 rad/s |
| 30–38 s | 0 / 0 / 0 |

There are 1,900 actual controls at 50 Hz and 950 frames at 25 fps: **1× playback**, with no interpolation or time warp. The full 19-body, 18-joint robot remains visible through the existing following camera. Ground geometry shows the integrated command path, blue translation arrows, gold heading/yaw arrows and an orange actual trail. The path is display geometry only; measured pose never generates actor commands or corrects path error. Labels distinguish requested, actor-observed and measured twist, branch/update count, checkpoint, requested torque and saturation. The banner states `DIRECT PPO PROGRESS - STAGE 2 INCOMPLETE`.

The inherited command ramp changes the command after scoring the current control. Each action therefore uses the command already in its observation; the recorder verifies that exact command in the pre-reset sample. Short zero-command clips show stopping behavior but cannot establish the native evaluator's separate 10-second quiet gate. Observation noise, normalizers and action processing remain those of native evaluation.

The first returned termination or truncation ends the loop. Its original pre-reset state and actions are retained, and a terminal banner uses the last valid RGB frame. No reset robot is rendered and no later segment runs. Such a partial video is explicitly incomplete and the process exits nonzero. Nonfinite data, rendering failures, wrong observations, input changes, writer failures and App-close failures also retain failure receipts and exit nonzero.

## Exact implementation boundary

`entry_adapter.py` asserts the exact source entry SHA before applying five counted AST seams: actual AppReady marker, metadata save wrapper, late render configuration, RGB constructor argument, and evaluator callback. The inverse-AST regression recovers the entire original native entry. In particular, the original 48-replica constant-evaluation selection and multiple-of-12 check execute unchanged; only immediately before environment construction does the rendering callback change the allocation to one. This override is recorded separately, and the output is not passed off as a native constant evaluation.

`matched_evaluation_environment.yaml` records the original 48-replica configuration. `environment.yaml` records the actual preview configuration. An exact field comparison permits only `scene.num_envs`, `video_recorder.window_width` and `video_recorder.window_height` to change. RGB and camera enablement are explicit additional rendering overrides. The 16/4 solver, external-force setting, 30/.6 PD, actuator 1.6 N·m/native effort limit 5.5, original rewards, .04-rad target slew, zero target filter, noise scale 1, mass/geometry, contact definitions and 20 ms control period remain the frozen native settings. Source009 is not used as the physical environment.

The original `runner.load` executes unchanged. Before and after recording, actual loaded actor/critic tensors, including normalization buffers, must exactly equal the selected checkpoint. Repeated deterministic inference on the same packet must be equal. No optimizer step occurs. Legacy imports must resolve to the exact 16-file `hexapod_rl` package with canonical digest `abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280`. The checkpoint is the newly trained direct315 actor, never the incompatible 495/498 velocity candidate or 846/849 reference policy.

The historical diagnostic field `quaternion_world_wxyz` is retained byte-for-byte, despite containing SDK XYZW. Explicit `quaternion_world_xyzw` and converted `quaternion_world_wxyz_converted` fields are added. Camera heading and display transforms explicitly use raw XYZW. Reported SDK rates and existing finite-difference diagnostic channels remain distinct; this adapter makes no velocity-fidelity claim.

`render_helpers.py` is byte-identical to the previously reviewed reference progress helper. It supplies bounded render-only warmup (maximum 24 calls / 30 seconds) and ground labels. `GroundDrawing`, body-twist integration and command slew are imported from the exact native source. No rigid bodies or collision geometry are added by these drawing helpers.

## Actual external CLI

The following is the complete adapter CLI. `SELECTED_FINAL_SHA256` must be supplied by the root owner from the selected campaign receipt. The example mount paths are literal container aliases; selection is not inferred from a filename.

```sh
/workspace/isaaclab/_isaac_sim/python.sh /recording/record_direct_preview.py \
  --source-root /source \
  --native-contract /contract \
  --pilot /pilot \
  --checkpoint /checkpoint/evaluated.pt \
  --checkpoint-sha256 "$SELECTED_FINAL_SHA256" \
  --admission /pilot/standing/admission.json \
  --output /output/recording \
  --seed 7057 --device cuda:0
```

`--preflight-only` performs standard-library identity and receipt verification without creating output, importing Torch or constructing AppLauncher. Headless mode and camera enablement are explicit fixed native arguments emitted by this adapter, not external parser options. `--kit_args` may supply the established startup telemetry options. The native selector remains `--mode evaluate --direct-evaluation constant`; native003's rejected video selector is not changed.

The root owner must mount the exact source/assets, contract, selected pilot, selected checkpoint and this frozen adapter read-only, including any aliases nested under a writable output parent. The output directory must be fresh and outside every immutable input. The host must pin this adapter's full freeze digest externally; its self-inventory check alone is not an authority to choose a new version. The existing owned-container supervisor, 90-second AppReady deadline, liveness/contact-buffer checks, locks, cleanup and external pause/restore guard remain host responsibilities. This bundle implements no replacement host framework and launches no jobs.

## Outputs and CPU checks

The output contains `rollout.mp4`, `video.json`, raw `trace.npz` with deterministic actions and joint names, command-reference display data, requested/observed/measured telemetry, first/last frames, RGB warmup, configuration delta, exact checkpoint readbacks, entry-seam counts, native argument list and provenance. All selected source/campaign/phase/checkpoint/admission bytes are reverified after recording and after native App close. Complete video requires exactly 1,900 recorded controls and 950 frames; a complete frame count is not a physical qualification result.

Run the focused CPU tests from the repository root using the already available analysis dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tmp/reference_residual_ppo_001/_deps \
  .venv/bin/python -B -m unittest discover \
  -s tmp/direct_omni_preview_adapter_002 -p 'test_*.py'
```

The tests exercise exact native AST parity, late configuration deltas, App-before-ready and standard-library-only startup, all phase/checkpoint/admission bindings, source inventory rejection, raw quaternion and packet conventions, exact checkpoint tensors, full-loop cadence, command timing, bounded render warmup, RGB dimensions, first-terminal preservation and failure finalization. The full-loop fixture is explicitly synthetic: it validates recorder sequencing, not Isaac rendering, contacts, motor behavior or walking quality. No selected actual pilot/checkpoint has yet been bound to a render run.

## Versioned change and project-site handoff

Adapter001 remains immutable. The rendering entry, AST seams, recorder loop, helper, source entry and native inference environment are byte-identical. This successor updates only native003/source002/host002 provenance, the source inventory, and the explicit pilot-only receipt restriction. Native003 changes value-preserving reload handling of inference-created normalizer buffers in the training finalizer; it does not change the matched inference controller or physics. `DELTA_REVIEW.json` and `test_successor.py` check this boundary.

This is ignored `tmp/` preparation, with no tracked or site edits. Under `docs/PROJECT_SITE.md`, root must add an append-only site update when publishing the prepared adapter, and update the media registry only after a real recording is verified. No new checkpoint is chosen by this bundle and no old video is presented as the selected pilot.

Final CPU result: **23 unique tests passed**. The initial 27-test log is retained and explicitly contains four duplicate receipt-test discoveries, removed from the final test runner.
