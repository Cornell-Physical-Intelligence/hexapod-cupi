# Native source009 stepping progress recording

This external adapter is ready for a bounded fresh Isaac recording. CPU preparation is complete; it has not rendered or run physics locally. Root owns the guarded Spark dispatch. The screen is a stepping reference with zero residual, not PPO, completed Stage2, terrain traversal, or hardware qualification.

The original009 screen admitted 11 measured landings across all six legs, 115.24 mm of forward progress in 24 seconds, and 12.46 seconds of scored quiet hold. Its stop request took **5.54 seconds** to reach reference quiet, followed by 2 seconds of excluded settling. The recording must pass the same checks on its own fresh physical trajectory. It is not a movie reconstructed from the earlier trace and must not be described as one.

## Immutable inputs and execution

The adapter requires source009 manifest `04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e`, exact completed campaign/standing/wave receipts, and all 550 admitted package files. Exact receipt hashes are in [recording_contract.py](recording_contract.py); [SOURCE_ORIGIN.json](SOURCE_ORIGIN.json) records the render-helper lineage. The full source tree, input tree, receipts, geometry identity, and external Python code are reverified after recording. Any incomplete or unverified result exits unsuccessfully.

`record_reference_video.py` compiles the unchanged `main` and `serializable` function ASTs directly from the hash-verified source009 entrypoint. Their function hashes are recorded. The frozen main retains its startup, zero-residual actions, target generation, measured contact/torque/support checks, quiet/progress scoring, state format, and first-terminal stop behavior. The adapter supplies only a rendering environment subclass and a shared pre-reset capture observer. The observer sees the same row populated with the frozen target and terminal flags before `env.step` returns. No root pose, joint pose, contact, target, or action is prescribed by the rendering code.

AppLauncher is constructed before importing Torch-dependent simulation code. A 90-second startup guard and bounded render-only RGB warmup run without physics steps. The sole environment already matches the original wave screen's single replica. Explicit rendering overrides are 1280 × 720, `rgb_array`, and enabled cameras; original and recording YAML files are saved separately. No solver, actuator, controller, collision, geometry, timing, or physical gate changes are made. An external temporary Python class binding enables rendering before the frozen environment builder's local import; source files remain read-only.

## Visible behavior and output

The unchanged sequence is 2 seconds of canonical-target startup, 2 seconds of settle, 24 seconds at forward 0.005 m/s, and 20 seconds of stop/quiet. There are 2400 real control steps at 20 ms and 1200 frames at 25 fps, yielding a 48-second video. The camera follows the actual robot while keeping the full C robot visible. Ground arrows and labels show the requested body twist to three translation decimals; an orange trail records the actual root path. The overlay says `STEPPING REFERENCE — NOT PPO / STAGE2 INCOMPLETE` and reports requested/measured speed, actual distal contacts, and the original slow-stop limitation.

On a terminal event, the last valid pre-reset render is explicitly labeled and recording stops; no auto-reset robot is shown as continued walking. On any other physical or rendering failure, the partial MP4, source failure state, trace, and `video.json` remain evidence of a failed screen. Only a complete unchanged gate pass, 2400 controls, 1200 frames, no terminal, and post-run identity verification can set `complete=true`. A partial recording is never a full job pass.

Outputs include `rollout.mp4`, `first_frame.png`, `last_frame.png` on complete runs, `video.json`, `provenance.json`, render configuration/warmup records, and all unchanged source009 state/trace/reference/substep artifacts. `video.json` contains the video SHA, source and recording identities, actual root trail telemetry, completion counts, source gate result, and explicit false flags for Stage2 completion, training started, and pose forcing. Successful playback is not an additional qualification.

## Invocation and verification

The root-owned external host mounts the frozen source at `/workspace/hexapod:ro`, exact completed run at `/qualified:ro`, this directory at `/recording:ro`, and fresh output at `/outputs:rw`. Its command is:

```sh
/recording/record_reference_video.py \
  --source-root /workspace/hexapod \
  --package /qualified/inputs/study \
  --campaign /qualified/campaign.json \
  --admission /qualified/standing/admission.json \
  --study-tree-receipt /qualified/inputs/study_before.sha256.json \
  --geometry-reference /workspace/hexapod/robot/hexapod_mkii_length_study/candidate_c_reference.json \
  --output /outputs/recording --headless --enable_cameras --device cuda:0 --info
```

The host provides the Isaac Python launcher, pinned runtime paths, startup supervisor, and workload restoration. The adapter emits `REFERENCE_SCREEN_APP_READY` and preserves the agreed source `state.json` and `video.json` schema. No standalone GPU launch is performed by this preparation.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tmp/reference_progress_recording_001 -p 'test_*.py'
```

Ten CPU tests cover the exact 926-file source and 550-file package/receipt preflight, changed evidence and protected-output rejection, unchanged function ASTs, bounded render-only warmup, frame cadence, actual trail, no post-reset rendering, command precision, incomplete/unverified failure, and application/import order. Local system Python supplies NumPy, Pillow, and imageio; the study virtual environment lacks Pillow. Runtime additionally needs the already used native Isaac RGB camera and H.264 writer. These CPU tests do not establish fresh physics or rendering success.
