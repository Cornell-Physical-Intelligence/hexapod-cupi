# External admitted-policy video capture

Prepared 2026-09-06. These tools have CPU verification only. They have not launched
Isaac Sim, recorded a robot, or established that a PPO checkpoint has completed.
The checkpoint, admission, training report and frozen source paths will be supplied
after a successful admitted training run.

The three capture modules live under `artifacts/`, outside the physical training
identity's source selection. Do not move them into `tools/`, `isaaclab/` or a
package in the frozen checkpoint source. They import that source without editing
it. Their own three file hashes and the imported host supervisor's hash are
recorded separately in every capture request/report.

## Invocation on Spark

Copy this folder to a separate tools directory. Invoke the host launcher with
absolute paths; the examples below are placeholders, not existing checkpoint claims.

```sh
python3 /ABSOLUTE/CAPTURE_TOOLS/capture_policy.py \
  --source-dir /ABSOLUTE/FROZEN_SOURCE \
  --checkpoint /ABSOLUTE/COMPLETED_RUN/checkpoint.pt \
  --admission /ABSOLUTE/QUALIFIED_CAMPAIGN/admission.json \
  --training-report /ABSOLUTE/COMPLETED_RUN/report.json \
  --output-root /ABSOLUTE/NEW_CAPTURE_OUTPUT \
  --seconds 15 --width 1280 --height 720 --timeout-seconds 1800 \
  --dry-run
```

Remove `--dry-run` only for the separately scheduled GPU capture. Dry-run performs
CPU lineage verification and prints a structured command without creating an
output directory or contacting Docker. It needs no checkpoint unpickling.
Duration is bounded to 10–20 seconds; the default is 750 policy transitions at
50 Hz. The host timeout is bounded to 60–1800 seconds. Actual rendering speed is
unmeasured. Output must be outside the frozen source directory.

The host launcher imports the exact source's existing `run-mkii-fourbar` resource,
canonical coordination-control, identity and ownership helpers. It holds the
hexapod FD9 GPU lock before container creation through exact-ID cleanup, checks
resources before and twice after creation behind a CPU barrier, polls active
resources and the shared control every five seconds, and never signals unrelated
jobs. A sharing request or ambiguous control fails the capture and stops only its
owned container. Compose consumes its existing `.env.base` without exposing it.
Automatic restart is explicitly rejected. Source, capture tools and checkpoint
inputs are mounted read-only; only the new output directory is writable.

## What is recorded and verified

- Requires full physical admission and completed, unpaused PPO with all requested
  iterations, verified checkpoint roundtrip, changed policy and completed finite
  inference probe. A short physical probe or paused checkpoint is rejected.
- Checks exact source identity, checkpoint bytes/sidecar, embedded checkpoint
  lineage, trained policy digest and full algorithm/normalization state digest.
  Rechecks source and inputs at the barrier and completion.
- Requires physical v5, 31 bodies, 30 coordinates, 18 active motors and 84
  observations. The one-environment recording must retain the exact admitted
  runtime manifest, including an admitted coincident-origin layout if present.
  Historical grid manifests retain their original layout. No capture override
  changes joint defaults, contact settings, controller, physical timing or gates.
- Uses deterministic inference and the original task's seeded command sampler.
  It neither calls `learn`/checkpoint `save` nor creates scripted joint motion.
  The complete algorithm state must remain unchanged throughout inference.
- Calls the frozen training physics guard at every physics substep. This retains
  that guard's existing scope and bounds; recording is not a new terrain,
  navigation or hardware admission, and falls/episode resets are recorded.
- Creates the Kit perspective RGB render product and waits for a populated frame
  using render calls only. Robot states and simulation counters must remain
  exactly unchanged during this bounded warmup. Blank frames are rejected.
- Streams an H.264 MP4 at policy frequency with a camera following root X/Y and
  fixed world Z. Camera motion never changes the robot. Encoder finalization and
  report persistence occur before environment/Kit shutdown. The host independently
  uses `ffprobe` to decode/count frames and verify rate, duration, size and codec.

Successful output contains `policy.mp4`, `states.npz`, `metadata.json`,
`report.json`, `supervisor.json`, `ffprobe.json`, progress/logs, input report copies,
and a source file hash inventory. Checkpoint bytes remain in their original
immutable directory. A failed or incomplete capture cannot pass merely because
Kit exits zero; existing partial output is preserved and its supervisor fails.

Each state row records the policy's actual 84-element observation, raw action,
pre/post commands, all 30 named joint positions/velocities, root position and
XYZW quaternion, processed target, final cached motor torque, reward, done flag,
camera transform, video index and simulation time. Command/observation equality,
array dimensions/finite values, quaternion normalization and frame/time alignment
are checked before saving. Post-transition state/frame may show an automatic
reset, which is explicitly marked; cached torque is not represented as a whole
interval peak. Every-substep physical metrics remain a separate report field.

## API and execution prerequisites

The installed Spark SDK was inspected through host source files, without starting
a container: `DirectRLEnv.render`, `VideoRecorderCfg`,
`isaaclab_physx.video_recording.isaacsim_kit_perspective_video`, and
`isaaclab_physx.renderers.kit_viewport_utils`. The RGB backend is the Kit
perspective camera/Replicator even with `--viz none`; the launcher explicitly
enables cameras. Its unpopulated initial image is black, hence the warmup check.
The installed RSL wrapper returns a TensorDict, not a Python dict.

RSL-RL 5.0.1's [runner](https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/runners/on_policy_runner.py)
returns the policy in evaluation mode; its
[MLP](https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/models/mlp_model.py)
supports `stochastic_output=False` and `reset(dones)`. The recorder uses the
[MoviePy streaming FFmpeg writer](https://github.com/Zulko/moviepy/blob/v2.2.1/moviepy/video/io/ffmpeg_writer.py)
API, declared by the installed Isaac Lab RL package. Container codec availability
and RTX capture still require the first live capture preflight; host `ffprobe`
is checked before container creation. No alternate physics backend is a fallback
for a rendering failure. Inspect the finished video's beginning/middle/end frames
before presenting it as a demonstration; quality of locomotion remains an
evaluation question, independent of whether recording succeeded.

## CPU validation

```sh
uv run python -m unittest discover \
  -s artifacts/mkii_fourbar_2026-09-06/policy_capture_tools -p 'test_*.py' -v
```

The 19 tests include real source/checkpoint byte corruption rejection, incomplete
training/admission rejection, source-identity exclusion, layout propagation,
command/time metadata misalignment, malformed/blank images, exact read-only
argument construction, MP4 validation, and mocked supervisor success/failure/
sharing-yield cleanup. All Docker/FFmpeg operations in lifecycle tests are mocked;
these tests make no GPU claim.
