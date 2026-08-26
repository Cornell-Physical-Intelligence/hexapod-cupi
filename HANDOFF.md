# Hexapod RL handoff for Claude

Snapshot: `2026-08-26T17:46:47Z` (`2026-08-26 13:46 EDT`)

This document supersedes every earlier handoff. It is based on a fresh audit of
the local Git worktree and the Spark, not conversational memory. Paths beginning
with `/Users/andreboufama` are on the Mac. Paths beginning with `/home/orionh`
are on `spark-e26c`. No password, token, key, or secret-file content is included.

## Read this first

The current research target is **Stage2C: stable anatomical-forward walking**.
There is no admitted Stage2C breakthrough yet and no new Stage2C video should be
claimed. The immutable best checkpoint is still useful, safe in the nominal
screen, and visually walks forward, but it misses the yaw and high-speed deck
stability gates.

At the snapshot there is **no hexapod process or container running**. An
unrelated root-owned workload is using the GPU:

```text
PID 217667  /root/cvnba/venv/bin/python sq/validate_nsva.py data/nsva \
  --out results/nsva_validation.json
ancestor: PID 215354 bash /root/nsva_dl.sh
GPU: about 76-82%, 1366 MiB at the audit
```

Do not signal, stop, modify, or compete with that workload. Do not resume the
hexapod merely because PID 217667 disappears: wait until `/root/nsva_dl.sh` and
all of its descendants are gone, then recheck the GPU, active Docker containers,
the systemd service, and the shared lock. The producer existed before the last
preflight and spawned its GPU child afterward; this is the post-launch race that
must be closed.

Once the unrelated producer and children are gone, the next scientific action is
to complete **exactly the same Probe21 causal arm** from the immutable parent:

```text
seed=99
num_envs=12288
rollout_steps=24
updates=12
learning_rate=2e-5
clip_param=0.06
inactive_bilateral_longitudinal_contact_moment_reward_scale=-2.0
inactive_bilateral_longitudinal_contact_moment_reference_nm=1.8
processed_joint_target_slew_limit_rad_per_20ms=0.040
```

Use a new unique attempt label. Never reuse or overwrite either failed Probe21
attempt. Do not change the reward, seed, parent, environment count, or learner
settings during the retry. Formal-screen all 12 children before interpreting it.

The user wants minimum Spark downtime and maximum useful training time. That
does not mean collision with unrelated GPU work or blind retry storms. The
correct operational objective is: detect pre-AppReady stalls within 45-90
seconds, preserve diagnostics, prove exact-container cleanup, and perform at
most one bounded identical retry while keeping the GPU lock for the logical run.

## 1. Product goal and current stage

Roadmap:

1. Stable, natural anatomical-forward/backward walking with a low, steady deck.
2. General joystick locomotion: forward, reverse, lateral, diagonal, and yaw in
   both signs, including command transitions for a navigation policy.
3. Varied terrain with camera/depth/LiDAR/IMU sensor fusion. A long Phase3
   sensor-fusion or difficult-terrain job still requires explicit user approval.

Current success definition for Stage2C:

- Commands: stand, `0.16`, `0.20`, and `0.30 m/s` anatomical forward.
- Zero falls and zero timeouts in the nominal formal screen.
- Achieved speed at the `0.30 m/s` command at least `0.240 m/s`.
- Yaw-rate RMSE at most `0.080 rad/s` at every moving command.
- Moving normalized deck composite at most `1.000`.
- RS05 limits: bounded continuous-duty fraction, burst length, raw peak demand,
  and worst-joint duty; never exceed the `5.5 Nm` raw-demand safety termination.
- Prefer a visibly lower, steadier, more natural gait. The user specifically
  wants less deck wiggle and a more upward-angled femur/first leg segment so the
  platform sits lower rather than being held high.

Current stage: reward/structure diagnosis and causal probing. Stage2D/Stage2E
configuration exists but has not been admitted. Phase3 is paused.

Simulator/task/framework:

```text
Simulator: Isaac Sim 6.0.1 + Isaac Lab DirectRLEnv / PhysX
Task: Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-Hexapod-RobStride-Direct-v0
RL: RSL-RL 5.0.1, PPO, actor/critic [256,256,128], ELU
Policy observation normalization: enabled/frozen on model-only recovery
```

## 2. Immutable current best

Remote checkpoint:

```text
/home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/
  hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/
  2026-08-25_23-44-14_accel_scale4_yaw160_trackguard_20260825T234500Z_seed86/
  model_2.pt
```

Local mirror:

```text
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/
  phase2_recovery_stage2c_stable_forward/checkpoints/current_best/model_2.pt
```

SHA-256:

```text
a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a
```

Formal seed-60 nominal measurements at the `0.040 rad / 20 ms` playback limiter:

| Command | Achieved forward | Yaw RMSE | Deck composite | Worst-joint duty | Falls |
| --- | ---: | ---: | ---: | ---: | ---: |
| stand | 0.0003 m/s | 0.0002 rad/s | 0.013 | 0.000 | 0 |
| 0.16 | 0.1599 m/s | 0.1070 rad/s | 0.765 | 0.206 | 0 |
| 0.20 | 0.1962 m/s | 0.1117 rad/s | 0.862 | 0.238 | 0 |
| 0.30 | 0.2420 m/s | 0.1378 rad/s | 1.032 | 0.223 | 0 |

Worst burst is `0.08 s`; peak raw demand is `2.691 Nm`. It is the parent for
all current probes. Do not promote over it without formal absolute-gate checks.
It is not hardware-ready.

## 3. Recent experiment ledger

The canonical local ledger is:

```text
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/
  phase2_recovery_stage2c_stable_forward/probes/README.md
```

| Probe | Intervention | Result | Decision |
| --- | --- | --- | --- |
| 15 | full ground-wrench yaw cost `-1`, ref `2 Nm` | parent ranked first | reject |
| 16 | joint-torque slew `-0.05` | tiny yaw gain; deck/peak regressed | reject |
| 17 | foot-slip `-3.0` | yaw unchanged; worst-joint duty regressed | reject |
| 18 | full ground-wrench yaw `-2`, ref `2 Nm` | best child looked slightly better | held for matched control |
| 19 | yaw-rate slew `-4`, ref `0.040` | pre-AppReady timeout; no policy | invalid attempt |
| 20 | matched zero-intervention continuation, seed 99 | reproduced Probe18 within ~1% | shows Probe18 gain was not causal |
| 21 cal A | bilateral longitudinal moment `-1`, ref `0.5`, LR 0 | signal live; mean/p50 `1.654/1.800 Nm` | recalibrate |
| 21 cal B | same, ref `1.8`, LR 0 | bounded cost mean/p50 `0.355/0.400` | calibration passes |
| 21 full | bilateral moment `-2`, ref `1.8` | two pre-AppReady attempts; no checkpoint | still scientifically untested |

Probe20 successful control:

```text
batch:
/home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c/
  accel_probe20_matched_zero_seed99_slew040_retry1_20260826T160100Z

run:
/home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/
  hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/
  2026-08-26_15-59-43_accel_probe20_matched_zero_seed99_slew040_retry1_20260826T160100Z

formal evaluation:
.../evaluation/formal_seed60_slew040_sharded_v1
```

It completed 12 updates and `3,538,944` samples in `45.67 s`. Best ranked child
was `model_8.pt`, SHA-256
`87570ffb2f870a9d8b90db5185fd34b8196786f0b09e4a9b2de987352d6d7cb6`:

```text
yaw RMSE at .16/.20/.30: .0977207 / .1024256 / .1339176
deck:                    .757930  / .866943  / 1.035952
mean moving deck:        .886942
tracking RMSE:           .0379985
max duty / burst:        .088889 / .10 s
peak raw demand:         3.34379 Nm
max-joint duty:          .237895
falls/timeouts:          0 / 0
```

It is operationally safe in this screen but fails the absolute yaw/high-speed
deck gates. It is a control child, not a promotion.

Probe21 calibration paths:

```text
successful ref-0.5 telemetry:
/home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c/
  accel_probe21_bilateral_cal_ref050_telemetry_retry1_seed99_20260826T171206Z

successful calibrated ref-1.8:
/home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c/
  accel_probe21_bilateral_cal_ref1800_seed99_20260826T171546Z

LR=0 calibration checkpoint only, not a candidate:
/home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/
  hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/
  2026-08-26_17-23-55_accel_probe21_bilateral_cal_ref1800_seed99_20260826T171546Z/model_0.pt
```

The first telemetry attempt at `...T170241Z` failed with a reset-batch indexing
shape mismatch in `env.py`. The helper was corrected to consume already-selected
aligned tensors; the full 266-test suite passes. Preserve the failed attempt.

## 4. Probe21 failed-start evidence and current process state

Attempt 0:

```text
label: accel_probe21_bilateral_long2_ref1800_seed99_slew040_20260826T172645Z
remote artifact: /home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c/<label>
local mirror: /Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/
  phase2_recovery_stage2c_stable_forward/probes/<label>
started: 2026-08-26T17:26:58Z
finished: 2026-08-26T17:33:58Z
overall/docker/train: 124 / 124 / 124
checkpoint/run directory: none
```

Its 706-byte log ends after container creation and the stock `train.py`
deprecation warning. It never printed `Loading user config`, the Kit clock, or
`[ISAACLAB] AppLauncher initialization complete`; Python slept in
`futex_do_wait`, CPU was near zero, and no GPU context existed. No NVRM XID,
OOM, or segfault was found.

The 420-second watchdog fired, but the then-current cleanup parser used a Go
template containing `\t`. Docker emitted literal backslash-t sequences, Bash
did not split the fields, and cleanup safely refused with `identity-mismatch`.
The exact known container was manually stopped and removed after its artifact
was preserved.

Attempt 1:

```text
label: accel_probe21_bilateral_long2_ref1800_seed99_slew040_retry1_20260826T174200Z
started: 2026-08-26T17:42:55Z
signal requested: 2026-08-26T17:45:02Z
cleanup finished: 2026-08-26T17:45:32Z
overall/docker/train: 143 / 130 / 130
cleanup: stopped, stop_status=0, postcheck=absent
checkpoint/run directory: none
```

Attempt 1 also remained pre-AppReady. A new unrelated `validate_nsva.py` process
claimed the GPU after the clean prelaunch check. Only the hexapod launcher was
signaled. The fixed cleanup validated the immutable container ID, stopped it,
and proved it absent; no unrelated process was touched. Preserve this artifact.

Current snapshot:

```text
hexapod process/container: none
hexapod-rl-training.service: inactive/dead
GPU user: unrelated validate_nsva.py via /root/nsva_dl.sh
next hexapod run: waiting for the producer and all descendants to finish
```

The Codex app heartbeat `resume-stage2c-after-shared-gpu-work` now checks this
gate every two minutes and is instructed to launch only one fresh exact Probe21
attempt, then formally screen it. Claude should first verify whether that watcher
has already acted before launching anything manually.

## 5. Isaac/Kit reliability problem

### Confirmed

- The failure is before AppLauncher, user config, scene allocation, checkpoint
  load, reward evaluation, or CUDA context creation.
- Probe19, Probe20 attempt 0, and both Probe21 full attempts reached this same
  boundary.
- An identical Probe20 seed/config retried successfully: `Loading user config`
  at about `+2 s`, AppLauncher complete at about `+12 s`, all 12 updates done.
  Therefore 12,288 environments and the reward config are not credible direct
  causes of the startup stall.
- Successful logs also show OmniHub reconnect and GB10/PyTorch capability
  warnings. Those warnings alone are not diagnostic.
- The old Docker healthcheck searches a questionable Kit-log tree and cannot be
  trusted as an AppReady signal without validation inside a successful live
  container.

### Fixed already

The exact-container cleanup in both hardened launchers now uses `|` delimiters,
not literal `\t`, and Docker's current `--timeout 30` option. The change is
synced to the Spark and passed all tests. Attempt 1 validated it in vivo.

```text
probe launcher SHA:       5f69625d7638e538f2c6f48c2907c9b42c61ac0329afa340362f1cfc1c5cc506
calibration launcher SHA: 810a4ed436f935f280043a8a54a0f2ae6e6af8a72e647af0dff632ca9b8c5ece
manifest SHA:             c104eb22024077f8c875b2d8a632d437ff47bb89c2c7da8999570de5bdec99c2
```

`sha256sum -c isaaclab/deploy/stage2_pipeline.sha256` verifies all 46 manifest
entries locally and remotely. `python3 -m unittest discover -s isaaclab/tests`
passes all 266 tests locally.

### High-priority fix not implemented yet

Implement an attempt-aware startup supervisor in the hardened launcher:

1. Hold `/tmp/hexapod-isaac-gpu.lock` across the entire logical run.
2. Gate not only current GPU/Docker/service use but producer scripts and
   descendants: `nsva_dl.sh`, `validate_nsva.py`, `score_clip.py`,
   `validate_b51.py`, and torch-compile/clip-scoring workers.
3. Recheck contention immediately after container creation and before PPO.
4. Set `PYTHONUNBUFFERED=1`.
5. Require `Loading user config` by 45 seconds and the AppLauncher completion
   marker by 90 seconds. Keep 420 seconds as the overall 12-update cap.
6. At 30 seconds without progress, atomically capture exact container inspect,
   logs, `docker top`, nvidia-smi, owned `/proc/*/{status,wchan}`, log size/last
   line, run-directory inventory, OOM state, and parent hash.
7. Only for a proven pre-AppReady stall with no run/checkpoint, unchanged parent,
   no traceback/OOM, and proven cleanup, allow one identical retry under a new
   immutable attempt label. Never retry user signals, post-AppReady failures,
   Docker errors, or unverifiable cleanup.
8. Before that one full retry, a pinned 64-environment, one-update, LR=0 runtime
   warmup may be used, but it must use the hardened exact-ID path and be marked
   diagnostic/non-policy. If it fails, stop. Do not use the legacy smoke script
   unchanged because it calls `docker rm -f` and lacks the shared lock contract.
9. Preserve `attempts/00`, `attempts/01`, `retry.decision`, hashes, and a root
   outcome that says whether a retry occurred. Overall success must not hide it.
10. For sharded evaluation, launch shard B only after shard A reaches AppReady;
    harden shard cleanup to immutable IDs before trusting it unattended.

Likely failure class: an intermittent Carbonite/Kit/import-entry deadlock or
runtime/cache/IPC race. Shared writable Kit/OV caches are a plausible lead, not
proof. Do not auto-delete caches or Docker volumes. Add flushed wrapper milestone
timestamps and opt-in `faulthandler`/SIGUSR1 dumps around the deferred import and
`runpy` boundaries before changing caches, IPC, or image layout.

Primary implementation targets:

```text
isaaclab/deploy/probe-stage2c-single-current-best
isaaclab/deploy/calibrate-stage2c-bilateral-current-best
isaaclab/deploy/screen-stage2c-probe-sharded
isaaclab/train_model_only_resume.py
isaaclab/deploy/hexapod-rl-training-stage2c-stable-forward.service
isaaclab/deploy/stage2_pipeline.sha256
isaaclab/tests/test_probe_stage2c_single_launcher_contract.py
isaaclab/tests/test_calibrate_stage2c_bilateral_launcher_contract.py
```

## 6. Exact next-run and evaluation commands

First audit without changing anything:

```bash
ssh orionh@100.82.166.9
cd /home/orionh/HEXAPOD
date -u
ps -eo pid,ppid,lstart,etime,%cpu,%mem,args --sort=-%cpu | head -80
pgrep -af 'nsva_dl|validate_nsva|score_clip|validate_b51|torch.*compile|train_model_only_resume|evaluate_checkpoint' || true
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
systemctl is-active hexapod-rl-training.service || true
nvidia-smi
flock -n /tmp/hexapod-isaac-gpu.lock true
sha256sum -c isaaclab/deploy/stage2_pipeline.sha256
sha256sum isaaclab/logs/rsl_rl/hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/2026-08-25_23-44-14_accel_scale4_yaw160_trackguard_20260825T234500Z_seed86/model_2.pt
```

Only after every resource check is clean, launch one new unique label through
the hardened launcher. Example shape; replace the timestamp once:

```bash
cd /home/orionh/HEXAPOD
label=accel_probe21_bilateral_long2_ref1800_seed99_slew040_retry2_YYYYMMDDTHHMMSSZ
out=/home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c/${label}.launcher.log
test ! -e "$out"
nohup ./isaaclab/deploy/probe-stage2c-single-current-best \
  "$label" 99 \
  env.inactive_bilateral_longitudinal_contact_moment_reward_scale=-2.0 \
  env.inactive_bilateral_longitudinal_contact_moment_reference_nm=1.8 \
  >"$out" 2>&1 </dev/null &
```

The launcher pins all other settings and verifies the parent hash. Do not append
unreviewed Hydra overrides.

After a successful exact 12-checkpoint inventory, resolve `child.run_path` and
run the formal sharded screen:

```bash
cd /home/orionh/HEXAPOD
./isaaclab/deploy/screen-stage2c-probe-sharded \
  <successful_probe21_batch_label> \
  <timestamped_child_run_basename> \
  2026-08-25_23-44-14_accel_scale4_yaw160_trackguard_20260825T234500Z_seed86 \
  model_2.pt \
  a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a \
  formal_seed60_slew040_sharded_v1 \
  0 1 2 3 4 5 6 7 8 9 10 11
```

Predeclared causal interpretation against checkpoint-index-matched Probe20:

- late checkpoints 8-11 median `0.30` yaw advantage at least `0.003 rad/s`;
  at least three of four improve by `0.001`;
- best `0.30` yaw at most `0.128918`;
- `0.16/0.20` yaw at most `0.099221/0.103926`;
- `0.30` deck at most `1.042313`; moving mean deck at most `0.891376`;
- zero falls/timeouts;
- max duty `<=0.093889`, burst `<=0.12 s`, peak `<=3.511 Nm`, worst-joint
  duty `<=0.247895`.

Stronger go signal: `0.30` yaw `<=0.125`, peak `<=3.0 Nm`, burst `<=0.10 s`,
worst-joint duty `<=0.24`. A causal gain is not automatic promotion; the absolute
Stage2C yaw gate remains `0.080`.

If Probe21 is a no-go, abandon bilateral shaping rather than scaling it blindly.
If inconclusive, use a seed-100 matched pair. Ground-yaw dose escalation is low
priority because Probe20 falsified the apparent Probe18 gain. The next distinct
portfolio candidate after resolving reliability is the uncompleted yaw-rate-slew
arm, not a stronger ground-yaw penalty.

## 7. Robot and task configuration

Assets:

```text
URDF: /home/orionh/HEXAPOD/robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf
USD:  /home/orionh/HEXAPOD/robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda
container USD: /workspace/hexapod/robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda
mass target: 1.5 kg body + six 0.8 kg complete legs = 6.3 kg
```

Runtime joint/action order:

```text
0 revolute_1_1   1 revolute_1_7   2 revolute_2_5
3 revolute_3     4 revolute_4     5 revolute_5
6 revolute_1     7 revolute_1_6   8 revolute_1_5
9 revolute_1_3  10 revolute_1_4  11 revolute_1_2
12 revolute_2   13 revolute_2_6  14 revolute_2_4
15 revolute_2_2 16 revolute_2_3  17 revolute_2_1
```

Indices 0-5 are coxa, 6-11 femur, 12-17 tibia. Do not infer physical leg order
from these names or deploy to hardware without a verified mapping.

RS05 approximation for all 18 joints:

```text
continuous/applied effort limit: 1.6 Nm
raw simulated demand ceiling: 5.5 Nm
nominal velocity: 50.27 rad/s; simulation velocity: 55.29 rad/s
stiffness: 30.0 Nm/rad; damping: 0.6 Nms/rad
armature: 0.0007 kg m^2; friction: 0.01; viscous friction: 0.002
soft joint limit factor: 0.95; self-collision disabled
```

Observations/actions/timing:

```text
observation dimension: 66, no privileged state
  root linear velocity 3, angular velocity 3, projected gravity 3,
  command [forward,lateral,yaw] 3, joint position error 18,
  joint velocity 18, current clipped action 18
action: 18 position offsets, clipped [-1,1], scale 0.20 rad
processed target slew: 0.040 rad per 20 ms
physics: 0.005 s / 200 Hz; decimation 4; policy rate 50 Hz
episode: 20 s; PPO rollout: 24 steps/update
```

Coordinate contract:

```text
anatomical/navigation forward = imported body -Y
navigation lateral             = imported body +X
navigation up                  = imported body +Z
```

This fixes the earlier visually sideways “forward” policy. Do not revert it.

Stage2C pose/commands:

```text
root reset height 0.185 m
coxa default 0; femur 0.6000 rad; tibia 2.2335 rad
moving/standing deck targets 0.181/0.177 m
20% stand, 80% forward; speed 0.16-0.32 m/s; hold 6-10 s
```

Current Probe21 effective overrides are recorded verbatim in each attempt's
`effective.params`. Important values include longitudinal signed progress `+2`,
normalized longitudinal error `-1`, inactive yaw rate `-160`, support shortfall
`-0.90`, rated-torque excess `-0.65`, worst-joint excess `-0.50`, saturation
`-1.50`, torque slew `-0.01`, foot slip `-0.50`, ground-yaw `0`, yaw-slew `0`,
and bilateral longitudinal moment `-2/ref1.8`.

Termination: base contact >5 N, height <0.055 m, upside/tilt projected-gravity
Z >-0.45, raw demand >5.5 Nm continuously for 0.10 s after 0.50 s grace, or
20-second truncation. Reset restores pose/velocity, perturbs joints +/-0.03 rad,
clears action/metric histories, and resamples command/timer.

Terrain is flat. Startup randomization: friction/restitution in 64 buckets and
root mass additive `[-0.20,+0.40] kg`. There is no rough terrain, latency,
encoder/IMU noise, voltage sag, thermal model, actuator-strength variation, or
push randomization in Stage2.

Known model gaps: complex visual-mesh collisions, no explicit foot pads,
self-collision disabled, estimated link/inertia/friction values, no backlash or
thermal/I-squared-t behavior, near-upper-bound tibia default, and no physical
observation/action parity test.

## 8. Environment

Spark:

```text
host: spark-e26c / 100.82.166.9 over Tailscale SSH
user: orionh
project mirror: /home/orionh/HEXAPOD (not a Git repository)
Isaac Lab source: /home/orionh/IsaacLab
OS: Ubuntu 24.04.3 LTS, aarch64, Linux 6.17.0-1031-nvidia
GPU: NVIDIA GB10; driver 580.173.02; driver CUDA 13.0
Isaac Sim: 6.0.1-rc.7+release.42383.32955d8d.gl3.0.0
Isaac Sim commit: 045ca8b59622b99a408092124377c66346e8d9c2
Isaac Lab: repository VERSION 3.0.0, commit ffff603eafc6b74264a5261cc0183d6a65390d78
Python: 3.12.13 inside Isaac container
PyTorch: 2.10.0+cu130; CUDA available; device NVIDIA GB10
RSL-RL: rsl-rl-lib 5.0.1
Gymnasium: 1.2.1
```

No Conda/venv is used for training. Docker Compose image `isaac-lab-base`
bind-mounts `/home/orionh/HEXAPOD` at `/workspace/hexapod` and launches via
`/workspace/isaaclab/_isaac_sim/python.sh`. Required non-secret environment:

```text
PYTHONPATH=/workspace/hexapod/isaaclab
PYTHONDONTWRITEBYTECODE=1
ISAACSIM_VERSION=6.0.1
ISAACSIM_ROOT_PATH=/isaac-sim
ISAACLAB_PATH=/workspace/isaaclab
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all
```

`/home/orionh/IsaacLab/docker/.env.base` is required; do not copy its contents
into logs or chat. A PyTorch warning says GB10 compute capability 12.1 is newer
than the build's advertised 12.0 ceiling, but the same image has trained many
successful jobs, so this is not a confirmed stall cause.

## 9. Repository state and transfer caveat

Authoritative Git worktree:

```text
/Users/andreboufama/Documents/CUPI/HEXAPOD
branch: codex/isaaclab-training-artifacts
origin: https://github.com/Cornell-Physical-Intelligence/hexapod-cupi.git
tracking: 0 ahead / 0 behind before this handoff edit
```

Latest commits:

```text
e93d911 Add Isaac Lab training stack and curated policy artifacts
1ed66eb Fix inverted left-leg joint axes; add tripod gait demo to viewer
7a62e45 Add Hexapod MKII mock URDF and React web viewer
```

Current meaningful modified files:

```text
.gitignore
HANDOFF.md
artifacts/README.md
artifacts/phase2_recovery_stage2c_stable_forward/checkpoints/current_best/README.md
isaaclab/analyze_stage2c_probe_sweep.py
isaaclab/deploy/stage2_pipeline.sha256
isaaclab/hexapod_rl/env.py
isaaclab/tests/test_analyze_stage2c_probe_sweep.py
isaaclab/tests/test_bilateral_contact_reward.py
```

Current untracked source/test files:

```text
isaaclab/deploy/calibrate-stage2c-bilateral-current-best
isaaclab/deploy/probe-stage2c-single-current-best
isaaclab/deploy/screen-stage2c-probe-sharded
isaaclab/merge_stage2c_probe_shards.py
isaaclab/tests/test_calibrate_stage2c_bilateral_launcher_contract.py
isaaclab/tests/test_merge_stage2c_probe_shards.py
isaaclab/tests/test_probe_stage2c_single_launcher_contract.py
```

Untracked artifact groups include baseline/Phase1 videos and models,
Stage1/Stage2 checkpoints, 21 Phase2-warmup checkpoints, Probe15-21 artifacts,
and the two newly mirrored failed Probe21 full attempts. Before the two newest
attempt mirrors the audit counted 213 untracked files totaling about 102 MB.
Global `*.log` ignore rules mean logs inside artifact directories do not enter a
normal commit unless force-added or the ignore policy is refined.

**Critical transfer caveat:** GitHub currently contains only through `e93d911`.
The Probe15-21 work, bilateral telemetry/reward, analyzer corrections, hardened
launchers, tests, and recent artifacts are dirty local work. A Claude session
that only clones the branch will not receive them. Give Claude this file plus a
fresh archive of the worktree, or first review/commit/push the dirty work. Do not
clean, reset, switch branches, or regenerate assets before preserving it.

Meaningful dirty changes:

- `env.py`: live full-contact yaw wrench and bilateral longitudinal contact
  moment reward; reset-safe mean/p50 bounded-cost telemetry.
- analyzer/merger: exact sharded candidate membership, parent equality, hashes,
  ranking, and causal comparison.
- probe/calibration launchers: immutable parent, allowlisted interventions,
  shared GPU flock, Docker/service gates, atomic artifacts, dense checkpoint
  inventory, exact-ID cleanup, and current delimiter fix.
- tests: reward math/reset batching and launcher/analyzer contracts.
- artifacts: retained failures, configs, models, hashes, evaluation reports, and
  selected child checkpoints. No failed-attempt evidence should be deleted.

## 10. What to give the Claude session

At minimum, attach this `HANDOFF.md` and state:

> Treat HANDOFF.md as the authoritative 2026-08-26 snapshot. Preserve the
> immutable current-best checkpoint and every failed-attempt artifact. First
> fix the pre-AppReady startup supervisor and post-launch resource race using
> the bounded fail-closed design in section 5. If you have Spark access, do not
> touch `/root/cvnba`; wait for `/root/nsva_dl.sh` and descendants to finish,
> prove the resource gate, then complete the exact Probe21 arm and formally
> screen all 12 children. Do not start Phase3. Optimize for minimum idle GPU time
> without sacrificing experiment identity or workload isolation.

Also provide one of:

1. A fresh archive of `/Users/andreboufama/Documents/CUPI/HEXAPOD`, including
   ignored Probe21 logs; or
2. A reviewed commit/push containing the dirty source, handoff, selected models,
   evaluation JSON, and failed-attempt metadata/logs.

A ready-to-upload 26 MB source/evidence bundle was created at:

```text
/Users/andreboufama/Documents/CUPI/HEXAPOD/claude_hexapod_handoff_20260826T174700Z.tar.gz
```

It contains this handoff, the current `isaaclab/` source and tests, robot assets,
the current-best Stage2C checkpoint, and the Stage2C probe evidence (including
ignored logs). Python caches and local `isaaclab/logs/` are excluded. It contains
no `.env`, credential, key, or token file found by the prepackaging scan.

If Claude runs on its own computer and cannot reach the Spark/Tailscale network,
it can review and improve the source but cannot honestly claim to have resumed,
monitored, evaluated, or recorded live training. In that case it must return the
exact patch and remote commands for execution by a session that has Spark
access. Do not give it credentials in chat; authenticate through the normal
Tailscale/GitHub flow.

## 11. Video and Phase3 policy

The last available anatomical-forward videos are Phase1 v5:

```text
artifacts/phase1_v5/video/model200_cmd0p30_single/rl-video-step-0.mp4
artifacts/phase1_v5/video/model200_cmd0p30_army144/rl-video-step-0.mp4
```

There is no admitted Stage2C stability video. The user asked for a new video
only after a real breakthrough/policy worth being proud of. When a checkpoint
passes or materially approaches the absolute gates with safe torque metrics,
record both a single-robot close view and a many-environment “army” view, verify
anatomical forward visually, and copy the MP4 plus frames and evaluation JSON
into the repository artifacts.

Phase3 prototypes reference an Intel RealSense D455 depth camera and a Livox
Mid-360-like near-hemispherical LiDAR/IMU. They are not fused into a trained
terrain task. Do not launch a long Phase3 overnight job without the user's
explicit approval.
