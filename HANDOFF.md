# Hexapod RL handoff for Claude

> **2026-09-03 addendum.** The training asset is now the CAD assembly
> `robot/hexapod_mkii_assy/` (`urdf/hexapod_mkii_serial.urdf`, 8.261 kg, RS05
> hard-set to 191 g, joints `<leg>_coxa_yaw|femur_pitch|tibia_pitch`, root link
> `body`). `isaaclab/hexapod_rl/asset_cfg.py` and `env_cfg.py` already point at
> it. Section 7 below (mock URDF/USD paths, `revolute_*` joint order, 6.3 kg
> budget) is historical. Follow `AGENTS.md` for the Isaac Sim import, contact
> reports and validation runbook; the USD still has to be generated on the
> Spark with `tools/import_urdf_to_usd.py`, and `phase2_cfg.py` stances must be
> re-seeded from `robot/hexapod_mkii_assy/stance.json` before training.

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

## 2026-09-09 — mock geometry length study (separate from training lineage)

At the user's request, `robot/hexapod_mkii_length_study/` uses the archived
mock's geometry with anatomically registered masses, COMs and full inertia
tensors from the current CAD serial URDF. The grid is 7 × 7: femur and tibia
50–110% in 10% steps, including the unchanged-length baseline. Coxa and mounts
are fixed. Intrinsic link tensors and masses are fixed across this geometry
ablation; longitudinal COM offsets follow length. See the study README and
manifest for the assumptions, precise mapping and hashes. Do not switch the
current CAD training task's USD path to these mock-convention assets.

Tailscale SSH to `spark` (100.82.166.9) verified. After the unrelated StormScope
CUDA job exited, the user instructed us to take the GPU immediately. All work
uses bounded per-job locks; no legacy training service, persistent reservation
or background queue was started. Isolated source and results are under
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/`.

The old `_urdf.ImportConfig` API is absent on the installed Isaac Sim 6 build.
`tools/simulate_length_study.py` now uses Isaac Lab's `UrdfConverter` with the
new importer. It explicitly authors source tensors onto new study USDs and
checks the tensor round trip with OpenUSD's own vector rotation. All 931 link
tensors pass (49 robots × 19 bodies); the separate synthetic CPU test also
passes, with maximum absolute tensor error 1.63e-9 kg m² or less.

The baseline smoke test (`smoke_003`) and first 49-robot batch (`batch_001`)
completed 1,000 physics steps at 400 Hz with finite joint states. This is a
standing comparison using explicit RS05 position control, **not PPO or a
training acceptance gate**. All 49 had the following observed joint order:

```text
revolute_1_1 revolute_1_7 revolute_2_5 revolute_3 revolute_4 revolute_5
revolute_1 revolute_1_6 revolute_1_5 revolute_1_3 revolute_1_4 revolute_1_2
revolute_2 revolute_2_6 revolute_2_4 revolute_2_2 revolute_2_3 revolute_2_1
```

Within each joint family that corresponds to `lf lr lm rr rf rm`; it is an
observation tied to these imports, never a runtime index contract. The runner
resolves positions by name and saves each articulation's actual names/order.
Batch 001 maximum applied torque was 1.600000024 N m (float32 tolerance),
maximum raw computed demand 3.632945 N m. This does not establish low continuous
duty, contact correctness or gait quality. Some common-posture static candidates
are explicitly flagged for insufficient non-foot clearance. The repeated
camera capture is `batch_002`; its final state and image are in
`artifacts/length_study_2026-09-09/`. No asset has been admitted to training by
this task.


### 2026-09-09: walking comparison queued; hardware decision scope expanded

The user requested walking policies for all sizes, an illustrated implementation
writeup when finished, and investigation of credible missed/out-of-bound
femur/tibia solutions. Coxa remains fixed. The original 49-size fixed-inertia
mock study is now explicitly only the first screening stage. Read
`artifacts/length_study_2026-09-09/DECISION_PROTOCOL.md` before continuing.
There is no hardware recommendation and no admitted/trained policy yet at this
entry's creation.

New CPU stance search `robot/tools/prepare_length_study_training.py` writes
`robot/hexapod_mkii_length_study/training_plan.json`. All 49 sizes have at least
one eligible static candidate with 0.20 rad action headroom, predicted hold
load <=1.3 N m, nonfoot clearance >=5 mm and root height >=70 mm. These are not
Isaac admission results. `tools/train_length_study.py` adapts the existing
HexapodEnv process-locally to mock link names and nested contact paths. It uses
400 Hz physics, 50 Hz control, 16/4 solver iterations, unchanged RS05 parameters,
exact transferred mass and fixed friction. Each size must pass 32x1000 control
steps of standing before scratch PPO (256 envs, 500x24 updates, seed57) and
matched 64-env tests at 0.1/0.2/0.3 m/s. `tools/rank_length_study.py` publishes
only completed evaluations and a constrained Pareto shortlist, not a made-up
weighted score or an implementation verdict.

Training entry point is `isaaclab/deploy/hexapod-rl length-study --source ...
--output ...`, implemented by `tools/launch_length_training_spark.py`.
Remote frozen source:
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/training_source_v1`.
The verified batch002 USDs were copied into its package's `training_usd/`.
130 source files are recorded in `campaign_source_hashes.json`. Do not edit
that snapshot during a campaign; preserve evidence and create a new version if
runtime adapter corrections are needed.

Active remote CPU coordinator PID2943469:
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/walking_campaign_001`.
Read its `campaign.json`, then the current job's `state.json`, `run.log`, and
`failure.json` if present. It waits every30seconds without a GPU reservation,
uses both shared locks for actual jobs, stops only its owned immutable Docker
ID if interrupted/competing/low-memory, and has a48hour total bound. A
`stop.request` file stops this campaign. Baseline is first, then the other48
sizes in seeded shuffled order. Failed static/dynamic candidates are not
quietly trained. Infrastructure errors stop the campaign for diagnosis.

At 2026-09-09T17:52Z the GB10 was96% utilized by unrelated StormScope nowcast
PID2942150 (`20260909T1730Z`), which held `/opt/wx/gpu.lock`; project lock was
free. Campaign status was `waiting_for_shared_gpu`, baseline validation queued.
The user was explicitly told no walking policy was training. Do not kill the
weather process or infer the GPU is free from low reported memory use.

A thread heartbeat `complete-hexapod-leg-length-study` is ACTIVE every15minutes
for diagnosis, continuation and the eventual decision writeup. It should stay
quiet on unchanged/non-actionable state, preserve unrelated workloads, and
stop after the full investigation/report or user cancellation. Its prompt
points to campaign001; update it if the campaign moves. This is an actual app
automation, not a promise of manual background monitoring.

CPU checks passed: 288 existing Isaac task tests +11 study tests (8 geometry,
3 training-input/ranking checks), Python compilation and deploy shell syntax.
The new runtime adapter still needs its first Isaac standing import/gate; do
not report runtime success based on these CPU checks.

User inspection viewer: `robot/tools/pack_length_study_viewer.py` packs all49
exact hash-checked URDFs and shared STL meshes into a <1MB fragment. Source
`robot/hexapod_mkii_length_study/preview_template.html`; response fragment
`/Users/andreboufama/.codex/visualizations/2026/09/09/01a086f9-03d8-7861-afc6-5a4ff715a1f2/hexapod-size-inspector.html`.
It supports complete robots, individual links, all49 leg profiles, baseline
outline, angle sliders, selected-link inertials and original URDF download.
Verified via CUA at736/360px light/dark, no JS errors, correct size selection
and mass/inertia picking. It labels mesh scaling and stretched mounting holes
explicitly; no detailed production Onshape CAD was edited.


#### Forecast GPU pause explicitly authorized and executed (2026-09-09T17:57Z)

The user explicitly granted permission to pause the current Spark forecasting
job and defer it, identifying https://github.com/CornellGeoData/Forecasting-Pipeline.
This supersedes earlier instructions against interrupting that particular job.
Only `stormscope-dispatch.timer`, `stormscope-scout.timer` and their two GPU
services were stopped via systemd --user. CPU monitor/verify/publish continue.
The model had no mid-inference checkpoint or completed stage at interruption;
its input and state files were retained. No promise of checkpoint resume was
made. The input run was `scope-202609091730-180` (PID2942150).

Pause evidence, pre-pause state and restoration script live at
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/forecast_pause_001/`.
Run its `resume_forecasting.py` to restore the two GPU timers when the hexapod
GPU work finishes or is cancelled. A native systemd user backstop timer,
`hexapod-restore-forecasting-20260909.timer`, automatically runs that script
48hours later (2026-09-11T17:57Z). It restores scheduling, not the interrupted
model's memory state. Do not silently extend/remove this backstop. The app
heartbeat was updated with this authorization and restoration responsibility.

The hexapod container `hexapod-length-policy-3f0362f583b0` then acquired the
GPU and launched baseline validation in campaign001. It passed import/contact
layout setup and reached800/1000controlsteps by about17:59Z. Read the completed
admission before claiming pass. Walking training follows only if it passes.


#### Walking campaign002: RSL-RL compatibility correction

Campaign001 baseline passed the full32x1000standing gate: 18joints/19bodies/
6feet; zero falls, truncations or nonfoot contact; computed-torque saturation
fraction0.00173611 (0.174% <0.5%); post-settle maximum computed2.157N m,
max applied1.600000024N m; settled root0.211264m. Full admission is archived
locally as `artifacts/length_study_2026-09-09/baseline_walking_admission.json`.
Do not hide the short over-rating samples or treat this as final hardware proof.

The first PPO constructor then failed before any learning with
`MLPModel.__init__() got an unexpected keyword argument stochastic`. Isaac
Lab's config includes deprecated pre-v5 fields; its native train.py calls
`isaaclab_rl.rsl_rl.utils.handle_deprecated_rsl_rl_cfg` before constructing the
runner. The study now uses that same official compatibility helper with the
installed `rsl-rl-lib` version, and explicitly assigns actor/critic to policy
observations. No physics, reward, geometry or PPO hyperparameter was changed.

Preserved failed campaign001/source_v1. New source `training_source_v2` and
active `walking_campaign_002` at the same remote study root. CPU coordinator
PID2949932. It reruns baseline admission then launches PPO, retaining all49
sizes. The app heartbeat now points to campaign002. Inspect actual state/logs
before reporting learning progress. The prior forecast pause and48hour
restoration backstop remain in force.


#### Walking campaign003: GPU ownership snapshot race hardened

Campaign002 completed baseline validation successfully, then its guard stopped
at shutdown claiming a competing CUDA process. An immediate independent check
showed0% GPU use, no CUDA process, and both forecast GPU timers still inactive.
The likely cause is non-atomic nvidia-smi/docker-top snapshots: Kit's CUDA PID
can disappear before the later Docker snapshot. The old guard did not preserve
the alleged foreign PID, so that explanation remains an inference.

`live_competitors` now ignores vanished processes, independently recognizes a
live process's exact owned Docker cgroup, and stops for every other live CUDA
process. On an interruption it records the raw GPU snapshot, Docker-owned PIDs,
foreign cgroups and timestamp. A CPU regression checks that stale/owned entries
are ignored while live weather work is still protected (4 training-input tests,
8 geometry tests and288 preexisting tests passed). Physics and policy settings
remain unchanged.

Preserved campaign002/source_v2. Current remote frozen source is
`training_source_v3`, campaign is `walking_campaign_003`, coordinatorPID2951839,
all under `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/`.
The heartbeat points to campaign003. It reruns the baseline gate and then PPO.
Do not call PPO running until learning updates/checkpoints appear.


#### Confirmed learning, campaign003

Baseline PPO is actually learning in256environments. At about18:07Z the log
reported iteration34/500, checkpoints `model_0.pt` and `model_25.pt`, about
1.37seconds/update and10m51s remaining. User was told about11minutes for the
baseline and a preliminary12–16hours for the initial49size sweep including
validation/evaluation; multi-seed/refinement/CAD work takes longer. No winner
or completed policy was claimed.

Post-processing correction: the LOCAL `tools/rank_length_study.py` additionally
requires MAE <=max(0.03m/s,25% of command) at all three test speeds before a
candidate is eligible. This prevents zero-motion low-power policies from
qualifying. Its regression test includes a standing-still candidate. It does
not change PPO or physics. The active frozen source_v3 keeps its original
ranking helper; do not mutate a running snapshot. Pull completed evaluations
and regenerate the authoritative analysis using the updated LOCAL helper,
and use the corrected helper for subsequent frozen campaigns. Four local
training/ranking tests pass, including live GPU ownership and standing-still
rejection. Total with geometry and preexisting tests:300.


#### Baseline-first video gate (2026-09-09, latest user steering)

User asked to verify one policy and provide its actual recording ASAP before
launching the long all-size sweep. Baseline f100_t100 completed500 updates
(3,072,000 transitions) at18:18:31Z, final checkpoint SHA-256
`9b6f826b2ca36d85979a1ddc9e8d9bdecd18fc8da0be6192d859ff3945870525`.
Latest training telemetry showed weak forward tracking; successful optimizer
completion is not proof of walking. The remaining48 sizes are held.

Campaign003 had already started baseline evaluation when the lock-waiting
helper could acquire the shared GPU lock (flock reacquisition can race a
waiter). After verifying final.pt against completed training state, explicitly
wrote stop.request to interrupt only campaign003's evaluation for video
priority. No trained checkpoint was lost. Its evaluation is partial and must
not be ranked. Campaign003 now has status stopped. The helper exited after
observing this and released its temporary lock.

`tools/train_length_study.py --mode video` loads that exact checkpoint and
admitted geometry, runs1 environment for12s with a0.2m/s forward command,
records actual Isaac RGB frames at1280x720/25fps, and writes trajectory and
checkpoint identity. Training physics/reward/optimizer settings are unchanged.
`tools/launch_length_pilot_spark.py` runs only baseline video then matched
evaluation using the same per-job GPU guards.

First pilot source_v4 failed before GPU launch because the inherited source
hash manifest included mutable Python bytecode. Preserved it. source_v5
excludes __pycache__/*.pyc and the CPU coordinator runs python -B. Active
remote pilot: `baseline_pilot_005`, source `training_source_v5`, both under
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909`. Systemd user unit
`hexapod-baseline-video-pilot5-20260909.service`. Inspect pilot.json and
f100_t100/video/{state.json,run.log,video.json,rollout.mp4}; recording must
complete and be visually checked before claiming a walking video. Matched
evaluation follows automatically. No other morphology is queued by this pilot.
Do not resume49-size training until evaluation/recording and policy behavior
are checked end to end; diagnose weak learning first if confirmed.

The interactive inspector now labels the actual current production reference
`hexapod_mkii_serial.urdf` and embeds its exact downloadable XML (hash prefix
6109956e9e3a). None of the49 mock variants is that production CAD model. A
star identifies only the original mock100/100 baseline, explicitly labeled.
No production CAD edit occurred. Updated fragment size822810bytes.
Twelve geometry/training-input CPU tests passed after video support was added.
Forecasting pause and restoration responsibilities remain unchanged.


Recorder retry: pilot005 hung before AppLauncher ready, CPU~0%, no CUDA,
Python futex wait, matching the older documented Spark pre-startup symptom.
Stopped only its owned container via its stop.request. Source_v6 adds the
existing deploy scripts' Kit telemetry startup flags and90second Python
faulthandler startup diagnostics, plus bounded GPU-lock retry in the pilot.
Current active pilot is `baseline_pilot_006`, source `training_source_v6`,
unit `hexapod-baseline-video-pilot6-20260909.service`. It passed AppLauncher
and generated actual RGB frames at18:22Z. Do not mutate its frozen source.
Both prior failed/hung pilots are preserved. Local300 CPU tests passed.


Baseline video complete: pilot006 recorded300 frames,1280x720 H.264,25fps,
12seconds real-time playback. Local copy:
`artifacts/length_study_2026-09-09/baseline_500_rollout.mp4`, trajectory in
`baseline_500_video.json`. ffprobe confirmed duration/frame count/codec and
first/10second frames were inspected. Mean forward velocity after2seconds
was0.03127m/s for0.2m/s requested; zero falls in this single12s rollout.
It moves slowly with lateral drift, not satisfactory command-tracking gait.
This is a diagnostic rollout, not a successful morphology or hardware result.
Baseline matched64-environment evaluation follows automatically in pilot006.
Keep remaining48 held and diagnose baseline learning before the full sweep.


#### Proposed controlled-gait screen, following user strategy question

User asks whether to hand-pick a few geometries for PPO or screen all without
training each from scratch. Recommendation is a hybrid: mechanical screening
across all49, short torque-limited dynamic trials of parameterized tripod/
ripple/wave controllers with equal tuning budgets, then4–6 promising designs
plus controls for detailed CAD and learned-policy validation. Proposed method
and sources are in `artifacts/length_study_2026-09-09/CONTROLLED_GAIT_SCREEN.md`.
It has not yet been implemented or run. Keep the full PPO sweep held.

Pilot006 evaluation failed at its second speed reset with an in-place update
to an inference tensor outside InferenceMode. Only0.10m/s completed, complete
is false; its very low speed and18.9% requested-torque saturation are not an
all-speed evaluation. Local copy `baseline_500_partial_evaluation.json`. The
local evaluate function now has @torch.inference_mode() so resets and stepping
share the same scope; compilation passes, Isaac rerun still required in a new
frozen source. Do not mutate source_v6 or mark this fix runtime-verified.


#### Stage 1 complete; candidate confirmation required (2026-09-09)

User authorized the controlled-gait strategy, then explicitly requested the
Stage 1 candidates for a smell test before proceeding. Stage 2 controlled
walking trials and all GPU training are held pending that confirmation.

Completed CPU mechanical screening for the original 49 sizes plus nine
boundary/intermediate probes: femur 40/45/55% crossed with tibia 50/60/80%.
Those nine URDFs were regenerated from
`artifacts/length_study_2026-09-09/boundary_stage1_config.json` into
`robot/hexapod_mkii_length_study_boundary/`; coxa is fixed, current mass 8.2608 kg
and transferred inertia tensors are held fixed. No production CAD was edited.

Tools: `screen_length_mechanics.py` provides URDF FK/IK, exact mesh extrema,
force/moment balance with unilateral friction-constrained contacts and minimax
joint load. `run_length_mechanical_paths.py` and
`parallel_length_mechanical_paths.py` evaluate the common pose-selection rules
and tripod/ripple/wave schedules. The full prescribed-motion inverse dynamics
includes COM and rotational accelerations, gyroscopic terms, joint armature
0.0007 kg m^2, Coulomb friction 0.01 N m and viscous friction 0.002 N m s.
The body motion is prescribed for these calculations: no walking controller
or floating-body dynamic trial has demonstrated the target motions yet.

The static search used 56 nominal poses per size. The path search evaluated
1,934 trajectory configurations at 0.05/0.10/0.20/0.30 m/s, two strides
(60/100 mm), 20 mm swing lift and three gait schedules. The shared pose
selection rules retain up to eight diverse poses per schedule. Constraints
include joint soft limits, support margin >=10 mm, nonfoot clearance >=5 mm,
foot-pad discrepancy >=-2 mm, and the 1.6 N m continuous motor region. The
review table imposes nominal belly clearance >=90 mm. Initial phase sampling
was 64; selected paths were rechecked at 256 samples. Candidate 0.20 m/s peak
torque changed by less than 0.6% under this refinement.

Authoritative output under `artifacts/length_study_2026-09-09/`:
- `mechanics_stage1_v3_static/` and `mechanics_stage1_boundary_static/`;
- `mechanics_stage1_motor_complete/` (49) and
  `mechanics_stage1_boundary_paths/` (9), including per-shard frozen source
  copies and hashes;
- `stage1_refined_motor_candidates.json` and
  `stage1_refined_boundary_candidates.json`;
- `stage1_candidate_review.json`, `STAGE1_CANDIDATE_REVIEW.md`,
  `stage1_candidate_profiles.png`, `stage1_torque_map.png`.
Earlier failed, partial and pre-armature screening artifacts are retained but
superseded. In particular, do not use `mechanics_stage1_complete/` or
`stage1_refined_candidates.json` as the final motor-complete screen.

Review candidates: femur/tibia mm, peak estimated N m at 0.20 m/s,
nominal belly clearance mm (256 phase samples):
- A f040_t050: 58/105, 1.0684, 96; lower-boundary probe, motor fit unresolved.
- B f050_t050: 72.5/105, 1.3429, 118; compact.
- C f050_t060: 72.5/126, 1.3807, 142; useful central candidate.
- D f050_t080: 72.5/168, 1.3486, 183; clearance comparison.
- E f050_t100: 72.5/210, 1.4468, 226; long-tibia energy comparison.
- F f055_t060: 79.75/126, 1.4673, 137; more femur room, less torque margin.

All 304 CPU tests passed (16 robot/study plus 288 Isaac tests). Four new
independent checks cover IK/FK and finite-difference Jacobians on all legs,
gravity torque against potential-energy derivatives, LP force/moment/friction
constraints, and rejection of unreachable IK targets. Candidate profile and
torque-map images were visually checked.

No sampled motion passed 0.30 m/s. The original mock 145/210 baseline passed
only a 0.05 m/s sampled case; this is not a proven speed limit. Production CAD
is a separate control, not the mock baseline. Shorter femurs still improve
the sampled boundary, so minimum manufacturable length remains unresolved.
Rigid motor/linkage/mount fit, actual CAD mass properties, self-collision,
impacts/slip/feedback tracking, disturbance recovery and thermal endurance
remain unverified. Mechanical work is not battery consumption. Do not claim
that a hardware optimum or even successful dynamic walking has been found.

While Stage 2 awaits confirmation, restored the Forecasting-Pipeline dispatch
and scout timers at 2026-09-09T19:22:54Z using the preserved
`forecast_pause_001/resume_forecasting.py` on Spark. Both timers verified
active; `forecast_pause_001/restored.json` records the restoration. The native
48-hour backstop was left unchanged. GPU utilization was 0% with no study
container before restoration. Before any later approved GPU work, recheck
ownership and coordinate the already-authorized forecast pause with both GPU
locks; do not assume the GPU remains free.

User smell-test update (2026-09-09): explicitly exclude option A / f040_t050
(58/105 mm) because the femur is too short to fit the motors. This is a
packaging rejection from the user, not a failed dynamics result. Preserve its
screening data for provenance, but remove it from the active candidate list.
Recorded in `artifacts/length_study_2026-09-09/stage1_user_review.json` and
consumed by the report generator. B–F remain under consideration. Assistant
recommendation is C (72.5/126) as the lead for further validation, D (72.5/168)
as the closest competitor, B as the compact comparison, F as a longer-femur
packaging fallback to check, and E as the long-tibia comparison. C offers
about 23 mm more nominal clearance and 16% lower positive-work proxy than B
with nearly identical worst-motor RMS torque. D is close enough that its
slightly lower peak torque and greater clearance warrant direct testing.
No measured dynamic-stability ranking exists. The user asked for a
recommendation; no Stage 2 candidate selection has yet been confirmed.

#### User stance concern: selection bias found and C/D audited

User questioned why longer tibia D has lower peak torque and why the shown
stances are tall. This was a request to investigate, not Stage 2 approval.
Audit found a material selection gap: minimum-static-torque seeds per knee
angle and minimum-height threshold could discard lower postures before the
motion screen. Constant body height/orientation in inverse dynamics also
does not assess actual steady-body motion. The previous recommendation was
too strong about overall project suitability; retain it only as a provisional
geometry lead. A remains excluded for hardware packaging.

New scripts `tools/audit_length_stances.py` and
`tools/report_length_stance_audit.py` generated
`artifacts/length_study_2026-09-09/stance_audit_all_low_poses/` with frozen
source copies/hashes, `audit.json`, `refined_lower_stances.json`,
`same_lengths_lower_stance.png`, and `STANCE_AUDIT.md`. The earlier two-seed
pilot is separately preserved in `stance_audit/` and is superseded.
No original simulation result or URDF was changed. No GPU work started.

Original C/D poses both use mock angles femur10/knee110 degrees. Extending
the inward-pointing tibia brings the nominal lf foot radially closer to the
hip: 61.45 mm C versus 53.66 mm D. The knee offset grows from 9.95 to17.74 mm.
Joint-load optimization also redistributes horizontal ground forces; longer
tibia therefore does not increase every joint moment together. The original
peak difference is only 2.3%. The limiting torque decomposition was rerun at
256 phases and its signed components checked to sum to the actual torque.
This is not proof of a realizable smooth contact-force controller.

Expanded CPU audit completed 459 trajectory configurations: every retained
original-grid pose with belly clearance 80–140 mm for C/D, tripod/ripple/wave,
40/60/100 mm strides, 20 mm lift, speeds0.05/0.10/0.20 m/s. Initial64 phase
samples; the best kinematically/contact-feasible 0.20 m/s case in the95–120 mm
height band for each geometry was rechecked at256 phases:
- C f050_t060: femur40/knee120deg, clearance107.53 mm, tripod100 mm stride;
  peak1.38709 N m at0.20 m/s, passes mechanical gates, work23.8695 J/m,
  worst-motor RMS0.77248 N m. Same geometry as the old141.55 mm /1.38074 N m
  pose. This lower pose was missed by the original seed selection.
- D f050_t080: femur60/knee120deg, clearance116.98 mm, same gait/stride;
  peak1.64994 N m at0.20 m/s, fails continuous-torque gate. At0.10 m/s this
  same tripod path is1.50934 N m and passes; it is not the best slower gait.
These heights are similar, not exactly matched. No global infeasibility or
hardware optimum follows. Both geometries can adopt lower poses; A's short
femur is not necessary for the crouched appearance. Figure uses exact mock
leg meshes with a schematic body block and was visually checked.

Further Stage 1 comparisons must explicitly include common body-height bands,
stance width and motion headroom across all remaining candidates. Do not
reuse the earlier pose pruning as an exhaustive crouch test. Stage 2 still
awaits the user's candidate confirmation, and forecasting stays restored.

#### C selected; full-robot walking policy authorized (2026-09-09, latest)

User: "proceed with C. I'd like to see a fully training walking policy as soon
as possible with the full robot". This supersedes earlier Stage 2 holds and
prioritizes a single C policy over the multi-size comparison. C is f050_t060,
72.5 mm femur /126 mm tibia, coxa fixed, full six-leg19-body18-joint mock-study
robot with current8.26081134 kg transferred mass/inertia. Production CAD was
not edited. Use lower femur40/knee120deg stance, root contact height0.1305325 m,
belly107.53 mm, reset root0.1365325 m. A remains excluded for motor fit.

Prepared separate inputs via `tools/prepare_candidate_c_training.py` into
`artifacts/length_study_2026-09-09/candidate_c_training_inputs/`. The original
49-size training plan is unchanged. New named-joint256-phase reference is
100 mm stance travel,20 mm lift, tripod duty0.65,6.5 cycles/m. The new
`length_reference_env.py` supplies position AND velocity targets through the
same capped RS05 motors, plus learned residual position offsets up to0.12rad.
Actual base motion/contact forces remain unconstrained physics. Each reset
ramps the reference over2s; the actor observes phase via68 observation values.
Reference-only stepping is labeled untrained and is tested before PPO.

`tools/launch_candidate_c_spark.py` through
`isaaclab/deploy/hexapod-rl candidate-c` runs full standing validation,
32-environment reference probe at0.10/0.20m/s, then PPO1024envs:300 initial
updates followed by600-update resumptions. Each chunk records an actual
checkpoint video BEFORE evaluation. All outputs are fresh and hashed. It
stops for review after two consecutive walking-gate passes or6300 updates;
that is not proof of full robustness or manufacturing readiness. Existing
GPU locks/ownership guards remain. SIGTERM requests an orderly stop.

All306 CPU tests passed (18 robot/study +288 existing Isaac), including new
checks of reference asset/name identity, joint range and full six-leg stance
foot velocity cancellation of the commanded forward body velocity. Video
now wraps reset and stepping in inference mode as evaluation already did.
Runtime tests remain authoritative; do not call the prior unverified issues
fixed until their relevant modes finish successfully.

Remote active source: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/training_source_v7`,
134 frozen source hashes. Campaign: `candidate_c_001`. Unit:
`hexapod-candidate-c-v7-20260909.service`. Started20:24:22Z, passed AppLauncher
startup and began validation. Never mutate v7. Status in `candidate.json`,
standing in `f050_t060/validate_0`, untrained probe in `f050_t060/probe`, and
training/video/eval chunks in `stage_000/f050_t060/{train,video,evaluate}` etc.
Copy and visually inspect the first completed trained video and deliver it
ASAP; do not wait for the whole campaign. Keep working on weak gait or errors.

Forecasting GPU dispatch was identified and paused again under the user's
existing explicit authorization. Evidence and restore script in
`forecast_pause_002/`. The candidate systemd unit has ExecStopPost to run
that restore script when it exits; original48-hour native backstop remains
unchanged. Verify restoration on exit. Do not interrupt other workloads.

C standing gate passed at20:25:43Z:32envs,1000steps,19bodies/18joints/6feet,
zero terminations/truncations/nonfoot contacts, zero post-settle saturation,
max post-settle computed torque0.56404 N m, mean root height0.129538 m.
Local evidence: `artifacts/length_study_2026-09-09/candidate_c_001/standing_admission.json`.
Observed joint order for C import: revolute_1_1, revolute_1_7, revolute_2_5,
revolute_3, revolute_4, revolute_5, revolute_1, revolute_1_6, revolute_1_5,
revolute_1_3, revolute_1_4, revolute_1_2, revolute_2, revolute_2_6,
revolute_2_4, revolute_2_2, revolute_2_3, revolute_2_1. Reference columns are
looked up by name against this runtime order. Probe followed automatically.

Full C reference-only probe completed successfully (still untrained):32envs,
20s each at0.10 and0.20m/s. Mean forward speeds0.09609 and0.20240m/s; zero
falls and nonfoot contacts. Tilt RMS0.2539/0.1815deg; speed absolute error
0.02720/0.05792m/s; requested-torque saturation2.6809/3.1551%. Thus average
speed is good but instantaneous tracking and saturation still miss the final
walking gate. Local `candidate_c_001/untrained_reference_evaluation.json`.
The prior repeated-reset inference failure did not recur in this completed
two-speed probe. Actual reference motor velocity targets were exercised.

PPO stage_000 started automatically with1024envs. Early iteration time~1.64s,
initial300 updates about8–9minutes plus recording startup. First trained clip
was estimated about10minutes from20:29Z, conditional on that measured pace.
Do not call the reference-only probe a trained policy or a final walking result.

First trained C recording ready and visually checked: stage_000 completed300
PPO updates (7,372,800 transitions), checkpoint SHA-256
`5ecbe978d659413dc050fec93b0e211a3eadf6ce1c807790d87c14044ff245a2`.
Actual12s1280x72025fps300-frame recording is locally at
`artifacts/length_study_2026-09-09/candidate_c_001/stage_000/rollout.mp4`;
checkpoint `policy.pt`, video trajectory, environment/agent configs, recording
audit, and frames1/5/10s are alongside it. ffprobe and checkpoint identity
verified; all three frames visually inspected. Mean forward speed after2s
is0.21994 m/s for0.20 commanded, zero falls in this single rollout. This is
an initial trained checkpoint, not a full convergence/robustness result.
Delivering this video immediately in the current user turn. Do not repost
the same stage_000 clip on unchanged heartbeat checks.

Current campaign mode after recording: stage_000 evaluation. It then continues
automatically into stage_001 with600 further updates unless stopped/failed.
Local helper `tools/sync_candidate_c_results.py --stage N` retrieves completed
identity-matched recordings and checkpoints, checks video metadata and extracts
inspection frames. The existing automation follows progress every5minutes.

#### Benchmark 1 preserved; mission scope restored (2026-09-09, latest)

The user selected the first C trained video as the first benchmark for the new
approach, then clarified that this is only a demo and should not absorb excessive
training time. The final goal remains omnidirectional walking over terrain using
onboard sensors, as described in `artifacts/project_review_2026-09-04/ROADMAP.md`.
They requested a documentation-grounded plan of action. This supersedes automatic
continuation of the forward-only C campaign to 6300 updates.

Benchmark `benchmark_01_c_300` is frozen at
`artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300/` and mirrored at
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/benchmarks/benchmark_01_c_300/`.
It contains read-only checkpoint/video/configs, source archive, URDF/reference,
evaluation and a SHA256SUMS manifest. All 134 frozen source hashes were checked;
all 24 benchmark payload hashes were checked again on Spark. Checkpoint remains
`5ecbe978d659413dc050fec93b0e211a3eadf6ce1c807790d87c14044ff245a2`.
Do not overwrite the benchmark or repost its unchanged recording.

The completed nominal evaluation uses 32 environments, 20s per command, seed7057.
At commands0.10/0.20m/s: actual means0.11027/0.21989m/s, absolute speed error
0.01487/0.02533m/s, tilt RMS2.777/2.620degrees, requested torque >1.6Nm in
4.4435/4.8063percent of sampled joint/environment/control-step values. Both tests
had zero falls and nonfoot contacts. Positive mechanical power2.299/3.462W is
not battery power. Torque demand still fails the0.5percent gate; benchmark
selection is a user-preferred demo milestone, not hardware qualification.

The original coordinator had already started stage_001. Its existing campaign
`stop.request` has now been written to stop only the owned extra forward-training
job and preserve all saved outputs. Verify unit exit and restoration of both
forecast timers through the existing ExecStopPost; do not restart this campaign.
The automation was updated to this scope and should be paused once restoration
is verified. Frozen sourcev7 remains unchanged.

Next plan: verify physically buildable C geometry, mass/COM and single-leg motor/
linkage response; extend control to stop/start and signed forward/reverse/lateral/
yaw plus combined commands; qualify bounded terrain, then deployable sensor-based
terrain observations and estimated motion; integrate Jetson runtime and survey
coverage navigation. Preserve the useful phase-guided baseline, and compare any
less constrained controller by evidence instead of treating the demo's fixed
tripod schedule as a permanent terrain requirement. The old roadmap's unguided
policy recommendation predates this accepted new demo; the final behavior and
physical deployment goals remain applicable. Sensor prototype mounts/accounting
are provisional and its old6.3kg budget must not replace current mass accounting.
The user has asked for a plan, not launched a new long terrain/sensor training job.

Cleanup verified at20:51Z: candidate.json status `stopped`, owned systemd unit
inactive, both forecast dispatch/scout timers active, and forecast_pause_002/
restored.json records restoration. The demo-training heartbeat is now PAUSED.
Benchmark 1 is preserved; no new training campaign has been launched.

#### Step 2 authorized: research-backed flat omnidirectional training (2026-09-09)

User explicitly asks to continue Step 2 now, research first, change architecture
if needed, train translation across all360degrees and both yaw signs, and keep
future sensor-based terrain adaptation and the final roadmap central. This
supersedes the previous planning-only hold, while Benchmark 1 remains frozen.
Terrain training is next and has not been launched.

Read `artifacts/omni_flat_2026-09-09/RESEARCH_AND_PLAN.md` for primary-source
research, curriculum, architecture, held-out tests and physical limitations.
Research supports commanded planar velocity/yaw, separate physical regularizers,
and a terrain-aware privileged teacher followed by a sensor-based student.
Quadruped research is method evidence, not proof of this hexapod's performance.

New architecture `omni_history_direct_v1`: scratch PPO,315 actor observations
(5frames of63 IMU/gravity/command/joint/action values),318 critic observations
(add simulator linear velocity only to critic); direct joint-position offsets,
no fixed gait clock or forward stepping reference. Same selected full C study
model,72.5/126mm,coxa fixed,8.2608kg transferred mass/inertia. Production CAD fit
remains unresolved. Named runtime action mapping and caps remain in use.

Commands continuously sample all bearings, pure yaw, mixed motion and standing;
targets change every3–6s with bounded vector/yaw acceleration. New rewards score
requested vector/yaw errors, bounded progress, motor demand, power, smoothness,
slip, nonfoot contact and limits. Flat posture terms must change for terrain.
Actor observation noise is provisional; field dynamics are not calibrated.

All311 CPU tests pass:288 existing Isaac plus23 robot/study tests, including
5new tests for bearing invariance, both yaw signs, overspeed/idle reward,
command coverage, reversal/stop acceleration and reset-safe observation history.
Runtime standing/adaptation/PPO/evaluation/recording still require pilot evidence.

Remote frozen source: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_source_v1`
(138source hashes). Campaign `omni_flat_001`, unit
`hexapod-omni-flat-v1-20260909.service`, deploy entry `hexapod-rl omni-flat`.
Status `omni_flat_001/omni.json`. It runs full32env1000step standing validation,
100-update pilot with1024env,90 static command/seed rows plus uninterrupted
transition evaluation and an actual40s labeled recording, then700-update chunks
up to1500updates for review. Neither budget completion nor average reward is a
success criterion. Watch the first pilot before allowing blind long retries.

Forecasting dispatch/scout paused in forecast_pause_003 under continuing user
authorization. Only Forecasting-Pipeline units were stopped; preflight checks
unrelated GPU users. Unit ExecStopPost restores timers via its resume script;
original native backstop remains. Verify restoration on exit. Frozen sourcev1
and Benchmark 1 must not be modified. If a runtime repair is needed, freeze a
new source version and create a fresh campaign output; preserve all evidence.

#### Arc-capable motion contract and labeled path demos (latest Step 2 steering)

The user asks to choose the best movement approach for future path planning,
combine arcs/turning/strafing, keep terrain adaptation central, continue, and show
example paths with labeled arrows on the ground in Isaac. Confirmed explicitly:
this is PPO (RSL-RL); a separate path follower supplies body-twist requests.

Keep the three-component forward/left/yaw command contract. Plan position and body
heading independently. Future candidate local planner is holonomic MPPI informed
by measured gait limits; current demo uses a bounded feedforward/P pose follower
with ideal simulator localization. Do not claim MPPI or real sensing is deployed.

Version1 runtime pilot completed100updates, standing gate,90-row evaluation and40s
video successfully. It mostly stands; only two static standing rows pass. No moving
capability is established. Evidence fetched under artifacts/omni_flat_2026-09-09/
omni_flat_001/stage_000/ (checkpoint1de5137cb9423fd8d82610279173b8f84a52cbc4e2fc520b71059a79542a0ad2).

Latest saved continuation was model425 from stage001. Copied with verified SHA
5ac83e8ce8909af4d33e3aebbe8aff4581ff930dd34228219cf8c0c19e9f074b to
`/home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_transition_001/resume.pt`;
provenance alongside. A stop.request was sent to version1 before migrating. Its
source/evidence and Benchmark1 remain immutable. Verify forecast restoration before
new pause/launch; use a new campaign and source for version2.

Prepared frozen remote omni_source_v2 (139source hashes): same315/318 actor/critic
architecture and PPO optimizer, combined-yaw samples through zero for gentle arcs,
77static scenarios ×2seeds (616env),70s continuous transitions with S-curve and
fixed-heading bend. New tools/omni_path_demo.py renders five separate actual-policy
path trials (69s total) with ground labels, blue travel/reference arrows, gold body
heading arrows and orange actual trail. The path follower cannot move the robot
except through requests to PPO. Failures and path error are displayed and recorded.
All315CPU tests pass. New rendering still needs an Isaac smoke/visual check.

Active continuation is now **omni_flat_002**, frozen **omni_source_v3** (139hashes),
unit **hexapod-omni-flat-v3-20260909.service**. Version2 was never launched; version3
adds initial renderer creation before camera/drawing operations. It resumes the
verified model425 checkpoint, revalidates the full robot under the new plan, runs
25 PPO updates to exercise the updated154-row evaluation and69s annotated path
video, then700-update chunks up to1425additional updates before review. Sourcev1,
v2,v3 are preserved; never mutate any frozen source.

Version1 stop completed and forecast timers were verified active. Continued user
authorization was used for forecast_pause_004; the new unit's ExecStopPost restores
both timers via that folder's resume_forecasting.py. Check restoration on exit.
Current status: omni_flat_002/omni.json; standing: f050_t060/validate_0; stage outputs:
stage_000/f050_t060/{train,evaluate,video}. Fetch completed identity-checked media
with `python3 tools/sync_omni_results.py --campaign omni_flat_002 --stage N`.
Inspect actual frames for leg behavior and visible ground arrows/text. Ground
annotations are non-colliding display meshes; the root/body must never be driven.
The demo's pose feedback uses simulator localization and must be labeled accordingly.

#### First five-path recording and corrected-label render queue

Campaign omni_flat_002 stage_000 completed the 25 resumed PPO updates, evaluation,
and 69-second actual simulator recording. Checkpoint SHA256 is
`3ed406ca46d948ff9c438ac9b9817a58df39d842c6264a3dc96dc78bedfd3ef8`.
Evidence is local under `artifacts/omni_flat_2026-09-09/omni_flat_002/stage_000/`.
All five path trials had zero terminations. P95 position errors were 0.110 m
straight, 0.058 m sideways, 0.103 m combined arc, and 0.098 m fixed-heading curve.
These closed-loop demonstrations do not establish qualification: all 154 static
command rows and all 14 transition gates failed. Requested motor torque saturation
and tracking remain material problems. Safety terminations are not necessarily falls.
Stage_001 is continuing 700 PPO updates under the existing source_v3 unit.

Actual-frame inspection caught mirrored ground text and clipped legends. Do not
present that first recording as the finished labeled demo. Frozen `omni_source_v4`
(140 hashes) fixes ground glyph orientation and widens the camera framing; it does
not change policy or training. `tools/queue_omni_preview.py` is running as
`hexapod-omni-preview-v4-20260909.service`, output `omni_preview_001/preview.json`.
It waits for the next free interval using both existing GPU locks, selects the
latest completed, evaluated checkpoint with matching SHA, and renders without
interrupting PPO. It has a 90-minute deadline. It does not independently pause
forecasting; the main campaign still owns forecast_pause_004 and restoration.

Fetch `omni_preview_001/f050_t060/video/{rollout.mp4,video.json,state.json}` plus
`preview.json` and `inputs/policy.pt`, verify checkpoint identity and video duration,
then inspect ground text and framing before posting the corrected clip. The existing
sync_omni_results.py handles campaign stages, not this separate preview layout.
The corrected renderer is syntax checked but not yet visually validated in Isaac.
Future source_v3 campaign recordings retain the old label renderer; use source_v4
or a fresh frozen successor for presentation. Preserve all source versions and
Benchmark 1. Any follow-up must track both training and this queued preview.
The existing local heartbeat automation file disappeared during this turn; the
attempt to update its prompt failed. It was not recreated. Do not assume scheduled
follow-up is active without verifying the automation state. The remote training and
render queue run independently of that local heartbeat.


#### Completed omni run, corrected recording, and parallel acceleration (9 September 2026)

User explicitly requests Stage2 fastest completion and terrain/perception in parallel, with as many agents as useful and no avoidable Spark idle time. Three bounded agents now own PPO diagnostics, terrain integration, and sensor mount screening; root alone coordinates GPU jobs. Avoid competing GPU workers. Forecast pause authorization persists.

omni_flat_002 finished 22:55:06 UTC after1425 resumed updates. Final stage002 checkpoint SHA2561971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8. All154 static and14 transition gates still fail. Final median planar error0.0463m/s, yaw error0.1143rad/s, requested torque saturation14.26%; applied torque still capped1.6Nm. Stand seed7057 has no terminations but15.12% requested saturation, indicating problems are not solely reset contamination. Budget completion is NOT Stage2 completion. tools/report_omni_progress.py and campaign PROGRESS_REVIEW.md record comparison.

omni_preview_001 completed23:00:44UTC using that exact final checkpoint and frozen sourcev4. Local artifacts/omni_flat_2026-09-09/omni_preview_001 contains69s1280x72025fps movie,policy,state,video,audit and frames. Five paths all0terms/0truncs, P95positionerror20–57mm. Frames05/33 inspected: text normal orientation; straight lowest orange caption partly clipped, arc legend fits. Posted actual movie to user. Follower uses ideal simulator localization; this is not terrain or perception qualification.

New remote unit hexapod-omni-diagnostics-001-20260909.service launched23:21:27UTC via deploy hexapod-rl omni-diagnostics. Root /home/orionh/HEXAPOD_runs/mock_length_study_20260909. Fresh frozen sources omni_diagnostic_source_001_{baseline,slew,solver},142 hashes each; no older source mutated. Each performs full32env1000step standing admission then48env12s pre-reset diagnostic on the immutable final checkpoint. Separate plan changes: baseline none; slew0.03rad/20ms; solver cfg.sim.physics.enable_external_forces_every_iteration=True (verified installed API). Output omni_diagnostics_001/diagnostic_campaign.json and comparison_00..02/f050_t060/evaluate/{diagnostics.json,diagnostic_trace.npz}. Agent adds observation/history, finite-difference translation/heading and per-joint traces; diagnostics cannot qualify a policy. Launcher's evaluation evidence check now distinguishes this explicit mode.

Forecast pause005 recorded exact StormScope GPU cmd/cgroup and unit states before stopping only dispatch/scout timers/services. Unrelated isim-web-viewer-1 left alone. ExecStopPost runs pause005/resume_forecasting.py, runtime bound5700s; verify restored.json and timer states when finished. Existing original native forecasting backstop remains. Both GPU locks and owned-container cleanup remain. After comparisons, choose measured repair before another long PPO run.

Detailed primary-source research plan: artifacts/project_review_2026-09-04/TERRAIN_AND_SENSING_PLAN_2026-09-09.md. User owns Mid360+D455; extra purchases unrestricted. Proposed LiDAR/forwardcamera plus screened near-foot cameras; no final sensor CAD claimed. Pure CPU prep in tools/terrain_readiness.py and prepare_terrain_readiness.py generated30 fixtures,108 initial mount candidates,payload ledger and uncertainty/age map contract. Separate mount agent expands actual-CAD screening. Training on terrain still requires physical/sensor smoke and exact payload admission.

#### Explicit Stage2 finish standard from user

The user says forward walking is the visual smoothness standard. DO NOT mark Stage2 complete until reverse, both strafes, all diagonal bearings, both yaw directions, combined arcs and path transitions match that level of smoothness, with quiet static stance after stopping. Small disturbance corrections are necessary, but habitual stepping/chatter at zero command is not acceptable. Require representative same-view/same-speed actual-policy videos plus objective motor, stability, tracking and stop/stand checks; one good forward clip, a median score, or budget exhaustion does not complete the stage.


#### Diagnostic conclusion and active quiet-stand repair

All3 controlled diagnostics completed. Source001 baseline requested saturation median14.93%, positive power4.94W,4base-contact terminations. Slew0.03rad/20ms:5.74%,2.30W,0terms. Solver external-forces-every-iteration=True did not help (15.26%,5.27W,4terms); retain False. Slew-only fast-forward and left-arc tracking regressed, so averages are insufficient. Actual standing pose/joint oscillation remains; it is not only noisy endpoint velocities. Evidence and inspected plots are under artifacts/omni_diagnostics_2026-09-09/{README.md,NEXT_EXPERIMENT.md,comparison_report.json,...}.

New opt-in rewards in omni_flat_env.py: stand_joint_velocity and stand_target_velocity, defaultzero for old behavior. Repair overrides: slew0.03; stand_joint_velocity-0.5,stand_target_velocity-0.15,stand_posture-2,action_rate-0.075,saturation-0.75,torque_excess-0.6,worst_torque_excess-0.2. Actor315/critic318 and PPO preserved. Full evaluation now measures before automatic resets, excludes each episode's first2s with per-replica denominators, and retains every failure count.

**Active unit hexapod-omni-repair-002-20260909.service**, launched23:37:55UTC; outputs remote omni_repair_002/repair.json. Frozen sources omni_repair_source_002_diagnostic and _full,142hashes each. First exact-plan32env1000step validation, held-checkpoint48env diagnostic, then100PPOupdates and comparison; up to300+300 more only with measured improvement, no material per-direction regression and further marginal improvement against previous stage. Guard is compute allocation only, not qualification. If no improved checkpoint clears the screen, return needs_repair without expensive known-poor full rerun. Selected improvement receives fresh full-plan standing admission,154static+14transition eval and actual path video; always stage2_complete=False pending visual/all-direction/quietstance/robustness requirements.

Pause008 owns forecasting restoration via ExecStopPost;4hbound+systemd14700s bound,KillMode=mixed. Verify pause008/restored.json and timer states on exit. Source/unit001 was stopped before any training while waiting for GPU; its evidence is preserved. Source002 adds per-direction and incremental allocation screens and cleanup of exact owned DockerID even if Docker client already exited. Focused omni tests19pass.

First terrain fixture runtime (source terrain_smoke_source_001,208hashes; terrain_fixture_smoke_001) stalled beforeCUDAinit. It was stopped to restoreStage2priority. Docker client exit initially left owned cb721d48dad84fcc4b855a247d314ee6f7ff314d9ef39d8fcbe841c621d1f660 running; explicitly stopped that exactID. No unrelated container touched. Local terrain launcher now reads fixtures/validation.json rather than state.json and checks containercleanup independently. Fresh fixture harness defers numpy/helpers untilafterSimulationApp, writes phases and90sfaulthandlertrace; startupcause remainsunproven. Never mutate source001. New fullC terrain standing runner is CPUchecked but needs passed fixture runtime and exactflatadmission beforeGPUuse. See isaaclab/hexapod_terrain/README.md.

Mount screen completed actualCADtriangle tests: first6hip-anchorD405views only16.1%targets visible acrosssampledattitude/lift; expanded120mmoutboard140mmaboveplate90degdown proposal78.1%mean,48.1%worstcase/sector. Neither is finishedCAD or sufficientcoverage. Details artifacts/sensor_mount_study_2026-09-09/README.md; productionCADscreen,notfinaldetailedC. Agentcontinuing CPUcalibrateddepth/pose/mapreplay; terrainagentcontinuing exactsupportqueries/curriculummanifest; PPOagentadding quietstand/stop andallbearingvideoeval.

Created local threadheartbeat **advance-hexapod-stage-2**, every5minutes, toreview/advanceauthorizedwork andnotifyonlymeaningfulchanges. The previously deleted automation was not restored; this is a fresh automation under the user's explicitcontinuousaccelerationrequest. Remote services run independently; localfollowupsdependonapp availability. AllGPUlaunches remainroot-coordinated.


#### Next review hooks and last verified live state

At the latest check, omni_repair_002 passed its new full standing admission and exact-settings diagnostic baseline, then started the100-update PPO pilot on the Spark. Observed iteration time was about1.7s; that is a pilot timing sample, not a Stage2 completion ETA. Baseline reproduced the slew comparison:0terms,5.744%medianrequested saturation,0.03951m/splanar error,0.09178rad/syaw error,2.301Wpositive power and0.750rad/sstandingjointvelocityRMS.

PPO agent completed isolated tools/omni_quiet_review.py and tools/omni_visual_review.py, documented in artifacts/omni_diagnostics_2026-09-09/REVIEW_RUNBOOK.md. They are NOT wired into the active frozen source and have not run in Isaac. Quiet review covers32sstanding plus12stop-from-motion cases; visual review46cases/334s spans16bearings at2speeds, turns, arcs, reversals, paths andquietstand. Integrate their dispatch hooks only into a fresh source; standing admission must match its plan. Proposed quantitative quiet thresholds need reference/standing validation, and visual review remains required.23focused omni tests and288Isaac CPUtests passed.

PPO allocation fixes requested by agent are already in active source002: namedper-direction regressions, improvement relative toprevious chunk and skipping full review when no candidate improves. Source001 never trained. Continue by inspecting omni_repair_002/repair.json, then fetchcompleteddiagnostics andselectedcheckpoint with hashes. Do not let a successful scheduled process overwrite the explicit user quality requirement or markStage2complete.


#### 100-update repair result and next active diagnostic

The omni_repair_002 pilot finished at Unix1788997483.5, after2m54s actual PPO training and about6m48s for the complete standing/baseline/train/evaluation cycle. It returned needs_repair without continuing: planar error improved0.03951→0.02883m/s, but requested saturation worsened5.744→6.640%, standing joint velocity barely changed0.750→0.745rad/s, and forward yaw error worsened0.0889→0.1344rad/s. There were no terminations. No candidate was promoted and Stage2 remains incomplete. Pause008 restoration and no owned leftover container were verified. User asked ETA; answered about5–10minutes per short cycle, with no reliable full smoothness-completion ETA yet.

PPO agent is analyzing exact pre/post traces and smaller decisive corrections. In parallel root launched **hexapod-terrain-smoke-002-20260909.service**, fresh frozen terrain_smoke_source_002 (208 hashes), output terrain_fixture_smoke_002. This traced retry uses deferred geometry imports, --info, phase state and90second stack dumps, five-minute job bound, correct fixtures/validation.json completion check and fixed owned-container cleanup. Pause009 owns restoration; verify restored.json on exit. Do not mutate either frozen terrain source. Terrain agent continues support-query/curriculum preparation.

Sensor/perception agent completed the synthetic CPU pipeline in tools/perception_replay.py: timestamped camera and point-cloud transforms, robot masking, uncertainty/age/observed map and student patch interface.14 targeted tests pass; synthetic steps/pits preserved, unknown/stale observations remain unusable. Artifacts/perception_readiness_2026-09-09/README.md documents limits: no ROS, actual sensors or actor integration yet.
