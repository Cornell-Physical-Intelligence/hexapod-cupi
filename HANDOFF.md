# Hexapod RL lead-agent handoff

Snapshot date: 2026-08-25. Local repository audit and Spark runtime audit were read-only. This file is the only file created by the handoff operation. No training process, service, container, log, checkpoint, configuration file, or existing uncommitted file was stopped, restarted, deleted, overwritten, or otherwise modified.

## Critical state at handoff

- The deployed Stage2C service is still `active (running)` on `spark-e26c`, exactly as requested. Do not infer progress from that state.
- Its training Python process has been idle since `2026-08-25 02:26:29 EST`: zero GPU presence, effectively zero CPU, sleeping in `futex_do_wait`, no nominal run directory, no event file, no checkpoint, and only the initial two-line deprecation warning in Docker logs.
- The newest completed Stage2C work is a 64-environment, one-iteration smoke run ending at `02:23:40 EST`; it produced only `model_0.pt`. It is not a trained or admitted policy.
- The authoritative Git working tree is the Mac repository at `/Users/andreboufama/Documents/CUPI/HEXAPOD`. The deployed Spark mirror `/home/orionh/HEXAPOD` is not a Git repository.
- Spark has an older, partial Stage2C source snapshot. It lacks Stage2D, Stage2E, the release manifest, and preflight script, and its core environment hashes differ from the local worktree. Do not synchronize while the current process is being preserved.
- The last demonstrated useful policy is Phase1 v5 `model_200.pt`: correct anatomical-forward walking. No Stage2 joystick, lateral, stabilized Stage2C, or terrain policy has been admitted.

## 1. Goal

### Intended behavior and success definition

The product goal is a realistic RobStride RS05 hexapod locomotion stack:

1. Walk in the robot's anatomical forward/backward direction, not the imported body's misleading long-axis convention.
2. Keep the main platform low and steady: minimize heave, roll/pitch rate, tilt, and height variation while maintaining a natural gait and safe foot contacts.
3. Accept navigation-frame joystick commands `(forward, lateral, yaw)` in both signs, including reverse, diagonal/combined commands, and uninterrupted direction changes.
4. Remain within the RS05 actuator contract in every joint: 1.6 Nm continuous/applied target, 5.5 Nm absolute simulated raw-demand ceiling, bounded duty/burst and RMS.
5. Only after flat-ground joystick admission, add depth camera plus near-hemispherical 360-degree LiDAR sensor fusion and varied-terrain locomotion. A long Phase3 terrain/sensor-fusion job requires explicit user approval.

Current Stage2C success is deliberately narrower: stand and walk anatomically forward at 0.16, 0.20, and 0.30 m/s in a lower stance, with zero falls/timeouts, accepted command tracking, absolute platform-stability gates, at least 15% moving-stability improvement over immutable Stage2 `model_25`, and nominal plus randomized RS05 admission. This gate has not yet been passed.

### Simulator, task, RL framework, and algorithm

- Simulator/task API: Isaac Sim + Isaac Lab `DirectRLEnv`, PhysX, flat plane.
- Current intended task: `Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-Hexapod-RobStride-Direct-v0`.
- Last demonstrated task: `Isaac-Velocity-Flat-Hexapod-RobStride-Phase1-Anatomical-V5-Direct-v0`.
- RL library: RSL-RL `5.0.1` through `isaaclab_rl`.
- Algorithm: on-policy PPO, actor/critic MLPs `[256, 256, 128]`, ELU, observation normalization.
- Current project stage: Phase2 reward/configuration tuning, training, and admission tooling. Stage2C nominal training is stalled before simulator/GPU initialization. Stage2D/E are locally implemented but have never run. Sim-to-real preparation is preliminary only.

## 2. Repository state

### Authoritative repository

```text
Absolute path: /Users/andreboufama/Documents/CUPI/HEXAPOD
Branch: main
Remote: https://github.com/Cornell-Physical-Intelligence/hexapod-cupi.git
```

No `AGENTS.md` was found in the repository or its parent tree.

The requested latest five commits produce only two because the repository has only two commits:

```text
1ed66eb Fix inverted left-leg joint axes; add tripod gait demo to viewer
7a62e45 Add Hexapod MKII mock URDF and React web viewer
```

Spark deployment state:

```text
/home/orionh/HEXAPOD
fatal: not a git repository (or any of the parent directories): .git
```

### Full `git status --short`

This is the complete default short status after creating this handoff:

```text
 M README.md
?? HANDOFF.md
?? artifacts/
?? isaaclab/
?? robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf
?? robot/hexapod_mkii_mock_assy/usd/
?? robot/sensors/
?? tmp/
?? tools/
```

Expanded status before this file was created contained 237 paths: one tracked modification and 236 untracked paths. Breakdown: 87 non-ignored artifact files, 34 Python bytecode files, 32 deployment files, 17 top-level Isaac scripts, four staged local checkpoint/audit files, 15 RL-package files, six Phase3 files, 20 tests, one generated URDF, nine USD files, four sensor-reference files, five temporary PDF-page renders, and two tools. Ignored but important outputs include `.pytest_cache/` and the `.log` files listed in the training/evaluation sections. Git cannot prove which untracked files predated this task; all work-relevant uncommitted paths are inventoried below.

### Files modified or created during the work

Tracked modification:

- `/Users/andreboufama/Documents/CUPI/HEXAPOD/README.md`: replaced the mock-only description with the physics/training asset contract, 6.3 kg mass target, RS05 assumptions, Isaac Lab curriculum, Spark usage, sensor approval gate, and sim-to-real limitations.

Robot/tooling additions:

- `/Users/andreboufama/Documents/CUPI/HEXAPOD/tools/generate_robstride_urdf.py`
- `/Users/andreboufama/Documents/CUPI/HEXAPOD/tools/enable_nested_contact_reports.py`
- `/Users/andreboufama/Documents/CUPI/HEXAPOD/robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf`
- `/Users/andreboufama/Documents/CUPI/HEXAPOD/robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/` and its nine USD/USDA payload files
- `/Users/andreboufama/Documents/CUPI/HEXAPOD/robot/sensors/` including the Livox Mid-360 manual and STEP references
- `/Users/andreboufama/Documents/CUPI/HEXAPOD/tmp/pdfs/` temporary rendered manual pages

RL package additions under `/Users/andreboufama/Documents/CUPI/HEXAPOD/isaaclab/hexapod_rl/`:

```text
__init__.py
asset_cfg.py
command_sampling.py
env.py
env_cfg.py
phase1_v2_cfg.py
phase1_v3_cfg.py
phase1_v4_cfg.py
phase1_v5_cfg.py
phase2_cfg.py
phase2d_cfg.py
phase2e_cfg.py
ppo_cfg.py
register.py
showcase_sequence.py
```

Top-level Isaac scripts:

```text
bootstrap_phase2_recovery.py
evaluate_checkpoint.py
evaluate_stage2_command_transitions.py
grade_stage1_recovery.py
grade_stage2_axis_acquisition.py
grade_stage2b_lateral_acquisition.py
grade_stage2c_robustness.py
grade_stage2c_stable_forward.py
grade_stage2d_homotopy.py
grade_stage2e_static.py
phase3_sensor_smoke.py
showcase_phase2.py
stage2_command_transition_contract.py
summarize_stance_validation.py
train_model_only_resume.py
validate.py
```

Phase3 prototype under `/Users/andreboufama/Documents/CUPI/HEXAPOD/isaaclab/hexapod_phase3/`:

```text
README.md
__init__.py
d455.py
mid360_pattern.py
sensor_cfg.py
sensor_model.py
```

Deployment/release additions under `/Users/andreboufama/Documents/CUPI/HEXAPOD/isaaclab/deploy/`:

```text
accept-phase1-v4
accept-phase1-v5
hexapod-rl
hexapod-rl-training.service
hexapod-rl-training-stage2b-lateral.service
hexapod-rl-training-stage2c-stable-forward.service
preflight-stage2-spark
record-phase1-v4
record-phase1-v5
record-phase1-v5-army
record-phase2-showcase
screen-phase1-v2
screen-phase1-v3
screen-phase1-v4
screen-phase1-v4-batch
screen-phase1-v5-batch
screen-phase2-command-transitions
screen-phase2-recovery-stage1-batch
screen-phase2-recovery-stage2-batch
screen-phase2-recovery-stage2b-lateral-batch
screen-phase2-recovery-stage2c-robust-best
screen-phase2-recovery-stage2c-stable-forward-batch
screen-phase2-recovery-stage2d-homotopy-stage
screen-phase2-recovery-stage2e-joystick-static
screen-phase2-warmup-batch
smoke-phase2-recovery-stage2
smoke-phase2-recovery-stage2b-lateral
smoke-phase2-recovery-stage2c-stable-forward
stage2_pipeline.sha256
train-phase2-recovery-stage2d-homotopy-stage
train-phase2-recovery-stage2e-joystick-stage
validate-stance-sweep
```

Twenty test modules under `/Users/andreboufama/Documents/CUPI/HEXAPOD/isaaclab/tests/` cover command sampling, recovery bootstrap/model-only resume, reward primitives, Stage2 configuration contracts, every grader, command transitions, sensor hardware accounting, showcase sequencing, and stance summarization. `.pytest_cache` contains 187 collected node IDs and an empty `lastfailed` map at 07:04 EDT; the exact final pytest invocation/output was not retained, so this is evidence of test collection/no cached failures, not proof that all 187 passed in one run.

Artifacts under `/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/` include every local checkpoint, evaluation JSON/log, video, frame, stance study, sensor-smoke report, and baseline export described in section 6. Local staged recovery seeds/audit JSONs also exist under `/Users/andreboufama/Documents/CUPI/HEXAPOD/isaaclab/logs/rsl_rl/`.

### Meaningful change summary

- Generated a physics-targeted RobStride URDF/USD at 6.3 kg: 1.5 kg body plus six complete 0.8 kg legs, including three 191 g motors per leg. Visual meshes were reused as collision meshes; inertias and mass distribution remain estimates.
- Added a DirectRLEnv locomotion task with 18 RS05 position actuators, nested per-leg contact sensors, torque telemetry, raw-demand safety gates, command-frame conversion, and deterministic evaluation.
- Corrected the forward-coordinate error in software: anatomical forward is imported body `-Y`; navigation vectors are `[-body_y, body_x, body_z]`.
- Added Phase1 safety/heading variants and Phase2 warmup/recovery curricula.
- Added model-only warm-start conversion preserving forward outputs while expanding command observations from one useful axis to `x/y/yaw`.
- Added lower-stance stability rewards, processed-target slew limiting, per-joint torque admission, signed lateral/yaw/reverse shaping, Stage2D oblique homotopy, Stage2E joystick transitions, and strict graders.
- Added hash/provenance-aware smoke, training, screening, and recording launchers plus a local release manifest.
- Added an isolated D455 + Livox Mid-360/IMU Phase3 sensor prototype; no sensor-fusion Gym task or terrain training exists.

### Important uncommitted-state warning

All Isaac/RL implementation is untracked. No commit contains the training environment, physics-ready asset, policies, graders, deployment scripts, or Phase3 prototype. The work remains uncommitted because it is experimental/WIP at the pause point: every post-Phase1 steering branch failed admission, Stage2C nominal training never initialized, remote/local sources diverged, and large generated checkpoints/videos/logs need an explicit versioning policy. Preserve the worktree before any cleanup, branch switch, sync, or commit operation.

Local manifest check:

```text
Command: shasum -a 256 -c isaaclab/deploy/stage2_pipeline.sha256
Working directory: /Users/andreboufama/Documents/CUPI/HEXAPOD
Result: success; all 39 listed entries matched.
Manifest SHA-256: 04f460a80b2fb61c987bd63a5d8e243f5da1a8d5fa4fb9db4ee42548209fa408
```

However, the manifest omits runtime dependencies `phase1_v2_cfg.py`, `phase1_v4_cfg.py`, `phase1_v5_cfg.py`, `ppo_cfg.py`, and `train_model_only_resume.py`. It can therefore pass while a required dependency is stale or absent.

## 3. Environment

### Local development machine

```text
OS: macOS/Darwin 24.6.0
Architecture: arm64 (Apple T8132)
Kernel: Darwin Kernel Version 24.6.0, RELEASE_ARM64_T8132
Python command: not installed on PATH as `python`
GPU training: not performed locally
```

### Spark training machine

```text
Host: spark-e26c
Project mirror: /home/orionh/HEXAPOD
OS: Ubuntu 24.04.3 LTS (Noble)
Architecture: aarch64
Kernel: Linux 6.17.0-1031-nvidia
GPU: NVIDIA GB10
Driver: 580.173.02
Driver-reported CUDA: 13.0
```

Full `nvidia-smi` snapshot:

```text
Tue Aug 25 11:12:48 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   39C    P0             11W /  N/A  | Not Supported          |      0%      Default |
+-----------------------------------------+------------------------+----------------------+

Processes: Xorg 76 MiB; GNOME Shell 65 MiB; Firefox 263 MiB; unrelated llama-server 197 MiB.
The Stage2C training PID is absent.
```

### Isaac/Python/RL versions

```text
Isaac Sim environment: 6.0.1
Isaac Sim VERSION file: 6.0.1-rc.7+release.42383.32955d8d.gl3.0.0
Isaac Sim source: /home/orionh/IsaacSim, detached 045ca8b59622b99a408092124377c66346e8d9c2
Isaac Lab source: /home/orionh/IsaacLab, detached ffff603eafc6b74264a5261cc0183d6a65390d78
Isaac Lab describe: v3.0.0-beta2.patch1; repository VERSION: 3.0.0
Installed package: isaaclab==6.1.14
isaaclab_rl==0.5.5
isaaclab_tasks==1.10.9
Python: 3.12.13
PyTorch: 2.10.0+cu130
PyTorch CUDA: 13.0
torch.cuda.is_available(): True
torch.cuda device: NVIDIA GB10
RSL-RL: rsl-rl-lib==5.0.1
Gymnasium: 1.2.1
```

PyTorch warns that the GB10 is compute capability 12.1 while this build advertises support through 12.0. Earlier jobs did train successfully in the same image, so this is a confirmed compatibility warning but not a confirmed cause of the current stall.

### Exact runtime/launch method

There is no Conda environment or ordinary venv in the training path. Systemd runs Docker Compose image `isaac-lab-base` (image ID `sha256:8ddc1623d70d5dd622fd728ce4eb3f59dea6ce1ea5cfbe858353dab15e8d0ef8`) from host working directory `/home/orionh/IsaacLab`. It bind-mounts `/home/orionh/HEXAPOD` at `/workspace/hexapod` and launches through `/workspace/isaaclab/_isaac_sim/python.sh`. Calling the Kit Python binary directly is invalid; it failed with `ModuleNotFoundError: No module named 'typing_extensions'`.

Required launch inputs/environment:

```text
/home/orionh/IsaacLab/docker/.env.base       required by Docker Compose; contents not included
PYTHONPATH=/workspace/hexapod/isaaclab
PYTHONDONTWRITEBYTECODE=1
HEXAPOD_USD_PATH                              optional; unset, so source default applies
ISAACSIM_VERSION=6.0.1
ISAACSIM_ROOT_PATH=/isaac-sim
ISAACLAB_PATH=/workspace/isaaclab
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all
```

Other non-secret environment names were collected without values: `ACCEPT_EULA`, `DEBIAN_FRONTEND`, `DOCKER_ISAACLAB_PATH`, `DOCKER_ISAACSIM_ROOT_PATH`, `DOCKER_NAME_SUFFIX`, `DOCKER_USER_HOME`, `HOME`, `HUB__ARGS__DETECT_ONLY`, `HUB__CACHE__PATH`, `ISAACSIM_BASE_IMAGE`, `ISAACSIM_PATH`, `LANG`, `MIN_DRIVER_VERSION`, `OMNICLIENT_HUB_EXE`, `PATH`, and `VK_DRIVER_FILES`.

## 4. Robot and task configuration

### Assets

```text
Source URDF:
/Users/andreboufama/Documents/CUPI/HEXAPOD/robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_mock_assy.urdf

Training URDF:
/Users/andreboufama/Documents/CUPI/HEXAPOD/robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf

Training USD:
/Users/andreboufama/Documents/CUPI/HEXAPOD/robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda

Spark host USD:
/home/orionh/HEXAPOD/robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda

Container USD default:
/workspace/hexapod/robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda
```

Local audited USD SHA-256: `f0e55f8415e028fcd807cc6bce718916fd7dbb73414f7df7d8aaf4bf370ca5b0`.

### Joint names and action order

Runtime articulation order, evidenced by evaluator per-joint telemetry:

```text
 0 revolute_1_1   1 revolute_1_7   2 revolute_2_5
 3 revolute_3     4 revolute_4     5 revolute_5
 6 revolute_1     7 revolute_1_6   8 revolute_1_5
 9 revolute_1_3  10 revolute_1_4  11 revolute_1_2
12 revolute_2    13 revolute_2_6  14 revolute_2_4
15 revolute_2_2  16 revolute_2_3  17 revolute_2_1
```

Indices 0-5 are coxa, 6-11 femur, 12-17 tibia. The names do not naturally encode anatomical leg order; do not reorder a policy vector without a verified hardware mapping.

### Actuator configuration

All 18 joints use `DCMotorCfg` for an RS05 approximation:

```text
continuous/applied effort limit: 1.6 Nm
simulation/short peak saturation: 5.5 Nm
nominal velocity: 480 rpm = 50.27 rad/s
simulation velocity: 528 rpm = 55.29 rad/s
stiffness: 30.0 Nm/rad
damping: 0.6 Nms/rad
armature: 0.0007 kg m^2
friction/static dynamic proxy: 0.01
viscous friction: 0.002
soft joint limit factor: 0.95
self-collision: disabled
```

Hard position limits: coxa about +/-0.872665 rad, femur 0-1.74533 rad, tibia 0-2.53073 rad.

### Observations

Policy observation dimension is 66, with no privileged state:

```text
root linear velocity in selected command frame        3
root angular velocity in selected command frame       3
projected gravity in selected command frame           3
command [forward, lateral, yaw]                        3
joint position minus default                         18
joint velocity                                       18
current clipped action                               18
TOTAL                                                66
```

No Phase2 observation noise is configured.

### Actions and timing

- Action space: 18 joint-position offsets.
- Raw policy action clipped to `[-1, 1]`.
- Stage2 action scale: `0.20 rad`; added to default joint pose, then clamped to soft limits.
- Stage2C processed target slew: at most `0.04 rad` per 20 ms policy step, reset-anchored to the actual randomized joint pose.
- PhysX step: `0.005 s` (200 Hz).
- Decimation: 4.
- Policy/control frequency: 50 Hz.
- Episode length: 20 s.
- PPO rollout: 24 policy steps per environment per update.
- Source default scene: 1,024 environments; completed large runs and current Stage2C service override to 4,096.

### Coordinate frame

The imported root's physical nose is body `-Y`, not body `+X`. Navigation conversion is:

```text
navigation forward = -body Y
navigation lateral = +body X
navigation up      = +body Z
```

Phase1 v5 and Phase2 use this mapping. Earlier Phase1 policies trained body `+X`, explaining the visually sideways locomotion. This software frame issue is fixed, but joint signs and the whole observation/action mapping remain unverified on restrained hardware.

### Stage2C pose and commands

```text
root reset height: 0.185 m
coxa defaults: 0
femur defaults: 0.6000 rad
tibia defaults: 2.2335 rad
moving deck target: 0.181 m
standing deck target: 0.177 m
command mix: 20% stand, 80% forward
forward command range: 0.16-0.32 m/s
command hold: uniform 6-10 s
```

### Stage2C rewards, resolved through inheritance

All per-step terms are multiplied by the 0.02 s policy step except the terminal fall event.

```text
x tracking Gaussian                         +4.5, std 0.15
y tracking Gaussian                         +3.0, std 0.12 (inactive in Stage2C command set)
yaw tracking Gaussian                       +3.0, std 0.20 (inactive in Stage2C command set)
lateral signed progress                     +2.0
yaw signed progress                         +2.0
inactive lateral velocity squared           -0.25
inactive yaw rate squared                   -0.10
alive                                       +0.10
vertical velocity squared                  -12.0
roll/pitch angular velocity squared         -1.5
applied joint torque squared                -2e-4
computed demand above 1.6 Nm squared        -0.14
fraction of motors over 1.6 Nm              -0.40
joint acceleration squared                  -4e-7
action delta squared                        -0.06
applied torque slew squared                 -0.01
foot air time                                0
foot slip                                   -0.50
support-contact shortfall                   -0.50
undesired contacts                          -1.0
projected-gravity tilt squared             -30.0
base-height error squared                 -800.0
joint-limit violation                       -0.25
bounded deck-stability score                +3.0
terminal fall event                         -8.0
```

Deck stability equally combines world vertical velocity, roll/pitch rate, projected-gravity XY, and command-conditioned height error. Support floors are five/four/three feet at low/medium/high commanded speed. Air-time shaping is disabled because its fixed 0.25 s target encouraged exaggerated high swings.

### Termination and reset

Termination:

- base contact force above 5 N;
- root height below 0.055 m;
- projected-gravity Z above -0.45 (upset/inverted);
- any raw computed joint demand above 5.5 Nm continuously for 0.10 s, after 0.50 s startup grace;
- 20 s timeout is a separate truncation.

Reset:

- restore root pose/velocity at environment origin;
- restore default joint velocities;
- perturb each joint position uniformly by +/-0.03 rad;
- clear action, applied-torque, processed-target, lateral EMA, command timer, and metric histories;
- sample a new command and timer;
- randomize initial episode phase on full-fleet initialization.

### Terrain, contact, randomization, and noise

- Terrain is a flat plane only.
- Startup material randomization: static friction 0.7-1.2, dynamic friction 0.6-1.0, restitution 0-0.02, 64 buckets.
- Root mass startup randomization: additive -0.20 to +0.40 kg.
- Six exact-path terminal-tibia contact sensors track air time/contact points because the nested imported hierarchy cannot be represented by one PhysX contact view. Separate root/coxa/femur sensors detect undesired contacts.
- No rough terrain, pushes, actuator-strength variation, latency, encoder/IMU noise, voltage sag, or thermal model is enabled in Phase2.
- Nominal evaluations disable startup randomization; robust Stage2C evaluation uses eight randomized copies per command.

### Known configuration issues

- Collision geometry duplicates complex visual meshes, has no explicit foot pads, and self-collision is disabled.
- Link masses, inertias, RS05 housing inertia/friction, battery behavior, thermal/I-squared-t behavior, backlash, and latency are estimates or absent.
- The tibia default 2.2335 rad is near its 2.53073 rad hard upper bound; a full +0.20 rad target leaves about 0.097 rad margin.
- Natural gait quality is under-specified by admission: foot clearance, cadence, shuffle/drag, and visual naturalness have no direct hard gate.
- Pure-yaw support shaping may favor pivot dragging.
- No observation/action parity test has been performed on physical hardware.

## 5. Commands already run

Shell history was not inspected because it may contain credentials. Therefore exact caller commands/caller working directories are not recoverable for every historical job. The table below distinguishes commands evidenced directly by service definitions, immutable scripts, logs, and artifacts from commands merely prepared for future use.

### Current service command: executed and still running

Host working directory: `/home/orionh/IsaacLab`.

```bash
/usr/bin/docker compose \
  --env-file docker/.env.base \
  -f docker/docker-compose.yaml \
  --profile base run --rm \
  --name hexapod-rl-training \
  -w /workspace/hexapod/isaaclab \
  -e PYTHONPATH=/workspace/hexapod/isaaclab \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/orionh/HEXAPOD:/workspace/hexapod:rw \
  --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
  isaac-lab-base \
  /workspace/hexapod/isaaclab/train_model_only_resume.py \
  --std-min 0.08 \
  --std-max 0.10 \
  --task Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-Hexapod-RobStride-Direct-v0 \
  --external_callback hexapod_rl.register.register_envs \
  --num_envs 4096 \
  --max_iterations 120 \
  --run_name phase2_recovery_stage2c_lower_stable_forward_from_stage2_model25_seed60 \
  --seed 60 \
  --resume \
  --load_run seed_from_stage2_model25 \
  --checkpoint 'model_25_stage2_seed[.]pt$'
```

Result: process remains active but has not reached simulator initialization or created a run. Learned: systemd/container launch alone is not evidence of PPO progress.

### Important executed workflows evidenced by outputs

| Command/workflow | Working directory | Result | What was learned |
|---|---|---|---|
| `python3 tools/generate_robstride_urdf.py` | local project root | Succeeded; generated URDF exists | Produced the 6.3 kg RobStride physics variant. |
| Isaac Sim URDF import plus `python tools/enable_nested_contact_reports.py .../hexapod_mkii_robstride.usda` | Isaac Sim environment | Succeeded; USD payload and contact reports exist | Imported floating-base asset works with nested contacts. |
| `isaaclab/deploy/screen-phase1-v2` | Spark project/Isaac Lab Docker | Succeeded as evaluator; candidates were not all safe | Model350 was the safest v2 candidate. |
| `isaaclab/deploy/screen-phase1-v4-batch` and `accept-phase1-v4` | Spark | Evaluation succeeded; randomized model405 failed safety | Heading refinement still had falls and >5.5 Nm raw demand. |
| `isaaclab/deploy/screen-phase1-v5-batch` and `accept-phase1-v5` | Spark | Succeeded | Identified Phase1 v5 model200 as the retained correct-forward checkpoint. |
| `isaaclab/deploy/screen-phase2-warmup-batch` | Spark | Evaluation succeeded; no checkpoint admitted | Omni warmup destroyed torque safety/tracking. |
| `bootstrap_phase2_recovery.py` model-only observation rebase | Spark/local staged output | Succeeded; audit JSON exists | Forward actor/critic outputs preserved to <2e-6 while new command columns began at zero. |
| `screen-phase2-recovery-stage1-batch` | Spark | Evaluator/grader succeeded; 0/7 accepted | Forward survived, but all steering directions collapsed. |
| `validate-stance-sweep` and playback evaluations | Spark | Succeeded | Deeper stance reduced platform motion in forward playback but failed mixed-axis safety. |
| `smoke-phase2-recovery-stage2` | Spark | First attempt incomplete; second smoke passed | Model-only Stage2 configuration could initialize. |
| Stage2 systemd training, 150 iterations | Spark `/home/orionh/IsaacLab` | Succeeded | Produced seven checkpoints; none acquired both steering signs. |
| `screen-phase2-recovery-stage2-batch` | Spark | Succeeded as grader; 0/7 accepted | Model25 retained only as conservative seed. |
| Two `smoke-phase2-recovery-stage2b-lateral` attempts | Spark | Both produced model0; second is the clean smoke | Stage2B configuration initialized. |
| Stage2B systemd training, 100 iterations | Spark | Succeeded in 174.04 s | Stronger Y rewards still yielded essentially no lateral policy and unsafe later checkpoints. |
| `screen-phase2-recovery-stage2b-lateral-batch` | Spark | Succeeded as grader; 0/11 accepted | Pure lateral was near zero; falls/torque escalated. |
| `phase3_sensor_smoke.py` attempt 1 | Spark Docker | Failed | `ModuleNotFoundError: isaacsim.sensors`. |
| `phase3_sensor_smoke.py` attempt 2 | Spark Docker | Incomplete/failed before report JSON | Robot, wall, RTX stack and D455 initialized, then depth-texture shape handling failed. No sensor-fusion training ran. |
| `isaaclab/deploy/smoke-phase2-recovery-stage2c-stable-forward` | Spark | Passed; 64 env, one update, model0 | Stage2C smoke worked with the then-deployed source snapshot. |
| `shasum -a 256 -c isaaclab/deploy/stage2_pipeline.sha256` | local project root | Passed all 39 listed entries | Local listed release is internally consistent, but manifest dependency coverage is incomplete. |

Important read-only audit commands executed for this handoff included `pwd`, `uname -a`, `git branch --show-current`, `git log --oneline -5`, `git status --short`, `nvidia-smi`, correct-container Python/Torch imports, `ps`, `tmux ls`, `screen -ls`, `df -h`, `free -h`, `systemctl status/show/cat`, `docker ps/stats/logs/inspect`, `/proc` state inspection, run/checkpoint enumeration, and local/remote SHA-256 comparisons. All succeeded except expected non-applicability/failures: remote project is not Git, local `python` is absent, local `free`/`nvidia-smi` are not macOS commands, and no tmux/screen session exists.

### Prepared but not executed after pause

Do not treat these as completed:

```bash
cd /home/orionh/HEXAPOD
sha256sum -c isaaclab/deploy/stage2_pipeline.sha256
isaaclab/deploy/preflight-stage2-spark
isaaclab/deploy/screen-phase2-recovery-stage2c-stable-forward-batch EXACT_RUN
isaaclab/deploy/screen-phase2-recovery-stage2c-robust-best EXACT_RUN
isaaclab/deploy/train-phase2-recovery-stage2d-homotopy-stage C0 RUN /absolute/seed.pt SHA256
isaaclab/deploy/train-phase2-recovery-stage2e-joystick-stage E0 RUN /absolute/seed.pt SHA256
```

The required files for these commands are not all present on Spark.

## 6. Training runs

All run-directory timestamp prefixes are UTC. Start/duration below use Spark local EST and are inferred from checkpoint modification times unless a training log supplied a duration. `$R` means `/home/orionh/HEXAPOD/isaaclab/logs/rsl_rl`; `$A` means `/home/orionh/HEXAPOD/artifacts`.

### Complete discovered run ledger

| Run directory | Configuration | Start / duration | Envs / seed | Checkpoints | Best/useful result and failure mode |
|---|---|---|---:|---|---|
| `$R/hexapod_robstride_flat_direct/2026-08-24_22-42-30_smoke` | base Direct task, 5 iterations | 17:42:41 / 3 s | 64 / 42 | 0,4 | Smoke passed; no policy claim. |
| `$R/hexapod_robstride_flat_direct/2026-08-24_22-48-35_smoke_continuous` | base task, 10 iterations | 17:48:46 / 7 s | 128 / 42 | 0,9 | Continuous-run smoke passed. |
| `$R/hexapod_robstride_flat_direct/2026-08-24_22-51-24_production_seed42` | original base PPO, 1,200 iterations, old mass | 17:51:39 / 14m22s | 1024 / 42 | 0,50,...,1150,1199 | Learned initial locomotion; exported baseline videos/policies, but asset mass was about 8.175 kg and body +X was later recognized as anatomically sideways. |
| `$R/hexapod_robstride_flat_direct/2026-08-24_23-34-47_mass6p3_capacity4096` | 6.3 kg capacity smoke, 2 iterations | 18:35:19 / 2 s | 4096 / 42 | 0,1 | Proved 4096 environments fit. |
| `$R/hexapod_robstride_flat_direct/2026-08-24_23-48-08_INVALID_zero_command_standing` | base task, intended 600 | 18:48:40 / 6m19s | 4096 / 42 | 0-250 every50 | Explicitly invalid: zero-command/standing setup did not represent walking objective; stopped before 600. |
| `$R/hexapod_robstride_flat_direct/2026-08-24_23-56-02_phase1_forward_mass6p3_seed42` | Phase1 body-frame forward, 600 iterations | 18:56:34 / 18m04s | 4096 / 42 | 0-599 every50 | Model350 was strongest nominal candidate: 3/3 fall-free, max raw 3.952 Nm, mean planar RMSE 0.0328. Behavior was anatomically sideways because body +X was commanded. |
| `$R/hexapod_robstride_phase1_forward_v2_direct/2026-08-25_00-26-33_phase1_v2_safe_forward_mass6p3_seed43` | safer action scale 0.20, 600 iterations | 19:27:06 / 18m37s | 4096 / 43 | 0-599 every50 | Model350 safest: 3/3 fall-free, max raw 4.757 Nm, mean RMSE 0.0282. Later models fell or reached 8.16-15.22 Nm. Still anatomically sideways. |
| `$R/hexapod_robstride_phase1_forward_v3_direct/bootstrap_from_v2_model350` | staged seed | 20:21:23 | n/a | copied model350 | Not a training run. |
| `$R/hexapod_robstride_phase1_forward_v3_direct/2026-08-25_01-25-45_phase1_v3_torque_heading_finetune_from_v2_model350_seed44` | torque/heading fine-tune, 200 updates from model350 | 20:26:15 / 5m54s | 4096 / 44 | 350,375,...,525,549 | No accepted endpoint retained; fine-tuning did not solve anatomical frame. |
| `$R/hexapod_robstride_phase1_forward_v4_direct/bootstrap_from_v2_model350` | staged seed | 20:49:18 | n/a | copied model350 | Not training. |
| `$R/hexapod_robstride_phase1_forward_v4_direct/2026-08-25_01-52-48_phase1_v4_heading_only_from_v2_model350_seed45` | heading-only, 75 updates | 20:53:36 / 1m58s | 4096 / 45 | 360,375,390,405,420,424 | Model405 nominal candidate; randomized 9-copy test had 2 falls, raw peak 7.979 Nm, over-rating fraction 0.226. Rejected. Video exists but motion is body-frame sideways. |
| `$R/hexapod_robstride_phase1_anatomical_forward_v5_direct/2026-08-25_03-04-58_anatomical_frame_smoke_seed51` | anatomical frame, 1 iteration | 22:05:08 | 8 / 51 | 0 | Smoke passed. |
| `$R/hexapod_robstride_phase1_anatomical_forward_v5_direct/2026-08-25_03-06-06_phase1_v5_anatomical_forward_scratch_seed52` | correct navigation frame, 500 iterations | 22:06:37 / 14m49s | 4096 / 52 | 0,25,...,475,499 | Retained model200. Randomized 9-copy/60 s: 0 falls, raw peak 5.490 Nm, max tilt 2.52 deg, max planar RMSE 0.0615. Accepted under the then-current Phase1 screen; later Stage2 per-joint/duty gates are stricter. |
| `$R/hexapod_robstride_phase2_omni_warmup_direct/bootstrap_from_phase1_v5_model200` | staged Phase1 seed | 22:42:58 | n/a | copied model200 | Not training. |
| `$R/hexapod_robstride_phase2_omni_warmup_direct/2026-08-25_03-43-20_phase2_sampler_resume_smoke_seed55` | sampler/resume smoke, 1 update | 22:43:31 | 32 / 55 | model200 | Smoke only. |
| `$R/hexapod_robstride_phase2_omni_warmup_direct/2026-08-25_03-43-53_phase2_omni_warmup_from_phase1_v5_model200_seed54` | full omni mixture, 500 PPO updates, about 49.15M samples | 22:44:25 / 15m03s | 4096 / 54 | 200,225,...,675,699 | 11 screened. Only model400 was fall-free across all nine commands, but raw peak 10.457 Nm and max planar RMSE about 0.244; all rejected/no formal accepted policy. |
| `$R/hexapod_robstride_phase2_recovery_stage1_direct/bootstrap_from_phase1_v5_model200_rebased` | 66-observation model-only rebase | 23:33:15 | n/a | model_200_rebased | Forward actor/critic equivalence errors <2e-6; new lateral/yaw columns zero. |
| `$R/hexapod_robstride_phase2_recovery_stage1_direct/2026-08-25_04-41-34_phase2_recovery_stage1_model_only_smoke3_seed56` | Stage1 model-only smoke | 23:41:44 | 32 / 56 | 0 | Third smoke succeeded after earlier wrapper/import attempts. |
| `$R/hexapod_robstride_phase2_recovery_stage1_direct/2026-08-25_04-55-23_phase2_recovery_stage1_stabilized_from_rebased_v5_model200_seed56` | conservative recovery Stage1, 125 updates | 23:55:53 / 3m35s | 4096 / 56 | 0,25,50,75,100,124 | 0/7 accepted. All evaluated checkpoints retained forward motion/fall-free behavior, but six steering commands failed; yaw/lateral essentially collapsed. Model25 was a conservative continuation seed, not an accepted policy. |
| `$R/hexapod_robstride_phase2_recovery_stage2_direct/seed_from_stage1_model25` | immutable Stage1 seed | 00:22:14 | n/a | model_25_stage1_seed | Not training. |
| `$R/hexapod_robstride_phase2_recovery_stage2_direct/2026-08-25_05-37-23_phase2_recovery_stage2_smoke_seed57_20260825T053709Z` | Stage2 smoke, 1 update | 00:37:34 | 64 / 57 | 0 | Clean smoke succeeded; preceding smoke log was incomplete. |
| `$R/hexapod_robstride_phase2_recovery_stage2_direct/2026-08-25_05-43-11_phase2_recovery_stage2_signed_axes_from_stage1_model25_seed57` | signed-axis recovery, 150 updates | 00:43:44 / 4m32s | 4096 / 57 | 0,25,50,75,100,125,149 | 0/7 accepted and 0/7 axis-safe. Model25 had zero falls and raw peak 4.068 Nm; lateral responses +0.0060/-0.0007 versus 0.040 floor and yaw +0.091/-0.059 versus 0.100 floor. Later checkpoints reached 7.6-20.8 Nm. Model25 retained only as the safest seed. |
| `$R/hexapod_robstride_phase2_recovery_stage2b_lateral_direct/seed_from_stage2_model25` | immutable Stage2 seed | 01:28:52 | n/a | model_25_stage2_seed | Not training. |
| `$R/hexapod_robstride_phase2_recovery_stage2b_lateral_direct/2026-08-25_06-30-59_phase2_recovery_stage2b_lateral_smoke_seed58_20260825T063045Z` | Stage2B smoke | 01:31:09 | 64 / 58 | 0 | Smoke produced model0. |
| `$R/hexapod_robstride_phase2_recovery_stage2b_lateral_direct/2026-08-25_06-32-38_phase2_recovery_stage2b_lateral_smoke_seed58_20260825T063225Z` | corrected/clean Stage2B smoke | 01:32:49 | 64 / 58 | 0 | Smoke passed. |
| `$R/hexapod_robstride_phase2_recovery_stage2b_lateral_direct/2026-08-25_06-39-22_phase2_recovery_stage2b_true_lateral_from_stage2_model25_seed58` | true-lateral branch, 100 updates | 01:39:54 / 2m49s (log: 174.04 s) | 4096 / 58 | 0,10,...,90,99 | 0/11 accepted and 0/11 axis-safe. Pure lateral stayed below 0.001 m/s; model0 had 3 falls and 5.047 Nm peak. Later candidates fell and raw peaks reached 61.408 Nm. Rejected. |
| `$R/hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/seed_from_stage2_model25` | immutable Stage2 seed | 02:22:56 | n/a | model_25_stage2_seed | SHA below; not training. |
| `$R/hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/2026-08-25_07-23-30_phase2_recovery_stage2c_stable_forward_smoke_seed60_20260825T072316Z` | lower stable-forward smoke, 1 update | 02:23:40 / reported 3.35 s | 64 / 60 | 0 | Functional smoke, zero falls. Not formally evaluated/admitted. |
| Intended nominal Stage2C run label `phase2_recovery_stage2c_lower_stable_forward_from_stage2_model25_seed60` | 120 updates | service start 02:26:29 / >9 h idle at audit | 4096 / 60 | none | No run directory; stalled before Kit/simulator/GPU initialization. |

### Retained checkpoints

Last demonstrated correct-forward policy:

```text
Remote: /home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/hexapod_robstride_phase1_anatomical_forward_v5_direct/2026-08-25_03-06-06_phase1_v5_anatomical_forward_scratch_seed52/model_200.pt
Local:  /Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/phase1_v5/checkpoint/model_200.pt
```

Current conservative Stage2 seed, not an admitted joystick policy:

```text
Remote source: /home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/hexapod_robstride_phase2_recovery_stage2_direct/2026-08-25_05-43-11_phase2_recovery_stage2_signed_axes_from_stage1_model25_seed57/model_25.pt
Remote staged: /home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/seed_from_stage2_model25/model_25_stage2_seed.pt
Local: /Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/phase2_recovery_stage2/checkpoint/model_25.pt
SHA-256: 2cb28a0f4f4388e709111a09568c72d0770390d2ba72e08ee17eaaf0f27b6893
```

Newest checkpoint by modification time is Stage2C smoke `model_0.pt`, not the best policy:

```text
/home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/2026-08-25_07-23-30_phase2_recovery_stage2c_stable_forward_smoke_seed60_20260825T072316Z/model_0.pt
```

### Stance studies and evaluation outputs

The deep Stage2C pose was evaluated by replaying the old Stage2 model25 without retraining:

- zero-action 32-copy/5 s: 0.17685 m deck, 0.000126 m height std, 0.000113 m/s vertical RMS, 0.00105 rad/s roll/pitch RMS, 0.085 deg tilt RMS, no falls/unwanted contacts;
- forward 0.20/0.30 m/s versus halfway stance: height variation improved 8.7%, vertical RMS 6.5%, roll/pitch RMS 4.1%, tilt RMS 7.3%, moving deck lowered from about 193.3 to 181.2 mm;
- deep 0.30 m/s remained fall-free with 4.094 Nm peak computed demand;
- eight-command playback failed negative lateral/oblique commands; raw peaks reached 6.224 Nm.

Evidence:

```text
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/stability_stance/README.md
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/stability_stance/20260825_deep_qf0p60_qt2p2335/eval_8cmd.json
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/phase2_recovery_stage1/stance_ab/comparison.json
```

### Videos

Baseline/older body-frame videos:

```text
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/baseline_8p175kg/policy_demo.mp4
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/baseline_8p175kg/policy_demo_close.mp4
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/phase1_v4/video/model405_cmd0p30/rl-video-step-0.mp4
```

Correct anatomical-forward Phase1 v5 videos:

```text
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/phase1_v5/video/model200_cmd0p30_single/rl-video-step-0.mp4
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/phase1_v5/video/model200_cmd0p30_army144/rl-video-step-0.mp4
```

No post-v5 stabilized/joystick policy video exists.

## 7. Currently running processes

Snapshot at `2026-08-25T11:22:24-05:00`; process intentionally left untouched.

Final preservation check at `2026-08-25T12:34:27-05:00`: the same systemd unit was still `active`, the same container was still `running`, and the same training PID `196888` remained alive after `10:07:57` at `0.0%` CPU, sleeping in `futex_do_wait`. No process action was taken.

### Service/container

```text
Unit: hexapod-rl-training.service
Description: RobStride RS05 hexapod Stage-2C lower stable-forward training
Fragment: /etc/systemd/system/hexapod-rl-training.service
Loaded state: disabled
Active state: active (running) since 2026-08-25 02:26:29 EST
Restart policy: on-failure
Systemd working directory: /home/orionh/IsaacLab
Container: hexapod-rl-training
Container ID: 673c004cd82a6d2e23a118d3c0f264709269bd657db8aee40684b2b1ddf9328f
Container status: running, health marked unhealthy
Container working directory: /workspace/hexapod/isaaclab
```

The Docker healthcheck is itself misconfigured: it searches `/isaac-sim/.nvidia-omniverse/logs/Kit`, which does not exist; runtime logs are under `/root/.nvidia-omniverse/logs`. Therefore `unhealthy` is not independent proof of the stall. The stall evidence is the idle process/no-output/no-run/GPU state.

### Process hierarchy and resources

```text
PID 196766  systemd MainPID, docker compose client
PID 196796  docker-compose plugin
PID 196827  containerd shim
PID 196850  container bash/init, container PID 1
PID 196888  training Python, container PID 11
```

Training Python:

```text
full command: /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 /workspace/hexapod/isaaclab/train_model_only_resume.py --std-min 0.08 --std-max 0.10 --task Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-Hexapod-RobStride-Direct-v0 --external_callback hexapod_rl.register.register_envs --num_envs 4096 --max_iterations 120 --run_name phase2_recovery_stage2c_lower_stable_forward_from_stage2_model25_seed60 --seed 60 --resume --load_run seed_from_stage2_model25 --checkpoint model_25_stage2_seed[.]pt$
cwd: /workspace/hexapod/isaaclab
elapsed at final audit: 08:55:54
CPU: 0.0%
RSS: 685292 kB
VSZ: 6483372 kB
threads: 21
state/wait: sleeping / futex_do_wait
voluntary context switches: 10
nonvoluntary context switches: 68
open descriptors: 8
GPU: absent from nvidia-smi and nvidia-smi pmon; no NVIDIA device descriptor open
```

`docker stats --no-stream`:

```text
hexapod-rl-training CPU=0.00% MEM=399.9MiB/121.7GiB MEMPCT=0.32% NET=0B/0B BLOCK=119kB/0B PIDS=22
```

### Logs and reconnection

There is no tmux/screen session and nothing interactive to reattach. Read-only reconnection/log viewing:

```bash
tailscale ssh orionh@100.82.166.9
journalctl -fu hexapod-rl-training.service
docker logs -f hexapod-rl-training
```

Outputs go to Docker stdout/stderr pipes and the systemd journal. Docker JSON log:

```text
/var/lib/docker/containers/673c004cd82a6d2e23a118d3c0f264709269bd657db8aee40684b2b1ddf9328f/673c004cd82a6d2e23a118d3c0f264709269bd657db8aee40684b2b1ddf9328f-json.log
```

### Latest 100 lines of active training log

Only two active-process lines exist:

```text
2026-08-25T07:26:29.822388512Z /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py:10: DeprecationWarning: scripts/reinforcement_learning/rsl_rl/train.py is deprecated. Use `./isaaclab.sh train --rl_library rsl_rl --task <TASK>` instead. Example: `./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-v0`.
2026-08-25T07:26:29.822421232Z   warnings.warn(
```

No active run/event/checkpoint log file is open or being written.

## 8. Known problems and hypotheses

### Confirmed problems/evidence

1. **Current Stage2C process is stalled before training initialization.** Evidence: no new run directory, no Kit/RSL log, no checkpoints, two lines of stdout over nine hours, zero GPU use, effectively zero CPU, futex wait, minimal context switches/I/O, no GPU FD.
2. **Spark source is stale/partial.** Current hashes:

   ```text
   file                         local SHA-256                     Spark SHA-256/status
   env.py                       414ad54c...                        3eac825d...
   env_cfg.py                   b65b2f57...                        2c5de48f...
   command_sampling.py          4e88c848...                        3401c3c2...
   phase2_cfg.py                2457bf66...                        dcea55af...
   phase2d_cfg.py               d1f609c9...                        missing
   phase2e_cfg.py               226a1e3b...                        missing
   train_model_only_resume.py   294c84b8...                        294c84b8... (matches)
   preflight-stage2-spark       ffeda4c7...                        missing
   stage2_pipeline.sha256       04f460a8...                        missing
   ```

3. **Release manifest coverage is incomplete.** It omits several imported/executed runtime dependencies, so a passing manifest check cannot guarantee a valid deployment.
4. **No accepted post-v5 policy exists.** Stage1 0/7, Stage2 0/7 and 0/7 axis-safe, Stage2B 0/11 and 0/11 axis-safe; Stage2C only smoke model0.
5. **Lateral acquisition collapsed.** Stage2 and Stage2B evaluations show near-zero signed Y response despite Y commands.
6. **Unsafe demand appears during attempted steering.** Later candidates reached raw computed peaks from 7.6 to 61.4 Nm, above the 5.5 Nm simulated peak contract.
7. **Current final showcase tooling is unsafe/incompatible.** `record-phase2-showcase` defaults to the legacy warmup task, excludes Stage2C/D/E in the parser, uses lateral +/-0.15 and yaw +/-0.35 outside E2's +/-0.10 and +/-0.28 envelope, accepts an arbitrary checkpoint, and is not bound to an accepted transition grade/hash.
8. **`hexapod-rl` helper is stale.** Its `latest` and `runs` commands hardcode `hexapod_robstride_flat_direct`, so they report the wrong experiment for Stage2.
9. **PyTorch/GB10 capability mismatch warning.** Build max is 12.0, hardware is 12.1. It did not prevent earlier jobs, but it must be tracked.
10. **Simulation-to-hardware model is incomplete.** No measured actuator thermal/I-squared-t, backlash, compliance, battery sag, CAN delay, encoder/IMU noise, or broad link mass/CoM uncertainty.
11. **Phase3 smoke is incomplete.** Attempt 1 import failure; attempt 2 depth-texture mismatch; no sensor-fusion task or training. LiDAR identity is unconfirmed: current Livox Mid-360 profile is 360x59 degrees, whereas a more hemispherical Unitree L1 is 360x90 degrees.
12. **All implementation is uncommitted/untracked.** A cleanup, checkout, or broad sync could lose it.

### Suspected causes/hypotheses, not confirmed bugs

- The active Stage2C stall may be an Isaac/Kit import initialization deadlock, wrapper interaction, container-runtime problem, or another pre-AppLauncher issue. No stack trace was attached because that could perturb the preserved process. The PyTorch capability warning is not sufficient to blame.
- Earlier mixed command distributions let a forward policy earn much of its return while ignoring lateral/yaw commands. Source comments and evaluations support this reward-shortcut hypothesis.
- A narrow lateral Gaussian can reward standing still or one-sided drift too generously around small commands. Stage2D responds with signed progress, normalized error, EMA shaping, and an oblique-to-pure-Y homotopy, but that curriculum has not run.
- Platform steadiness improved from the lower pose, but it may still produce shuffling or foot drag because naturalness/clearance/cadence lack direct admission gates.
- Pure-yaw five-foot support may encourage pivot dragging despite slip penalties.

### Unresolved questions

- Which physical 360-degree LiDAR is actually available, and what are its exact beam pattern, mount, mass, and latency?
- What are measured body/leg masses, link CoMs/inertias, RS05 friction/backlash/thermal behavior, CAN latency, encoder/IMU characteristics, and supply sag?
- What is the canonical hardware joint order/sign/zero calibration?
- Should generated binaries/logs/checkpoints live in Git LFS, object storage, or remain outside Git?
- Does the lead want to preserve the currently stalled process for debugging/stack inspection, or authorize stopping it before source synchronization? This handoff intentionally makes no choice.

## 9. Recommended next action

Do not run a new major experiment until the lead agent reviews this handoff.

Recommended lead-controlled sequence:

1. Preserve the current process and all files until the lead decides whether to capture additional non-destructive diagnostics. Do not sync into its bind mount while it runs.
2. Review the uncommitted work and create a preservation strategy/branch or archive before any cleanup. Separate source/config from large generated artifacts.
3. Once explicitly authorized, the lead should decide how to end the stalled service safely. It has `Restart=on-failure` and an `ExecStartPre` that removes the generic container; stopping/restarting it is intentionally outside this handoff operation.
4. Repair the release manifest to include the full imported/executed dependency graph. Only after the service is inactive, synchronize the reviewed local release to Spark and verify exact hashes plus `preflight-stage2-spark`.
5. Reproduce Stage2C initialization first with a bounded smoke/diagnostic run and capture a stack trace/log if it stalls. Do not launch the 4096-environment nominal job until the cause is understood.
6. Train Stage2C from the exact immutable Stage2 model25 SHA above. Require both nominal and randomized robustness grades to return `accepted: true` before staging a seed.
7. Proceed strictly through Stage2D C0-C5 and Stage2E E0-E2, with an accepted checkpoint/hash at every stage. Do not describe omni walking as proven before E2 static and transition admission pass.
8. Fix the showcase recorder before producing a final E2 video: bind it to the accepted transition grade/checkpoint/hash, select the exact E2 task, and keep commands inside the trained envelope.
9. Keep Phase3 isolated and obtain user approval before any long terrain/sensor-fusion job.

## Appendix A: requested command outputs

### Local repository commands

```text
$ pwd
/Users/andreboufama/Documents/CUPI/HEXAPOD

$ uname -a
Darwin dhcp-vl2041-15994.eduroam.cornell.edu 24.6.0 Darwin Kernel Version 24.6.0: Mon Jul 14 11:30:40 PDT 2025; root:xnu-11417.140.69~1/RELEASE_ARM64_T8132 arm64

$ git branch --show-current
main

$ git log --oneline -5
1ed66eb Fix inverted left-leg joint axes; add tripod gait demo to viewer
7a62e45 Add Hexapod MKII mock URDF and React web viewer

$ git status --short
 M README.md
?? HANDOFF.md
?? artifacts/
?? isaaclab/
?? robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf
?? robot/hexapod_mkii_mock_assy/usd/
?? robot/sensors/
?? tmp/
?? tools/

$ python --version
zsh: command not found: python
```

### Correct Spark environment commands

```text
$ cd /home/orionh/HEXAPOD && pwd
/home/orionh/HEXAPOD

$ uname -a
Linux spark-e26c 6.17.0-1031-nvidia #31-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 24 22:03:06 UTC 2026 aarch64 aarch64 aarch64 GNU/Linux

$ python --version   # through /workspace/isaaclab/_isaac_sim/python.sh in container
Python 3.12.13

$ python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device")'
2.10.0+cu130
True
NVIDIA GB10
```

### Process listing

The full top-40 snapshot was collected. Relevant entries, preserving the exact requested columns:

```text
PID     PPID     ELAPSED %CPU %MEM CMD
5789    3487    20:26:15  1.2  0.5 /usr/lib/firefox/firefox
5897    5861    20:26:14  1.0  0.2 /usr/lib/firefox/firefox -contentproc ...
1995       1    20:37:31  0.4  0.0 /usr/bin/containerd
2307       1    20:37:31  0.4  0.0 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
1959       1    20:37:31  0.2  0.0 /usr/sbin/tailscaled --state=/var/lib/tailscale/tailscaled.state --socket=/run/tailscale/tailscaled.sock --port=41641
43963  43933    18:09:13  0.2  0.6 /app/llama-server [unrelated command arguments omitted]
3332    3327    20:37:18  0.1  0.0 /usr/lib/xorg/Xorg ...
3487    3266    20:37:17  0.0  0.3 /usr/bin/gnome-shell
227837 227825    03:31:48  0.0  0.1 /usr/bin/python /home/orionh/.local/bin/batchspawner-singleuser jupyterhub-singleuser
227981 227837    03:31:24  0.0  0.4 /usr/bin/python -m ipykernel_launcher ...
196827      1    08:46:18  0.0  0.0 /usr/bin/containerd-shim-runc-v2 ...
196888 196850    08:46:18  0.0  0.5 /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 /workspace/hexapod/isaaclab/train_model_only_resume.py --std-min 0.08 --std-max 0.10 --task Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-Hexapod-RobStride-Direct-v0 --external_callback hexapod_rl.register.register_envs --num_envs 4096 --max_iterations 120 --run_name phase2_recovery_stage2c_lower_stable_forward_from_stage2_model25_seed60 --seed 60 --resume --load_run seed_from_stage2_model25 --checkpoint model_25_stage2_seed[.]pt$
196796 196766    08:46:18  0.0  0.0 /usr/libexec/docker/cli-plugins/docker-compose compose ... Stage2C command above
```

Unrelated desktop/kernel rows were omitted here because they add no handoff value; the active training hierarchy is complete above.

### tmux/screen

```text
$ tmux ls 2>&1 || true
error connecting to /tmp/tmux-1000/default (No such file or directory)

$ screen -ls 2>&1 || true
No Sockets found in /run/screen/S-orionh.
```

### Disk

```text
$ df -h
Filesystem                 Size  Used Avail Use% Mounted on
tmpfs                       13G  2.6M   13G   1% /run
efivarfs                   256K   21K  236K   9% /sys/firmware/efi/efivars
/dev/nvme0n1p2             3.7T  138G  3.4T   4% /
tmpfs                       61G     0   61G   0% /dev/shm
tmpfs                      5.0M  8.0K  5.0M   1% /run/lock
/dev/nvme0n1p1             298M  6.4M  292M   3% /boot/efi
tmpfs                       13G  132K   13G   1% /run/user/1000
192.168.8.22:/export/home  147G   12G  128G   9% /export/home
192.168.8.22:/export/data  147G   12G  128G   9% /export/data
```

### Memory

```text
$ free -h
               total        used        free      shared  buff/cache   available
Mem:           121Gi       6.3Gi        22Gi        63Mi        94Gi       115Gi
Swap:           15Gi        28Ki        15Gi
```

## Appendix B: newest paths

Newest completed run/log directory:

```text
/home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/2026-08-25_07-23-30_phase2_recovery_stage2c_stable_forward_smoke_seed60_20260825T072316Z
```

Newest smoke console log:

```text
/home/orionh/HEXAPOD/artifacts/phase2_recovery_stage2c_stable_forward/smoke/phase2_recovery_stage2c_stable_forward_smoke_seed60_20260825T072316Z.log
```

Newest checkpoint by time:

```text
/home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/2026-08-25_07-23-30_phase2_recovery_stage2c_stable_forward_smoke_seed60_20260825T072316Z/model_0.pt
```

Newest substantive local evaluation log:

```text
/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/stability_stance/20260825_deep_qf0p60_qt2p2335/eval_8cmd.log
```

Best demonstrated policy remains Phase1 v5 model200; safest current continuation seed remains Stage2 model25, both listed above.
