# Training and task design

Durable design reference for the hexapod locomotion task: the simulator stack,
the observation/action/timing contract, the coordinate contract, the staged
curriculum, the reward structure, the Stage2C acceptance gates, and the known
sim-to-real gaps.

This file describes design that is expected to stay true across runs. It does
not record run results or current status; those live in `STATUS.md` and in
`artifacts/`.

## 1. Simulator and framework stack

```text
Simulator: Isaac Sim 6.0.1 + Isaac Lab DirectRLEnv / PhysX
RL:        RSL-RL 5.0.1, PPO, actor/critic [256, 256, 128], ELU
Policy observation normalization: enabled; frozen on model-only recovery
```

The task package is `isaaclab/hexapod_rl`. Gym task IDs are registered in
`isaaclab/hexapod_rl/register.py` and are frozen: existing IDs must not be
renamed, because checkpoints, launchers, evaluation payloads, and manifests
reference them by string. The current research task ID is:

```text
Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-Hexapod-RobStride-Direct-v0
```

Assets used by the task:

```text
URDF: robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf
USD:  robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda
container USD path: /workspace/hexapod/robot/hexapod_mkii_mock_assy/usd/hexapod_mkii_robstride/hexapod_mkii_robstride.usda
mass target: 1.5 kg body + six 0.8 kg complete legs = 6.3 kg
```

Asset v1 (ADR-0001), the CAD assembly, is registered under its own task ID so
every ID above keeps loading the mock it was trained on:

```text
task ID: Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0
URDF: robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf
USD:  robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda  (tools/import_urdf_to_usd.py)
container USD path: /workspace/hexapod/robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda  (override: HEXAPOD_MKII_V1_USD_PATH)
mass: 8.261 kg from Onshape per-part properties, every RS05 hard-set to 191 g; no sensor payload yet
spec: packages/hexapod_env/hexapod_env/assets/spec.py :: MKII_V1_ASSET
runtime joint order: packages/hexapod_core/hexapod_core/joints_v2.py (provisional until read back from the imported articulation)
```

Each robot model is described once by a `HexapodAssetSpec` (USD, root link,
joint and link names, limits, stance, foot-pad geometry, expected runtime joint
order). An env config points at a spec; `HexapodEnv` reads link names from the
config and, when the config names an expected runtime joint order, refuses to
construct if the imported articulation's `joint_names` differ. Adding a robot
is a new spec plus a new env config under a new task ID.

## 2. Observation, action, and timing contract

```text
observation dimension: 66, no privileged state
  root linear velocity          3
  root angular velocity         3
  projected gravity             3
  command [forward, lateral, yaw] 3
  joint position error         18
  joint velocity               18
  current clipped action       18
action: 18 joint position offsets, clipped to [-1, 1], scale 0.20 rad
processed joint-target slew limit: 0.040 rad per 20 ms
physics: 0.005 s / 200 Hz; decimation 4; policy rate 50 Hz
episode: 20 s; PPO rollout: 24 steps per update
```

The 66-dimensional layout and the 18-action interface are the deployed
interface. A task that changes either one cannot load an existing checkpoint
model-only; the Stage2G scratch arm is the deliberate exception and uses 68
dimensions (see §4).

The `0.040 rad / 20 ms` processed-target slew limiter is both a training
setting and the playback setting used by every formal screen recorded so far.
Comparisons across checkpoints are only valid at a common limiter value.

Runtime joint/action order:

```text
 0 revolute_1_1   1 revolute_1_7   2 revolute_2_5
 3 revolute_3     4 revolute_4     5 revolute_5
 6 revolute_1     7 revolute_1_6   8 revolute_1_5
 9 revolute_1_3  10 revolute_1_4  11 revolute_1_2
12 revolute_2    13 revolute_2_6  14 revolute_2_4
15 revolute_2_2  16 revolute_2_3  17 revolute_2_1
```

Indices 0-5 are coxa, 6-11 femur, 12-17 tibia. Do not infer physical leg order
from these names, and do not deploy to hardware without a separately verified
mapping.

RS05 actuator approximation applied to all 18 joints:

```text
continuous/applied effort limit: 1.6 Nm
raw simulated demand ceiling:    5.5 Nm
nominal velocity: 50.27 rad/s; simulation velocity: 55.29 rad/s
stiffness: 30.0 Nm/rad; damping: 0.6 Nms/rad
armature: 0.0007 kg m^2; friction: 0.01; viscous friction: 0.002
soft joint limit factor: 0.95; self-collision disabled
```

## 3. Coordinate contract

```text
anatomical/navigation forward = imported body -Y
navigation lateral            = imported body +X
navigation up                 = imported body +Z
```

This contract fixed the earlier policy that walked visually sideways while
reporting "forward" progress. It is exposed by the `command_frame:
"navigation"` profile introduced in Phase1 v5 and inherited by every later
stage. Do not revert it.

## 4. Curriculum stages

Each stage is a separate registered task and a separate log/checkpoint
namespace. Later stages inherit the earlier stance, limiter, and safety
structure unless stated otherwise.

### Phase 1 — forward walking in the body frame, then in the anatomical frame

| Stage | Configuration | Intent |
| --- | --- | --- |
| Phase1 v2 | `phase1_v2_cfg.py`: action scale 0.20, linear-velocity reward 4.0, yaw-rate reward 1.0, yaw std 0.20, fall penalty -1.0 | Safer second pass at forward walking; reduce raw position demand and explicitly protect heading and falls |
| Phase1 v3 | `phase1_v3_cfg.py`: yaw-rate reward 2.0, yaw std 0.10, rated-torque excess -0.12, saturation -0.5, 200 iterations | Torque-aware, heading-stable fine-tune from the strongest v2 gait; v2 action semantics preserved so v2 checkpoints resume safely |
| Phase1 v4 | `phase1_v4_cfg.py`: yaw-rate reward 2.0, yaw std 0.10, rated-torque excess -0.03, saturation 0.0, 75 iterations | Heading-only refinement that leaves the learned speed/torque tradeoff alone |
| Phase1 v5 | `phase1_v5_cfg.py`: `command_frame = "navigation"`, 500 iterations, scratch run | Corrected anatomical nose-forward axes (§3). Its `model_200.pt` is the last policy with admitted anatomical-forward video evidence |

### Phase 2 — recovery and axis acquisition

| Stage | Command structure | Intent |
| --- | --- | --- |
| Phase2 warmup | opt-in omnidirectional velocity-command profile | Bridge from Phase 1 reset-only commands to sampled velocity commands |
| Stage1 recovery | 55/20/20/5 episode-stable gait-preserving buckets | Re-anchor to the accepted anatomical-forward gait under the Phase 2 command contract |
| Stage2 | 40/30/30 forward/yaw/lateral buckets | Force balanced signed y and yaw acquisition while retaining the forward gait |
| Stage2B lateral | 40/40/10/10 forward / pure-y / forward+y / forward+yaw | Acquire signed lateral gait without sacrificing the safe lower stance |
| Stage2C stabilized forward | stand plus forward-only commands with deliberate stationary anchors | Lower, hardware-aware forward gait with deck steadiness as an explicit objective. This is the current research target |
| Stage2D homotopy (C0-C5) | six short stages, one fixed-bucket command per episode, oblique heading rotated toward pure lateral | Rotate the dominant command from forward to pure navigation-frame lateral while keeping Stage2C stance and limits. **Configured, not admitted** |
| Stage2E joystick (E0-E2) | three short stages; reverse-x grows 0.02-0.05, then 0.04-0.10, then 0.08-0.20 m/s; fixed holds shorten 8 s to 5 s to 3 s | Bridge an accepted Stage2D C5 pure-y policy to a real navigation-frame x/y/yaw joystick with command transitions. **Configured, not admitted** |

Stage2D and Stage2E exist as configuration and grading code only. No Stage2D or
Stage2E policy has been admitted, and Stage2C has not been cleared, so those
stages have no valid parent yet.

### Stage2C pose and command profile

```text
root reset height 0.185 m
coxa default 0; femur 0.6000 rad; tibia 2.2335 rad
moving/standing deck targets 0.181 / 0.177 m
20% stand, 80% forward; speed 0.16-0.32 m/s; command hold 6-10 s
```

The stance values come from the controlled stance sweep, not from tuning by
hand.

### Stage2G — insect-gait rework (two arms)

Rationale, from `isaaclab/hexapod_rl/phase2g_cfg.py`: Stage2C's
speed-conditioned five/four/three-foot support floor combined with heavy
deck-stillness costs structurally rewards a statically supported shuffle. A
perfectly frozen deck and a lively tripod gait are mutually exclusive
objectives. Stage2G replaces that structure with the alternating-tripod
organization insects use at speed:

- a per-environment gait clock paced by commanded speed
  (`gait_cycles_per_meter = 8.0`, clamped to 1.2-3.0 Hz, duty factor 0.5),
- smooth expected stance/swing windows per tripod with runtime-geometric tripod
  assignment,
- a swing-apex clearance target (`0.030 m`, tolerance `0.015 m`),
- the fixed-target air-time bonus disabled, since the phase reward supersedes it,
- the speed-conditioned support schedule replaced by a plain three-foot support
  floor kept only as a safety bias (`support_shortfall_reward_scale = -0.25`),
- relaxed deck-stillness weights so a stepping rhythm is admissible.

RS05 torque costs and the 5.5 Nm raw-demand termination are inherited unchanged.

| Arm | Task ID | Observation | Action authority | Purpose |
| --- | --- | --- | --- | --- |
| Scratch | `Isaac-Velocity-Omni-Stage2G-Insect-Gait-Hexapod-RobStride-Direct-v0` | 68-dim (adds the two-value gait-phase observation) | action scale 0.25, slew limit 0.08 rad / 20 ms | Learn swing-speed leg motion from scratch with a phase-clock input |
| Adapt | `Isaac-Velocity-Omni-Stage2G-Insect-Gait-Adapt-Hexapod-RobStride-Direct-v0` | 66-dim, Stage2C-compatible | action scale 0.20, slew limit 0.06 rad / 20 ms | Keep the deployed interface so the immutable current-best checkpoint loads model-only; only the reward landscape changes |

The adapt arm's `0.06 rad / 20 ms` training limiter is looser than the
`0.040 rad / 20 ms` limiter used by every formal screen on record. A Stage2G
candidate must be screened at the common `0.040` limiter before its numbers can
be compared with the Stage2C ledger.

Stage2G has no evaluation artifacts, no mirrored checkpoint, and no SHA-256 in
this repository. See `STATUS.md` for the open contradiction this creates.

## 5. Reward philosophy

The reward is a small set of task terms plus a large set of bounded safety and
stability costs, with every candidate intervention term defaulting to zero
scale. Default scales live in `isaaclab/hexapod_rl/env_cfg.py`; per-stage
profiles override them; each probe activates exactly one term so that a change
in outcome is attributable.

Term groups:

- **Tracking.** Commanded linear velocity and yaw rate; in Stage2C also signed
  longitudinal progress and a normalized longitudinal tracking error.
- **Deck stability.** Deck-stability composite, vertical velocity, angular
  velocity, flat orientation, base height error. Stage2C weights these heavily;
  Stage2G relaxes them deliberately.
- **Contact and support.** Support shortfall against the stage's support target,
  foot slip, undesired body contact, and (Stage2G only) gait-phase contact
  agreement and swing-apex clearance.
- **RS05 torque safety.** Joint torque magnitude, rated-torque excess above the
  1.6 Nm continuous limit, worst-joint rated-torque excess, torque saturation,
  torque slew, joint acceleration, action rate, and joint-limit proximity.
  These are what keep continuous-duty fraction, burst length, raw peak demand,
  and worst-joint duty inside the RS05 envelope.
- **Probe terms, inactive by default.** Inactive lateral velocity, inactive yaw
  rate, inactive yaw-rate slew, inactive ground-contact yaw moment, and
  inactive bilateral longitudinal contact moment all default to scale 0.

Stage2C effective override values recorded verbatim in each probe attempt's
`effective.params` include: longitudinal signed progress `+2`, normalized
longitudinal error `-1`, inactive yaw rate `-160`, support shortfall `-0.90`,
rated-torque excess `-0.65`, worst-joint excess `-0.50`, saturation `-1.50`,
torque slew `-0.01`, foot slip `-0.50`, ground-yaw `0`, yaw-slew `0`, and (for
the Probe21 arm) bilateral longitudinal moment `-2` with reference `1.8 Nm`.

### Termination and reset

```text
terminate on: base contact > 5 N
              base height < 0.055 m
              projected-gravity Z > -0.45 (upside-down / severe tilt)
              raw torque demand > 5.5 Nm continuously for 0.10 s,
                after a 0.50 s grace period
truncate on:  20 s episode limit
```

Reset restores the stage pose and velocity, perturbs joints by `+/-0.03 rad`,
clears action and metric histories, and resamples the command and its timer.

### Domain randomization present in Stage 2

Terrain is flat. Startup randomization covers friction and restitution in 64
buckets and an additive root mass in `[-0.20, +0.40] kg`. There is no rough
terrain, no latency, no encoder/IMU noise, no voltage sag, no thermal model, no
actuator-strength variation, and no push randomization.

## 6. Stage2C acceptance gates

A candidate replaces the current best only by passing all of these in the
canonical formal screen (deterministic nominal playback, seed 60, 25 warmup
steps, 500 requested steps, `0.040 rad / 20 ms` final target limiter, commands
`(0,0,0)`, `(0.16,0,0)`, `(0.20,0,0)`, `(0.30,0,0)`):

- Yaw-rate RMSE at most `0.080 rad/s` at every moving command.
- Moving normalized deck composite at most `1.000`.
- Zero falls and zero timeouts.
- Achieved forward speed at least `0.240 m/s` at the `0.30 m/s` command.
- RS05 limits held: bounded continuous-duty fraction, bounded burst length,
  bounded raw peak demand, bounded worst-joint duty, and never exceeding the
  `5.5 Nm` raw-demand safety termination.

Beyond the numeric gates, the stated qualitative goal is a visibly lower,
steadier, more natural gait: less deck wiggle, and a more upward-angled
femur/first leg segment so the platform sits lower rather than being held high.

Screen classes are not interchangeable. Short 6-second probe screens carry 275
post-warmup samples and are diagnostic only
(`formal_admission_eligible=false`); the canonical formal evaluation is 10
seconds and 475 samples. Only the formal screen can replace the current best.

Sharded formal screens compare the independently repeated parent reports and
wrapper action-processing records for exact parsed-JSON equality before a
fail-closed merger accepts the exact parent plus `model_0.pt` through
`model_11.pt` membership and ordering.

## 7. Known sim-to-real gaps

Model gaps present in the current task:

- Collision geometry comes from complex visual meshes; there are no explicit
  foot pads and self-collision is disabled.
- Link masses, inertias, and friction are estimates. The 1.5 kg body and 0.8 kg
  complete-leg figures are user-specified mass targets, not measurements of an
  assembled robot; 191 g is the published per-actuator mass. RS05 housing
  inertia, joint friction, latency, and thermal parameters are not published,
  and the URDF generator records the explicit estimates it used.
- No backlash and no thermal / I-squared-t behavior.
- The default tibia angle sits near its upper bound.
- There is no physical observation/action parity test.

### Sim-to-real gates still required

- Replace or identify the assumed body/leg mass distribution, housing inertia,
  and friction.
- Add measured CAN/control delay, encoder/IMU noise, voltage sag, and an RS05
  thermal / I-squared-t model.
- Refine link collisions into primitives plus explicit foot pads, and validate
  self-collision.
- Verify observation parity and joint signs on a restrained robot before
  powered walking.
- Enforce the 1.6 Nm training limit, joint limits, watchdog, and E-stop on
  hardware.

No checkpoint in this repository is hardware-ready, including the current best.

## 8. Validation gate

`isaaclab/validate.py` checks 18 joints, 19 rigid bodies, six foot sensors,
finite observations, 1,000 standing steps, falls and timeouts, raw pre-clip
torque demand, and post-settling saturation. It is a structural gate on the
asset and task, not a policy screen.

## 9. Roadmap beyond Stage2C

The mission the locomotion work serves is in `dar.md`: survey an
operator-drawn bounded area while providing a steady platform for data
collection (ADR-0004). For training that means:

1. Stable anatomical forward/backward walking with a low, steady deck on
   asset v1 (the CAD assembly). The deck-stability composite is the primary
   grade and speed is secondary; thresholds are re-derived from `dar.md` §2
   once the team lead fills in the deck-steadiness numbers.
2. General joystick locomotion: forward, reverse, lateral, diagonal, and yaw in
   both signs, including the command transitions a coverage planner emits at
   the ends of sweep lines.
3. The `dar.md` terrain class (grass, gravel, a stated slope) with a
   body-frame height scan (contract C2, ADR-0002); the lidar is a mapping and
   localization sensor, never a policy input.

Milestones, gates, and ownership are in `docs/PLAN.md`. Phase 3 prototypes
reference an Intel RealSense D455 depth camera and a Livox Mid-360-like
near-hemispherical LiDAR/IMU. They are not fused into a trained terrain task.
Phase 3 is paused, and a long sensor-fusion or difficult-terrain job requires
explicit user approval before launch.
