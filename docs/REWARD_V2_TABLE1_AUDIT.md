# Reward v2: Table I audit

This audit covers the first item of issue #18's reward sub-issue. It audits
[Liu et al., Table I, p. 3](https://arxiv.org/pdf/2511.03167v1#page=3) against
version 1 of the training reward, resolves the tracking-exponent sign and the
stationary-translation criterion, declares the actuator adaptation and
separates the paper reward from command-scaled variants. Every choice below is a
proposal for the named reviewer; nothing here changes `locomotion/task.py`.
[`locomotion/tests/test_reward_v2_table1_formulas.py`](../locomotion/tests/test_reward_v2_table1_formulas.py)
checks each number in sections 4 and 5.

Sources: Table I, read at 400 DPI from the source PDF; version 1,
`measured_reward` in [`locomotion/task.py`](../locomotion/task.py)
(`canonical_quiet_quadratic_log_tail_v2`); the motor model and target limiter in
[`locomotion/env.py`](../locomotion/env.py) and
[`locomotion/env_config.py`](../locomotion/env_config.py). The candidate
[`locomotion/paper_reward.py`](../locomotion/paper_reward.py) implements the
paper reward with the adaptations in section 3.

## 1. Table I as printed

```text
Task r^g
  Linear velocity   1    * exp( ||v_t,xy - v_t,xy^des||_2 / 0.15 )
  Angular velocity  0.5  * exp( ||omega_t,z - omega_t,z^des||_2 / 0.15 )
Style r^s
  D Score           1    * max[0, 1 - 0.25*(d_t^score - 1)^2]
Penalty r^l
  Linear velocity              -1      * v_t,z^2
  Angular velocity             -0.08   * ||omega_t,xy||_2
  Joint torque                 -2e-6   * ||tau||_2
  Joint acceleration           -1.5e-7 * ||q_ddot||_2
  Action rate                  -0.01   * ||a_t - a_t-1||_2
  Collisions                   -0.05   * n_collision
  Joint torque limits          -0.05   * ||max(|tau_t| - tau^limit, 0)||_2
  Joint velocity limits        -0.5    * ||max(|q_dot_t| - q_dot^limit, 0)||_2
  Contact force                -0.1    * ||max(|f_t| - f^limit, 0)||_2
```

`||.||_2` is the unsquared Euclidean norm; on the scalar yaw error it is an
absolute value. The style term needs the AMP discriminator
([`locomotion/amp.py`](../locomotion/amp.py), Step 5) and stays outside reward v2
until the AMP owner agrees its inputs.

## 2. Findings on the printed formulas

**Sign.** As printed, `exp(||e||/0.15)` grows with tracking error. Resolution:
use `exp(-||e||/scale)`, which equals the weight at zero error and decays with
error, and declare the substitution.

**Shape.** Table I divides an unsquared norm by a scale. Version 1 divides a
squared error by a variance (`exp(-||e||^2/0.0009)`), a different kernel. Several
penalties differ in the same way; version 1's `effort` is a normalized mean
square of applied torque, not `||tau||_2` in N·m.

**Weights.** Version 1 differs from Table I without a declared reason: yaw
tracking 0.3 against 0.5, vertical velocity 0.05 against 1.0, roll and pitch
rate 0.01 against 0.08.

## 3. What the simulation supplies for each penalty

The reward reads only what `env.py` exports after a control. A change to
`env.py` voids standing admission; a change confined to `task.py` does not.

| Table I term | Status | Source and adaptation |
| --- | --- | --- |
| Vertical velocity | Direct | `linear_velocity_nav[:, 2]` at the root-link origin |
| Roll and pitch rate | Direct | `angular_velocity_body[:, :2]` |
| Joint torque | Direct | Per-joint RMS of applied torque over the eight substeps, from `torque_square_sum_400hz` |
| Joint acceleration | Derived in `task.py` | Joint-velocity change between consecutive controls over 0.02 s; zero after a reset |
| Action rate | Derived in `task.py` | Raw policy `action` against the previous raw action, before clipping and the limiter |
| Collisions | Approximated | One event when the largest non-tibia ground force exceeds 1 N; self-collisions are not reported |
| Torque limits | Direct | Requested torque (`requested_torque_abs_max_400hz`) against the 1.6 N·m cap; applied torque never exceeds the cap |
| Velocity limits | Direct | Endpoint joint velocity against the URDF limit, 50.27 rad/s |
| Contact force | Omitted | Control traces record no foot force and the paper states no limit |

## 4. Stationary-translation criterion

[TRAINING](TRAINING.md#step-1-tracking-reward-criterion) asks for an under-10%
condition on the commanded translation component. Criterion: for every nonzero
translation command in `command_bank`, a motionless robot earns under 10% of the
translation tracking term's maximum. The audit applies the same condition to
commanded yaw as a declared extension.

The condition is on the commanded component, not on combined tracking. For a
straight translation command, a motionless robot matches the zero yaw command
exactly, so the combined reward keeps a floor of `w_yaw / (w_lin + w_yaw)`,
23.1% in version 1, whatever the translation kernel.

Motionless share of each commanded component:

| Command | Version 1 | Paper, fixed 0.15 | Command-scaled, k = 0.4 |
| --- | ---: | ---: | ---: |
| Translation 0.025 m/s | 49.9% | 84.6% | 8.2% |
| Translation 0.05 m/s | 6.2% | 71.7% | 8.2% |
| Yaw 0.2 rad/s | 36.8% | 26.4% | 8.2% |
| Arc translation 0.04 m/s | 16.9% | 76.6% | 8.2% |
| Arc yaw 0.15 rad/s | 57.0% | 36.8% | 8.2% |

Version 1 fails at 0.025 m/s and for yaw. The paper's fixed scale fails
everywhere here: `exp(-c/0.15) < 0.1` needs `c ≥ 0.15 ln 10 = 0.345 m/s`, a speed
range this robot is not commanded to reach. The scale suits the paper's robot,
not this one.

## 5. Tracking variants

Reward v2 names three variants and keeps them distinct.

**A. Paper.** `w · exp(-||v - c|| / 0.15)` per control, Table I weights 1 and 0.5.
The reproduction baseline; it fails the criterion.

**B. Command-scaled.** `w · exp(-||v - c|| / sigma(c))` with
`sigma(c) = k · max(|c|, c_min)`. A motionless robot then keeps `exp(-1/k)` of the
term for every nonzero command, so any `k < 1/ln 10 ≈ 0.434` passes; `k = 0.4`
gives 8.2%. `c_min` is the smallest nonzero command in `command_bank`: 0.025 m/s
for translation and 0.15 rad/s for yaw. At a zero command a motionless robot
earns the full weight, and `sigma` is continuous at the smallest command.

**C. Command-scaled on stride-averaged velocity.** Variant B applied to the mean
velocity over one 1.2 s tripod period, restarted at each command change and
reset.

Per-control tracking pays a policy that oscillates at the command's speed each
time its velocity passes through the kernel, and it penalizes a gait whose
velocity swings within each stride. Linear tracking at the 0.05 m/s command on
recorded rollouts:

| Rollout | Mean along command | Forward std | Version 1 | Paper | B, per control | C, 1.2 s average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trajectory-optimizer replay (tripod walk) | 0.0488 m/s | 0.038 m/s | 0.364 | 0.802 | 0.254 | 0.967 |
| Test fixture (smooth tripod walk) | 0.0491 m/s | 0.008 m/s | – | – | 0.858 | 0.970 |
| PPO example arm, update 1200 | -0.0006 m/s | – | 0.484 | 0.743 | 0.349 | 0.109 |
| PPO scratch arm, update 1200 | 0.0066 m/s | – | 0.429 | 0.745 | 0.297 | 0.131 |

B ranks the smooth walk well above both failed policies but ranks the
optimizer replay, whose velocity swings within each stride, below them. C pays
both walks and separates them from the failed policies by a wide margin. The
choice is whether in-stride velocity swing should cost reward (B) or only the
mean velocity counts (C). C also makes the reward depend on up to 60 controls of
history, beyond the actor's five-frame observation, and delays the tracking
signal.

**Penalty weights.** Table I's penalties total about 1% of tracking on this
7.47 kg robot. `paper_reward_calibrated` rescales the continuous penalties to 80%
of mean tracking; under the paper kernel a motionless robot then outscores the
walk (1.089 against 0.938 per control). Recalibrate the penalties only after the
tracking variant is chosen.

## 6. Actuator adaptation

The paper adds 18 policy offsets to a nominal pose and drives each joint with
the cascaded law `tau = Kp2 * (Kp1 * (q_des - q) - q_dot)`, gains unstated. This
project keeps its approved motor model and limiter unchanged:

| Stage | This project |
| --- | --- |
| Policy output | 18 values at 50 Hz, clipped to [-1, 1] |
| Joint target | `neutral + 0.35 · a`, clipped to joint limits, then limited to ±0.040 rad per 20 ms control |
| Motor law | `tau_req = 12 · (q_target - q) - K_D · q_dot`; `K_D` = 0.442, 0.246 and 0.106 N·m·s/rad for coxa, femur and tibia |
| Torque limit | Applied torque clipped to a speed-dependent ceiling from the 48 V curve, capped at 1.6 N·m and zero at 480 rpm (50.27 rad/s) |
| Timing | 400 Hz physics, eight substeps per control |

Consequences for reward v2: the torque-limit term reads requested torque, the
action-rate term reads raw actions, and neither the 1.6 N·m cap nor the
0.040 rad / 20 ms limiter changes.

## 7. Proposed reward v2

1. Tracking: variant B or C, per the reviewer's decision; C's window and reset
   rule under test if chosen.
2. Penalties: Table I forms from section 3, contact force omitted, weights
   recalibrated after tracking is fixed.
3. A new `REWARD_VERSION` in `task.py` beside version 1, which stays selectable
   and unchanged.
4. CPU tests for tracking error, the section 4 criterion, zero commands and
   command transitions, including the reset of previous action, previous joint
   velocity and the stride window.
5. Offline checks with `locomotion.reward_scorer` and `locomotion.reward_viewer`
   on the recorded rollouts before any native experiment.

## 8. Decisions for the reviewer

- `k`, `c_min`, and whether the yaw extension of the criterion is adopted.
- Variant B, which costs in-stride velocity swing, or variant C, which scores
  only stride-mean velocity; for C, the window length and restart rule.
- Penalty weights after the tracking decision.
- Reward inputs agreed with the AMP owner, including the AMP state and whether
  the style term joins reward v2.

## Reproduce

```sh
uv run python tools/archive.py restore --destination <dir> artifacts/trajectory_optimizer_20260917/replay_001/standing/evaluation/control_trace.npz
uv run python tools/archive.py restore --destination <dir> artifacts/forward_example_ppo_20260917/scratch_evaluate_update001200_001/run/standing/evaluation_00/control_trace.npz
uv run python tools/archive.py restore --destination <dir> artifacts/forward_example_ppo_20260917/example_evaluate_update001200_001/run/standing/evaluation_00/control_trace.npz
uv run python -m locomotion.reward_scorer <dir>/artifacts/**/control_trace.npz --nominal-height 0.09780231400684256 \
  --reward paper=locomotion.paper_reward:paper_reward
uv run python -m unittest locomotion.tests.test_reward_v2_table1_formulas
```

The restored traces carry the SHA-256 values recorded in
`paper_reward.CALIBRATION`. The nominal height is the training value recorded
in the paired PPO run's `task_definition.json`.
