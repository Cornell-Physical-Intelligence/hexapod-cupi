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
([`locomotion/amp_discriminator.py`](../locomotion/amp_discriminator.py), Step 5) and stays outside reward v2
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
reset. It is implemented causally, as training computes it: before a full
window has passed since the latest command change, the mean covers only the
controls so far.

Per-control tracking pays a policy that oscillates at the command's speed each
time its velocity passes through the kernel, and it penalizes a gait whose
velocity swings within each stride. Linear tracking at the 0.05 m/s command on
recorded rollouts:

| Rollout | Mean along command | Forward std | Version 1 | Paper | B, per control | C, 1.2 s average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trajectory-optimizer replay (tripod walk) | 0.0488 m/s | 0.038 m/s | 0.364 | 0.802 | 0.254 | 0.932 |
| Test fixture (smooth tripod walk) | 0.0491 m/s | 0.008 m/s | – | – | 0.858 | 0.940 |
| PPO example arm, update 1200 | -0.0006 m/s | – | 0.484 | 0.743 | 0.349 | 0.109 |
| PPO scratch arm, update 1200 | 0.0066 m/s | – | 0.429 | 0.745 | 0.297 | 0.127 |

The B and C columns are `task_v2`'s `linear_tracking` component, scored with
`locomotion.task_v2:scorer_reward` on each trace.

B ranks the smooth walk well above both failed policies but ranks the
optimizer replay, whose velocity swings within each stride, below them. C pays
both walks and separates them from the failed policies by a wide margin. The
choice is whether in-stride velocity swing should cost reward (B) or only the
mean velocity counts (C). C also makes the reward depend on up to 60 controls of
history, beyond the actor's five-frame observation, and delays the tracking
signal.

**Penalty weights.** Table I's penalties total about 1% of tracking on this
7.47 kg robot. `paper_reward_calibrated` rescales the continuous penalties to 80%
of mean tracking under the paper kernel (variant A), and version 2 keeps those
weights with kernel C; under the paper kernel a motionless robot then outscores the
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
   recalibrated after tracking is fixed. Table I has no termination term, but
   PPO bootstraps zero after a termination, so a policy earning negative reward
   would gain by falling; version 2 charges 75 per terminated control, more
   than the discounted value of continuing at the worst recorded mean reward
   (0.721 / (1 - 0.99) = 72.1).
3. A new `REWARD_VERSION` beside version 1, which stays selectable and
   unchanged. `test_recorded_parity` freezes `task.py` byte for byte, so version
   2 lives in [`locomotion/task_v2.py`](../locomotion/task_v2.py) as a subclass
   of `TrainingTask`; `train.py --reward-version 2` selects it. Its defaults
   (variant C, `k` = 0.4, the section 4 floors, calibrated penalties, the
   termination term) are decided in the review of
   [PR #42](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/pull/42).
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

The review of [PR #42](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/pull/42)
records the reviewer's decisions on variant C with its 60-control window and
command-change restart, `k` = 0.4, `c_min` = 0.025 m/s and 0.15 rad/s, the yaw
extension, the calibrated penalty weights and the termination term. The style
term remains open with the AMP owner.

## 9. Failed-policy comparison after the first flat PPO run

You can inspect the [capture audit](../site/assets/ppo_action_audit_20261002_001/reward_audit.json)
and its [analysis source](../site/assets/ppo_action_audit_20261002_001/reward_audit.py).
The audit uses reward v2 from commit
`13f55d4efa40ac8497812a96c7b7af4741fd7666` and four retained native captures
at a forward command of 0.05 m/s. It verifies 68 input files and matching native
model and physics identities. Both optimized walks belong to the admitted AMP
dataset. The tripod and both optimized walks pass their recorded motion screens;
the PPO capture fails.

| Capture | Mean reward per control, 2 < t <= 20 seconds |
| --- | ---: |
| Failed PPO policy | 0.258359 |
| Same PPO trajectory with recorded actions clipped to [-1, 1] | 0.272557 |
| Current tripod reference | 0.987156 |
| Admitted optimized walk, phase 0 | 1.247595 |
| Admitted optimized walk, phase 0.5 | 1.251683 |

The audit reconstructs the 60-control tracking history from control zero and
uses the recorded eight torque samples per control. It scores 900 controls after
the two-second settling window. The original force matrix is absent, so the audit
allows the full collision penalty range of [-0.05, 0] on each trajectory. The
smallest walking advantage over the clipped-action PPO trajectory remains 0.664599
per control under that range.

Clipping the recorded actions reproduces the recorded joint targets without a
difference. This counterfactual changes the action-rate reward input; it does not
predict a new policy's motion. CPU float32 reductions can differ from the native
CUDA runtime. The receipt records the reward components and input hashes.

Retain reward v2 for the bounded-mean PPO comparison. These fixed trajectories
establish a reward ranking at one command. They do not establish that PPO can
discover walking or rule out poor exploration and delayed tracking feedback.
Evaluate the fresh trained policy before proposing a reward revision. Preserve
the original pilot and its checkpoints as separate evidence.

You can reproduce the audit from a checkout containing the pinned commit. Use
an external directory for the raw captures; the fetch requires SSH access to
`spark`. Use a fresh output path for a repeat audit.

```sh
uv run python -B site/assets/ppo_action_audit_20261002_001/reward_audit.py \
  --repository . --workspace "$HOME/hexapod-evidence/reward-audit-20261002" --fetch
```

## 10. Matched training after bounding the action mean

The bounded-mean PPO run completed 2,000 updates and 6,144,000 transitions with
128 robots, seed 20260917 and reward v2. Source commit `13f55d4e` supplies the
action change; the reward and physics files match the original PPO run. Native
execution and cleanup passed, and the checkpoint hashes match their sidecars.

The [training comparison](../site/assets/ppo_action_audit_20261002_001/training_comparison.json)
pools the last 200 updates from each run, or 614,400 controls across the robots.
It uses interval totals and counts; it does not average cumulative prefixes.

| Last 200 updates | Original PPO | Bounded mean |
| --- | ---: | ---: |
| Speed along translation commands | -0.000022 m/s | 0.000922 m/s |
| Requested translation speed | 0.038007 m/s | 0.037517 m/s |
| Mean reward per control | -0.0341 | -0.1732 |
| Joint-speed RMS during zero commands | 0.6442 rad/s | 0.7408 rad/s |

The bounded run reached 2.46% of requested translation speed in this window.
Its means stayed within [-1, 1], but raw Gaussian samples exceeded those bounds
on 22.53% of sampled joint controls. The last rollout reached 27.53%; eight joints
spent at least 90% of that rollout at a mean magnitude of 0.95 or more. The
original run has no matching rollout action statistics.

These measurements describe stochastic training with changing commands. The
zero-command sample includes transitions and resets. The last recorded rollout
precedes the final optimizer update. Use the final checkpoint evaluation for
motion and load comparisons; these measurements establish neither walking nor
the flat-pilot decision. The trajectory ranking in section 9 does not prove that
the learner can discover walking under this reward.

You can reproduce the comparison with its
[analysis source](../site/assets/ppo_action_audit_20261002_001/training_comparison.py):

```sh
python3 -B site/assets/ppo_action_audit_20261002_001/training_comparison.py \
  --workspace "$HOME/hexapod-evidence/training-comparison-20261002" --fetch
```

## 11. Final bounded-mean evaluation

We evaluated checkpoint `8e0ff06ece3ed348129afba633dab1ab35dbd4002a1c68189c6b8538334451c3`
from the completed run on its frozen source. The
[evaluation comparison](../site/assets/ppo_action_audit_20261002_001/evaluation_comparison.json)
covers all 13 probes. The
[training receipt](../site/assets/ppo_action_audit_20261002_001/training_completion_verification.json)
and [evaluation receipt](../site/assets/ppo_action_audit_20261002_001/evaluation_completion_verification.json)
record the source and input checks and owned-container cleanup. We verified
163 native evaluation files, including captures and force/torque records.
Both launchers exited with code zero and reported no native errors.

Eight translation probes and both turns fail tracking. The 20-second and
32-second quiet tests pass their numerical screens. The forward-to-stop screen
passes, but the policy makes no useful forward progress before the stop command.
That result does not establish stopping from walking. Three translation probes,
at 135, 180 and 225 degrees, also fail the computed-demand-over-rating fraction.

| Forward probe | Original PPO | Bounded mean |
| --- | ---: | ---: |
| Mean forward speed; command 0.05 m/s | -0.0000383 m/s | 0.00000612 m/s |
| Planar tracking error; limit 0.025 m/s | 0.050238 m/s | 0.049995 m/s |
| Body tilt RMS | 5.1779 degrees | 0.7270 degrees |
| Deterministic joint actions outside [-1, 1] | 61.09% | 0% |
| Mean per-joint target span, 2 < t <= 20 s | 0.030139 rad | 0.001378 rad |

The bounded policy holds a level stance. Eight of its 18 joint actions stay
at magnitude 0.95 or more throughout the scored forward window. You can inspect
the [forward video](../site/assets/media/ppo_bounded_mean_2000_20261002.mp4).
These deterministic action fractions differ from the sampled training actions
in section 10; do not compare the two as one measure.

The bounded policy's normalized tracking score `E` is 1.000132. The original
evaluation lacks ten probes, so its primary `E` remains unavailable. Human
walking labels remain pending, and we compute no primary `W`. Failed movement
does not support a load comparison at matched walking speed against the tripod.
The raw load records remain available. No AMP comparison or Stage 2 acceptance
follows from this diagnostic.

Bounding the mean removes out-of-range deterministic actions and improves the
quiet screens, but it does not produce walking at the frozen budget. The reward
audit still ranks recorded walking above the failed trajectory. These results
support investigation of exploration and reward credit assignment; they do not
isolate a reward-weight defect or justify a specific reward revision. We retain
reward v2. Further learner or reward changes need a separate declared comparison.

You can reproduce the analysis with its
[source](../site/assets/ppo_action_audit_20261002_001/evaluation_comparison.py).
The stance metadata comes from the reward audit fetch in section 9. The analysis
requires complete captures and verifies their recorded hashes; it starts no run.

```sh
uv run python -B site/assets/ppo_action_audit_20261002_001/evaluation_comparison.py \
  --workspace "$HOME/hexapod-evidence/evaluation-comparison-20261002" \
  --stance "$HOME/hexapod-evidence/reward-audit-20261002/reward_audit_inputs/stance.json" \
  --fetch
```

## 12. Separate normalization drift from reward timing

We tested `--observation-normalization none` first and retained reward v2.
The bounded-mean policy failed to walk, but neither the training totals nor
the fixed-trajectory reward ranking identifies the cause. We keep immediate
tracking as a separate future reward version if further evidence warrants it.
The model, motor limits and evaluation gates remain unchanged.

### Native normalization result

We completed 2000 updates with 128 robots, seed 20260917 and tanh action means
from source `bc16207b213efafd2ce203225d37f2f076e99f46`. The
[training receipt](../site/assets/ppo_learning_recovery_20261002_001/no_norm_training_verification.json)
verifies 6144000 transitions and all 40 checkpoint pairs. The
[evaluation receipt](../site/assets/ppo_learning_recovery_20261002_001/no_norm_evaluation_verification.json)
verifies all 13 captures, finite force/torque records, the declared forward
video and exact-container cleanup. Both launchers exited with code 0.

| Measure at update 2000 | Running normalization | No normalization |
| --- | ---: | ---: |
| Movement probes passed | 0/10 | 0/10 |
| Quiet probes passed | 2/2 | 0/2 |
| Stop screen passed | 1/1 | 0/1 |
| Normalized tracking error E | 1.000132 | 1.000658 |
| Forward speed for a 0.05 m/s command | 0.00000612 m/s | 0.00002469 m/s |
| Forward deterministic actions outside [-1, 1] | 0% | 0% |
| Mean per-joint forward target span, 2 < t <= 20 s | 0.001378 rad | 0.017164 rad |

The no-normalization policy fails all 13 probes. Its 20-second quiet probe
has maximum joint-speed RMS 0.654534 rad/s against the 0.03 rad/s limit.
The 225-degree translation probe exceeds the native speed bound at six physics
steps; the 270-degree probe exceeds it at one step. Their captures remain complete;
capture integrity does not grant physical acceptance. Eight forward joint
actions remain at magnitude 0.95 or more throughout the scored action window.
You can inspect the [forward video](../site/assets/media/ppo_no_norm_2000_20261003.mp4).

Pre-update mean Gaussian divergence stays zero across the 2000 collected
rollouts. Removing normalization eliminates this measured distribution drift.
The policy still fails walking and regresses quiet standing and stopping.
One seed does not isolate all causes of learning failure. These results support
the separate immediate-tracking reward comparison in
[PR #58](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/pull/58).
That comparison retains the penalty weights and physics; reward v2 remains
unchanged. Human gait labels remain pending, and ten original unbounded-policy
probes remain missing. Failed motion cannot establish a load comparison at
matched walking speed against the tripod.

The [comparison](../site/assets/ppo_learning_recovery_20261002_001/normalization_evaluation.json)
contains each probe's metrics and force/torque summaries. Its
[source](../site/assets/ppo_learning_recovery_20261002_001/normalization_evaluation.py)
reuses the hash-pinned action audit and checks the completed capture files.
Use the admitted stance file fetched in section 9 and a fresh external workspace:

```sh
uv run python -B site/assets/ppo_learning_recovery_20261002_001/normalization_evaluation.py \
  --workspace "$HOME/hexapod-evidence/normalization-evaluation-20261003" \
  --stance "$HOME/hexapod-evidence/reward-audit-20261002/reward_audit_inputs/stance.json" \
  --fetch
```

### CPU normalization and timing audits

We replayed stored actor observations from all 13 completed probes on the CPU
in the [normalization audit](../site/assets/ppo_learning_recovery_20261002_001/normalization_audit.json).
For each window, we initialized an actor with seed 20260917 and held its weights
fixed. We assigned probe `i % 13` to replica `i` across 128 replicas, then updated
normalization statistics from the next observation after each control. We
compared the stored Gaussian distributions with distributions from the same
inputs after the 24-control window. We took no optimizer steps.

| Prior normalization updates | Mean KL, empirical normalization | Mean KL, none |
| --- | ---: | ---: |
| 0 | 1.050361 | 0 |
| 100 | 0.003022 | 0 |
| 500 | 0.00002642 | 0 |

We measured less drift in the later windows. This replay establishes that
normalization can change the action distribution while actor weights stay
fixed. It uses fresh actor weights and retained observations; actions do not
drive a simulator. It does not measure native training KL or establish why
later native updates stayed small. A low learning rate remains a symptom.
You can inspect the [replay source](../site/assets/ppo_learning_recovery_20261002_001/normalization_audit.py)
and its source/input hashes before reproducing the comparison.

We also compared immediate kernel B with current kernel C using the four
captures from section 9. We retained the exact penalty means and verified
81 source/input files in the
[timing receipt](../site/assets/ppo_learning_recovery_20261002_001/reward_timing_audit.json).
We computed
both tracking terms from completed-control native velocity measurements,
reconstructed C from control zero, then scored the 900 endpoints from 2.02
through 20 seconds. We allowed the full unknown collision penalty of [-0.05, 0].

| Recorded trajectory with actions clipped to [-1, 1] | Mean reward interval, B | Mean reward interval, C |
| --- | ---: | ---: |
| Failed original PPO | [0.064183, 0.114183] | [0.222557, 0.272557] |
| Current tripod | [0.671832, 0.721832] | [0.937156, 0.987156] |
| Admitted walk, phase 0 | [1.070787, 1.120787] | [1.197595, 1.247595] |
| Admitted walk, phase 0.5 | [1.076220, 1.126220] | [1.201683, 1.251683] |

We verified that clipped actions reproduce the recorded targets. These
comparisons use recorded motion; clipping changes the action-rate reward input. Under both
kernels, retained walking earns more than failure despite collision uncertainty.
B costs more reward for speed and yaw oscillation within a stride.

At a 0.05 m/s command, we filled C's 60-control window with zero velocity and
replaced one endpoint with 0.005 m/s forward motion. B increased linear reward
by 0.023314; C increased it by 0.000343, a factor of 68.03. This calculation
changes tracking velocity alone and predicts no physical response to an action.
PPO collects 24 controls per rollout, while C uses 60 controls of velocity
history. The actor receives five proprioceptive frames and the current command.
It also receives the prior held-target offset divided by 0.35 rad, after clipping
and slew limiting. The critic adds
current linear velocity. Neither input contains the 60-control reward state.
This timing difference warrants investigation; it does not prove failed learning.

You can reproduce the timing comparison with its
[source](../site/assets/ppo_learning_recovery_20261002_001/reward_timing_audit.py).
Set `--base` to the retained bounded-action audit workspace, including its
original reward receipt, raw captures and frozen source directories. Use a fresh
external output path; the script rejects an existing output and starts no run.

```sh
uv run python -B site/assets/ppo_learning_recovery_20261002_001/reward_timing_audit.py \
  --base "$HOME/hexapod-evidence/ppo_bounded_actions_20261002_001" \
  --output "$HOME/hexapod-evidence/reward-timing-repeat-20261002.json"
```

## 13. Experimental immediate tracking reward: version 3

You select `--reward-version 3` to test immediate translation and yaw feedback
in a fresh PPO attempt. The candidate uses kernel B from section 12 and retains
version 2's weights and command scale. It uses each completed control's velocity
instead of the 60-control velocity mean. The model, observations, command
sampling and physical acceptance gates remain unchanged. Version 2 retains
its original source and defaults.

This candidate addresses the hidden reward history and delayed feedback in
section 12. The recorded-trajectory ranking supports a controlled experiment;
it does not establish that PPO can learn walking with this reward. Keep the
current no-normalization reward-v2 attempt frozen. Complete its 2000 updates
and final 13-probe evaluation, including force and torque records, before
deciding whether to dispatch version 3.

For that comparison, retain 128 robots, 2000 updates and seed 20260917, with
tanh action means and no running observation normalization. Change the reward
version alone. Evaluate the final checkpoint from its own frozen source and
compare the full probe set. Numerical gate results and human gait acceptance
remain separate requirements.

The candidate calls the existing penalty implementation through
`TrainingTaskV3`. Its declaration identifies an experiment and retains the
version 2 review as provenance. It does not claim that the version 2 reviewer
approved immediate tracking. The termination coefficient remains 75 for this
comparison; the historical worst-return bound describes version 2 and needs
reassessment under version 3.

CPU checks cover endpoint feedback after different velocity histories, exact
penalty and observation parity, and checkpoint reward identity. The existing
unknown-version tests now use `unknown` because `3` names a supported candidate.
No assertion tolerance or physical threshold changed. Native version 3 results
remain pending.

## Reproduce the Table I audit

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
