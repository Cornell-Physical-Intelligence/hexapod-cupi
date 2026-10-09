# Locomotion training and evaluation

You run the approved robot through [`locomotion/`](../locomotion/README.md).
Read that file for the control loop and entry points. The physical robot is
unbuilt; simulation results do not establish hardware calibration.

## Current contract

| Item | Contract |
| --- | --- |
| Model | [`robot/active_model.json`](../robot/active_model.json), 19 bodies and 18 direct-drive joints; preserve its exact URDF and mass corrections. |
| Control | 400 Hz physics, 50 Hz policy, eight substeps, 0.35 rad action scale and 0.040 rad target-change bound per control. |
| Motor | Named damping and the existing speed-dependent effort curve, with a provisional 1.6 N·m software cap. Hardware characterization remains open. |
| Policy input | 231 actor values: five 42-value proprioception frames, the three-value velocity command and 18 executed-action values. The critic adds measured planar/vertical velocity for 234 values. |
| Learning | Stock RSL-RL 5.0.1 PPO. [`task.py`](../locomotion/task.py) owns command sampling and reward version 1, the default; [`task_v2.py`](../locomotion/task_v2.py) owns reward version 2, selected with `train.py --reward-version 2` ([audit](REWARD_V2_TABLE1_AUDIT.md)); [`task_v4.py`](../locomotion/task_v4.py) owns reward version 4, the first version with a native walking and stopping result ([audit section 15](REWARD_V2_TABLE1_AUDIT.md#15-vanilla-ppo-stability-and-reward-version-4)); [`ppo.py`](../locomotion/ppo.py) owns its adapter and settings. |
| Admission | Recomputed native standing passes at one robot and the exact intended batch size, bound to the same source, model, neutral stance and geometry. |

The current sampler trains translation commands at 0.025 and 0.05 m/s, yaw and
combined commands, plus quiet intervals. The 0.05 m/s forward benchmark came
from the experiment configuration; the program lead did not prescribe that speed. Keep
benchmark choice separate from navigation requirements.

The learner predicts joint-position offsets. With the baseline options it
receives no gait-phase state or optimized reference target. The
reference-plus-residual design remains an unimplemented proposal.

Reward version 4 is an opt-in path with its own learner options: a gait clock
input, a scheduled action deviation, fixed observation scales, a two-control
action mean and noise on the actor's joint-velocity inputs. By default it draws
forward commands and zero commands alone, and `--reward-options
forward_draw_fraction=0` restores the full command bank. It trains 10 s
episodes and uses no demonstration, no discriminator and no optimized
reference target.

## Results and research order

Read [STATUS](https://cornell-physical-intelligence.github.io/hexapod-cupi/#findings) for measured results and original evidence. The
optimized forward target sequence passes its native screen. The paired
action-initialization experiment found no benefit in its one seed, and that
result does not evaluate AMP.

Vanilla PPO with reward version 4 passes the forward probe and the
forward-to-stop probe in native simulation on seeds 20260917 and 20260918 at
update 2000. Seed 20260917 also passes both quiet probes; seed 20260918 fails
them on one toe-contact check. The nine probes in other directions and in yaw
fail, because training draws forward commands alone. These policies remain
unqualified for Stage 2. [Audit section 15](REWARD_V2_TABLE1_AUDIT.md#15-vanilla-ppo-stability-and-reward-version-4)
gives the causes of the earlier failures, the controlled evidence and each
departure from the paper.

## Parallel-training reference

[Rudin et al. 2022](https://proceedings.mlr.press/v164/rudin22a.html) is the
recipe behind RSL-RL: on-policy PPO with time-out bootstrapping, 24 steps per
robot and update, and thousands of robots on one GPU. The table states this
repository's parity with it. Read it before you change a learner setting.

| Setting | Rudin et al. | This repository |
| --- | --- | --- |
| Algorithm | PPO with GAE, time-out bootstrapping, an adaptive rate at KL 0.01, clip 0.2, 5 epochs and 4 mini-batches | The same values through RSL-RL 5.0.1. The rate cap and the command-boundary bootstrap are additions. |
| Batch per update | 4096 robots and 24 steps, 98,304 samples. 128 robots is the smallest count in the authors' sweep and gives their lowest final reward. | 128 robots and 24 controls, 3,072 samples. The standing admission fixes the replica count. |
| Observations | Base linear and angular velocity, gravity, joint positions and velocities, the previous action and 108 terrain heights | Five proprioception frames without base linear velocity, the command and the executed action. The critic adds measured velocity. |
| Actions | Joint position targets to a PD controller with an action-rate penalty and no rate clamp | Joint offsets with a 0.35 rad scale and the 0.040 rad per control limiter |
| Reward | Nine gait-free terms with an exponential tracking kernel and a feet air-time term. The policy converges to a trot. | Reward v4 adds a tripod schedule, swing travel and quiet terms. Version 1 is the gait-free baseline. |
| Curriculum and randomization | Terrain and command curricula, friction in [0.5, 1.25], pushes every 10 s and measured observation noise | Flat ground, fixed friction and no pushes. Joint-velocity noise on the actor is the one noise channel. |

The batch row is the largest gap. With the replica count fixed, the paper's
lever is more controls per robot and update. That changes the frozen
24-control budget and needs a program-lead decision.

## Proposed reproduction sequence

[Liu et al.](https://arxiv.org/abs/2511.03167) use PPO with an adversarial motion
prior (AMP) and an asymmetric critic. You can follow the nine steps below to
implement the simulation method. The kernel carries the AMP discriminator,
velocity estimator and memory encoder with CPU tests; no native pilot has used
them. The stock MLP actor's flattened 231-value input does not implement the
paper's network. Keep paper reproduction and named adaptations
separate; this robot has mass 7.47 kg against the paper's 25.5 kg.

Declare the actuator adaptation before Step 1. The authors use the CSP law
`tau = Kp2 * (Kp1 * (q_des - q) - q_dot)`
([§III, p. 2; Fig. 2, p. 3](https://arxiv.org/pdf/2511.03167v1#page=2)).
Retain this project's approved motor model and target-change limiter; document
their differences from that controller with the new task configuration.

Preserve version 1 and its gates. Give changed tasks, models and results fresh
identities. The program lead approves native steps. Source preparation starts no research
allocation and supplies no native admission.

| Step | Work and prerequisite | Depends on | Spark | Issue | Status (29 Sep 2026) |
| --- | --- | --- | --- | --- | --- |
| 1. Reward v2 | Audit [Table I, p. 3](https://arxiv.org/pdf/2511.03167v1#page=3), for formulas, units, signs and weights before implementing a versioned paper reward with CPU tests. Resolve its printed positive tracking exponent against the intended decreasing reward. Define command-scaled tracking as a separate adaptation; resolve the stationary-reward criterion below. | None | No | [#35](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/35) | Done: #42 merged reward version 2 in `task_v2.py`; version 1 stays the default. |
| 2. Throughput profile | Measure physics, contact, observation and learner cost plus memory across replica counts. The current guards cap replicas at 128 and updates at 2,000; the paper uses 4,096 robots. Extend scale through a named configuration and isolation tests. Change the 153 SDF colliders only if measurements justify an asset variant, then repeat one-robot and batch admission. | None | Yes | [#35](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/35) | Done to the 128-robot guard: #44 merged the profiler and the 1, 32 and 128 profiles; AMP costs pending. |
| 3. Omni motion dataset | Use the [optimizer and dataset pipeline](../locomotion/priors/README.md) for eight bearings, both yaw directions and the arcs in `command_bank`. Require alternating tripod demonstrations on the approved URDF, with a consistent cycle across directions ([§III-A–B, p. 3](https://arxiv.org/pdf/2511.03167v1#page=3)). Issue #36 uses optimized motions alone; native and human review determine admission. | None | No | [#36](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/36) | Done: #36 closed. |
| 4. Native motion validation | Replay each motion with [`replay_native.py`](../locomotion/priors/replay_native.py) and require its matching screen before dataset admission. Construct AMP states from native replay. Match feature order, frames, units and the 20 ms interval; exclude terminal-to-reset pairs. | 3 | Yes | [#36](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/36) | Done: 18000 pairs admitted; #36 closed. |
| 5. Flat AMP pilot | Resolve the gradient-penalty ambiguity below, then implement the least-squares discriminator and style reward from Eqs. (1)–(2) with the current [256, 256, 128] MLP actor and critic. Update PPO and the discriminator together ([§IV-B, p. 4](https://arxiv.org/pdf/2511.03167v1#page=4)). The [AMP learner](../locomotion/README.md#amp-learner-and-paper-networks) implements this with CPU tests; no native pilot has run. Run on flat ground and compare with the tripod baseline through the same scorer and force metrics; defer the final reward comparison to Step 9. Use the 13 learning probes as diagnostics and the full gate for qualification. | 1, 4; 2 for full scale | Yes | [#37](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/37), [#50](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/50), [#51](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/51) | CPU done in #43. The native pilot runs under #50 and its baseline and scoring under #51; the 29 September standing admission covers the physics source. |
| 6. Network architecture | Implement Table III's velocity estimator, memory encoder over five proprioception frames, low-level actor, privileged encoder and critic. Train the estimator with supervised simulation velocity labels ([§IV-A–B; Table III, p. 4](https://arxiv.org/pdf/2511.03167v1#page=4)). Supply the full 42-value privileged state, including base height and perturbations, plus collision states ([§III, p. 2](https://arxiv.org/pdf/2511.03167v1#page=2)). Test shapes and gradient paths; keep privileged inputs out of the actor. [`paper_networks.py`](../locomotion/paper_networks.py) implements these networks with CPU tests; friction and perturbation slots hold declared constants until Step 7. Add the terrain encoder with Step 8. Repeat the Step 5 pilot with these networks before Step 7. | None to build; 5 to compare | No | [#37](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/37), [#50](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/50) | CPU done in #43. The paper-network run follows the pilot under #50; #51 scores both. |
| 7. Domain randomization | Implement Table II with frozen model-specific values. Declare adaptations for the different robot and simulator before comparing runs. | 5, 6 | Yes | — | Not started; no issue. |
| 8. Terrain curriculum | Add terrain levels, the critic height scan and terrain encoder. Both stay critic-only: the GPS-only robot carries no terrain sensor. Retain admitted flat behavior. | 7 | Yes | — | Not started; no issue. |
| 9. Reward and robustness comparisons | After Steps 7–8, compare task plus penalty, task plus style, and task plus style plus penalty with matched transitions and five seeds. Assess flat tracking and terrain progression ([§V-A; Figs. 3–4, pp. 4–5](https://arxiv.org/pdf/2511.03167v1#page=4)). Test push disturbances and reproduce the baseline controllers as separate comparisons ([§V-B, p. 5](https://arxiv.org/pdf/2511.03167v1#page=5)). Keep RMA/MPC work separate from the reward ablation. | 7, 8 | Yes | — | Not started; no issue. |

Steps 1, 2, 3 and 6 need no earlier step, so you can assign them in parallel
through GitHub issues. Each linked issue records its step's progress in its checkboxes;
Steps 7–9 have no issue yet. The Step 5 pilot keeps the current MLPs: if it fails, the
Table III networks cannot be the cause.

Freeze the chosen PPO settings and terrain curriculum thresholds in the task
configuration before training. The authors omit these details from
[§IV-B–V, p. 4](https://arxiv.org/pdf/2511.03167v1#page=4); label them as implementation
decisions. The authors train with randomization and a terrain curriculum
([Table II, p. 3; §IV-B, p. 4](https://arxiv.org/pdf/2511.03167v1#page=3)).

### Frozen flat pilot: issues #50 and #51

You use protocol `flat_pilot_v1` for the first native comparison. The
[program lead approved the budget and decision rule](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/issues/37#issuecomment-5920656379).
You retain reward version 1 as the default outside these packs.

| Arm | Preparation options | Common budget |
| --- | --- | --- |
| PPO control | `--learner ppo --networks mlp` | 128 robots, 2000 updates, seed 20260917, reward version 2 |
| Step 5 AMP | `--learner amp --networks mlp` | Same budget |
| Step 6 AMP | `--learner amp --networks paper` | Same budget |

You retain 24 controls per update, or 6,144,000 transitions per arm. Freeze
`ppo.py` and the resolved AMP configuration with the source. The AMP arms use
the admitted `amp_demonstrations_001` bank, style weight 1, 20 discriminator
updates per PPO update and discriminator seed 20260925. Keep the approved
model, motor limits and 0.040 rad / 20 ms target limiter.

You compare checkpoint update 2000 for the decision. Preserve checkpoints at
50-update intervals and evaluate each through the 13 learning probes; those
earlier comparisons describe learning progress and cannot replace update 2000.
Score PPO and Step 5 before Step 6. Record native contact force and torque
before starting the next training arm. A timeout or learner fix requires a
fresh attempt; this runner supports no training resume.

#### Tracking and appearance

You use the eight 0.05 m/s translation probes and two 0.20 rad/s yaw probes.
Read `planar_error_mps` and `yaw_error_rad_s` from the unchanged evaluator,
which scores controls 100 through 999. Compute the dimensionless error:

```text
E = (sum(planar_error_mps / 0.05 over eight translations)
     + sum(yaw_error_rad_s / 0.20 over two turns)) / 10
```

You give each command equal weight. Report both error channels per command;
retain the two quiet probes and the stop probe as separate diagnostics. Do not
mix the reward scorer's reward ranking with this tracking metric.

The program lead reviews each motion video with its checkpoint and video
hashes. Record `walking` for sustained stepping with commanded progression,
`standing` for a held stance without steps, and `jittering` for oscillation
without sustained commanded progression. Record a fall or unsafe motion as
`failed`; leave an unreviewed or missing video `pending`. Count a walking case
only when its unchanged numerical motion/contact checks pass and the program
lead labels it `walking`. Let `W` be this count out of ten. Keep failed cases
in the denominator. Missing captures or pending reviews block the decision;
do not compute a primary score from a failed prefix.

#### Contact and motor comparison

You prepare a fresh tripod `stop_stride`, candidate 0, `screen` allocation
from the same source and admitted inputs. Freeze its binding before dispatch.
Its commands are forward 0.05 m/s and yaw +/-0.20 rad/s. Record reference
report, capture and video hashes before scoring a learned checkpoint. Keep
historical tripod results under their original identities.

You compare the full 400 Hz window `2 < t <= 20` seconds. Include swing zeros.
Require complete captures and passing motion screens for both controllers.
Match achieved behavior: the absolute difference in mean along-command speed
or mean signed yaw rate must be at most 10% of the command magnitude. This
condition defines comparable load rows; it changes no motion gate. Mark other
directions and unmatched rows `unavailable`, with a reason.

For each supported command, compare each foot's normal-force p95 and peak,
and each joint's applied absolute-torque RMS and peak. Require each AMP value
to be at most its corresponding tripod value, using unrounded values without
an added tolerance. Report requested torque and total support as diagnostics.
All supported rows must be comparable and within the reference loads to pass
this pilot comparison. These relative criteria add no Stage 2 safety limits.

#### Decision and run preparation

The action diagnostic selects `--action-mean tanh` to bound the Gaussian mean
before sampling. You retain reward version 2 and the frozen pilot budget for
this comparison. The default `--action-mean unbounded` preserves the original
pilot. Use fresh attempt names and record the selected distribution in the
checkpoint configuration. Match the option in training and evaluation; an
existing unbounded checkpoint cannot serve as a bounded-mean training result.
This diagnostic tests the learner change before any reward revision. It does
not replace an original pilot arm or change the physical acceptance gates.

The [retained-capture reward audit](REWARD_V2_TABLE1_AUDIT.md#9-failed-policy-comparison-after-the-first-flat-ppo-run)
ranks the current tripod and two admitted forward walks above the failed PPO
trajectory under reward v2. The [final bounded-mean evaluation](REWARD_V2_TABLE1_AUDIT.md#11-final-bounded-mean-evaluation)
fails all ten movement probes and passes the quiet screens. The action change
does not produce walking at the frozen budget. We retain v2 because these tests
do not isolate a reward defect or establish a specific replacement.

The next PPO diagnostic selects `--action-mean tanh
--observation-normalization none`. You retain 128 robots, 2000 updates, seed
20260917 and reward v2. This tests the effect of disabling running observation
statistics in both networks. You retain the stock optimizer and adaptive
learning-rate schedule, and record policy divergence before and after each
update. The [CPU replay and reward timing audit](REWARD_V2_TABLE1_AUDIT.md#12-separate-normalization-drift-from-reward-timing)
motivate this test; they supply no native learning result. Use a fresh attempt
and evaluate its final checkpoint on all 13 probes with force and torque
records before another training sequence. Preserve the full Stage 2 gate and
human gait acceptance. Any reward revision needs its own version and trial.

You compare each AMP arm with PPO at update 2000. Lower `E`, higher `W` and
passing load comparisons mean a pilot win. Higher `E` and lower `W` mean a
loss on both; fix the reward or learner before Step 7. A tie, mixed result,
load exceedance or unavailable evidence holds Step 7 for review. A pilot win
supplies no full Stage 2 qualification; the 96-case gate and human acceptance
remain.

You prepare training with `--allocation-profile flat_pilot_v1
--max-wall-seconds 21600`. The supervisor retains a 400-second cleanup margin,
for a 22000-second cap. These are finite operational limits, not measured AMP
completion times. The native runner saves the last complete update and load
summary on a deadline failure. Preserve the failure; do not lower the budget
or call a shorter checkpoint a completed arm. Standard allocations keep their
existing bounds. The same profile supports probe evaluation; it extends no
replica, update or physical gate.

You use W&B project `hexapod-amp` in offline mode for these packs. Freeze the
input declaration from the 29 September 128-robot admission, including the
mounted one-robot and batch reports. Verify its physics and asset hashes at
preflight. Copy this protocol into the run archive with its SHA-256, retain
`PACK.json` and `binding.json`, and transfer each fresh pack without macOS
metadata. Use [OPERATIONS](OPERATIONS.md) for host preflight and dispatch.
Preparation and host preflight start no native training.

### Run tracking

Training logs to W&B through RSL-RL when you prepare it with W&B options. The
Spark image includes wandb 0.28.2, and the `tracking` group pins that version on
your machine. W&B receives the PPO losses, the task reward components and each
checkpoint; Git keeps the acceptance summary.

```sh
uv run python -m locomotion.prepare --mode train ... --logger wandb --wandb-project <project>
uv run --group tracking wandb sync <allocation>/run/standing/wandb/offline-run-*
```

Runs default to `--wandb-mode offline` and write their W&B files inside the run
output; the second command uploads them after you retrieve the run. With
`--wandb-mode online`, export `WANDB_API_KEY` in the launcher's shell on Spark.
The launcher passes the key to the container by name and records no value.

### Step 1: tracking-reward criterion

A motionless robot under version 1 earns about 62% of maximum combined tracking
reward at a straight 0.025 m/s command and 28% at 0.05 m/s, before penalties.
Both figures include full reward for matching the zero yaw command. Narrowing
the linear kernel alone leaves a `0.3 / 1.3 = 23.1%` combined reward floor.
Define the proposed under-10% condition for the commanded translation component,
or declare a further reward adaptation. Test zero commands and transitions.
The paper's 0.15 kernel and command range require a separate comparison.

### Step 2: throughput profile

[`throughput.py`](../locomotion/throughput.py) profiles training at one
admitted replica count per allocation. It runs `train.py`'s allocation: task
version 1, stock PPO, 20-second episodes and the same per-update records. It
accepts one, 32 or 128 robots with matching one-robot and batch standing
admission, and it keeps the 128-replica and 2,000-update guards. The program
lead approves each native profile.

```sh
uv run python -m locomotion.prepare --mode throughput --num-envs 32 \
  --warmup-updates 2 --updates 5 --inputs <admitted inputs.json> \
  --output <fresh local pack> --remote-root <fresh remote root>
uv run python -m locomotion.throughput summarize --run <allocation>/run/standing \
  --run <other allocation>/run/standing --output <fresh summary.json>
```

The profiler discards the warmup updates, which absorb startup allocation and
the stock runner's first model save. Measured updates then alternate between
two passes. A throughput update has no hooks and waits for the GPU only at its
boundaries; it gives transitions per second. A component update waits for the
GPU at each component boundary. Each label owns its exclusive time, so the
labels sum to the update's wall time. Parent labels such as `env_control` also
hold most barrier time. The report counts each label's barriers and subtracts
an estimate from the measured idle barrier cost; compare shares after that
correction.

`physics_step` covers PhysX broadphase, SDF contact generation and the solver
when `platform.cuda_contexts.shared` is true. Otherwise the barriers may not
wait for PhysX kernels, and you compare the sum of `physics_step`,
`state_readback` and `contact_readback`. A contact census after the timed
updates counts reported floor patches against buffer capacity. The report
records PyTorch allocator peaks, Warp and PhysX GPU heaps where their
interfaces exist, process memory and host-wide memory. On GB10 unified memory,
process memory omits CUDA allocations, and host-wide memory includes other
processes.

`profile.json` binds the frozen source, model, stance, geometry, admission,
PPO configuration and RSL-RL source hashes. `summarize` needs each run's
launcher records beside it in `run/`. It rejects changed bytes, failed jobs or
contact audits, binding arguments that differ from the profile, repeated
replica counts, and mixed identities, platforms or settings. It extrapolates to
no unmeasured count. The report labels the learner `stock_ppo`; AMP learner
costs remain pending until a contributor integrates the AMP learner.

The [29 September profiles](../site/assets/throughput_profile_20260929_001/summary.json)
ran frozen source `56ebf907` (#44 with the #46 launcher) on the GB10, with 2
warmup and 20 measured updates per pass. An idle `qwen38-server` shared the
GPU. PhysX shared the synchronized CUDA context at each count.

| Robots | Transitions/s | Update | PhysX scene step | PhysX share | PPO share |
| --- | --- | --- | --- | --- | --- |
| 1 | 12.9 | 1.77 s | 6.4 ms | 70% | 7% |
| 32 | 319 | 2.17 s | 8.0 ms | 70% | 6% |
| 128 | 930 | 3.21 s | 13.4 ms | 80% | 3% |

From 32 to 128 robots, each added robot adds about 0.056 ms to the scene step
and about 1 MB of GPU memory. The measurements support the current 128-robot
guard. Extend replicas only through a named configuration with isolation
tests. The 80 m floor mesh holds about 400 robots at 2 m spacing, and the
128-robot standing capture took 21 minutes and 11 GB, so larger counts need a
wider floor and a scalable admission capture. Change the 153 SDF colliders
only when measurements justify an asset variant, then repeat one-robot and
batch admission.

### Paper ambiguities

The authors print a parameter gradient, `grad_phi D_phi(T_s)`, in
[Eq. (1), p. 4](https://arxiv.org/pdf/2511.03167v1#page=4). The proposed input-gradient
penalty differentiates with respect to the transition instead. Resolve that
difference against the cited AMP method or author code before implementation;
record the chosen formula and its source.

The authors call the AMP state 61 values in
[§III-A, p. 3](https://arxiv.org/pdf/2511.03167v1#page=3), but their foot-height description
accounts for fewer values. The retained kernel uses six 3D foot positions.
The program lead retained that 61-value interpretation for issue #36.
[`amp.py`](../locomotion/amp.py) specifies the order, frames and units; the dataset
audit verifies the shared extractor against native telemetry before admission. The optimizer's point contacts omit
mesh patches and impacts, so native replay remains a prerequisite.

## Tripod baseline

The prescribed [tripod controller](../locomotion/README.md#prescribed-tripod-controller)
reproduces Zhang et al. on the approved robot. It covers forward commands up
to 0.10 m/s and yaw commands at ±0.20 rad/s; it does not cover the full
omnidirectional bank. Each trial records the 61-value AMP state per control.
Issue #36 uses the optimizer for its demonstrations. The tripod screens and
force metrics give Step 5 a comparison through the same scorer.

## Foundation commands

```sh
uv sync --locked
uv run python -m locomotion.inputs check
uv run python -m unittest discover -s locomotion/tests
uv run python -m unittest discover -s locomotion/priors/tests
uv run python -m locomotion.inputs pack --help
uv run python -m locomotion.prepare --help
```

`inputs pack` requires a fresh local output and an explicit remote input root
under `/srv/cupi/hexapod/inputs/`. Historical restart paths remain supported.
It copies the approved USD, geometry and neutral stance, verifies their hashes,
and writes `inputs.json`. Transfer the directory to that root, then use the
file with `prepare --inputs`. Preparation launches no compute. The new frozen
source still requires matching one-robot and intended-batch admission before
training. Historical admissions and checkpoints retain their original sources.

## Acceptance and measurements

[`evaluation.py`](../locomotion/evaluation.py) owns the unchanged numerical
gates. [`evaluate.py`](../locomotion/evaluate.py) records actual policy actions
and native responses. The 13 learning probes are additional diagnostics. The
full Stage 2 manifest has 96 cases, including signed directions, turns,
transitions and quiet stops. Missing cases remain missing; an interrupted
trial retains its failed prefix. Human visual acceptance remains required.

Each evaluation records contact-normal force and requested/applied motor torque
at 400 Hz in `force_metrics.json`. Report total support and per-foot force,
plus motor means, RMS, percentiles and peaks. Swing zeros remain in averages.
These are descriptive loads, with no new numerical acceptance limits. Contact
measurements exclude unrecorded tangential friction. Compare loads at matched
achieved behavior before making an efficiency claim.

Bind each checkpoint, video and report to its exact source and settings. Keep
raw captures and failed attempts. The [experiment record](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/forward_example_ppo_20260917/README.md)
explains the saved files and complete Spark archives. Preserve historical
checkpoint identities; a source move cannot admit a new runtime.

## Historical contracts

Use `locomotion/env_config.py` for the approved joint order. Hardware mapping
requires its own verification. Historical joint layouts remain in the pinned
pre-cleanup reference below.

C-study, mock, four-bar and custom PPO/AMP runs keep their original model,
observation layout, source and gates. Use their frozen source packs or the
[pinned pre-cleanup training reference](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/docs/TRAINING.md).
The [archive index](../configs/archive.json) identifies retired source and context. Historical decisions do not authorize new work or override
the approved robot. The current status lives in [STATUS](https://cornell-physical-intelligence.github.io/hexapod-cupi/#findings).
