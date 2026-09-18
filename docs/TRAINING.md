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
| Learning | Stock RSL-RL 5.0.1 PPO. [`task.py`](../locomotion/task.py) owns command sampling and the measured reward; [`ppo.py`](../locomotion/ppo.py) owns its adapter and settings. |
| Admission | Recomputed native standing passes at one robot and the exact intended batch size, bound to the same source, model, neutral stance and geometry. |

The current sampler trains translation commands at 0.025 and 0.05 m/s, yaw and
combined commands, plus quiet intervals. The 0.05 m/s forward benchmark came
from the experiment configuration; James did not prescribe that speed. Keep
benchmark choice separate from navigation requirements.

The learner predicts joint-position offsets. It receives no gait-phase state
or optimized reference target. The reference-plus-residual design remains an
unimplemented proposal. Cleanup preserves the existing learning problem and
adds no curriculum, reward redesign or motion-prior algorithm.

## Results and research order

Read [STATUS](../STATUS.md) for measured results and original evidence. The
optimized forward target sequence passes its native screen. The tested PPO
policies remain unqualified; the paired action-initialization experiment found
no benefit in its one seed. That result does not identify the failure cause or
evaluate AMP. More replicas or updates have not established a walking solution.

## Proposed reproduction sequence

[Liu et al.](https://arxiv.org/abs/2511.03167) use PPO with an adversarial motion
prior (AMP) and an asymmetric critic. You can follow the nine steps below to
implement the simulation method. The current kernel has no AMP discriminator,
velocity estimator or memory encoder. Its flattened 231-value actor input does
not implement the paper's network. Keep paper reproduction and named adaptations
separate; this robot has mass 7.47 kg against the paper's 25.5 kg.

Declare the actuator adaptation before Step 1. The authors use the CSP law
`tau = Kp2 * (Kp1 * (q_des - q) - q_dot)`
([§III, p. 2; Fig. 2, p. 3](../hexapod_locomotion.pdf#page=2)).
Retain this project's approved motor model and target-change limiter; document
their differences from that controller with the new task configuration.

Preserve version 1 and its gates. Give changed tasks, models and results fresh
identities. James approves native steps. Source preparation starts no research
allocation and supplies no native admission.

| Step | Work and prerequisite | Spark |
| --- | --- | --- |
| 1. Reward v2 | Audit [Table I, p. 3](../hexapod_locomotion.pdf#page=3), for formulas, units, signs and weights before implementing a versioned paper reward with CPU tests. Resolve its printed positive tracking exponent against the intended decreasing reward. Define command-scaled tracking as a separate adaptation; resolve the stationary-reward criterion below. | No |
| 2. Throughput profile | Measure physics, contact, observation and learner cost plus memory across replica counts. The current guards cap replicas at 128 and updates at 2,000; the paper uses 4,096 robots. Extend scale through a named configuration and isolation tests. Change the 153 SDF colliders only if measurements justify an asset variant, then repeat one-robot and batch admission. | Yes |
| 3. Omni motion dataset | Extend [`optimize.py`](../locomotion/priors/optimize.py) from a forward cycle to eight bearings, both yaw directions and the arcs in `command_bank`. Require alternating tripod demonstrations on the approved URDF, with a consistent gait cycle across directions ([§III-A–B, p. 3](../hexapod_locomotion.pdf#page=3)). | No |
| 4. Native motion validation | Replay each motion with [`replay_native.py`](../locomotion/priors/replay_native.py) and require its matching screen before dataset admission. Construct AMP states from native replay. Match feature order, frames, units and the 20 ms interval; exclude terminal-to-reset pairs. | Yes |
| 5. Network architecture | Implement Table III's velocity estimator, memory encoder over five proprioception frames, low-level actor, privileged encoder and critic. Train the estimator with supervised simulation velocity labels ([§IV-A–B; Table III, p. 4](../hexapod_locomotion.pdf#page=4)). Supply the full 42-value privileged state, including base height and perturbations, plus collision states ([§III, p. 2](../hexapod_locomotion.pdf#page=2)). Test shapes and gradient paths; keep privileged inputs out of the actor. Add the terrain encoder with Step 8. Current actor/critic networks are plain [256, 256, 128] MLPs. | No |
| 6. AMP integration | Resolve the gradient-penalty ambiguity below, then implement the least-squares discriminator and style reward from Eqs. (1)–(2). Update PPO and the discriminator together ([§IV-B, p. 4](../hexapod_locomotion.pdf#page=4)). Check integration on flat ground; defer the final reward comparison to Step 9. Use the 13 learning probes as diagnostics and the full gate for qualification. | Yes |
| 7. Domain randomization | Implement Table II with frozen model-specific values. Declare adaptations for the different robot and simulator before comparing runs. | Yes |
| 8. Terrain curriculum | Add terrain levels, the critic height scan and terrain encoder. Retain admitted flat behavior. | Yes |
| 9. Reward and robustness comparisons | After Steps 7–8, compare task plus penalty, task plus style, and task plus style plus penalty with matched transitions and five seeds. Assess flat tracking and terrain progression ([§V-A; Figs. 3–4, pp. 4–5](../hexapod_locomotion.pdf#page=4)). Test push disturbances and reproduce the baseline controllers as separate comparisons ([§V-B, p. 5](../hexapod_locomotion.pdf#page=5)). Keep RMA/MPC work separate from the reward ablation. | Yes |

Freeze the chosen PPO settings and terrain curriculum thresholds in the task
configuration before training. The authors omit these details from
[§IV-B–V, p. 4](../hexapod_locomotion.pdf#page=4); label them as implementation
decisions. The authors train with randomization and a terrain curriculum
([Table II, p. 3; §IV-B, p. 4](../hexapod_locomotion.pdf#page=3)).

### Step 1: tracking-reward criterion

A motionless robot under version 1 earns about 62% of maximum combined tracking
reward at a straight 0.025 m/s command and 28% at 0.05 m/s, before penalties.
Both figures include full reward for matching the zero yaw command. Narrowing
the linear kernel alone leaves a `0.3 / 1.3 = 23.1%` combined reward floor.
Define the proposed under-10% condition for the commanded translation component,
or declare a further reward adaptation. Test zero commands and transitions.
The paper's 0.15 kernel and command range require a separate comparison.

### Paper ambiguities

The authors print a parameter gradient, `grad_phi D_phi(T_s)`, in
[Eq. (1), p. 4](../hexapod_locomotion.pdf#page=4). The proposed input-gradient
penalty differentiates with respect to the transition instead. Resolve that
difference against the cited AMP method or author code before implementation;
record the chosen formula and its source.

The authors call the AMP state 61 values in
[§III-A, p. 3](../hexapod_locomotion.pdf#page=3), but their foot-height description
accounts for fewer values. The retained kernel uses six 3D foot positions.
Declare that interpretation before discriminator training and verify the same features on
native demonstrations and policy rollouts. The optimizer's point contacts omit
mesh patches and impacts, so native replay remains a prerequisite.

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
under `/home/orionh/HEXAPOD_runs/restart_20260914/`.
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
the approved robot. The current status lives in [STATUS](../STATUS.md).
