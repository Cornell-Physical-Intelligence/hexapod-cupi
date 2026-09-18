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

## Results that affect the next research decision

The [optimized reference](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/ppo_reference_comparison_20260917/review_smooth_001/RESULT.json)
passes the forward screen through the native controller. It is an executed
target sequence, with no learned feedback policy or omni qualification.

The [standard PPO baseline](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/ppo_reference_comparison_20260917/README.md)
completed 1,200 updates but failed forward, quiet and stop probes. The subsequent
[paired example-initialization test](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/forward_example_ppo_20260917/COMPARISON_001.json)
gave both arms 1,200 updates and 3,686,400 transitions with seed 20260914.
Both arms failed their forward screens at updates 0, 300, 600 and 1,200.
Final mean speeds were 0.004811 m/s from scratch and 0.000961 m/s with example
initialization, against 0.05 m/s. The copied actor failed before PPO began.
This pair establishes no benefit from that initialization. It does not resolve
other motion-prior methods or the cause of the locomotion failure.

The [training diagnostics](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/forward_example_ppo_20260917/TRAINING_DIAGNOSTICS_001.json)
record weak forward motion despite rising reward and frequent target limiting.
Use those measurements to choose the next bounded investigation. More replicas
or updates have not established a solution.

## Proposed reproduction sequence

[Liu et al.](https://arxiv.org/abs/2511.03167) train a velocity-commanded hexapod
policy with PPO, an adversarial motion prior (AMP) and an asymmetric critic. The
kernel's 231-value actor input and 61-value AMP state follow that paper. This
sequence orders the remaining simulation work. It is a proposal. James approves
each native step. No step changes an existing gate, an admission or the
version 1 task; each new task, model and result takes a fresh name. Record each
departure from the paper as a named adaptation, separate from the reproduction.

| Step | Work | Spark |
| --- | --- | --- |
| 1. Reward v2 | Add a new task schema that copies the paper's Table I without changes. Cover it with CPU tests. Command-scaled tracking is an adaptation with its own schema name: the paper's 0.15 kernel assumes its command range, and this kernel commands 0.025 and 0.05 m/s. The adaptation sets kernel width against command magnitude so a motionless robot earns under 10% of the maximum tracking reward. Under version 1 that robot earns 62% at 0.025 m/s and 28% at 0.05 m/s. | No |
| 2. Throughput profile | The paper trains 4096 robots against 128 here, and the recorded runs have not tested that scale. Measure step time and GPU memory per robot count, split between physics, contact reporting, observation and learner. Simplify the 153 SDF colliders per robot only if the profile names them as the limit. Changed physics source needs a new one-robot and batch admission. | Yes |
| 3. Omni motion dataset | Extend [`optimize.py`](../locomotion/priors/optimize.py) from one forward cycle to the eight bearings and both yaw directions in `command_bank`, then its arcs. | No |
| 4. Native motion validation | Replay each optimized motion through native physics with [`replay_native.py`](../locomotion/priors/replay_native.py). Admit a motion to the AMP dataset after it passes its matching screen. Build the 61-value AMP states from the native replay, because the optimizer's point contacts omit mesh patches and impacts. | Yes |
| 5. Network architecture | Implement the paper's Table III: the velocity estimator and memory encoder over five proprioception frames, the low-level actor, the privileged encoder and the critic. Give the critic contact forces and friction. Cover shapes and gradient paths with CPU tests. The current actor and critic are plain [256, 256, 128] networks, so an ablation on them does not test the paper's method. | No |
| 6. AMP style term | Add the least-squares discriminator with gradient penalty and its style reward. Run the flat two-arm ablation, task plus penalty against task plus style plus penalty, on the 13 learning probes. | Yes |
| 7. Domain randomization | Randomize the parameters in the paper's Table II. | Yes |
| 8. Terrain curriculum | Add the terrain levels, the critic's height scan and its terrain encoder. Retain admitted flat behavior. | Yes |
| 9. Robustness comparison | Run push-disturbance tests and the paper's baseline controllers. | Yes |

## Foundation review of the proposed order

You can start with the reward audit after the CPU checks below. Keep the proposed
sequence as the research order, with these prerequisites:

- **Step 1:** Distinguish a Table I reproduction from a command-scaled reward.
  The current 62% and 28% figures include the full yaw reward for a straight
  command. Changing the linear kernel alone leaves a 0.3 / 1.3 = 23.1% floor.
  Define the under-10% condition for the commanded translation component, or
  declare an additional reward change. Cover zero commands and transitions.
  Table I prints a positive tracking exponent; resolve that ambiguity against
  the intended decreasing tracking reward before implementing version 2.
- **Step 2:** Profile the current collider recipe first. Record collection versus
  learning cost from the existing metrics, memory use and contact completeness.
  The present guards also cap replicas at 128 and updates at 2,000. Increase
  scale through a named configuration and replica-isolation checks. Simplify
  colliders only if measurements justify a separate asset variant; preserve
  the approved source model and require fresh native admission.
- **Steps 3–5:** The optimizer runs on CPU, but each selected directional motion
  needs native replay before it becomes demonstration data. Validate the same
  state features and 20 ms transition interval on priors and policy rollouts;
  do not pair terminal states with the next episode's reset. Add the actor's
  velocity estimator and short-term memory encoder before interpreting a
  paper-method comparison. The current flattened history is not that network.
  The full critic needs the paper's privileged inputs and terrain encoder.
- **Steps 6–8:** Freeze model-specific randomization and terrain settings before
  comparisons. The paper's 25.5 kg robot differs from this 7.47 kg model.
  Add task-plus-style to the proposed two-arm comparison for the complete
  reward ablation, with matched transitions and five seeds. Separate additional
  RMA/MPC controller reproduction from the core method and reward ablations.

The paper names a 61-value AMP state, but its description of foot heights
accounts for fewer values. The retained kernel uses six 3D foot positions.
Declare this interpretation and test feature order, frames and units before
training a discriminator. Preserve the paper's input-gradient penalty intent
from its cited AMP method. This foundation implements no new learning method.

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
