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

The [optimized reference](../artifacts/ppo_reference_comparison_20260917/review_smooth_001/RESULT.json)
passes the forward screen through the native controller. It is an executed
target sequence, with no learned feedback policy or omni qualification.

The [standard PPO baseline](../artifacts/ppo_reference_comparison_20260917/README.md)
completed 1,200 updates but failed forward, quiet and stop probes. The subsequent
[paired example-initialization test](../artifacts/forward_example_ppo_20260917/COMPARISON_001.json)
gave both arms 1,200 updates and 3,686,400 transitions with seed 20260914.
Both arms failed their forward screens at updates 0, 300, 600 and 1,200.
Final mean speeds were 0.004811 m/s from scratch and 0.000961 m/s with example
initialization, against 0.05 m/s. The copied actor failed before PPO began.
This pair establishes no benefit from that initialization. It does not resolve
other motion-prior methods or the cause of the locomotion failure.

The [training diagnostics](../artifacts/forward_example_ppo_20260917/TRAINING_DIAGNOSTICS_001.json)
record weak forward motion despite rising reward and frequent target limiting.
Use those measurements to choose the next bounded investigation. More replicas
or updates have not established a solution.

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
raw captures and failed attempts. The [experiment record](../artifacts/forward_example_ppo_20260917/README.md)
explains the saved files and complete Spark archives. Preserve historical
checkpoint identities; a source move cannot admit a new runtime.

## Historical contracts

The core contract tests retain this historical Runtime joint/action order:

```text
 0 revolute_1_1   1 revolute_1_7   2 revolute_2_5
 3 revolute_3     4 revolute_4     5 revolute_5
 6 revolute_1     7 revolute_1_6   8 revolute_1_5
 9 revolute_1_3  10 revolute_1_4  11 revolute_1_2
12 revolute_2    13 revolute_2_6  14 revolute_2_4
15 revolute_2_2  16 revolute_2_3  17 revolute_2_1
```

Indices 0–5 are coxa, 6–11 femur and 12–17 tibia. Use the canonical joint names
in `locomotion/env_config.py` for the approved robot. Hardware mapping needs
its own verification.

C-study, mock, four-bar and custom PPO/AMP runs keep their original model,
observation layout, source and gates. Use their frozen source packs or the
[pinned pre-cleanup training reference](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/docs/TRAINING.md).
The [source inventory](../configs/source_inventory.json) identifies removed
code and context. Historical decisions do not authorize new work or override
the approved robot. The current status lives in [STATUS](../STATUS.md).
