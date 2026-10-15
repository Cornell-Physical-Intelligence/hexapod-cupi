# Locomotion goals and acceptance

You build a controller that turns a forward, lateral and yaw velocity command
into stable motion on the approved hexapod. It must walk in either direction,
turn both ways, change commands and stop with a steady payload deck.
[ARCHITECTURE](../ARCHITECTURE.md) defines the mission and system boundaries.

## Intended behavior

You first qualify omnidirectional motion on flat ground. Test combinations and
transitions as well as isolated directions. Under a zero command, the robot
must settle and hold its stance. Judge gait appearance alongside measured
tracking and loads.

You then extend the demonstrated operating envelope to disturbances and terrain
while retaining accepted flat-ground behavior. Define the conditions and pass
criteria before training. Keep the policy's inputs compatible with the sensors
you intend to use on the robot; privileged simulation inputs belong in the critic.

Hardware transfer requires measured motor behavior and verified joint mapping.
Simulation results alone cannot establish hardware readiness. Use the
[single-leg stand](LEG_STAND_HARDWARE.md) to test actuator assumptions before
full-body hardware trials.

## Current simulation baseline

| Item | Contract |
| --- | --- |
| Robot | [Approved model](../robot/active_model.json): 19 bodies, 18 direct-drive joints, with its recorded mass and motor corrections. |
| Control | 400 Hz physics, 50 Hz policy actions and eight substeps per action. |
| Actions | Joint-position offsets, baseline scale 0.35 rad; retain the 0.040 rad per 20 ms target-change limit for comparisons. |
| Motors | Named damping and the approved speed-dependent effort curve; provisional 1.6 N·m software cap. |
| Observations | Baseline actor: 231 values from five proprioception frames, velocity command and executed action. Critic: 234 values with measured velocity. Record optional inputs, including gait-clock values, in the task configuration. |
| Learning | RSL-RL PPO with versioned rewards. Record learner options and reward version with each result. |
| Admission | Native standing passes at one robot and the intended batch size, matching the source and model/input identities. |

You can change the implementation through reviewed experiments. Give changed
models and tasks fresh identities and preserve historical checkpoints with
their original contracts. Renew native admission for the changed source or
model. A package move alone cannot establish compatibility.

## Method and comparisons

You compare learned controllers against a PPO baseline and the prescribed
[tripod controller](../locomotion/README.md#prescribed-tripod-controller).
The AMP research path tests whether a motion prior improves the gait; the
paper-network path tests its proposed architecture. Treat those as hypotheses
that need matched comparisons and repeated seeds.

Freeze each comparison's configuration and budget before dispatch. Record the
robot count and controls per update, reward terms and learner settings. Keep
paper reproduction separate from adaptations for this robot. Use the
[implementation guide](../locomotion/README.md) for supported options and the
[prior pipeline](../locomotion/priors/README.md) for demonstration admission.

## Acceptance

You qualify Stage 2 against the full 96-case manifest and the unchanged gates
in [the scorer](../locomotion/evaluation/scoring.py). The 13 learning probes
provide intermediate diagnostics; they do not replace full qualification.
The program lead must also accept the gait's appearance.

Each native evaluation must capture ground contact and requested/applied motor
torque before another learner sequence. Compare loads at matched achieved
motion. These measurements add no new acceptance limits. Preserve failed
cases and missing evidence; do not change thresholds to admit a result.

Bind a result to its model, source, settings and checkpoint. Preserve the raw
captures and visual evidence. Use [SPARK](../SPARK.md) to run an assigned
experiment and [OPERATIONS](OPERATIONS.md) for dispatch and recovery.

## Results and historical protocols

Read the [progress site](https://cornell-physical-intelligence.github.io/hexapod-cupi/)
for current results and GitHub issues for assignments. Keep this guide focused
on goals and acceptance.

The [training reference at df551b90](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/df551b9018505daef56f48e0078282b608d9db45/docs/TRAINING.md)
preserves the reproduction sequence, pilot v1/v2 protocols, throughput results
and tracking commands. Use that revision to interpret those experiments;
its recorded state does not describe a new run. The
[reward audit](REWARD_V2_TABLE1_AUDIT.md) retains the formulas and diagnostic
history. The [source archive guide](PIPELINE_LINEAGES.md) explains how to find
older implementations and evidence.
