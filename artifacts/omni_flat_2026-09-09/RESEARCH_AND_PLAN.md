# Flat omnidirectional locomotion toward sensor-based terrain walking

9 September 2026. User authorization: continue Step 2 now, research the approach,
change controller architecture if useful, and keep future terrain adaptation central.

The final product remains the [survey autonomy roadmap](../project_review_2026-09-04/ROADMAP.md):
all-direction terrain locomotion from onboard sensing, stable survey payload,
Jetson deployment, and accurate bounded coverage without an operational RTK base.
This experiment advances the motion controller; it does not train terrain or mission planning.

## Current execution

Active campaign: `omni_flat_002`, frozen `omni_source_v3`, continuing the verified
model425 PPO checkpoint. The revised command curriculum includes gentle through
tight arcs; the77-scenario suite and path demos below supersede the initial pilot
counts. Full standing validation passed:32environments,19bodies,18joints,6feet,
zero falls/truncations/non-foot contacts or post-settle saturation; maximum
post-settle requested torque0.53668Nm. A25-update continuation pilot is exercising
the expanded evaluator and the new renderer before700-update training chunks.

The displayed demo will be actual PPO in the control loop. The future navigation
architecture remains modular: coverage route and desired heading → local pose/path
controller → forward/left/yaw commands → learned18-joint controller. Simulator
pose is used only by the test path follower and critic; real estimation is still
required before deployment. The new ground renderer is not yet visually verified.

## Research findings and decisions

1. **A commanded body twist is the appropriate flat-ground task.** The released
   [legged_gym configuration](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)
   samples signed planar velocities and yaw commands and separates velocity tracking
   from regularization. [Rudin et al.](https://proceedings.mlr.press/v164/rudin22a.html)
   demonstrate parallel PPO with terrain curricula. Their timing and ANYmal results
   do not predict training duration or hardware success for this hexapod.

2. **Reward requested motion, not one preferred axis.** The
   [Isaac Lab velocity task](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
   uses planar velocity and yaw tracking plus physical regularizers. For this run,
   planar vector error is invariant to translation bearing. Overspeed receives
   tracking error too. Signed progress is bounded and active only when requested.
   Turning penalties use yaw error, so intended turning is not penalized as drift.
   Stopping and transitions are sampled explicitly and tested without resets.

3. **Terrain and perception should be developed as distinguishable stages, with
   a shared interface.** [Miki et al.](https://arxiv.org/html/2201.08117) train a
   privileged terrain-aware teacher and a student using noisy available observations.
   Their recurrent encoder combines proprioception and exteroception. An elevation
   map separates the controller from a particular camera or LiDAR. This supports
   our proposed proprioceptive terrain baseline, terrain-aware teacher, then
   sensor-based policy with age/validity and observation failures. It does not
   establish that a blind controller alone can handle our intended terrain, or
   guarantee transfer from a quadruped to this hexapod.

The user’s intuition is therefore consistent with established training practice.
Flat and terrain control retain velocity/yaw tracking, motor constraints and
smoothness. Terrain later changes the training distribution, available observations,
clearance/support objectives and allowable body posture; it is not simply a new
forward reward or a LiDAR input attached to an otherwise unchanged demo.

## Initial pilot architecture (retained in the current continuation)

- Same selected C study asset: 72.5/126 mm femur/tibia; coxa unchanged; 19 bodies,
  18 joints; transferred current CAD mass/inertia, approximately 8.2608 kg.
- Benchmark 1 remains frozen. This is a separate scratch policy because its action
  semantics and observation dimensions differ from the reference-guided demo.
- MLP PPO with five frames of angular velocity, gravity, command, encoder and
  previous-action observations (315 actor values). Only the critic also receives
  simulator linear velocity (318 critic values). Short history is an initial
  architecture choice, not proof that recurrence is unnecessary on terrain.
- Direct bounded joint-position offsets; no gait clock, forced tripod contact
  schedule, forward trajectory reference, or prescribed base motion. Actual motor
  torque caps, dynamics and ground contacts determine movement.
- Commands: continuous uniform translation bearing; speed 0.06–0.20 m/s;
  yaw magnitude 0.12–0.40 rad/s. Mixture: 10% stand, 20% pure yaw, 45% translation,
  25% combined. Targets change every 3–6 s, with 0.25 m/s² vector acceleration
  and 0.8 rad/s² yaw acceleration limits. Both yaw signs are independently sampled.
- Flat reward: command tracking/progress, tilt/heave, torque demand and positive
  mechanical power, action/acceleration smoothness, slip, non-foot collisions and
  joint limits. Contact-event airtime shaping imposes no periodic schedule.
- Provisional small IMU/encoder observation noise. Mass/material and actuator
  dynamics remain nominal in this pilot. Hardware-informed randomization follows;
  the actor schema being implementable does not make it physically calibrated.

## Initial pilot acceptance evidence (expanded below)

A short 100-update pilot first exercises standing admission, PPO, evaluation and
recording before 700-update continuation chunks. Initial cap is 1500 updates,
then inspect evidence and adapt. A completed budget is not success.

The held-out nominal suite has 45 scenarios: 16 bearings at each of 0.10/0.20 m/s,
stand, four pure-yaw commands and eight combined commands. Each has eight replicas,
repeated with two held-out seeds. Report each direction independently. Continuous
forward/left/reverse/right/left-turn/right-turn/diagonal-turn/reversal/stop segments
share one episode. Record an actual 40-second labeled rollout, including failures.

Initial per-scenario gates: zero terminations/truncations, non-foot contact fraction
≤0.1%, requested torque above 1.6 Nm in ≤0.5% of sampled joint values, planar error
≤max(0.025 m/s, 25% of command magnitude), yaw error ≤max(0.06 rad/s, 25% of command),
tilt RMS ≤5°, vertical velocity RMS ≤0.04 m/s. Transition end-state errors are
scored separately from ramp tracking. These are simulation development thresholds;
thermal duty and field mission acceptance require additional measurements.

## Keep the terrain path open

The public command contract remains forward/left/yaw in a heading-following body
frame: forward=-body Y, left=+body X, yaw about +body Z. Preserve joint-name mapping.
Future changes can replace the encoder with recurrent memory and append local
terrain features, sample age and validity without changing navigation’s command
contract. Add terrain-relative clearance and posture objectives; never force the
flat-world root-height target on a slope. Compare against the saved flat policy
and a proprioceptive terrain baseline before attributing benefit to perception.

Production C CAD fit, real linkage mapping, completed payload mass/COM and measured
actuator dynamics remain outstanding. They should advance alongside software; this
scaled mock study is not the production model or manufacturing approval. Run the
single-leg test-stand identification loop from the roadmap as data becomes available.

## Updated navigation and arc contract (latest user steering)

The selected approach plans planar **position and body heading independently**.
The coverage layer chooses passes, exclusions and their ordering; a local controller
turns a time-varying pose trajectory into body-frame `(v_forward, v_left, yaw_rate)`.
The walking policy converts that command into joint targets. Constant nonzero
translation/yaw makes an arc, zero yaw allows straight strafing, and zero translation
allows turn-in-place. A curved route can also retain constant body heading by varying
the translation direction. No radius-only command can express all of these cases
without special cases; the three-component twist contract can.

For constant body-frame translation and yaw, ideal planar kinematics give radius
`sqrt(v_forward² + v_left²) / abs(yaw_rate)`. This is a trajectory relationship,
not a promise that the current policy achieves it. Unit tests independently check
quarter-circle endpoints, sideways arcs, straight and turn-in-place limits, heading
rotation, finite zero-yaw behavior and follower command bounds.

[Nav2 MPPI](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/controller_plugins/mppi_controller/configuring_mppic/)
supports an Omni motion model with forward, lateral and yaw controls and acceleration
constraints. It is a candidate local planner to benchmark on the Jetson using our
measured response limits/delay; it has not been deployed here. The coverage server's
[documented turn connectors](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/others/configuring_coverage_server/)
include turning-radius and car-like path options. We should evaluate its coverage
ordering separately from those connectors, preserving the hexapod's ability to
strafe and select body heading for sensor visibility and survey stability.

Version2 samples combined yaw continuously through zero, adding the gentle long-radius
curves missing from version1's0.12rad/s minimum moving yaw. The static suite expands
to77 scenarios ×2seeds (154rows,616environments). It includes eight translation
bearings for gentle arcs (0.15m/s,±0.05rad/s) and tighter arcs (0.20m/s,±0.40rad/s).
A70-second uninterrupted sequence adds gentle/sideways arcs, an S curve, a curved
translation with fixed heading, and stopping. Feedforward path error is recorded
alongside velocity errors; it is not a claim of closed-loop navigation.

The requested demonstration now uses five separate labeled simulator trials:
straight pass, sideways pass, combined arc, fixed-heading curve, and turn-in-place.
A bounded feedforward/P pose follower generates twist requests from ideal simulator
localization. PPO still generates all joint actions; no root pose or contact force
is scripted. Non-colliding ground meshes draw the desired path and blue travel
arrows, gold body-heading arrows, and an orange actual trail. Ground labels, current
path error and failure counts make failure visible. This is a test follower, not
an unimplemented MPPI controller relabeled as complete. The total video is69seconds.

The100-update version1 pilot completed standing admission, training,90-row evaluation
and its40-second recording. It was mostly stationary and failed moving commands;
only two standing rows passed. That is runtime-path validation, not a walking
milestone. Continuation reached a saved model425 checkpoint. It is preserved and
resumed into version2, retaining learned weights/optimizer while changing command
sampling and evaluation. Sourcev1 and its evidence remain frozen. All315 CPU tests
now pass (288Isaac +27robot/study). The new ground-annotation renderer still needs
actual Isaac recording and visual inspection before being claimed correct.

The resumed campaign is `omni_flat_002`, using frozen source_v3 (v2 was never
launched). Its first 69-second, five-path recording completed with zero terminations
across those trials. P95 path errors were 11.0 cm straight, 5.8 cm sideways, 10.3 cm
combined arc and 9.8 cm fixed-heading curve. This uses ideal simulator localization
for the outer path follower. The broader held-out tests still passed 0/154 static
command gates and 0/14 transition gates; motor demand and tracking need improvement.
The next 700-update PPO continuation is running. Update counts alone cannot qualify it.

Visual inspection found mirrored ground text and a clipped legend in that first
recording. Frozen source_v4 corrects glyph orientation and camera framing, with
an independent render queued in `omni_preview_001` for the next free GPU interval.
Training remains on source_v3; it is not interrupted for this visual correction.
Inspect the corrected recording before delivering it as the requested labeled demo.
