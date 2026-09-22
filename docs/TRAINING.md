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
([§III, p. 2; Fig. 2, p. 3](https://arxiv.org/pdf/2511.03167v1#page=2)).
Retain this project's approved motor model and target-change limiter; document
their differences from that controller with the new task configuration.

Preserve version 1 and its gates. Give changed tasks, models and results fresh
identities. James approves native steps. Source preparation starts no research
allocation and supplies no native admission.

| Step | Work and prerequisite | Spark |
| --- | --- | --- |
| 1. Reward v2 | Audit [Table I, p. 3](https://arxiv.org/pdf/2511.03167v1#page=3), for formulas, units, signs and weights before implementing a versioned paper reward with CPU tests. Resolve its printed positive tracking exponent against the intended decreasing reward. Define command-scaled tracking as a separate adaptation; resolve the stationary-reward criterion below. | No |
| 2. Throughput profile | Measure physics, contact, observation and learner cost plus memory across replica counts. The current guards cap replicas at 128 and updates at 2,000; the paper uses 4,096 robots. Extend scale through a named configuration and isolation tests. Change the 153 SDF colliders only if measurements justify an asset variant, then repeat one-robot and batch admission. | Yes |
| 3. Omni motion dataset | Extend [`optimize.py`](../locomotion/priors/optimize.py) from a forward cycle to eight bearings, both yaw directions and the arcs in `command_bank`. Require alternating tripod demonstrations on the approved URDF, with a consistent gait cycle across directions ([§III-A–B, p. 3](https://arxiv.org/pdf/2511.03167v1#page=3)). | No |
| 4. Native motion validation | Replay each motion with [`replay_native.py`](../locomotion/priors/replay_native.py) and require its matching screen before dataset admission. Construct AMP states from native replay. Match feature order, frames, units and the 20 ms interval; exclude terminal-to-reset pairs. | Yes |
| 5. Network architecture | Implement Table III's velocity estimator, memory encoder over five proprioception frames, low-level actor, privileged encoder and critic. Train the estimator with supervised simulation velocity labels ([§IV-A–B; Table III, p. 4](https://arxiv.org/pdf/2511.03167v1#page=4)). Supply the full 42-value privileged state, including base height and perturbations, plus collision states ([§III, p. 2](https://arxiv.org/pdf/2511.03167v1#page=2)). Test shapes and gradient paths; keep privileged inputs out of the actor. Add the terrain encoder with Step 8. Current actor/critic networks are plain [256, 256, 128] MLPs. | No |
| 6. AMP integration | Resolve the gradient-penalty ambiguity below, then implement the least-squares discriminator and style reward from Eqs. (1)–(2). Update PPO and the discriminator together ([§IV-B, p. 4](https://arxiv.org/pdf/2511.03167v1#page=4)). Check integration on flat ground; defer the final reward comparison to Step 9. Use the 13 learning probes as diagnostics and the full gate for qualification. | Yes |
| 7. Domain randomization | Implement Table II with frozen model-specific values. Declare adaptations for the different robot and simulator before comparing runs. | Yes |
| 8. Terrain curriculum | Add terrain levels, the critic height scan and terrain encoder. Retain admitted flat behavior. | Yes |
| 9. Reward and robustness comparisons | After Steps 7–8, compare task plus penalty, task plus style, and task plus style plus penalty with matched transitions and five seeds. Assess flat tracking and terrain progression ([§V-A; Figs. 3–4, pp. 4–5](https://arxiv.org/pdf/2511.03167v1#page=4)). Test push disturbances and reproduce the baseline controllers as separate comparisons ([§V-B, p. 5](https://arxiv.org/pdf/2511.03167v1#page=5)). Keep RMA/MPC work separate from the reward ablation. | Yes |

Freeze the chosen PPO settings and terrain curriculum thresholds in the task
configuration before training. The authors omit these details from
[§IV-B–V, p. 4](https://arxiv.org/pdf/2511.03167v1#page=4); label them as implementation
decisions. The authors train with randomization and a terrain curriculum
([Table II, p. 3; §IV-B, p. 4](https://arxiv.org/pdf/2511.03167v1#page=3)).

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
[Eq. (1), p. 4](https://arxiv.org/pdf/2511.03167v1#page=4). The proposed input-gradient
penalty differentiates with respect to the transition instead. Resolve that
difference against the cited AMP method or author code before implementation;
record the chosen formula and its source.

The authors call the AMP state 61 values in
[§III-A, p. 3](https://arxiv.org/pdf/2511.03167v1#page=3), but their foot-height description
accounts for fewer values. The retained kernel uses six 3D foot positions.
Declare that interpretation before discriminator training and verify the same features on
native demonstrations and policy rollouts. The optimizer's point contacts omit
mesh patches and impacts, so native replay remains a prerequisite.

## Terrain-adaptive tripod reproduction contract

You reproduce the prescribed controller in [Zhang et al. (2024), §§3.3.1,
4.1–4.2](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1426269/full).
The authors state that the gait algorithm “stops when the force sensors detect
that the leg has touched the ground” (§3.3.1). You interpret this as stopping
the descending foot trajectory. The authors do not specify debounce or recovery.
You retain RS05 torque control and the approved geometry. You make no claim
about their hardware results or Stage 2 completion.

The [supplement, Data Sheet 1, version 3](https://public-pages-files-2025.frontiersin.org/articles/1426269/file/Data_Sheet_1.docx/1426269_data-sheet_1/3)
has SHA-256 `029f0d0c5edeaff94b75bdd9cac051704f5e7f0323cd79a111570eb24b4f3df1`.
Its six figures show the prototype and Simscape blocks, including prescribed
revolute-joint motion in Figure S6. It supplies no controller code or additional
gait parameters. Its linked Google Drive video was inaccessible through the
research tool on 21 September 2026; video inspection remains incomplete.

### Equations and geometry adaptation

You implement Eqs. (1)–(3) with phase `u = omega*t + phi`:

```text
hip   = hip0 + A*sin(u)
knee  = knee0 + B*g(u)
ankle = ankle0 + C*g(u)
g(u)  = (1 + cos(2*u))/2 if cos(u) > 0, else 0
```

The authors print `2*t` and `4*t`; you expose `omega = 2*pi/period`
as a parameter. You start at `u = -pi/2`, where the lift and its derivative
are zero. You use joint order `lf,lm,lr,rf,rm,rr`, with each leg ordered
`coxa_yaw,femur_pitch,tibia_pitch`. Tripod A (`lf,lr,rm`) has phase 0;
tripod B (`lm,rf,rr`) has phase pi. The native joint axes define positive
rotation. Forward means native -Y, left means +X, and left yaw means +Z.
For forward swing, hip signs are `(-,-,-,+,+,+)`; left turns use positive
hip amplitudes on both sides. You verify these signs against approved-model
forward kinematics. The authors describe adjacent phase opposition and
same-direction hip turning in §4.2; the named mapping is our adaptation.

You use the existing low stance `(0,-0.30,0.40)` rad. The raised stance adds
`(0,-0.15,+0.15)` rad. Approved-model kinematics predict an 11.0 mm increase
in root height for fixed foot height, from 97.8 to 108.8 mm. You measure root
height here; the authors report 60/180 mm chassis clearances (Figure 4).
You require a measured raised-minus-low root height of at least 8 mm after
settling. You report chassis clearance from mesh geometry. Low swing offsets
are `(B,C)=(+0.35,-0.30)` rad; raised offsets are `(+0.45,-0.35)` rad.
You target at least 12 mm and 16 mm toe lift above each mode's stance plane.
You keep the 0.35 rad action envelope and 0.040 rad/20 ms limiter.

Eqs. (4)–(5) describe the authors' link geometry and approximate inverse
relations. You use the approved model transforms to check our offsets before
dispatch; you solve no inverse kinematics in the control loop. Eqs. (6)–(8)
hold hip angle and prescribe sinusoidal pitch changes. Their printed
`delta/T * sin((t-t0)/(2*T))` does not specify a final angle or a hold rule.
You use endpoint interpolation `h(s)=(1-cos(pi*s))/2`, `s in [0,1]`, over
one second with six supporting feet and fixed hip targets. You label this
endpoint rule as an adaptation. Clearance changes come from declared commands;
you omit Figure 5's obstacle perception and detour logic.

### Controller state and contact adaptation

You retain phase, active command, clearance mode, transition time, per-foot
contact debounce and touchdown pitch targets. You reset these values per trial.
You require six contacts before startup and blend to the first sweep endpoint
over one second. You run each half-cycle with one swing tripod. You accept
touchdown on descent after measured liftoff, freeze that foot's pitch, and
continue its hip sweep. You hold the next half-cycle until the landing tripod
supports the robot. During that tripod's support half-cycle, you return its
retained touchdown pitch to the mode stance with a cosine blend. This restores
the zero-lift part of Eqs. (2)–(3) before its next swing and avoids an angle jump.

You use distal contact-normal force from the preceding eight native samples,
with mean magnitude at least 2 N for contact and at most 1 N for release.
You require two controls for each change. These are controller thresholds;
they do not replace the existing contact acceptance checks. At a missing
touchdown you hold phase and lower the missing foot at 0.10 rad/s, at most
0.04 rad, for 0.40 s. Failure to land, failure to lift, or sustained support
loss latches a fault and holds the last target. You require a trial reset
after a fault. You record the fault as a failed attempt.

You support zero, forward commands in `(0,0.10]` m/s and pure yaw commands
with magnitude at most 0.20 rad/s. You reject commands outside this envelope.
You treat a command or clearance change as a stop, complete the current swing,
return to the mode stance over one second, then start the new command or
clearance transition. This startup/stop behavior is a project adaptation.
You declare stopped after six-foot support and completion of the stance return;
native stop acceptance still requires the unchanged velocity and contact gates.

### Frozen experiment matrix and acceptance

You declare a finite sweep before dispatch: periods 1.2 and 1.6 s crossed
with hip amplitudes 0.12 and 0.18 rad, four candidates total. At 0.05 m/s
or 0.20 rad/s you use that amplitude; forward amplitude scales with command
and caps at 0.30 rad. You screen forward and both yaw signs for each candidate
on flat ground. You select the first candidate that passes all three screens,
in the declared order, then freeze it for qualification. You keep these four
timing/hip choices for each source correction. Before a geometry revision,
you record its native evidence and declare its offsets. You preserve motor/model limits.

| Phase | Cases and unchanged checks |
| --- | --- |
| CPU | Named joint mapping, phase opposition, reset determinism, boundary continuity, contact debounce, missed/early touchdown, command changes, faults and geometric reach. |
| Admission | Fresh one-robot and 32-robot standing runs with matching physics, model, stance and geometry, recomputed by `locomotion.admission`. |
| Flat screens | Forward 0.05 m/s and pure yaw ±0.20 rad/s; use existing static scorer and full native contact/motor checks. |
| Flat qualification | Existing forward 0.10 m/s, yaw ±0.20 rad/s, quiet 20/32 s and stand/forward/both-yaw stop cases. Repeat from two independent resets. Report other commands as unsupported or outside this reproduction. |
| Clearance | Repeat low/raised forward and yaw cases, plus low→raised→low commands while moving; no native joint violations or nonfoot contacts. Require the geometry targets above. Keep all stop checks. |
| Terrain, after flat pass | One fixed bar, 10 mm high, 40 mm along travel, 0.80 m across, centered 0.60 m ahead. Command raised mode at 2 s and low mode after the full robot crosses. Require crossing within 30 s, 10 s stopped afterward, no nonfoot contact or joint violations, and two reset replicas. Freeze a terrain-specific contact capture before dispatch. |

You record video and 400 Hz ground-contact force plus requested/applied torque
through the existing evaluation pipeline. You bind source, model, input and
configuration hashes, retain failures, and report each parameter change.
For the 36-second clearance-switch capture, you apply the unchanged 20-second
static scorer to controls 0–999 and 800–1799. The overlapping windows cover
the scored interval after the initial two seconds. Both windows must pass;
the native contact and motor checks retain the full 36-second capture.
You do not infer physical success from CPU checks. Failed flat qualification
blocks terrain; failed terrain leaves terrain reproduction incomplete.
You may use accepted native transitions as future AMP demonstrations after
separate dataset review. This work starts no AMP or learned-control work.

You prepare each allocation with `python -m locomotion.prepare --mode tripod`,
the admitted `--inputs`, a fresh `--output` and `--remote-root`, plus
`--candidate 0..3` and `--suite screen|qualification|clearance`.
You complete the declared screen order before qualification and require its
two-reset pass before clearance testing. The launcher retains both GPU locks
and the existing reservation checks. The runner enforces standing admission
and records one actual video per trial. You must inspect prior suite results
before dispatch; the runner does not select a candidate for you.

### Reproduction result on 21 September 2026

You can inspect the [source-bound result](../site/assets/tripod_20260921/result.json).
Seventeen CPU checks pass criterion 1. Approved-model kinematics predict
15.28 mm low-mode toe lift and 19.09 mm raised-mode toe lift. Criteria 2–7
remain incomplete because no native trial ran. The host preflight rejected a
released reservation marker. After authorizing GPU job recovery, the user
instructed: “Stop the GPU jobs; leave native runs pending.” Both idle jobs
have preserved restart records; the reservation service remains unchanged.
The flat-ground runner includes torque/contact capture and video recording,
but it supplies no measured native evidence yet. Terrain capture and testing
remain pending behind flat qualification. Future AMP work must wait for
accepted native demonstrations, including supported turns and stops.

### Native corrections declared on 22 September 2026

The one-robot and 32-robot standing captures pass after reservation restoration.
The first controller retained early-touchdown pitch throughout stance; nine
screens ended with a liftoff fault. You now return that offset to zero during
the support half-cycle. With this correction and the original lift offsets,
candidate 0 completed its forward and left-turn screens but failed tracking.
The forward capture showed 3–11 mm toe-height ranges and 0.051 m/s mean planar
error against the unchanged 0.025 m/s limit. These are failed native results.

You declare a second geometry adaptation before dispatch: low pitch offsets
`(+0.35,-0.30)` rad and raised offsets `(+0.45,-0.35)` rad. Forward kinematics
predict 25.7 mm and 34.4 mm of lift; native measurement must still establish
12 mm and 16 mm. You retain the four period/hip choices above and independent
reset requirements. You preserve the original captures and source packs,
including prepared choices that a source correction supersedes before execution.

### Complete-cycle diagnosis on 22 September 2026

You can inspect the [cycle measurements](../site/assets/tripod_cycle_20260922_001/cycle_metrics.json)
from candidate 0's larger-lift forward run. Controls 190–262 cover the first
complete A/B cycle starting after 3.5 seconds. Replay matches all 1,000 logged
controller targets, and approved-model forward kinematics match the captured
toe positions within 0.3 micrometres.

We measure a 1.46-second cycle against the 1.2-second setting, including 0.26
seconds of touchdown waits. Rear-foot touchdown precedes front-foot touchdown;
the controller extends the front legs during recovery. Median forward speed
is 0.0081 m/s, and 23.3% of controls have negative forward speed in this cycle.
Eleven complete cycles retain the surge and recoil pattern.

The commanded support triangle has edge-length ranges of 27.9–28.6 mm on
two edges per half-cycle. Fixed ground contacts cannot follow these targets
through rigid body motion. The contact-point calculation estimates median
tangential speeds of 0.23–0.59 mm/s per foot, with brief spikes. This estimate
uses consecutive tibia poses and current contact patches; the recorder does
not supply tangential friction force. Toe-landmark displacement includes
rotation of the foot shape. We retain joint tracking and touchdown timing as
coupled contributors; these measurements do not isolate one cause.

The [completed sweep](../site/assets/tripod_sweep_20260922_001/result.json)
retains 24 failed motion trials across the source corrections. Candidate 2
completed its slower-period forward run with 0.03736 m/s planar error against
the 0.025 m/s limit. It also exceeded the native joint-speed bound on two
physics steps. Candidate 3 stopped its forward run at control 615 with
`touchdown_timeout`. Both slower-period candidates failed both turn screens.
We verified exact-container cleanup and retained the Spark reservation.

Declare the next geometry correction before another native candidate. Keep
changes to the paper's joint equations distinct from a parameter calibration.
Retain the approved model, actuator limits and acceptance gates. Qualification
and terrain remain pending because no candidate passed the three screens.

### Geometry correction declared on 22 September 2026

You retain the paper-equation variant and declare a separate `geometry`
variant before native dispatch. You add a sine term to both pitch joints.
This changes Eqs. (2)–(3); it is a geometry adaptation, not a parameter fit
of those equations. You retain uniform phase and contact-triggered transitions.

At each mode stance, you compute each toe position `p` and its joint Jacobian
`J` from the approved model transforms. You solve `J*a = d`, where
`d = gain*period/4 * ((0,-forward,0) + yaw*(-p_y,p_x,0))`.
You command `q = stance + a*sin(u) + lift*g(u)`. For pure forward motion,
the ideal stance displacement gives the requested average speed over a
half-period. This calculation omits actuator tracking and touchdown waits.
Native speed measurements must establish the mapping under load.

You derive pitch lift coefficients from `J*lift = (0,0,height)`, then set
the yaw lift coefficient to zero. You use heights of 22 mm in low mode and
28 mm in raised mode. You change the raised stance offset to
`(0,-0.15,+0.10)` rad to leave room for support motion within the action
envelope. These are kinematic inputs; the measured 12/16 mm toe-lift gates
and 8 mm root-height-increase gate remain unchanged.

At touchdown, you hold the pitch target through the rest of that swing.
During support, you blend from that target to the next sweep endpoint with
the existing cosine rule. You retain the two-control contact debounce and
bounded recovery. You check trajectory and transition bounds on the CPU
before any native run. You preserve the original controller variant and tests.

You declare two candidates, in order: gains 1.0 and 1.1 with period 1.2 s.
You screen forward 0.05 m/s and both yaw signs for each candidate, then use
the qualification and clearance matrix above for the first complete pass.
You select this sweep with `--tripod-adaptation geometry`; the default
`paper` selection retains the four earlier candidates. You retain failed
attempts and require a new declaration before another parameter revision.

You can inspect the [CPU declaration](../site/assets/tripod_geometry_20260922_001/declaration.json).
The first candidate's 0.05 m/s trajectory has a maximum stance-edge range of
1.3 mm. Both candidates remain within 0.341 rad of the low neutral stance
across the declared commands and modes. CPU contact fixtures cover early
touchdown, missed touchdown and stops. Native tracking remains unqualified.

### Geometry correction result on 22 September 2026

You can inspect the [six native screens](../site/assets/tripod_geometry_native_20260922_001/result.json)
and the [recorded cycle](../site/assets/tripod_geometry_native_20260922_001/cycle_metrics.json).
Both candidates complete all three captures without controller faults. Their
native motor, joint and contact checks pass, but all six fail motion tracking.
The forward mean speeds are 0.0466 and 0.0513 m/s for the 0.05 m/s command;
their planar errors are 0.0490 and 0.0535 m/s against the 0.025 m/s limit.
Both turn directions fail the 0.06 rad/s yaw-error limit.

The first candidate's measured cycle lasts 1.20 s without touchdown waits.
Its target stance-edge range falls to 1.83 mm, while 35% of cycle controls
retain backward velocity. The controller detects liftoff at 23–27% of the
swing interval. The feet retain load during the initial forward target sweep.
You retain these six failures and the preceding 24 attempts. Geometry and
average speed corrections have not established accepted tracking under load.

### Damping correction declared on 22 September 2026

You declare one `damping` candidate before dispatch. You retain geometry
candidate 0 and add joint-velocity feedforward to its motor target:

```text
dq_reference = (reference_now - reference_previous) / 0.02
motor_target = reference_now + KD * dq_reference / 12
```

The approved motor law remains `torque = 12*(target-q) - KD*dq`, with the
same torque-speed limit and native physics. At exact reference tracking,
this target term cancels the motor's damping demand. It does not compensate
gravity, inertia or ground forces. You label the term as an RS05 tracking
adaptation to the prescribed joint motion in paper §5.2.1.

You retain the geometry reference for phase and touchdown state. You record
that reference and the compensated motor target as separate channels. You
check the full compensated target against the existing joint/action bounds
and apply the same 0.040 rad/20 ms slew bound to the emitted motor target.
You reject an out-of-envelope target; you do not enlarge the action range.

You screen the single candidate with `--tripod-adaptation damping --candidate 0`
after the two geometry candidates finish. You retain the forward and both-turn
screens, then the existing qualification and clearance matrix if it passes.
You require a new evidence-backed declaration before another correction.

You can inspect the [frozen damping declaration](../site/assets/tripod_damping_20260922_001/declaration.json)
and [CPU replay](../site/assets/tripod_damping_20260922_001/cpu_replay.json).
Replay uses the first geometry candidate's recorded contact inputs; it does
not predict the new contact sequence. The maximum added offset is 0.0332 rad,
and the emitted targets remain within the existing action and slew bounds.

The [three native damping screens](../site/assets/tripod_damping_native_20260922_001/result.json)
complete 1000 controls each with native motor and contact checks passed.
Forward mean speed reaches 0.04979 m/s, but planar tracking error remains
0.04852 m/s against 0.025. Left and right yaw errors remain 0.15315 and
0.15304 rad/s against 0.06. You retain all 33 failed motion attempts.

### Phase-clock correction declared on 22 September 2026

You retain the uniform-clock variants and declare one `retimed` candidate.
You retain the geometry path and damping correction, with period 1.2 s and
sweep gain 1.0. You change the phase clock in Eqs. (1)–(3); this is a method
adaptation, distinct from a reproduction of the printed time law.

For an ideal rigid stance and exact joint tracking, the uniform-clock speed
is proportional to `sin(pi*s)` over half-cycle fraction `s`. Its minimum
mean absolute error against unit speed is `1/3`, even after amplitude
calibration. At 0.20 rad/s this gives 0.0667 rad/s, above the existing 0.06
gate. Compliance and nonlinear geometry can change this ideal result; the
calculation does not establish a bound on native motion.

You define a normalized displacement `h(s)` with cosine acceleration ramps
over the first and last 10% of each half-cycle, and constant speed between
them. With `r=0.10` and `v=1/(1-r)`, you use:

```text
h(s) = v/2 * (s - r/pi*sin(pi*s/r))                  for s < r
h(s) = v * (s-r/2)                                  for r <= s <= 1-r
h(s) = 1-v/2 * (1-s-r/pi*sin(pi*(1-s)/r))             for s > 1-r
phase = -pi/2 + half*pi + acos(1-2*h(s)) + leg_phase
```

You keep the sine and cosine-squared joint path. You use `1-h(s)` for the
touchdown-offset return, which preserves the blend to the next stance
endpoint. Contact thresholds and recovery limits remain unchanged. You
check emitted targets against the same joint/action and slew bounds.

You screen this single candidate with `--tripod-adaptation retimed --candidate 0`
after the failed damping screen. You retain the existing forward/yaw
gates and the qualification, stopping and clearance matrix. Native success
must establish the effect of the changed clock under load. You can inspect
the [declaration](../site/assets/tripod_retimed_20260922_001/declaration.json)
and [ideal-speed calculation](../site/assets/tripod_retimed_20260922_001/error_bound.json).

You retain the [interrupted first attempt](../site/assets/tripod_retimed_20260922_001/interrupted.json).
The launcher detected unrelated CUDA compute after preflight and removed
its owned container. The closed native chunk contains 800 substeps over
controls 0–99; no final motion score exists. A follow-up cleanup check
verified container absence and the retained reservation. You must resolve
allocation contention before a fresh attempt. The 33 completed motion
trials remain failed; this partial attempt supplies no acceptance result.

### Retimed native result and joint feedback declared on 22 September 2026

You retain the [three completed retimed screens](../site/assets/tripod_retimed_native_20260922_001/result.json)
and the [measured cycle comparison](../site/assets/tripod_retimed_native_20260922_001/cycle_comparison.json).
Forward error falls to
0.03031 m/s, and left/right yaw errors fall to 0.08860/0.08894 rad/s. These
values exceed the unchanged 0.025 m/s and 0.06 rad/s limits. Native motor
and contact checks pass. You retain 36 failed completed motion trials and
one interrupted attempt.

In forward controls 233–292, you measure 17% negative-speed controls. The
mean support-joint errors are about -2.90° at the femur and +3.83° at the
tibia; swing errors are smaller. A finite-difference estimate from the
nominal support feet has 0.00795 m/s error, against 0.03047 m/s from actual
joint motion. This estimate omits angular correction and uses interval
velocities. It supports a joint-tracking diagnostic without assigning the
full error to one cause.

You declare two `feedback` candidates with gains 0.5 and 1.0. Both retain
the retimed path, period and lift. You add bounded proportional joint-error
feedback through the existing motor-target input:

```text
offset = clip(blend * gain * (q_reference - q_measured), -0.070, +0.070)
target = clip(q_reference + KD*qdot_reference/12 + offset, target_bounds)
```

You read the last measured joint positions from the preceding native control.
You ramp `blend` from zero to one during the existing startup cosine blend,
and from one to zero during settling. You use one during walking and zero
during idle or clearance changes. You retain the joint limits and neutral
±0.35 rad target envelope, then apply the 0.040 rad / 20 ms limiter. You
record the feedback offset and target clipping for each control.

You keep the actuator gains and torque-speed limits unchanged. This outer
feedback changes closed-loop stiffness and is a declared controller
adaptation to the prescribed-motion simulation in paper §5.2.1. It does not
force measured joint positions or change physics. You screen gain 0.5 first,
then gain 1.0 if needed, before qualification and clearance suites. You
retain the existing acceptance limits and report unexecuted candidates.
You can inspect the [feedback declaration](../site/assets/tripod_feedback_20260922_001/declaration.json)
and [recorded-input replay](../site/assets/tripod_feedback_20260922_001/cpu_replay.json).

### Contact-timing lift declared on 22 September 2026

You retain the [failed gain-0.5 screen](../site/assets/tripod_feedback_native_20260922_001/result.json).
Forward error rises to 0.03333 m/s; yaw errors remain 0.08784/0.08813 rad/s.
The forward trial also fails the native speed and contact checks. The
[contact incident](../site/assets/tripod_feedback_native_20260922_001/contact_incident.json)
records a 202 N rear-right toe impulse and a tibia speed excursion, with
the applied-torque cap active. You leave gain 1.0 unexecuted. You retain
39 failed completed trials and one interrupted attempt.

You rejected an initial `cos(phase)` lift in CPU checks: the 0.10 m/s
command exceeded the unchanged joint/action envelope. You preserved that
source and failure outside the checkout and did not dispatch it.
You declare one revised `liftoff` candidate from the retimed configuration
with joint feedback disabled. You keep the same maximum lift and replace
the positive swing lift factor `cos(phase)^2` with `cos(phase)^1.5`, or
`lift^0.75` in the existing waveform. You retain zero lift outside swing. This changes
the pitch path from paper §4.2; it is a declared contact-timing adaptation.

The retained cycle shows force on the outgoing tripod during the first
13–20% of swing and touchdown near 90–93%. During these intervals, swing
targets move opposite to the planted-foot motion needed for forward travel.
The new profile raises the foot sooner and lowers it later at the same
peak height. The retimed phase clock preserves zero endpoint lift velocity.
Native contact measurements must establish whether this reduces braking;
the geometric profile supplies no physical acceptance.

You retain the period, hip sweep, damping feedforward and touchdown rules.
You check both clearance modes against the existing target and slew bounds
before dispatch. You run one forward/left/right screen with the unchanged
motor, contact and tracking gates before qualification or terrain work.
You can inspect the [lift declaration](../site/assets/tripod_liftoff_20260922_001/declaration.json)
and [recorded-input replay](../site/assets/tripod_liftoff_20260922_001/cpu_replay.json).

### Velocity feedback declared on 22 September 2026

You retain [three failed lift screens](../site/assets/tripod_liftoff_native_20260922_001/result.json).
Forward error is 0.02979 m/s; left
and right yaw errors are 0.08488 rad/s. Native motor and contact checks
pass. You retain 42 failed completed trials and one interrupted attempt.

You compare 12 complete forward cycles through measured joint positions,
root poses and toe positions in the [velocity decomposition](../site/assets/tripod_liftoff_native_20260922_001/velocity_decomposition.json).
Joint-tracking differences contribute
0.0318–0.0334 m/s RMS to body speed; planted toe-marker motion contributes
0.0006–0.0020 m/s. Toe-marker motion includes rotation at the contact.
These finite-difference RMS values differ from the acceptance scorer's
mean absolute error and endpoint COM velocity.

You test one velocity-feedback candidate from `retimed`, with position
feedback off and the original squared-cosine lift. A fixed-height stance
linearization gives forward damping ratio 0.204 and natural period 0.342 s.
This approximation omits body rotation, changing support and leg inertia;
it supports a damping test without proving native behavior.

You add `clip(blend * 2 * KD * (reference_velocity - measured_velocity) / 12,
-0.070, 0.070)` to the damping-compensated motor target. You compute reference
velocity from successive nominal targets at 20 ms and read measured velocity
from the prior native control endpoint. You use the existing startup/settling
blend and zero correction during idle and clearance changes. You clip targets
to the existing joint/action envelope and retain the 0.040 rad slew limit.
You record the correction and target clipping in each controller row.

This feedback changes closed-loop damping through controller inputs. You
preserve the actuator gains, torque-speed limits, physics and acceptance gates.
The paper prescribes joint motion; this controller adds velocity feedback to
track that reference under load. You test CPU motor-law and transition checks,
then one native forward/left/right screen before qualification.
You can inspect the [velocity-feedback declaration](../site/assets/tripod_velocity_20260922_001/declaration.json),
[CPU replay](../site/assets/tripod_velocity_20260922_001/cpu_replay.json) and
[stance approximation](../site/assets/tripod_velocity_20260922_001/stance_linearization.json).

### Sampled velocity filter declared on 22 September 2026

The completed unfiltered forward trial meets the planar-error gate at
0.02195 m/s but fails yaw tracking at 0.11454 rad/s. Its yaw spectrum has
96.3% of power at or above 15 Hz, with a 25 Hz peak. The prior retimed trial
has 3.1% in that band. The unfiltered velocity feedback introduces this
oscillation at the 50 Hz controller sample rate.

You check the approved free-leg inertia at the neutral pose with a fixed
base and an exact 20 ms zero-order hold. The unfiltered linear model has
spectral radius 1.524. A 5 Hz one-pole filter lowers it to 0.854. This
model omits contact and saturation; it supplies no native acceptance.

You declare one `velocity_filtered` candidate with the same gain and bounds.
You filter reference and measured joint velocity with
`alpha = 1 - exp(-2*pi*5*0.02)` before subtracting them. You reset both
filter states to zero while the feedback blend is zero. You retain the
unfiltered candidate and its captures. You test filter attenuation and
transitions, then run one native screen after the current allocation ends.
You retain the [completed unfiltered screen](../site/assets/tripod_velocity_native_20260922_001/result.json):
all three trials fail yaw tracking and pass native motor/contact checks.
You retain 45 failed completed trials and one interrupted attempt.
The [filter declaration](../site/assets/tripod_velocity_filtered_20260922_001/declaration.json)
links its CPU replay and sampled-loop inputs.

### Combined feedback declared on 22 September 2026

The filtered screen passes forward motion at 0.02292 m/s planar error
and 0.01674 rad/s yaw error. Both turns fail yaw tracking near 0.077 rad/s.
You retain one passed screen, 47 failed completed trials and one interrupted
attempt. Support pitch errors remain near 3–4 degrees in the completed
forward and left trials. Qualification and clearance work remain pending.

You declare one `pd_filtered` candidate that combines position gain 0.5
with velocity gain 2 and the 5 Hz filter. You add both bounded corrections
before clipping the combined target to the existing joint/action envelope.
You retain the 0.070 rad bound on each correction and the 0.040 rad slew limit.
You use the existing startup and settling blend for both feedback terms.

You check 2,520 fixed-base free-leg models from captured and CPU-generated
poses. The combined candidate's maximum spectral radius is 0.802. You reject
position gain 1 in this combination because its neutral-pose spectral radius
is 1.050. These linear checks omit contact and saturation. You run one native
forward/left/right screen with unchanged physical and tracking gates.
You retain the [filtered screen](../site/assets/tripod_velocity_filtered_native_20260922_001/result.json)
and its selected forward video. The [combined-feedback declaration](../site/assets/tripod_pd_20260922_001/declaration.json)
links the CPU replay and sampled-model inputs.

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
