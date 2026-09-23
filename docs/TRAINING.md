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

### Current evaluated variant

You select the current candidate with `--tripod-adaptation stop_stride
--candidate 0`. You retain the original equations under `paper` and the
intermediate variants for replay of their frozen results. The passing screens
and stop repeats use the adaptations below; they do not validate the original
sinusoidal targets on this robot.

| Component | Current declared behavior |
| --- | --- |
| Geometry | Compute joint sweep coefficients from the approved toe Jacobians, the command and a 1.2 s period. Use low stance `(0,-0.30,0.40)` rad and raised stance `(0,-0.45,0.50)` rad. |
| Support motion | Use a linear support sweep. Return the swing feet between half-cycle fractions 0.15 and 0.80, with cosine speed ramps for forward motion and a cubic return for yaw. Retain measured touchdown and bounded recovery. |
| Forward lift | Use pitch offsets `(+0.26,-0.20)` rad at commands through 0.05 m/s, interpolate to `(+0.25,0)` at 0.10 m/s, and use `(+0.28,-0.10)` in raised mode. Yaw retains the geometry-derived lift. |
| Servo tracking | Add nominal damping compensation, position-error gain 0.5 and velocity-error gain 2 with a 5 Hz filter. Preserve the motor limits, action envelope and target limiter. |
| Start and stop | For forward starts, hold the stance for one second, then grow stride over one cycle. For yaw starts, blend to the first sweep endpoint. At a stop, finish the current swing and reduce stride over one full cycle before the standing hold. Use this stop for command and mode changes. |

The current variant computes each 20 ms target from half-cycle progress `p`,
which advances by `2*0.02/1.2` per control:

```text
S_support(p) = 1 - 2p
S_swing(p)   = -1 - 2p + 4*r(u),   u = clip((p - 0.15)/0.65, 0, 1)
r(u)         = h(u) for forward motion; 3u^2 - 2u^3 for yaw
h(s)         = v/2*(s - r0/pi*sin(pi*s/r0))      for s < r0
             = v*(s - r0/2)                       for r0 <= s <= 1 - r0
             = 1 - h(1 - s)                       for s > 1 - r0;  r0 = 0.10, v = 1/(1 - r0)
J*a          = period/4 * ((0, -forward, 0) + yaw*(-y, x, 0))   for each toe at (x, y)
q_ref        = stance + a*S + c_lift*g(phi)
phi          = -pi/2 + half*pi + acos(1 - 2*h(p)) + leg_phase
g(phi)       = (1 + cos(2*phi))/2 if cos(phi) > 0, else 0
e_p          = clip(0.5*b*(q_ref - q), -0.070, 0.070)
e_v          = clip(2*b*KD*(F(dq_ref) - F(dq))/12, -0.070, 0.070)
target       = clip(q_ref + KD*dq_ref/12 + e_p + e_v, joint limits and neutral +/- 0.35)
```

`J` is the approved toe Jacobian at the mode stance, and `c_lift` is the lift
row in the table above. The blend `b` is 0 at idle, 1 while walking and a
one-second cosine ramp during start and settling. `q` and `dq` come from the
preceding native control. `F` is a 5 Hz one-pole filter with
`alpha = 1 - exp(-2*pi*5*0.02)`, and its states reset while `b` is 0. A forward
start scales `a` by `(1 - cos(pi*min(t/T, 1)))/2`, and a stop scales it by
`s0*(1 + cos(pi*t/T))/2`, where `T` is the 1.2 s period. The emitted target
then changes by at most 0.040 rad per control. The controller log records
`nominal_target_rad` and `target_rad` as separate channels.

You can inspect the [three passing motion screens](../site/assets/tripod_speed_lift_native_20260922_001/native_result.json)
and the [passing stop suite](../site/assets/tripod_stop_stride_native_20260922_001/native_result.json).
[STATUS](../STATUS.md) records the native result of each variant. The
[closeout](../site/assets/tripod_reproduction_closeout_20260922_001/closeout.json)
records four interrupted qualification attempts. Full qualification, raised
clearance and terrain remain incomplete. You can inspect the
[forward comparison](../site/assets/tripod_qualification_recovery_20260922_001/forward_tracking_comparison.png)
and its [source identities and measurements](../site/assets/tripod_qualification_recovery_20260922_001/forward_tracking_comparison.json).
The [analysis provenance](../site/assets/tripod_review_corrections_20260922_001/analysis_provenance.json)
binds that comparison and two diagnostic figures to their scripts and inputs.

### Original equations and geometry adaptation

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
loss latches a fault and holds the last target. An out-of-envelope target
latches the `target_bounds` fault in the same way. You require a trial reset
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

The reset writes a fixed pose and zero velocity, and the controller draws no
random numbers. The recorded seed changes no state, so the second repeat of a
case replays the first trace. Each repeat pair in the
[stop suite](../site/assets/tripod_stop_stride_native_20260922_001/native_result.json)
has identical trace hashes: the six passing trials hold three distinct
trajectories. The [repeat identity record](../site/assets/tripod_review_corrections_20260922_001/repeat_identity.json)
lists each pair. Identical replays do not meet the independent-reset requirement.
A qualifying repeat needs a reset difference declared before dispatch.

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
`--tripod-adaptation stop_stride --candidate 0` for the current variant and
`--suite screen|forward_high|stops|qualification|clearance`. The default
`--tripod-adaptation paper` selects the four original candidates.
You complete the declared screen order before qualification and require its
two-reset pass before clearance testing. The launcher retains both GPU locks
and the existing reservation checks. The runner enforces standing admission
and records one actual video per trial. You must inspect prior suite results
before dispatch; the runner does not select a candidate for you.

### Declared variants

`--tripod-adaptation` selects a frozen variant, so you can replay each recorded
result. Each variant keeps the changes of its parent. The linked records hold
the equations, CPU checks and frozen inputs; [STATUS](../STATUS.md) holds each
native result. Declare a new variant and its CPU checks before native dispatch.
Keep changes to the paper equations separate from parameter calibration.

| Variant | Parent | Declared change | Record |
| --- | --- | --- | --- |
| `paper` | none | Eqs. (1)–(3) with the four period and hip choices above. Touchdown pitch returns to stance during support; lift offsets `(+0.35,-0.30)` and `(+0.45,-0.35)` rad. | [CPU result](../site/assets/tripod_20260921/result.json), [native corrections](../site/assets/tripod_native_20260922_001/result.json) |
| `geometry` | `paper` | Jacobian sweep coefficients with gains 1.0 and 1.1, 22/28 mm Jacobian lift and raised offset `(0,-0.15,+0.10)` rad. | [declaration](../site/assets/tripod_geometry_20260922_001/declaration.json) |
| `damping` | `geometry` gain 1.0 | Motor target adds `KD*dq_ref/12`. | [declaration](../site/assets/tripod_damping_20260922_001/declaration.json) |
| `retimed` | `damping` | Displacement clock `h` with 0.10 endpoint ramps. | [declaration](../site/assets/tripod_retimed_20260922_001/declaration.json) |
| `feedback` | `retimed` | Position feedback with gains 0.5 and 1.0, each within ±0.070 rad. | [declaration](../site/assets/tripod_feedback_20260922_001/declaration.json) |
| `liftoff` | `retimed` | Swing lift `g^0.75` with joint feedback off. | [declaration](../site/assets/tripod_liftoff_20260922_001/declaration.json) |
| `velocity` | `retimed` | Velocity feedback with gain 2, within ±0.070 rad. | [declaration](../site/assets/tripod_velocity_20260922_001/declaration.json) |
| `velocity_filtered` | `velocity` | 5 Hz one-pole filter on reference and measured velocity. | [declaration](../site/assets/tripod_velocity_filtered_20260922_001/declaration.json) |
| `pd_filtered` | `velocity_filtered` | Adds position feedback with gain 0.5. | [declaration](../site/assets/tripod_pd_20260922_001/declaration.json) |
| `overlap` | `pd_filtered` | Yaw support-overlap sweeps with the cubic return. | [declaration](../site/assets/tripod_overlap_20260922_001/declaration.json) |
| `startup` | `overlap` | Forward one-second standing hold, then a one-cycle stride ramp. The yaw touchdown residual returns over the rest of its half-cycle. | [declaration](../site/assets/tripod_startup_20260922_001/declaration.json) |
| `forward_overlap` | `startup` | Forward support-overlap sweeps with `h`; forward lift `(0.25,0)` rad and raised `(0.28,-0.10)` rad. | [declaration](../site/assets/tripod_forward_overlap_20260922_001/declaration.json) |
| `speed_lift` | `forward_overlap` | Low-mode forward lift `(0.26,-0.20)` rad through 0.05 m/s, interpolated to `(0.25,0)` at 0.10 m/s. | [declaration](../site/assets/tripod_speed_lift_20260922_001/declaration.json) |
| `stop_stride` | `speed_lift` | Stride decays over one cycle after the current swing, before the standing blend. | [declaration](../site/assets/tripod_stop_stride_20260922_001/declaration.json) |

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
