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

Finish the declared slower-period screens before choosing another adaptation.
Keep any change to the paper's joint equations distinct from a parameter
calibration. Retain the approved model, actuator limits and acceptance gates.

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
