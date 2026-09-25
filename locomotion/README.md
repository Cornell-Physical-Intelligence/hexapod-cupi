# Locomotion kernel

You run canonical locomotion through this package. The package uses the approved
19-body, 18-joint robot and stock RSL-RL 5.0.1 PPO. You can inspect its control
loop without reading old experiment launchers or a second environment port.

| Read in this order | Responsibility |
| --- | --- |
| [`../robot/active_model.json`](../robot/active_model.json), [`env_config.py`](env_config.py) | Select the URDF, masses, joint order, motor coefficients and fixed timing. |
| [`env.py`](env.py) | Load the native robot, form observations, limit joint targets, apply motor torque and advance eight physics steps. The actor receives 231 values and returns 18 offsets. |
| [`task.py`](task.py) | Sample held velocity commands, compute the training reward, detect failed episodes and reset selected robots. This file owns the reward. |
| [`ppo.py`](ppo.py), [`train.py`](train.py) | Adapt the task to stock PPO, record updates and loads, save checkpoints and load them for evaluation. |
| [`amp.py`](amp.py), [`amp_ppo.py`](amp_ppo.py), [`paper_networks.py`](paper_networks.py) | Share the 61-value AMP feature contract, score transitions with the discriminator, add the style reward inside PPO and supply the Table III networks. See [AMP learner](#amp-learner-and-paper-networks). |
| [`evaluate.py`](evaluate.py), [`evaluation.py`](evaluation.py) | Record actual policy rollouts and apply the existing walking, stopping, contact and motor gates. Missing cases remain missing. |
| [`force_metrics.py`](force_metrics.py), [`camera.py`](camera.py) | Report contact-normal force and motor torque, and record an Isaac camera video from the policy rollout. |
| [`admission.py`](admission.py) | Recompute one-robot and batch standing captures and require matching model, source and geometry before training. |
| [`prepare.py`](prepare.py), [`launch.py`](launch.py), [`reservation.py`](reservation.py) | Freeze named package files and explicit inputs, hold the existing Spark locks, supervise one container and verify its cleanup. |

The controller retains 400 Hz physics, 50 Hz policy actions, a 0.35 rad action
scale, the 0.040 rad target-change limit per control and the provisional 1.6 N·m
motor cap. [`task.py`](task.py) owns reward coefficients and command sampling;
[`evaluation.py`](evaluation.py) owns the numerical gates. Read
[TRAINING](../docs/TRAINING.md) for the proposed paper-method extensions.

## Run and verify

From the repository root:

```sh
uv sync --locked
uv run python -m unittest discover -s locomotion/tests
uv run python -m locomotion.prepare --help
uv run python -m locomotion.admission --help
uv run python -m locomotion.train --help
```

Use `prepare` to create a fresh allocation. Diagnostic mode supports one, 32 or
128 robots. Training uses the existing 128-robot configuration. Evaluation uses
one robot, with `--eval-scope focus` for the forward/quiet/stop subset, `probes`
for all 13 learning probes, or `full` for the 96 required Stage 2 cases.

Start the frozen launcher on Spark from its `source` directory with
`python3 -B -m locomotion.launch`, the binding path and its SHA-256. Read
[OPERATIONS](../docs/OPERATIONS.md) before dispatch. Each native result records
its exact source and inputs. Use `admission` on complete one/batch results to
produce the input declaration for a subsequent pack. Existing admissions cannot
admit changed physics source. Checkpoint loads require matching model, physics,
configuration and implementation; use the original frozen entry for historical
checkpoints that carry a different identity.

Each evaluation saves `control_trace.npz`, 400 Hz capture chunks, `report.json`
and `force_metrics.json`. A selected video also saves `rollout.mp4`. The reports
bind the checkpoint and file hashes. Training writes `metrics.jsonl` and load
summaries; mean contact force includes swing zeros. Contact forces exclude
unrecorded tangential friction. A completed recording does not accept walking.

## Larger project boundary

You need this loop to establish walking and stopping before adding terrain.
The model import and geometry tools remain under `robot/` and `tools/assets/`.
Hardware execution needs a measured actuator profile and a canonical runtime
binding. Navigation can then consume velocity commands and stopping status.
The navigation example consumes shared commands. Historical runtime bindings
remain in Git until a canonical replacement receives its own parity evidence.

The [AMP demonstration pipeline](priors/README.md) supplies an optimizer and
native replay with a reviewed dataset exporter. `amp.py` preserves the shared
61-value feature contract. Package
canonical inputs with `python -m locomotion.inputs pack --help`. Use the
[archive guide](../docs/PIPELINE_LINEAGES.md) for historical checkpoints and
source packs; preserve their original identities.

## AMP learner and paper networks

You select the online motion prior with `prepare --mode train --learner amp`
and the Table III networks with `--networks paper` ([TRAINING Steps 5–6](../docs/TRAINING.md#proposed-reproduction-sequence)).
[`amp_ppo.py`](amp_ppo.py) subclasses stock RSL-RL PPO. Each control pairs the
AMP state before the action with the state after it, before any reset, scores
the pair with the discriminator and adds the Eq. (2) style reward to the task
reward with weight 1. Each PPO update first trains the discriminator with the
Eq. (1) least-squares loss and input-gradient penalty against uniform samples
from the [admitted demonstration bank](priors/datasets/amp_demonstrations_001/manifest.json),
then runs the unchanged PPO update, then fits the velocity estimator by
supervised regression on the stored rollout. `test_amp_ppo` checks exact
parameter parity with stock PPO when the style weight and discriminator
updates are zero.

[`paper_networks.py`](paper_networks.py) implements the estimator [64, 32],
memory encoder [512, 256, 128], low-level actor [256, 128, 64], privileged
encoder [64, 32] and critic [512, 256, 128] over the existing 231-value
observation. The critic reads a 42-value privileged state: body velocity,
height, declared friction, six toe forces, zero perturbation slots and 13
non-tibia collision indicators. Friction and perturbations become measured
values with Step 7; the terrain latent belongs to Step 8. Privileged inputs and
the velocity label never enter the actor's observation groups.

Checkpoints add the discriminator, its optimizer and the bank identity, and a
load with another bank fails. `amp_learner.json` in each run records the
configuration and every declared decision. No native AMP run exists; the pilot
needs reward version 2, matching standing admission and the program lead's
approval. `DECISIONS` in each module lists the choices the paper leaves open.

## Prescribed tripod controller

You can evaluate the Zhang et al. sinusoidal tripod controller with
[`tripod_evaluate.py`](tripod_evaluate.py). [`tripod.py`](tripod.py) owns
the equations and contact state; [`tripod_config.py`](tripod_config.py) owns
parameters. You use `prepare --mode tripod --tripod-adaptation stop_stride`
with matching standing admission. [TRAINING](../docs/TRAINING.md#tripod-baseline)
keeps this controller as a comparison baseline. Issue #36 uses optimized
demonstrations for its motion bank.

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

You can inspect the [three passing motion screens](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_speed_lift_native_20260922_001/native_result.json)
and the [passing stop suite](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_stop_stride_native_20260922_001/native_result.json).
[STATUS](https://cornell-physical-intelligence.github.io/hexapod-cupi/#findings) records the native result of each variant. The
[closeout](../site/assets/tripod_reproduction_closeout_20260922_001/closeout.json)
records four interrupted qualification attempts. Full qualification, raised
clearance and terrain remain incomplete. You can inspect the
[forward comparison](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_qualification_recovery_20260922_001/forward_tracking_comparison.png)
and its [source identities and measurements](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_qualification_recovery_20260922_001/forward_tracking_comparison.json).
The [analysis provenance](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_review_corrections_20260922_001/analysis_provenance.json)
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
[stop suite](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_stop_stride_native_20260922_001/native_result.json)
has identical trace hashes: the six passing trials hold three distinct
trajectories. The [repeat identity record](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_review_corrections_20260922_001/repeat_identity.json)
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
the equations, CPU checks and frozen inputs; [STATUS](https://cornell-physical-intelligence.github.io/hexapod-cupi/#findings) holds each
native result. Declare a new variant and its CPU checks before native dispatch.
Keep changes to the paper equations separate from parameter calibration.

| Variant | Parent | Declared change | Record |
| --- | --- | --- | --- |
| `paper` | none | Eqs. (1)–(3) with the four period and hip choices above. Touchdown pitch returns to stance during support; lift offsets `(+0.35,-0.30)` and `(+0.45,-0.35)` rad. | [CPU result](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_20260921/result.json), [native corrections](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_native_20260922_001/result.json) |
| `geometry` | `paper` | Jacobian sweep coefficients with gains 1.0 and 1.1, 22/28 mm Jacobian lift and raised offset `(0,-0.15,+0.10)` rad. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_geometry_20260922_001/declaration.json) |
| `damping` | `geometry` gain 1.0 | Motor target adds `KD*dq_ref/12`. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_damping_20260922_001/declaration.json) |
| `retimed` | `damping` | Displacement clock `h` with 0.10 endpoint ramps. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_retimed_20260922_001/declaration.json) |
| `feedback` | `retimed` | Position feedback with gains 0.5 and 1.0, each within ±0.070 rad. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_feedback_20260922_001/declaration.json) |
| `liftoff` | `retimed` | Swing lift `g^0.75` with joint feedback off. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_liftoff_20260922_001/declaration.json) |
| `velocity` | `retimed` | Velocity feedback with gain 2, within ±0.070 rad. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_velocity_20260922_001/declaration.json) |
| `velocity_filtered` | `velocity` | 5 Hz one-pole filter on reference and measured velocity. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_velocity_filtered_20260922_001/declaration.json) |
| `pd_filtered` | `velocity_filtered` | Adds position feedback with gain 0.5. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_pd_20260922_001/declaration.json) |
| `overlap` | `pd_filtered` | Yaw support-overlap sweeps with the cubic return. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_overlap_20260922_001/declaration.json) |
| `startup` | `overlap` | Forward one-second standing hold, then a one-cycle stride ramp. The yaw touchdown residual returns over the rest of its half-cycle. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_startup_20260922_001/declaration.json) |
| `forward_overlap` | `startup` | Forward support-overlap sweeps with `h`; forward lift `(0.25,0)` rad and raised `(0.28,-0.10)` rad. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_forward_overlap_20260922_001/declaration.json) |
| `speed_lift` | `forward_overlap` | Low-mode forward lift `(0.26,-0.20)` rad through 0.05 m/s, interpolated to `(0.25,0)` at 0.10 m/s. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_speed_lift_20260922_001/declaration.json) |
| `stop_stride` | `speed_lift` | Stride decays over one cycle after the current swing, before the standing blend. | [declaration](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/05f706eb30ef98941b7f449228fc5b8f177d07b4/site/assets/tripod_stop_stride_20260922_001/declaration.json) |
