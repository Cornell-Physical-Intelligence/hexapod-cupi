# CPU surrogate of the native environment

You use this subpackage to try a reward or learner design on a laptop before you spend a
native run. It runs this repository's task, reward, PPO adapter and gate code and replaces
Isaac's rigid-body dynamics and floor contact with MuJoCo 3.14.0. **Native runs are the only
acceptance evidence.** A surrogate result supports a design choice and admits no checkpoint.

`locomotion/prepare.py` freezes `locomotion/*.py` without descent, so no native pack copies
this directory and no native source hash changes. The kernel imports nothing from it. The unit
suite checks both facts.

| File | Purpose |
| --- | --- |
| [`env.py`](env.py) | `SurrogateEnv` with the interface of `LocomotionEnv`, `ContactModel`, the touchdown model and an offscreen renderer. |
| [`train.py`](train.py) | PPO through `ppo.ppo_config`, `VanillaVecEnv` and RSL-RL, with the learner flags of `locomotion/train.py`. |
| [`evaluate.py`](evaluate.py) | The learning probes through `locomotion.evaluate`, a contact-model consensus for static probes and three stress flags. |
| [`replay.py`](replay.py) | Open-loop trace replay, scripted noise and reward scoring of a stored surrogate or native record. |
| [`calibrate.py`](calibrate.py), [`self_check.py`](self_check.py) | Comparisons with native reference copies that you supply, the writer of the two training extracts and a check of your machine. |
| [`provenance.json`](provenance.json) | The scratch sources of this port with their hashes, the changes of the port, the origin of each native reference and the hash of each record. |
| [`records/`](records) | The self-check record and the scratch-against-port equivalence record behind the tables below. |

## Train and evaluate

Run each command from the repository root after `uv sync --locked`. Keep run directories
outside the checkout.

```sh
# Reward v4 with the learner options of the native runs of 2026-10-05.
uv run python -m locomotion.surrogate.train --output <new dir> --task locomotion.task_v4:TrainingTaskV4 \
    --action-mean tanh --observation-normalization none --observation-scaling fixed \
    --command-segments bootstrap --learning-rate-max 3e-4 --action-std 0.15 --action-std-final 0.05 \
    --gait-clock 60 --action-smoothing mean2 --velocity-noise 0.5 --updates 2000 --threads 2
# A surrogate or native checkpoint on the forward, quiet and stop probes.
uv run python -m locomotion.surrogate.evaluate --checkpoint <run>/checkpoint_update002000.pt \
    --output <new dir> --video none
# Stress options on one probe.
uv run python -m locomotion.surrogate.evaluate --checkpoint <checkpoint> --output <new dir> --video none \
    --cases learning:forward_0.05_to_stop --velocity-input-gain 1.25 --action-mean2 --foot-load
# Score a stored probe record under a reward; check a machine; run the unit suite.
uv run python -m locomotion.surrogate.replay score --telemetry <probe>/telemetry.npz --command 0.05 0 0 --rewards v2 v3
uv run python -m locomotion.surrogate.self_check --output <new dir> [--native-ref <reference dir>]
uv run python -m unittest discover -s locomotion/surrogate/tests
```

- The trainer writes the native record layout: `state.json`, `ppo_config.json`, one
  `metrics.jsonl` row per update and a checkpoint pair every 50 updates. `--task` takes
  `module:Class` or `module:factory(text)`. `--contact FIELD=VALUE` overrides one
  `ContactModel` field. `--resume` continues the update count and the deviation schedule of a
  checkpoint. Resume checks the checkpoint hash and its paired JSON record
  before loading weights. The source hashes and training configuration must
  match, including the task, wrapper, contact model and replica count. The
  identity also hashes the source file of each task, wrapper, distribution,
  network or algorithm class that lives outside `locomotion/*.py` and this
  package, so an edit to such a class stops a resume. The
  record pins the parent checkpoint and identity. The deviation schedule keeps
  its original horizon; additional updates hold its final value after that
  horizon. The simulator and random generators restart from the recorded seed,
  so resume does not replay an uninterrupted trajectory. Use the original source
  checkout for a historical checkpoint.
- A reward with a contact-schedule term needs `--gait-clock` equal to its period. The trainer
  stops otherwise, as `locomotion/train.py` does.
- The wrapper option `command_segments: bootstrap` needs an algorithm class derived from
  `locomotion.rate_schedule:CommandBootstrapPPO`. The trainer stops when a
  `--ppo-config-overrides` value pairs that option with another algorithm: stock PPO would
  train the pre-action target under a bootstrap record.
- The evaluator reads the checkpoint's JSON record, so a native checkpoint needs no learner
  flag. It builds the policy input as the evaluation branch of `locomotion/train.py` does:
  fixed scales, gait clock, actor mean and the recorded two-control mean, with no velocity noise.
- `--velocity-input-gain` multiplies the 90 joint-velocity inputs. `--action-mean2` applies the
  two-control mean to a checkpoint that trained without it. `--foot-load` writes the floor
  force on each foot after each control. A wrapper with its own `policy_for_evaluation`
  replaces the policy path, and the evaluator refuses the two stress flags for it.
- Quiet and stop probes run under three contact models. `contact_consensus` gives `pass` or
  `fail` where the three agree and `unresolved` otherwise. Agreement shows that a surrogate
  verdict does not rest on the friction cone or the friction value. It is no native verdict;
  see limit 1.
- `--video first` renders through MuJoCo's offscreen renderer. Pass `--video none` on a
  machine without a GL context.
- Each tool takes `--threads` from 1 to 6. Each tool refuses an existing result: a run
  directory, a `calibrate` label, a sweep variant or a `replay score` output file.
- `self_check` exits 1 where the shaft classification fails, where a training ends with a
  status other than its expected one or where the evaluation fails. It writes its record after
  each stage. The native comparisons are measurements and set no bound.

## Native references

`calibrate` and `self_check --native-ref` read seven files under one reference directory. The
repository holds none of them. Spark holds each origin under `/srv/cupi/hexapod`, and
`provenance.json` holds each hash.

| Reference file | Origin on Spark |
| --- | --- |
| `standing_one/standing/substeps_000.npz`, `substeps_009.npz` | `evidence/legacy_runs/restart_20260914/amp_20260923_001/standing_one/run/standing/standing/` |
| `tripod_trial_000/control_trace.npz` | `runs/james/flat_pilot_20261002_001/tripod_reference/run/standing/trial_000/` |
| `noise_probe/plan.json`, `noise_probe/telemetry.npz` | `runs/james/ppo_noise_probe_20261004_001/run/standing/probe/` |
| `metrics_A.jsonl`, `native_noise_updates_6_20.json` | Written by `calibrate extract` from `run/standing/metrics.jsonl` of four runs under `runs/james/`: A `flat_pilot_20261002_001/ppo_mlp`, B `ppo_bounded_mean_20261002_001`, C `ppo_no_running_norm_20261003_001`, D `ppo_immediate_tracking_20261003_001`. |

```sh
uv run python -m locomotion.surrogate.calibrate extract --output <reference dir> \
    --run A=<A metrics.jsonl> --run B=<B metrics.jsonl> --run C=<C metrics.jsonl> --run D=<D metrics.jsonl>
```

The scratch study read two extracts that had no writer. The command above replaces them. On
2026-10-05 its output from the four Spark files equalled the scratch extracts within 3e-16 on
each value that the checks read.

## Calibrated contact model

The scratch calibration of 2026-10-04 set each `ContactModel` default. The port changes none;
[`records/equivalence.json`](records/equivalence.json) records the equality check. The action
path, gains, limits and masses carry no tuning.

| Parameter | Value | Native evidence |
| --- | --- | --- |
| `position_blend` | 33/64 | Free-flight substeps show `q += dt (v_old + 0.5156 (v_new - v_old))`. |
| `impact_euler` | true | A replica with a new floor contact keeps the Euler position update for that substep. |
| `noslip_iterations` | 4 | Settled joint angles differ from native by 1.8e-4 rad at most and 1.4e-4 rad in the mean; the static torque norm is 1.129 N·m on both sides (source S below). |
| `cone`, `friction` | elliptic, (1.0, 0.005, 0.0001) | Native friction is 1.0. |
| `solref`, `solimp` | (0.005, 1), (0.9, 0.95, 0.001, 0.5, 2) | The stiffest standard MuJoCo contact at 400 Hz, with critical damping. |
| `limit_margin` | 0 | A loaded joint rests 0.5 mrad beyond its stop, so the 2e-6 rad termination fires. |
| `touchdown_overshoot`, `overshoot_phase`, `overshoot_gain` | true, (0.75, 0.90), 15 | See "Touchdown model". |
| `joint_speed_clamp` | true | The step clips joint speed to 50.265 rad/s, as PhysX does. |

### Touchdown model

`SurrogateEnv._touchdown` adds the friction overshoot of a late toe touchdown. The scratch
study measured it on the 400 Hz capture of the native tripod trial (source T below):

1. The capture holds 65 toe touchdowns. Most stop the toe in the substep of contact. The
   others leave that substep with a slide against the incoming slide, at up to 2.2 m/s and a
   tibia rate up to 16.5 rad/s.
2. The reversed slide equals a share of a full Coulomb impulse: friction times the normal
   impulse through the toe's tangential mobility. The share follows the phase of the substep at
   which the toe reached the floor. It is 0.05 or less for the 25 touchdowns above 14 N that
   arrive before phase 0.70, 0.10 for one arrival at phase 0.79, and 0.75 to 1.03 for the five
   arrivals at phases 0.87 to 0.96.
3. The reversed slide stays below 18 times the incoming slide.

The model applies this rule to each new toe-cap contact. MuJoCo reports a contact one substep
after the toe crossed the floor, so penetration over approach speed gives the arrival phase.
The share rises from 0 at phase 0.75 to 1 at phase 0.90, and `overshoot_gain` caps the
reversed slide. The impulse acts at the lowest toe point through the full mass matrix and
replaces the tangential impulse of that substep. The model holds no random number.
`--contact touchdown_overshoot=False` keeps MuJoCo's soft contact alone.

## Measured parity

Three sources carry the numbers below.

- **S** is [`records/self_check.json`](records/self_check.json): the self check of 2026-10-05
  with this package and the native references above.
- **P** is [`surrogate_parity.json`](../../site/assets/ppo_walking_diagnosis_20261005_001/surrogate_parity.json):
  four retained native checkpoints, one trial per probe and simulator.
- **T** is the scratch study of 2026-10-04 at revision `875a3123`. This port reran no T
  value. `provenance.json` holds the hash of the scratch README that states each one, and
  `records/equivalence.json` shows that the scratch and the ported environment step to equal
  states.

| Measurement | Native | Surrogate | Source |
| --- | --- | --- | --- |
| Total mass | 7.466088 kg | 7.466088 kg | S |
| Neutral standing: root height, static torque norm | 0.09576 m, 1.129 N·m | 0.09575 m, 1.129 N·m | S |
| Reset with zero action: first toe force | 32.5 ms | 35.0 ms | S |
| Tripod open-loop replay: forward speed, planar error, yaw error | 0.0542 m/s, 0.0108, 0.0128 | 0.0543 m/s, 0.0104, 0.0121 | S |
| Noise probe, 12 noisy cells, ratio to native: planar speed, vertical velocity, roll-pitch rate, yaw rate | 1 | 1.02 to 1.06, 0.94 to 1.02, 0.91 to 1.02, 1.00 to 1.06 | S |
| Noise probe, ratio to native: applied torque RMS, joint velocity RMS | 1 | 0.88 to 1.01, 0.31 to 0.99 | S |
| Probe at neutral, std 0.15: tibia rate RMS, joint acceleration norm | 1.22 rad/s, 198.8 rad/s² | 0.83 rad/s, 173.4 rad/s² | S |
| Probe at crouch, std 0.15: the same two | 0.69 rad/s, 131.8 rad/s² | 0.45 rad/s, 121.4 rad/s² | S |
| Probe at raised femurs, std 0.15: the same two | 1.68 rad/s, 270.0 rad/s² | 0.82 rad/s, 168.9 rad/s² | S |
| Probe at std 0, four poses: joint velocity RMS | 0.002 to 0.026 rad/s | below 2e-6 rad/s | S |
| Reward v2 per control at std 0.15: neutral, raised femurs | -0.250, -0.245 | -0.223, -0.148 | S |
| Reward v1 per control, 16 cells: largest gap | 0 | 0.003 | S |
| Training with default options against native run A, updates 6 to 20: reward | -0.2114 | -0.2160 | S |
| Forward probe, action-wall update 1500: speed, planar error, verdict | 39.9 mm/s, 0.0264, fail | 40.7 mm/s, 0.0264, fail | P |
| Forward probe, action-wall update 2000 | 46.8 mm/s, 0.0222, pass | 47.0 mm/s, 0.0224, pass | P |
| Forward probe, velocity-noise seed 20260917 | 53.6 mm/s, 0.0077, pass | 54.6 mm/s, 0.0085, pass | P |
| Forward probe, velocity-noise seed 20260918 | 50.9 mm/s, 0.0080, pass | 52.2 mm/s, 0.0084, pass | P |
| Quiet probe verdict, the four checkpoints in that order | fail, fail, pass, fail | pass, pass, pass, pass | P |
| Stop probe verdict, the four checkpoints in that order | fail, fail, fail, fail | fail, fail, pass, pass | P |
| Stop probe, action-wall update 2000, input gain 0.9, 1.0, 1.25 | no native run | pass, fail, fail under the default contact model | P |

The pass state agrees on 7 of the 12 probes in P and the failed-bound set on 5: each of the
four forward probes, one quiet probe and no stop probe. Read the two stop rows that show
`fail` on both sides as no agreement. For update 1500 native fails the joint range, the joint
velocity, the target step and the six-toe count, and the surrogate fails three
torque-saturation bounds. For update 2000 both fail the joint velocity and the target step,
and native adds the six-toe count.

At gain 1.0 the default contact model alternates its action on consecutive controls (lag-1
autocorrelation -1.00) and fails. The pyramidal-cone and friction-0.9 runs pass that probe, so
its consensus is `unresolved`. Gains 1.25 and 1.5 fail in each of the three runs. Gains 0.5 and
0.75 pass under the default model with an `unresolved` consensus. A forced two-control mean
passes at gains 1.0 and 2.0 in each run.

## Known limits

Each limit names the decision that it bears on.

1. **Rest behaviour (P).** Native fails seven of the eight quiet and stop probes. The
   surrogate's default contact model fails two. `contact_consensus` resolves five probes,
   reports `pass` on four that native fails and equals native on one. Read no surrogate pass
   on a quiet or stop probe as a native pass, with or without a consensus.
2. **Raised-femur poses under noise (S).** Native joint rates exceed the surrogate's (joint
   velocity ratio down to 0.31). Reward v2 pays a motionless robot 0.05 to 0.10 more per
   control in the six noisy raised-femur cells than native does. Score a pose term on the
   native probe record with `replay score` before you trust it, and treat a drift of a
   surrogate policy toward raised femurs as no evidence for or against a design.
3. **Squared joint rate in each pose (S).** At std 0.15 the tibia rate RMS is 0.83 against
   1.22 rad/s at neutral and 0.45 against 0.69 rad/s at crouch. A term in squared joint rate
   therefore reads 0.46 and 0.43 of its native value in the two poses that agree best. T
   records a reward v4 `quiet_joint_rate` term of -0.085 against -0.140 at neutral, under the
   reward v4 of its revision.
4. **Tibia collapse (S).** Native noise folds a tibia more than 0.25 rad below its target in
   five probe cells: neutral std 0.15 (1.5 events per robot-minute), raised femurs std 0.05
   (2.25) and std 0.10 (0.75), beyond-bound femurs std 0.05 (0.75) and std 0.15 (2.25). The
   surrogate records none. Std 0.05 is the last value of the deviation schedule above.
5. **Static cells (S).** At std 0 native shows a residual joint velocity of 0.002 to 0.026
   rad/s. The surrogate stays below 2e-6 rad/s.
6. **Reset landing (S, T).** The first toe force arrives one substep late. T records that the
   zero-action landing costs 1.21 times the native sum of squared vertical velocity.
7. **Reward v4 on the native probe record (S).** The record of 2026-10-04 holds no
   `computed_torque_nm`, so reward v4 has no native score at this revision and no reward v4
   noise floor has native support.
8. **Joints near a stop (T).** A pose with a tibia within 0.06 rad of its stop ended by the
   joint-limit rule on one simulator and not on the other, in both directions: native ended a
   trial that the surrogate held 0.060 rad from the stop, and the surrogate ended a trial that
   native held 0.037 rad from it. Treat a `joint_limit_margin` below 0.06 rad as a native
   termination risk.
9. **Bound checks.** The native check `native_capture_or_original_physical_bounds` is
   unverified here; each summary says so.
10. **Tripod loads (T).** The requested torque peak of the tripod replay is 2.58 against 2.05
    N·m, and the over-rating fraction of the closed-loop tripod at 0.10 m/s is 0.0004 against
    0.0017. Reward v4 charges over-rating torque (`over_rating_weight` 10 at this revision),
    so confirm that term on native.
11. **Torque split under noise (T).** At neutral the femur share of the applied torque is 0.83
    to 0.90 of native and the tibia share 1.07 to 1.20. The total at neutral agrees within 1.2
    percent (S). A per-joint torque term sees the shift.
12. **Training statistics (T).** Joint-limit termination counts spread by a factor 1.2 to 4.5
    between seeds of one design, so the surrogate ranks no two designs whose counts differ by
    less than a factor 3. Treat a speed gap below 6 mm/s after update 400 as unresolved. With
    unscaled observations, the trainer default, the KL after update 1 is 0.8 to 1.2 against 0.36
    to 0.39; updates 2 to 20 agree. Pass `--observation-scaling fixed` for a surrogate study.
13. **Added telemetry.** `surrogate_toe_cap_force_min_400hz`, `surrogate_shaft_force_max_400hz`,
    the state keys `toe_cap_force_n` and `shaft_force_n`, and the touchdown and clamp counters
    exist on the surrogate alone. A reward that reads one cannot run on native.
14. **Geometry.** Convex hulls replace the collision meshes. The model has no self-collision,
    and the floor is a plane. Contact telemetry holds normal forces alone.
15. **Coverage.** Each native reference is one trial or one seed. The T values rest on scratch
    scripts that this port left out: the 800-update trainings, the 54 probes of eight older
    checkpoints and the closed-loop tripod. `provenance.json` names each script and each
    validation directory with its hash.

Use the surrogate for three questions:

- forward travel of a gait (within 1.4 mm/s and 0.0008 planar error on the four checkpoints in P);
- noise floors of body velocity, body rate, target step, height and total torque (ratios 0.88
  to 1.06 over the 12 noisy cells in S);
- the first updates of a learner comparison with fixed observation scaling.

Confirm on a native run each quiet or stop verdict, each term in squared joint rate or joint
acceleration in each pose, each over-rating or per-joint torque term, each reward v4 noise
floor and each walking result.
