# Bounded moving residual PPO candidate

This is a new, unadmitted training consumer. It has not run on Spark. The CPU
tests exercise the actual RSL 5.0.1 PPO optimizer, frozen wave005/residual002 and
846/849 observation builder with explicitly synthetic physical measurements.
No completed episode or recovered failure is a physical acceptance receipt.

The initial command envelope is exactly forward 0.005 m/s and zero command.
One quarter of replicas hold quietly; other replicas receive 24 s forward then
20 s stop. The forward/left/yaw interface and all 846/849 fields remain intact,
but reverse, strafe, arcs and turning are not allocated by this candidate. The
successful left-turn result can support a separately reviewed command extension.
Current speed is a feasibility envelope, not the Stage2 performance objective.

## Learning and reset semantics

`MovingSession` leaves the exact source009 physical step, actuator, joints,
contact thresholds and SDK update cadence intact. Its training-only done hook
calls the original predicate once and captures original flags before deferring
the physical reset. Finite failures on active rows become terminal transitions:
the applied action remains learning data, its event penalty is 3 once, and its
bootstrap is zero. Recovery controls are never learned. A 44 s time limit uses
the final critic packet from that episode before reset, not the new reset state
or the value of the previous observation.

The original `_reset_idx` writes only ended rows. The source-bound sensor reset
kernel must have cleared those actual clocks and marked those rows outdated;
unaffected rows must retain their actual clocks. A fresh 2 s C2 joint-target
return to the canonical stance plus 2 s settling follows. The row must regain
the unchanged contact/motor conditions and pass measured wave initialization.
Only then are its reference, history and interval-angle state initialized.
Other rows keep their histories and episodes. The actor receives finite zero
placeholders for disabled rows, explicitly excluded from normalization,
returns and PPO minibatches. The initial resumed angle-difference sample is
marked invalid; raw SDK rates remain independently present.

Reference feasibility is predicted once, before the next actor packet. The
pending result is derived from the fully observed current reference/measurement
and next encoded command. A failed next knot closes the transition that led to
it; it is not charged to an action that was never executed. Pending state is
bound to episode, control age and command and cannot cross a reset.

Nonfinite physics/data/critic outputs, stale sensors, changed source/layout,
an applied torque beyond float 32(1.6), an unexpected native 90 s timeout, or a
failed reset recovery reject the entire current rollout. They do not become
successful episodes. The original evaluation always aborts on its unchanged
physical/contact/motor/quiet gates and has no recovery adapter.

## Reward and policy

The actor/critic are fresh 256/256/128 ELU MLPs,846/849 observations and 18 actions.
The actor mean head is exactly zero, its explicit per-joint initial standard
deviation is 0.02, and Adam starts fresh at 0.0001. The installed PPO loss uses
5 epochs,4 minibatches, gamma 0.99, lambda 0.95, entropy 0 and fixed learning rate.
Old smoke/direct-joint/velocity checkpoints are incompatible and rejected.
The finite position residual remains 0.02*tanh(action), with the existing
0.25 rad/s residual rate,2 rad/s² residual acceleration and formal 0.04 rad/20ms
combined target bound. No physical bound or position gain changes.

The narrow reward callback preserves every inherited non-scope term and weight.
Every nonzero requested motion activates requested-goal tracking/progress and
airtime; contact-governor pauses cannot redefine that user goal as zero speed.
A requested finite stop tracks its filtered reference tail without progress
reward. Settled reference quiet activates all four existing posture/raw-action/
joint-rate/target-rate costs. This does not prove reward optimality or SDK-rate
fidelity. The pause counterfactual is assigned-state evidence, not an observed
exploit. Interval-angle rates remain separate and do not replace raw-rate costs
or any acceptance metric.

## Bounded executable phases

`run_moving_ppo.py` accepts the same exact dependency flags as consumer002:

```
--source-root SOURCE 009 --run COMPLETED 009 --device-run DEVICE 001
--bridge BRIDGE 001 --observation-bundle OBSERVATION 005 --package STUDY 550
--standing CAMPAIGN/standing/admission.json --output CAMPAIGN/PHASE
--mode MODE --device cuda:0 --headless --info --kit_args KIT_ARGUMENTS
```

The early parser has `allow_abbrev=False`; `--device` is parsed only after
AppLauncher registers it. `--preflight-only` imports no AppLauncher and verifies
all frozen inputs. Actual phases require fresh same-campaign 32 × 1000 standing.
Outputs must be outside every immutable input and must not exist beforehand.

1. `calibrate`, output `calibration`:32 replicas,200 startup controls then
  1000 zero-mean and 1000 sampled quiet controls. Both original per-row scores
  must pass before an immutable new-lineage `initial.pt` is written.
2. `profile_32`:200 startup plus 512 actual zero-residual
  moving/stop-scheduled controls, full raw 400 Hz and 50 Hz evidence. This phase must
  complete without row outcomes before this entrypoint allocates training.
  Profile results are not full-cycle or learned-policy admission. Compare
  actual transition throughput, wall time and memory before choosing scale.
  The pilot stays 32 replicas. `profile_128` is separately bounded and optional
  after the initial learning evidence; it cannot block 32-replica training or
  grant permission to increase its replica count.
3. `train_10`:256 controls/update,10 updates, immutable `decision_010.pt`.
  This gives 51.2 s per replica beyond startup, enough for the individual 44 s
  motion/stop trajectory if it survives. It is a decision point, not Stage2.
4. Cold `evaluate_initial`, `evaluate_010` are each the unchanged 1×2400-control
  forward/stop retention screen. `quiet_010` adds 32 × 1000 quiet controls after
  startup. Any rejection stays rejected and prevents continuation.
5. `train_25` requires `--decision-receipt` binding the exact four preceding
  phase states and an explicit measured-improvement/no-regression review.
  It loads only the same-source 10-update checkpoint/normalizers/Adam, starts
  fresh physical episodes and adds 15 updates. It writes `decision_025.pt`;
  `evaluate_025` and `quiet_025` must then run cold. There is no automatic
  successor allocation. Gate passage alone does not prove improvement.

All training raw samples are bounded by 25 updates, retain per-row failure flags,
requested commands, reward components, reference state, sensor clocks and every
physical substep. Reset discontinuities have explicit episode IDs; the exporter
does not integrate their jumps as physical motion. Partial rows have explicit
field-presence masks rather than silently dropping a whole evidence column.

The outer host is intentionally not included as an executed job. Root should
reuse the exact source009 `run_owned` supervisor, both locks, bounded unit,
90 s AppReady deadline, owned-container cleanup and forecast restoration. Mount
this frozen consumer and every input read-only. Run phases sequentially, preserve
failed prefixes and validate exact state/checkpoint/source hashes before each
dependent phase. Root owns dispatch and the measured continuation receipt.

## Scope and next decisions

The residual can alter joint offsets, contact loading and body/foot tracking
using measured feedback; it is not rewarded for copying reference targets.
Its small finite range cannot change the gait order or rescue every reference
failure. If 10/25 updates do not show useful changes while retaining quiet motion,
do not keep training this setup unchanged. Faster support-pattern experiments
remain parallel work; eventual foothold/clearance/timing actions need an explicit
new observable reference state and physical admission.

Terrain will require measured/teacher terrain-relative foot/body heights,
surface normals, support eligibility and age/unknown/confidence fields, with
contact-confirmed anchor reconciliation. The current flat preload and reference
are not terrain-admitted. Adding those inputs changes the checkpoint schema;
this 846/849 state is instrumented simulator information, not a deployable
proprioceptive actor or a perception-qualified policy.

## CPU reproduction

From the repository root with the already isolated local RSL wheel:

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tmp/reference_residual_ppo_001/_deps \
  .venv/bin/python -m unittest discover -s tmp/reference_moving_ppo_001 -p 'test_*.py'
```

The real RSL regression uses 512 synthetic controls and exact save/load of all
17 Adam entries. The actual-class recovery test uses 415 controls, including a
single row failure and 200-control recovery; the other 31 histories stay intact.
All 23 CPU tests pass. CPU fixtures do not replace fresh physical admission.
All six RSL module hashes match the Spark image. The storage source was copied
from a never-started, CPU-only container, which was then removed by its exact ID;
its absence was independently read back. See `RSL_STORAGE_SPARK_READBACK_002.json`.
The actual entrypoint rechecks all six installed module hashes before use.
