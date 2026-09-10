# Bounded reference plus position-residual PPO smoke

This is a new consumer of the immutable wave005 reference, residual002 action
servo, and 846/849 simulator observation. It does not change their prototype
training flags, production robot, old policies, or acceptance thresholds.
It authorizes only the separately identified, admitted two-update experiment.

The reference has demonstrated slow full-C forward motion and stopping in
actual reference009. The actual device001 run has demonstrated both 1 and 32
replicas with the complete observation and sensor clock contracts. Neither
result establishes omnidirectional PPO or terrain readiness.

## Execution contract

A guarded host must first run the unchanged source009 standing entrypoint for
32 replicas and 1000 controls in the new campaign's `standing/` directory.
All original physical and all 32 quiet gates must pass. The external consumer
then runs these sequential phases, stopping on any failure:

1. `smoke`: 32 replicas, 200 canonical startup controls; 1000 deterministic
   zero-mean calibration controls; 1000 sampled calibration controls; exactly
   two PPO updates of 24 controls; strict checkpoint reload; 1000 deterministic
   quiet controls. Each calibration and final quiet report scores every replica.
   Total: 3248 controls, 25,985 substep samples including the initial sample.
2. `evaluate_initial`: a cold single replica, 200 startup controls followed by
   1200 controls at +0.005 m/s forward and 1000 stopping controls, using the saved
   initial checkpoint. Total 2400 controls.
3. `evaluate_final`: the identical cold motion using the final checkpoint.

No phase automatically schedules more training. Any failed calibration prevents
both checkpoint initialization for optimization and PPO updates. An online
reference, contact, freshness, torque or reset failure aborts that phase while
retaining its failing pre-reset data. A completed PPO run that loses quiet
standing remains rejected and cannot enter the matched retention phases.

```
python run_residual_ppo.py --mode smoke \
  --source-root /source009 --run /completed_reference009 \
  --device-run /completed_device001 --bridge /device_bridge001 \
  --observation-bundle /observation005 --package /admitted_study \
  --standing /campaign/standing/admission.json --output /campaign/smoke \
  --headless --device cuda:0
```

For retention, choose `--mode evaluate_initial` or `evaluate_final`, set the
corresponding fresh output directory, and add `--smoke-output /campaign/smoke`.
Inputs must be read-only, complete and exactly hash-bound. The source009 guarded
host, job locks, isolated container cleanup and forecasting restoration remain
host responsibilities. This bundle does not launch a GPU job itself.

`physical_contract.verify_inputs(args, require_standing=False)` verifies all
immutable dependencies before a forecasting pause. The normal entrypoint
requires the same-campaign fresh standing receipt. `validate_result(output,
identity)` gives the host a strict, standard-library result/checkpoint contract
and file hash receipt. It validates consumer-scored per-replica reports; it does
not claim to independently recompute the raw numerical physics trace.

## Policy and action

PPO uses fresh 256/256/128 ELU actor and critic networks, fresh normalizers and
Adam, a zero actor mean head, fixed learning rate 0.0001, and zero entropy reward.
Initial Gaussian standard deviation is 0.02 per joint. That implies approximately
0.0004 rad small-signal residual goals, but is only a proposal until actual
sampled calibration passes. Exploration is measured rather than assumed safe.
The command interface remains forward/left/yaw. This first optimization is
stand-only; its retention envelope is forward 0..0.005 m/s and stopping. Wider
commands require the separately prepared directional physics evidence.

Actions request finite position offsets `0.02*tanh(action)` about the contact
reference. They do not integrate velocity indefinitely. The existing residual
servo bounds position, velocity and acceleration; formal 0.04 rad/20 ms is kept
separate from the historical 0.03 diagnostic intervention. The servo retains
feedback at zero body command. The original PD, torque caps, reward computation,
physics configuration and quiet gates are unchanged. A residual can deliberately
be nonzero, so the old zero-residual screen's zero-lag check is replaced by the
explicit new action contract's 0.02 rad residual radius; this is a new lineage,
not a relaxed zero-residual qualification.

The first requested command is copied and encoded only after canonical startup.
A queued command enters the next observation after the current action has been
executed with its previously observed command. The following action/reference
uses that new command. This explicit 20 ms queue latency never rewrites cached
same-step history. A latched failure cannot advance either reference or physics.

Observations are instrumented simulator state, including full reference state,
contact/preload/landing/stop state, executable residual state and history-valid
flags. Raw reported joint velocities remain alongside separate interval-average
angle differences. Their physical discrepancy remains unresolved; neither is
substituted into the original scoring gates. There is no additional actor input
noise in this bounded integration proof and no deployment qualification.

## CPU verification and limits

`real_rsl_cpu_regression.py` executes actual RSL-RL 5.0.1 with the frozen 846/849
encoder and real bounded residual core. Its fixed measured joints and contacts
are an explicitly synthetic fixture. Two updates change actor weights; strict
reload restores actor, critic, normalizers and all 17 Adam state entries, with
exact deterministic action equality. Wrong sources, schemas and checkpoint or
sidecar overwrites are rejected. The five RSL implementation files match the
installed Spark hashes in `RSL_SOURCE_PARITY.json`.

The physical scoring regressions use documented, byte-value-preserving subsets
of actual009 traces. They reproduce the original 32-replica quiet and full
forward/stop verdicts and reject a single noisy replica, nonfinite metric, wrong
final contact mode, source or campaign. `inputs/PROVENANCE.json` binds the full
original traces and the derived subsets.

```
PYTHONPATH=<isolated RSL501 dependencies> python real_rsl_cpu_regression.py \
  --observation-bundle /observation005 --output /fresh_CPU_report
```

The independent command/history/RSL receipt is separate; it does not qualify
this new physical entrypoint. The next Spark dispatch still needs root review,
the actual calibration and the terminal evidence. Stage 2 is not complete.
