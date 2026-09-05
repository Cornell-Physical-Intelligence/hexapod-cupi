# First flat-ground policy evaluation

This is a proposed evaluation, not an implemented runner or a claim that PPO has
started or passed. After full PPO starts, evaluate only completed, hash-verified
checkpoints; repeat on its final verified checkpoint. A finite 100-step inference
probe establishes software continuity, not walking skill. See the
[training runbook](MKII_FOURBAR_TRAINING.md) for admission and continuation.

## Preserve the trained system

Load the exact archived training source, selected USD/dependencies, admission,
motor contract and runtime manifest. Retain the trained observation/action order,
target scheduling, physics timestep, solver, contacts, reset and headroom rules.
Use deterministic actor inference with normalization in evaluation mode, no
exploration sampling and no optimizer updates. Verify learner-state hashes before
and after evaluation.

The current checkpoint identity hashes all `tools/*.py` and `isaaclab/*.py` source
recursively. Adding an evaluator to a new checkout changes that identity. A future
evaluator must therefore run **outside a read-only frozen training snapshot** and
import the task/guards from that snapshot. Verify its unchanged training contract
against the checkpoint sidecar; separately archive/hash the evaluation driver,
scenario configuration and dependencies. Record both identities and the actual
loaded module paths. Do not rewrite checkpoint lineage or claim that edited source
matches it. The external command-schedule adapter is an explicit evaluation
intervention: set commands before constructing policy observations, suppress random
command resampling, and record the commands actually consumed at every transition.

## Fixed first comparison

Run 14 scenarios concurrently, one environment per scenario, for 10 seconds each;
repeat with seeds **101, 202 and 303**: 42 trials and 1,500 batched policy steps
per checkpoint/baseline.
Keep the admitted zero-jitter reset and initial motor budget. Deterministic repeats
may coincide; these are reproducibility checks, not 42 independent terrain samples.
Compare the earliest verified scratch checkpoint, the selected full-run checkpoint
and a zero-action standing baseline under the same scenarios.

Commands are `(forward, left, yaw)` in the instantaneous anatomical navigation
frame: forward is body −Y, left +X, positive yaw about +Z.

| Scenarios | Commands and schedule |
| --- | --- |
| Stand (1) | Zero for all 10 s |
| Signed axes (6) | Forward ±0.10 m/s, left ±0.10 m/s, yaw ±0.20 rad/s; other components zero |
| Diagonals (4) | All sign combinations of `(0.10/√2, 0.10/√2, 0)` m/s |
| Reversals (3) | Each signed axis above: 2 s zero, 3 s positive, 3 s negative, 2 s zero |

For each non-reversal motion: 2 s zero, 4 s commanded motion, 4 s zero. These
speeds are within current training ranges; they are test inputs, not mission
requirements. Record startup, motion and stopping windows separately. Capture the
terminal state/reason before automatic reset, mark that trial interrupted, and
exclude subsequent respawn motion from its tracking statistics. Preserve failure
counts and exposure time rather than reporting only surviving trajectories.

## Evidence and next decision

Retain command/trajectory plots and per-trial results, including median, p95 and
worst errors across completed motion windows:

- Signed velocity response and forward/left/yaw tracking RMSE; unintended lateral
  motion/yaw. Integrate commands into a world-frame reference from the initial
  heading to report cross-track error and displacement without mixing frames.
- Stop displacement and residual velocity; reversal sign-crossing delay, recording
  “not reached” when appropriate. Report actual velocity, not just reward.
- Roll/pitch and base acceleration RMS, height range, loaded-foot slip, support
  distribution, nonfoot contacts, falls, resets and termination reasons.
- Raw demand versus applied torque, clipping, continuous overload, burst exposure,
  minimum headroom and motor speed; label electrical-current estimates as proxies.
  Include reward components to expose a policy that earns standing rewards while
  ignoring nonzero commands.

Retain the frozen trainer's physical guard at **every physics substep**, before
resets, with its existing closure/axis/envelope/finite-state bounds and exact sample
coverage. A physical-model violation invalidates the evaluation; falls and poor
tracking are measured policy outcomes. Do not loosen those bounds for a checkpoint.

Evidence of useful learning is signed response in every commanded direction,
improvement over the paired baselines, and controlled stops/reversals without
hiding failures in aggregate reward. Publish deficiencies by direction; no new
numeric locomotion pass threshold is declared here. Survey cross-track tolerance,
coverage, terrain/slope/obstacle limits, endurance and payload stability remain
separate requirements to agree with the teams. This flat-ground screen establishes
neither all-terrain capability nor hardware or autonomous survey readiness.
