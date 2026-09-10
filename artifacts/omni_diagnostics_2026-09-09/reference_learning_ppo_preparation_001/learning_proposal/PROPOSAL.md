# Proposed admission for a recoverable moving learner

This is an unimplemented proposal for review. It does not change consumer001,
the minimal reset-context successor002, or any physical acceptance gate.

The next useful step is a bounded demonstration that the learner can correctly
recover from actual finite task failures, followed by the existing ten-update
decision point. Requiring an untrained 32-replica controller to have no task
failures in a timing profile prevents collecting the transitions that the new
masked learner is intended to improve. Fresh standing/calibration and strict
evaluation serve different purposes from training episode termination.

## What the actual failure establishes

At control 224, replica 6 had five contacting feet in total while LF was the
planned swing. RR force fell from 3.63 N to 0.598 N over seven controls. Only
four of the five required stance feet remained. The source-bound wave therefore
correctly returned failure code 4. The residual offset was exactly zero.
Replaying the actual measured/state pair on CPU reproduces the exact failure
vector. This is not evidence of a timestamp, observation or batch-index error.

The subsequent reset exception was separate: DirectRLEnv allocates `reset_buf`
inside inference mode. Successor002 preserves the same mode around the deferred
original reset. Its old-code regression reproduces the actual exception; the
fixed path completes a selected reset plus 200 recovery controls while retaining
the other 31 histories. This is CPU evidence; actual recovery is still untested.

## A separately named learning-admission campaign

Use a new protocol and host phase named `learning_recovery_32`. Do not relabel
or reinterpret the failed `profile_32` result as a pass. Preserve the same exact
source009 physics, 32 replicas, forward 0.005 m/s and stop envelope, actor schema,
calibration, finite residual, requested-goal reward and motor bounds.

1. Require fresh original 32-replica standing and zero/sample calibration again.
2. Run 200 startup plus 512 zero-residual controls. This supplies actual timing
   and allows finite support/reference failures to end only affected training
   episodes. Preserve every terminal sample and the original -3 event penalty.
3. Require actual proof of at least one correctly classified finite event and
   completed 200-control recovery before any moving PPO allocation. Replica 6
   is the measured expected trigger; if this deterministic replay instead
   differs, preserve the new physical sequence and inspect it. Do not inject
   a hidden physical disturbance or manufacture an event to satisfy the check.
4. Validate reset/clock/history/critic evidence described below. Native data,
   source, actuator or recovery failure still rejects the entire campaign.
5. Review the complete per-row event and recovery totals, valid transition share,
   progress and timing before allocating ten updates. A finite task failure is
   neither a physical admission nor by itself an infrastructure rejection.
   No numerical event-rate cutoff is proposed from the incomplete 24-control
   moving prefix; assigning one now would be unsupported. Excessive repeated
   resets or nearly absent moving experience require a measured review instead
   of automatic continuation.

The event-free rule is replaced only in this explicit *learning infrastructure*
admission phase. Formal standing, motor/contact, cold forward/stop and quiet
acceptance gates are not changed. A successfully recovered fall or support
failure must still appear as a failed episode in all summaries.

## Required actual evidence

The current raw exporter already covers every 400 Hz torque/angle/pose sample,
every 50 Hz measurement and outcome, actual reset clock readback, reference state,
episode identity, active-learning flag, command, event reward and update counts.
The independent checker reconstructs those paths and prevents reset jumps from
being integrated as motion. Keep raw SDK rates and interval-angle evidence
separate; neither is a substitute for the other's acceptance metric.

Before adopting the new protocol, add a small explicit terminal/recovery ledger
to its new consumer (not to either frozen predecessor):

- At each terminal or timeout, record the actual final critic packet, episode,
  validity mask and value used by the learner; true terminal bootstrap must be
  zero, timeout bootstrap must use that final same-episode packet. The first
  post-reset packet cannot be used in its place.
- At reset and the first completed recovery, record the selected row's actor
  packet, history-valid flags, interval-angle validity, episode and step index.
  Record corresponding history/episode metadata for unselected rows before and
  after the reset to prove isolation without exporting every ordinary packet.
- Reconcile these entries with the raw event/reset-clock ledger, per-update
  normalization and learning counts. No recovery row may enter normalization,
  GAE or PPO minibatches. A failed recovery remains fatal.

These additions are evidence, not new observations or control state; the actor
stays 846/849. Consumer001 does not export those tensors, so its raw clock and
episode evidence alone cannot prove their GPU contents correct.

## Ten-update decision and speed limitations

If actual recovery/integrity passes review, keep the initial allocation at
32 replicas, ten updates and 256 controls per update. This is 51.2 simulated
seconds per replica, long enough for a surviving 44-second motion/stop episode;
the old 24-controls/update smoke could not establish that. Preserve actual
timeout-bootstrap evidence when the first complete episode ends.

Cold initial and updated 2400-control forward/stop evaluations plus 32-replica
quiet must still pass the original acceptance gates. Compare actual progress,
all six confirmed foot cycles, torque/substep peaks, slip, failure/reset count,
stop latency, quiet target motion and per-joint ranges; gate passage alone does
not prove a useful improvement. Keep immutable checkpoints at update 10 and
allow 25 only after a separately bound review. Do not train failed reverse,
strafe or arc directions under this initial command contract.

This can establish whether feedback residual learning corrects small support
and body-tracking errors. It cannot turn the 0.005 m/s scripted reference into
project-speed locomotion: its fixed finite offsets cannot substantially change
gait timing or support order. The faster support-pattern work remains critical
parallel work, with its own physical proof and later observable timing/contact
action contract. Terrain will additionally need qualified terrain-relative
height, surface and support-state inputs rather than a flat-reference relabel.
