# Motor target scheduling candidate

This document describes the isolated target-scheduling change frozen at
`9cd8d4c`. The later named Kd 0.30 controller experiment is documented in
[the training runbook](MKII_FOURBAR_TRAINING.md); it preserves this scheduler
and records its separate gains in the runtime manifest.

The 50 Hz policy still produces the same eighteen clipped, soft-limit-bounded,
0.040 rad / 20 ms slew-limited motor endpoints. The new candidate delivers each
endpoint to the explicit PD controller as sixteen equal position increments at
800 Hz. The final increment equals the endpoint exactly; the intermediate
targets stay inside the segment joining the two endpoints. Passive joints never
receive position commands. Reset changes only the selected environments.

The old zero-order hold can change the position target by 0.040 rad in a single
physics update. At the provisional 30 N·m/rad gain this changes requested
proportional torque by 1.2 N·m per motor instantaneously. Linear delivery limits
the target-induced P increment to 0.075 N·m per physics update. Actual torque
also depends on measured position/velocity and remains subject to the unchanged
RS05 speed/voltage and overload limits.

Desired velocity remains zero. Adding a 2 rad/s velocity feedforward with the
0.6 N·m·s/rad derivative gain would add an immediate 1.2 N·m term, defeating the
purpose of this isolated comparison. No motor gains, inertia, friction,
contact parameters, mechanism constraints, validation motions or acceptance
thresholds change.

The motivating v5 128/1 group trace records four environment-samples of zero
support, each exactly 1.25 ms, at the first physics update of the second negative
knee target step. Reconstructed pad bottoms are micrometres above the plane.
This supports testing target delivery; it does not prove the candidate will pass
or reproduce every event in the full 32-environment sequence.

This scheduler is implemented in the simulation environment and covered by
independent endpoint, reversal, limit and partial-reset tests. Before hardware
transfer, the CAN/embedded control team must implement and measure equivalent
setpoint scheduling and latency; the existing runtime action adapter produces
50 Hz endpoints only. No hardware parity is claimed. Every new qualification
and learner records the scheduling semantics in its runtime manifest, and old
admission reports/checkpoints cannot authorize this source revision.
