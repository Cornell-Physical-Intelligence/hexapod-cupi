# Reference004: 7 mm reference lift and substep measurement

This prepares a bounded full-C physics screen, with no new physical result claimed. Source003 remains immutable. The only motion change is the sensor agent's reviewed nominal lift from 5 to 7 mm. Actual003's RF leg had real flight, passed the planned apex and showed descent, but lifted only 1.445 mm relative to the immediate preflight sample, below the unchanged 2 mm physical requirement. The 7 mm reference passed 23 synthetic command cases and 16 owner tests; the rejected 8 mm alternative and its joint-margin evidence remain preserved by the owner. Neither CPU result guarantees real clearance or torque.

Independent actual003 analysis also found 36.402 mm forward link displacement versus 25.879 mm integrated reported link velocity over the partial 8.34-second moving window. The 10.541 mm full-vector discrepancy is unresolved. The existing 50 Hz metric and 5 mm complete-run gate remain **unchanged**. New 400 Hz telemetry is diagnostic; it cannot silently replace a failing old measurement or be selected because it gives a more favorable result.

Installed SDK readback confirms `root_pos_w` is root-link pose, and PhysX computes `root_link_lin_vel_w` from COM velocity plus the angular-velocity contribution of the COM offset. These are intended matching actor-frame quantities. The actual PhysX kernel, concrete data-property getters and DirectRLEnv loop are preserved with source hashes in `installed_sdk_contract.json` and `installed_velocity_kernel.json`. No coordinate-frame correction is justified by the current evidence. Last-substep sampling, contact-position correction and other possible causes remain hypotheses until runtime substep data distinguishes them.

The observer temporarily wraps `scene.update`, calls the original exactly once and reads state afterward. It requires `_physics_handles_decimation == False`, eight real updates at 2.5 ms per 20 ms control, consecutive simulation counters and SDK articulation timestamps. It records initial state and every actual substep: root-link/COM position and velocity, raw SDK XYZW, angular velocity, joint velocity and computed/applied torque. It checks the eighth sample against the unchanged pre-reset control sample and restores the method on exit. It never calls `sim.step`, writes physical state, changes a target, replaces DirectRLEnv.step or manufactures eight samples from one backend update.

New outputs:

- `physics_substeps.npz`: complete raw samples, explicit relative/control/substep indices, SDK timestamps/counters and runtime joint names.
- `physics_control_integrals.npz`: per-control displacement, all-eight-sample left/right/trapezoid velocity integrals and means, per-joint torque extrema and excess counts.
- `physics_substep_review.json`: explicit link/COM displacement comparisons, interval definitions and substep motor extrema; no substituted acceptance verdict.

Partial and initial nonfinite samples remain exportable when instrumentation rejects a run. Existing `trace.npz`, state, failure, progress and quiet outputs remain intact. The first eight-substep measurement is after physics; captured torque is the actuator demand/applied value for that simulation interval, while position/velocity is post-update state. Initial capture is labeled separately and excluded from per-control torque counts.

Six focused CPU tests pass, independently reviewed: exact calls/order/timestamps/endpoints, method restoration, backend-decimation rejection, missing/extra/wrong-dt detection, an intentionally aliased last-sample signal whose full substep integral matches displacement, a hidden within-control torque spike, and initial/later nonfinite evidence retention. These tests validate measurement logic, not the cause of the actual003 discrepancy.

Startup, motor gains/limits, geometry, solver, residual core, contact classifications, host, and all existing physical/quiet/progress gates are unchanged. Source004 still needs fresh 32 × 1000 standing before its 1 × 2400 wave. Root owns exact remote verification, pause030 allocation, guarded dispatch and cleanup/restoration. No GPU, main or Git action is taken by this preparation.
