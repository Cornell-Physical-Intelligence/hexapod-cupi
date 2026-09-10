# Terrain reset diagnosis001

This bounded source preserves attempt003's incorrect quaternion unchanged so the
simulator can report the actual root orientation and termination reasons before
automatic reset. It first requires a fresh exact-plan flat32×1000 standing
admission, then executes five zero-command terrain controls. The diagnostic
wraps the original `_get_dones` method, calls it once, records the pre-reset
state and returns the unmodified termination/truncation tensors. It does not
change limits, thresholds, controller code, rewards, geometry or reset rules.

The output explicitly uses `diagnostic_completed`, `gate.passed=false`,
`terrain_standing_passed=false`, and `ready_for_terrain_training=false`.
It cannot admit standing or training. A CPU fake-environment check verified
exactly five original done calls and five captured pre-reset records.

Frozen source: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_robot_reset_diagnostic_source_001`.
The661-file source manifest SHA-256 is
`93ab3251b736f7828491648c837390ad5fafe664218a3581906f20402fe4a7dc`.
Its parent is frozen `terrain_robot_source_003`, preserving the published9d61077
study runtime and exact original full-review training plan. The original30
fixture catalog and actual fixture attempt003 admission are unchanged. No
derived fixtures, policy checkpoint, optimizer or production four-bar are used.

Output: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/terrain_robot_reset_diagnostic_001`.
Unit: `hexapod-terrain-robot-reset-diagnostic-001-20260909.service`.
Forecast pause016 has exact-unit exit restoration and a30-minute fallback.
Both GPU locks are held per job, with workload/headroom/coordination checks;
the per-phase deadline is10minutes and service deadline22minutes. Only owned
immutable container IDs are stopped. Source stays read-only; the separate
per-run asset copy is writable for flat admission only, then read-only.

The diagnostic completed. Actual reset quaternion XYZW was
`[0.70710677,0,0,0.70710677]`; projected gravity was
`[0,-0.99999994,3.42e-8]`. The unchanged upside-down predicate was already true
before the first control. The first pre-reset row had gravity Z+0.0287853,
upside-down and base-contact predicates true, too-low false, torque-excess
duration zero, and termination true. The intended root translation and zero
environment origin read back correctly. Five original done evaluations were
captured; the fresh full flat prerequisite passed. This confirms the rotation
convention fault, not a steady standing overload or terrain-gait failure.

`results/run/terrain/reset_diagnostic.json` holds all five pre-reset records.
`results/post_run_verification.json` verifies the source661 and admitted550 asset
files unchanged, no extra source files, both owned containers absent, no CUDA
processes, both locks free, and previously active forecast timers restored.
The next separate source004 fixes XYZW yaw and repeats full standing admission
without diagnostic mode or threshold changes.
