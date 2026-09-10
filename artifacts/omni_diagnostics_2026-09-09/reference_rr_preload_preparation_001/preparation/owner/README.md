# RR first-landing preload diagnostic 001 — prepared, not physically admitted

This is a separately bound **fresh 32 × 1000 standing/quiet screen followed by one 1 × 2400 left-strafe trial**. The requested motion is exactly `[forward=0, left=0.005 m/s, yaw=0]`: 4 s canonical startup, 24 s motion and 20 s stop. Source931 manifest is **`9b71ad4e4be3e75ec34735725e0f815ac4ab0f4c40871150384658e2fdfc9268`**. No GPU job, actor, checkpoint, hardware qualification or production change occurred during preparation.

The experiment tests one measured hypothesis from rejected directional002: RR lost approximately 0.492 mm of downward target-to-toe offset after its qualified landing, while its force fell from 7.746 N to 0.946 N during LM swing. The preceding [frozen diagnosis](../reference_strafe_support_review_001/README.md) is unchanged (14-file freeze `8670b8e6acc2becbc7fa55f9de9f8056a5da5902029e53978a1031681303cf23`). Its alternative target calculation was not a physical result.

**This is not an arc fix.** Actual directional003's arc lost RM support (0.883063 N) while RR still carried approximately 8.02679 N. That different loaded foot is explicitly recorded in the new contract; turn, arc, reverse and training commands are rejected by the new host and scalar reference.

## The only target intervention

After the **first measured, confirmed RR landing**, lower the RR stance world anchor by **0.5 mm** using `s(u)=10u³−15u⁴+6u⁵` over the **existing 0.3 s contact hold**. Position matches at the beginning and endpoint, and analytic correction velocity and acceleration are zero at both ends. The endpoint is held; no later landing receives another scheduled correction. On reset, the one-shot state clears and the exact incoming executable target is retained.

The trigger is the original controller's landing-completion decision, before advancing target time. On the actual directional002 inputs, that occurs at measured time **8.40 s**; correction ends at **8.70 s**, and the first changed target is emitted for **8.42 s**. The existing next-liftoff schedule is unchanged. The earlier CPU sensitivity proposal began its interpolation at 8.42 s (first nonzero change 8.44 s); it remains frozen as preliminary evidence. This adapter uses the actual runtime's existing hold interval.

Measured anchors and contact classifications are never rewritten. Only the RR virtual anchor and its explicitly corresponding reference-minus-latched-measurement preload are adjusted. The 12 mm original-endpoint bound and 25 mm preload bound are checked before arming. During the active correction, RR must continue to report actual distal support; a lost RR contact, contact-region displacement beyond 12 mm, invalid clock, or existing reference/physical failure rejects the run. No contact loss is converted into a touchdown or an admitted support. The amplitude is a fixed diagnostic, not a force-controller gain or a tuning range.

A stop request suppresses further liftoffs through the original reference logic. An active correction must finish before the reference declares quiet; the quiet-target preservation branch cannot discard its final knot. The subsequent stop and measured quiet gates remain the original gates. Active-correction failures latch and return no target. This does not invent a physically admitted recovery trajectory.

## Preserved scope and limits

- Coxa, C femur/tibia lengths (72.5/126 mm), masses, inertias, stance, joint order/signs, actuator gains and 1.6 N m torque checks are unchanged.
- The 1 N distal classification, five-support requirement, actual 2 mm flight, apex/descent, landing bounds, original 5 mm pose/rate consistency gate, collision checks, and final measured quiet gates are unchanged. Fresh standing still requires all 32 replicas to pass the original six-support physical and quiet checks.
- Gait order, 2 s swing, 7 mm lift, 80% horizontal timing, existing 0.3 s hold, requested velocity and command filter are unchanged. No IK clipping, hidden command derating or body-pose forcing is introduced.
- Reference budgets remain 1.75 rad/s and 6 rad/s²; the unchanged zero-residual core reserves 0.25 rad/s and 2 rad/s² within combined 2 rad/s and 8 rad/s². These are target experiment budgets, not measured motor speed limits.
- Raw SDK XYZW, SDK joint rates, the separate 400 Hz joint-angle observer, contact timestamps and finalization paths remain unchanged.

The scalar output adds **`diagnostics.rr_preload_diagnostic`**, containing proposal identity, started/completed/rejected state, trigger/end/sample times, original virtual anchor, measured anchor, original swing endpoint, applied offset and analytic correction P/V/A, contact displacement, trigger touchdown count and failure reason. Existing base `state` fields remain intact. The historical 846/849 observation lineage and hashes remain recorded, but **that old packet/actor is not bound to this new stateful diagnostic**. No packet is emitted or checkpoint loaded here; later PPO adoption would require an explicit new observation/source contract.

## CPU evidence

[tests_003.log](tests_003.log): **16 tests passed**, including C2 endpoints, one-shot and exact case scope, exact old targets before the trigger, unchanged other-five-leg targets, unchanged measured anchors and cadence, reset continuity, support/non-foot interruption, genuine reference stop filtering with synthetic stationary measurements, and the unchanged residual core emitting every zero-residual reference exactly.

[ACTUAL_PREFIX_REPLAY.json](ACTUAL_PREFIX_REPLAY.json) reuses the 305 actual input rows through the old rejection solely as an input/target test. Every new target is reachable; the minimum named soft-joint margin is **0.085422 rad**, maximum discrete reference velocity **0.583159 rad/s**, and acceleration **1.303773 rad/s²**. Maximum target change is **0.010219 rad**. When fed the original rejected row, the new reference still rejects **“Fewer than five measured support contacts.”** Reusing those original contact forces does not predict altered physics or establish a successful continuation.

A fixed-measurement proportional-term difference is at most **0.306561 N m** at the 30 N m/rad gain. This is **not** a future torque or load prediction. The original LR torque approached 1.517596 N m, leaving only 0.082404 N m headroom. Root must independently review the torque implications before deciding whether to run this trial; a fresh run must preserve all original gates.

[DELTA_REVIEW.json](DELTA_REVIEW.json) and [wave_reference.patch](wave_reference.patch) show the exact change. Of 930 parent payloads, **925 are byte-identical**; one helper is added. The base wave AST is exactly equal after stripping only the explicitly enumerated diagnostic hooks and exact-case/config guards. Every solver configuration/readback function, the owned-container supervisor and phase-allocation code are AST-identical. `run_directional_physics.py`, all physical metrics, residual core/wrapper, startup, observer and assets are byte-identical. Solver metadata binds the new wave hash while retaining the actual solver settings and functions.

[INTEGRATION_PREFLIGHT_FINAL.json](INTEGRATION_PREFLIGHT_FINAL.json) binds current source `9b71ad4e…`, rejects actual directional002 standing admission, rejects synthetic failed quiet, and permits only the one named case with a fresh exact-source receipt. No AppLauncher/GPU was used. `DRAFT_INTEGRATION_PREFLIGHT_001.json` is explicitly older preparation metadata (`fba1c13…`), not final admission. `tests_001.log` preserves a test-harness filename error (`reference_metrics.py` instead of `screen_metrics.py`); subsequent logs pass. Neither issue was a physical result.

## Owner handoff

Root owns independent review, the external pause/restoration guard, GPU dispatch and any Git/publication. The prepared source's host entrypoint is `source_rr_preload_001/tools/launch_directional_physics_spark.py`. It keeps the original 90 s AppReady deadline, owned-container identity/cleanup, read-only source and asset mounts, source rechecks before/after each phase, and stops after any rejection. The case list has only `left_strafe`; the controller independently rejects all other nonzero commands.

The proposed remote binding supplied by root is `reference_rr_preload_source_001`, output `reference_rr_preload_001`, under a separately owned pause048 guard. These are intended paths, not evidence of an active or completed job. No dispatch has been performed by this agent.

Read-only tests after freeze:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tmp/reference_rr_preload_diagnostic_001 -p test_preload.py
```

`build_source.py` only reconstructs into a fresh source directory from the exact directional002 parent. `delta_review.py`, `check_integration.py` and `replay_actual_prefix.py` write receipts and should run only in a fresh copied preparation directory. Original source and frozen historical evidence must remain byte-immutable.
