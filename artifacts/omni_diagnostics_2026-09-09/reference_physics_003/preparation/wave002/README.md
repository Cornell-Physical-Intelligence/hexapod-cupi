# Wave002: measured descending contact with a bounded landing blend

2026-09-10. CPU-prepared successor to frozen wave001. **No new physics or walking admission.** Reference002's actual32-replica standing passed, then its firstLF swing was rejected by wave001's scheduled early-touch gate. This version preserves that evidence and replaces the timing assumption with an explicit provisional landing state.

## Actual evidence

The frozen trace has50consecutive below-threshold contact samples, including genuine zero-force flight. LF returned1.874N contact at5.48s, after the planned5.0s apex and before the6.0s endpoint. Actual toe rise was2.691mm from the sample immediately before flight, or2.719mm from settled sample199. The robot moved2.030mm forward while its command was still ramping. Other five supports remained, and requested torque peaked1.1452N·m. Only one returned-contact sample was recorded before rejection: it was **not** a confirmed touchdown.

`report.json`, `landing_diagnosis.png` and the byte-copied original trace/reference states preserve the evidence. The old wave001 and source002 files are unchanged.

## Changed transition

The original virtual swing, planned endpoint and coefficients remain recorded. A returned contact can start a landing only after two consecutive flight samples, measured≥2mm toe rise, passage through the original apex, measured descent and returned toe speed≤0.04m/s. A small transient force drop cannot count as a completed step.

The landing starts at the **existing virtual trajectory's exact position, velocity and acceleration**. It does not jump to the measured foot. A zero-lift quintic Hermite curve ends with zero world-foot velocity/acceleration. Duration is the remaining original swing time, bounded to0.10–0.50s; the actual early contact gets0.50s. Its XY endpoint stays at the trigger's virtualXY. Its Z endpoint is the measured toeZ plus the recorded preceding stance's vertical virtual-minus-measured preload. Restoring the preceding horizontal preload was rejected because it would move the firstLF target backward6.4mm.

Keeping the virtualXY endpoint still causes a small **braking overshoot and return** because the starting tangential velocity is nonzero. For the actual trigger this is1.697mm, explicitly measured from the new reference curve; its physical smoothness/slip remain unproven. Total target excursion is2.218mm. The original-planned-endpoint versus measured/preload discrepancy is10.363mm, within the unchanged12mm bound. Both the new endpoint correction and original contact discrepancy are checked before replacement; the original plan remains in state.

The contact is provisional during the blend. Losing support for more than100ms rejects; brief gaps reset the confirmation count. Completion requires the blend to finish followed by at least three stable measured contacts, speed≤0.04m/s, bounded3D displacement from the candidate contact, full3D consistency with the explicitly recorded landing preload, and absolute preload≤25mm. Only then are reference/measured anchors latched separately and the next-contact dwell begun. Missing touchdown, insufficient lift, ascending contact and excessive endpoint changes still fail closed.

All emitted joint knots remain name ordered, reachable, within actual soft limits plus0.02rad residual margin, and bounded by reference1.75rad/s and6rad/s². The physical wrapper retains combined2rad/s and8rad/s², actual torque/support/contact gates and zero-residual exactness. The generator does not prescribe robot pose, weaken physical gates or inject an actor. Requested zero suppresses new liftoffs, completes the pending landing/support confirmation, then reaches a finite reference quiet hold with feedback intact. Actual quiet standing still requires the separate≥10s physical gate.

## Verification and limits

Sixteen CPU tests pass: nine retained geometry/interface tests and seven landing tests. The latter replay the exact actual prefix, verify all pre-trigger references unchanged, check exact footP/V/A splice and endpoint, retain original endpoint bounds, reject premature/no-clearance or lost contact, require three post-blend contacts, and verify finite stop without new liftoffs. Independent PPO review also passed the seven new tests and found no concrete blocker to a bounded physical screen.

Post-trigger continuation in the tests is **synthetic prescribed support**, not physics. It reaches one confirmed synthetic touchdown and finite quiet reference5.50s after zero request, with maximum reference speed0.150rad/s and acceleration0.944rad/s². No full all-bearing or terrain qualification follows. New landing/flight/preload state must enter any future Markov observation design; output target shapes remaining compatible does not imply actor/checkpoint/schema compatibility.

## Runtime and reproduce

Exactly five runtime files are required, as before:

- `wave_reference.py` — changed controller.
- `wave_math.py`, `serial_geometry.py` — byte-identical geometry/math dependencies.
- `geometry/candidate_c_reference.json`, `geometry/f050_t060.urdf` — byte-identical immutable inputs.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tmp/omni_reference_wave_002 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 tmp/omni_reference_wave_002/make_report.py
```

`FREEZE_SHA256.json` covers this prepared bundle. The support-supervisor prototype was paused for this physical diagnosis and remains untested/unintegrated.
