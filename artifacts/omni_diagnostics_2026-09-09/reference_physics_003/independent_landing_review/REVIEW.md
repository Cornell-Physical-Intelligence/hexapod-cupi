# Independent landing successor review

**Clear for one fresh, bounded physical screen; not physically admitted.** Reviewed `tmp/omni_reference_wave_002/wave_reference.py` at SHA-256 `ef92745daf4c3bf544b6c6f9f65e18f19745d9e84811c8888a464312f864a56b`. No frozen source, main code, gate or GPU job was changed by this review.

The actual reference002 trace establishes a plausible landing event: LF unloaded for 50 samples, lifted 2.691 mm, descended, then returned 1.874 N contact at 74% of its planned swing. Only one contact sample exists at the old rejection. Allowing a provisional landing phase is justified; immediately declaring touchdown would not be. The new code preserves that distinction.

The successor retains the original swing and endpoint. Before starting the blend it requires the observed flight, at least 2 mm lift, passage through the nominal apex, actual descent, bounded returned-foot speed and original endpoint consistency. It seeds a separate trajectory at the outgoing reference's exact world position, velocity and acceleration. The physical body and measured joint/contact state are never prescribed.

Two issues found during independent review were corrected before clearance:

1. Landing completion now checks full 3D consistency against the declared landing preload, the existing 25 mm absolute preload bound, and a 12 mm measured contact region. It does not merely check vertical error before silently latching an arbitrary horizontal offset.
2. Lost contact remains provisional for at most 100 ms; longer loss rejects. Completion requires three stable measured samples after the blend endpoint. A timer or one returned sample cannot count as a completed step.

The blend duration is explicitly bounded to 0.10–0.50 seconds using remaining original swing time. The actual early trigger uses 0.50 seconds. The endpoint choice avoids restoring obsolete horizontal preload, but a same-XY stopping endpoint with nonzero incoming velocity produces approximately 1.697 mm planar excursion and return in this case. This is disclosed and tested; C2 continuity is not proof of visually smooth or slip-free physical landing. All emitted joint position/rate/acceleration bounds, support, torque and measured-progress gates remain required.

The original real-input prefix replays unchanged to double-precision tolerance against saved references; differences from emitted float32 targets are only cast-scale. Independent execution of all seven new landing tests passed. They cover the real trigger splice, rejection before apex or without lift, retained original endpoint error, loss after a single contact, three-sample completion, 3D contact consistency, and stopping during landing without another liftoff. The after-trigger continuation is an explicitly synthetic fixture, not a physical result. The author additionally reports all 16 inherited/new tests pass.

Future PPO must use a new observation schema covering the landing phase and its complete state. See `OBSERVATION_IMPLICATIONS.md`. The provisional 525/528 plan is not silently compatible. This screen has no actor, so implementing that future encoder is separate from the immediate physical feasibility test.

Independent actual-wave measurements are frozen separately in `tmp/reference_wave002_independent_review`, manifest SHA-256 `d96f176408d2081f4d5591555009638166f42c55b02d9b34223c5ccf1dbcccba`. The actual trace has SHA-256 `a1e2a96251325b49431c27ce80506117ae713359d680bfd55d4892f72dd12d67`.
