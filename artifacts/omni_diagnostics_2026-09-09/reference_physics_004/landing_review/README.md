# Reference004: RF landing-bound diagnosis

10 September 2026. The 7 mm reference cleared the previous RF clearance failure. It completed three measured steps (LF, RR, LM), then rejected RF at 12.50 s because its original planned foothold was too far from the returned measured foot after accounting for the old preload. This review preserves the rejection and the 12 mm bound.

The raw source004 trace contains 45 consecutive RF flight samples (0.90 s), followed by contact at swing phase 0.71. RF rose 3.44176 mm above its immediately preceding measured contact height. Contact returned after the apex and measured descent, at only 1.377 mm/s measured toe speed. These satisfy the previously failed clearance and flight predicates.

The precise failed value is `norm(original endpoint − measured toe − preceding stance preload) = 12.23450 mm`. The other check, `norm(candidate landing endpoint − original endpoint) = 10.49362 mm`, passes. The original planned stride is 69.900 mm; at returned contact its horizontal trajectory still has 10.49231 mm to travel and a speed of 44.451 mm/s. The old preload contributes a 3.495 mm lateral correction, and the combined lateral error is 5.407 mm. Neither removing the old preload nor replacing the endpoint solely to pass this datum would establish a coherent physical solution.

At the same instant, actual body translation differs from the virtual desired pose by 2.316 mm. Translation contributes −0.764 mm vertical offset, body rotation contributes −2.366 mm at RF, and joint deflection contributes another −0.327 mm. The virtual reference is therefore 3.456 mm above the actual toe. This is why increasing nominal lift alone does not ensure that physical touchdown occurs near the end of the horizontal quintic.

## Minimal proposed intervention

Complete horizontal motion in 80% of the existing 2 s swing, using the same endpoint-matched quintic. Retain the full 2 s, 7 mm vertical curve. Horizontal position, velocity and acceleration settle continuously at 1.6 s, before the vertical arc finishes. The landing blend and all actual flight, support, speed, contact, preload and correction bounds remain unchanged. The requested body command remains 0.005 m/s; this change does not derate or replace it.

At the observed return time the proposed horizontal reference has 0.835 mm remaining. If the actual foot retained exactly its previous tracking offset despite the timing change, the original endpoint error would be 5.496 mm. That is an explicit, unverified counterfactual. Faster target motion can change dynamics and contact timing. The unchanged actual contact still fails at 12.23450 mm; CPU replay cannot transform it into a physical success.

The proposed implementation lives in the separately versioned `omni_reference_wave_004` bundle. Its 23 synthetic bearing/yaw/arc/stop cases and 19 unit tests pass. The worst reference rate is 1.4065 rad/s and acceleration is 3.2388 rad/s², within the formal reference allocation of 1.75 rad/s and 6 rad/s². The more restrictive 0.03 rad/20 ms comparison is a separate diagnostic and is not claimed to pass for every case. No parameter here is a measured motor speed limit.

## Evidence

[report.json](report.json) records the exact vectors, baselines, source hashes and counterfactual assumptions. [analyze.py](analyze.py) reads the audited raw source004 trace from the root workspace and re-derives the actual values; it does not rerun physics. All lengths, mass, stance, torque gates and prior frozen controllers remain unchanged. The root owns raw-result publication and GPU dispatch.
