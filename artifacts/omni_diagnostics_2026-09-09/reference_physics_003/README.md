# Reference physics003: three confirmed steps, then clearance rejection

Fresh standing passed. The bounded wave completed measured LF, RR and LM flight/landing sequences, then rejected the RF swing because its measured lift did not reach the unchanged 2 mm requirement. Overall wave admission failed. No final six-leg cycle, full-duration progress gate or quiet-stop window was completed; this is not a learned walking policy, Stage 2 completion or terrain qualification.

| Measured result | Standing | Wave before rejection |
| --- | ---: | ---: |
| Replicas × recorded controls | 32 × 1000 | 1 × 617 |
| Postsettle measurement duration | 16 s | 8.34 s |
| Minimum distal foot supports | 6 | 5 |
| Peak postsettle requested torque | 1.462980 N·m | 1.145178 N·m |
| Postsettle requested saturation | 0 | 0 |
| Nonfoot / base contacts after settling | 0 | 0 |
| Terminations / truncations anywhere | 0 / 0 | 0 / 0 |
| Reported incomplete contact/friction warnings | 0 | 0 |
| Gate | Passed | Rejected |

The standing trace is byte-identical to reference002. The only executable change was the reviewed contact-triggered C2 landing reference; source identity changed accordingly. The original preparation, its documented 1.6968 mm planned landing overshoot/return, and its CPU-only claims remain unchanged inside this terminal wrapper.

**Exact RF failure:** distal contact was absent over indices 584–615 inclusive (32 samples) and returned at 616 / 12.34 s with 2.06758 N vertical normal force. The generator had already observed real flight and actual descent, and time was past the planned midpoint/apex. The failed conjunct was measured lift: **1.445192 mm < 2 mm**. Its peak reference-point world Z at 602 / 12.06 s was 1.089721 mm; the immediately preflight baseline at 583 was −0.355471 mm. Peak world height and lift relative to that baseline are different values. These are measured link reference points, not complete footpad or shaft clearance. The new returned contact was not declared a completed fourth landing.

The partial moving window reports mean forward velocity 0.003113 m/s for a 0.005 m/s request. An additional independent check over the same 8.34-second interval measured 36.402 mm forward root-link displacement but only 25.879 mm from integrating the reported link velocity; the full 3D discrepancy is 10.541 mm. There were no resets. This discrepancy exceeds the existing complete-run 5 mm consistency tolerance and needs measurement/sampling/physics diagnosis. The complete progress gate was never reached; neither partial statistic is promoted to a pass, and the tolerance is not changed.

Primary evidence:

- [Fresh standing admission](results/run/standing/admission.json), [standing trace](results/run/standing/trace.npz).
- [Rejected wave state](results/run/wave/state.json), [wave trace](results/run/wave/trace.npz), [reference states](results/run/wave/reference_states.json).
- [Independent numeric/force/baseline review](INDEPENDENT_REVIEW.json), [root remote audit](results/remote_audit.json), [campaign terminal state](results/run/campaign.json).
- [Standing contact-data audit](results/run/jobs/standing_contact_data_audit.json), [wave contact-data audit](results/run/jobs/wave_contact_data_audit.json), complete logs under `results/run/logs/`.
- [Immutable compact preparation and reconstruction](preparation/README.md), [independent landing review](independent_landing_review/REVIEW.md), [pause029 restoration](results/forecast_pause/restored.json).

All 24 raw remote payload hashes independently matched. The terminal audit verified all 924 source files and 550 admitted assets unchanged, both exact owned container IDs and names absent, and checked cleanup for both phases. Pause029 records restoration of the two forecasting timers. No statement about current GPU occupancy or locks after a later job is implied.

Source manifest SHA-256: `7c75f0372abcea9eace3a280a2c204164179b8c433fc9d6280f60dd189737e24`. The preparation bundle SHA remains `927931b03e7ec1ffd4811da5befc454c1e50f37d9102c81d261cdbce7c2d1fb0`. All actual failures and prior source versions remain immutable. Any successor requires a new source and fresh standing admission with the same physical acceptance gates.
