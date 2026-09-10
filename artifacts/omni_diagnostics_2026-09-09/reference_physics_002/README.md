# Reference physics002: standing passed; wave rejected

The target-only startup correction restored six-foot standing across all 32 replicas under the unchanged gate. The subsequent reference wave was rejected at its first early touchdown. This is a completed diagnostic, not a walking controller admission, Stage 2 completion or terrain result.

| Measured result | Standing | Wave before rejection |
| --- | ---: | ---: |
| Replicas × recorded controls | 32 × 1000 | 1 × 274 |
| Postsettle measurement duration | 16 s | 1.48 s |
| Minimum distal foot supports | 6 | 5 |
| Peak postsettle requested torque | 1.462980 N·m | 1.145178 N·m |
| Postsettle requested saturation | 0 | 0 |
| Nonfoot / base contacts after settling | 0 | 0 |
| Terminations / truncations anywhere | 0 / 0 | 0 / 0 |
| Contact/friction truncation warnings | 0 | 0 |
| Gate | Passed | Rejected |

The standing trace independently confirms the emitted targets reach the exact named canonical stance after two seconds and remain there. Physical reset noise, motor gains, model and gates were unchanged. Source001's permanent randomized-target hold remains a rejected result; this successor does not overwrite it.

The wave recorded one real LF flight lasting 50 control samples, with 2.691 mm measured reference-point lift. Contact returned at control index273 (5.48 s physical time, 1.48 s after the four-second startup/settle), before its planned swing endpoint. The generator rejected it as `Unexpected early touchdown; no unbounded endpoint correction`. No three-sample confirmed landing, completed six-leg cycle, final quiet stop or complete forward-displacement gate was achieved. A physical toe reference point is not a full-pad clearance measurement. The original failure and partial physical trajectory are preserved.

Primary evidence:

- [Fresh standing admission](results/run/standing/admission.json) and [standing trace](results/run/standing/trace.npz).
- [Rejected wave state](results/run/wave/state.json), [wave trace](results/run/wave/trace.npz) and [reference states](results/run/wave/reference_states.json).
- [Campaign terminal state](results/run/campaign.json), [independent review](INDEPENDENT_REVIEW.json) and [root remote audit](results/remote_audit.json).
- [Standing contact-data audit](results/run/jobs/standing_contact_data_audit.json), [wave contact-data audit](results/run/jobs/wave_contact_data_audit.json), and both complete logs under `results/run/logs/`.
- [Exact startup overlay and reconstruction](preparation/README.md), [source manifest](preparation/source_identity/campaign_source_hashes.json), and [pause028 restoration](results/forecast_pause/restored.json).

The independent copy review matched all 24 raw remote payload hashes. Root's terminal audit verified all 924 source files and 550 admitted assets unchanged; both exact owned container IDs and names were absent, and both host job records report checked cleanup. Pause028 records restoration of the two forecasting timers. These are terminal audit facts, not a claim about current GPU occupancy or locks after a later launch.

Source manifest SHA-256: `34390c6162f462baa4c531f1d5aff346ffbc98d4d0266a68ea797f5e5b992ee0`. The compact preparation includes the three changed runtime files and full identity, retaining the published reference001 parent without another full source copy. The next landing-reference investigation requires its own source and fresh standing admission. Existing support, torque, actual-flight, landing, progress and quiet gates remain in force.
