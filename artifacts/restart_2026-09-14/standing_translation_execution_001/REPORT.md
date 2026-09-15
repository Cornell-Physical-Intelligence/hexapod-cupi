# Canonical standing restart: reproducible placement-dependent failure

Spark was reserved under the user's explicit authority, the exact mass-corrected model was verified, and three fresh native diagnostic runs completed. All three recorded 1,000 controls and 8,000 explicit physics steps; the motor, solver, geometry and pass conditions were unchanged.

| Run | Standing gates | Missing support after settling | Worst joint speed RMS |
| --- | --- | --- | --- |
| Origin | Pass | 0 steps | 0.002949 rad/s |
| (14,4) | Fail | 5 steps | 0.074617 rad/s |
| (14,4) repeat | Fail | 5 steps | 0.074617 rad/s |

The joint-speed limit remains 0.03 rad/s. Each translated run has 17 isolated one-step zero-force events overall, including 5 in the 16-second quiet window after 4 seconds of settling. The quiet-window losses affect the middle feet. Every event reproduces the saved signature of 128 inactive zero tuples. Neither translated run exceeds the torque cap, contacts the floor with a non-foot body, or terminates early.

All 12 raw trace hashes—10 physics archives, the control trace and the contact stream—match between the two translated runs. The fresh origin traces also match the earlier successful source005 origin run. This establishes repeatability for these recorded conditions, not general GPU determinism. The original review's illustrative random-trial probabilities are not calibrated estimates for these deterministic repeats.

A batch is therefore unnecessary to reproduce this contact-loss signature. Changing placement from origin to (14, 4) reproduces the problem in the single-robot topology. This does not isolate floating-point precision, the location relative to the floor's triangle seam, or retained solver/warmstart state. No corrective dynamics or collision change has been selected from this observation alone. The [bounded CPU comparison](../standing_translation_comparison_001/README.md) finds tiny origin/translated joint differences in the first 2.5 ms step, contact-force differences at 32.5 ms touchdown, and a first support difference at 87.5 ms. At the same translated position, single and batch environment 23 reset values and early traces match until 37.5 ms, after ground contact. These observations narrow the reproducer without identifying a unique cause.

These diagnostic identities explicitly prohibit standing/batch/training admission. Walking and later terrain/survey qualification remain blocked by unresolved standing on the confirmed model. Historical standing, walking and pause records are unchanged; no old checkpoint is relabelled or reused as a new-model result. All native containers have exited and the exclusive compute controls remain in place.

RESULTS.json contains the compact measurements. Each results_* directory preserves selected byte-verified receipts, full remote inventory and the readout that reaggregated all 8,000 contact samples against the sealed force arrays. Large native traces, scene exports and logs remain intact on Spark at the paths in those inventories. The original and failed results stay separate.

Run `python3 -B -S verify_results.py` for offline verification of 73 selected receipts, three completed acquisitions, preserved pass/fail gates and 24 recorded repeat-trace hash matches. The verifier does not contact Spark or re-read omitted remote raw files. FINAL_SPARK_STATE.json records the later empty GPU/container query, available locks and persistent reservation.
