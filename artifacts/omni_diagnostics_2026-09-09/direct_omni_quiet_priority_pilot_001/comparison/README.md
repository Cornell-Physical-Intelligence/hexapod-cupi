# Actual quiet-priority50 versus CAPS50

Both matched-budget pilots completed 50 updates; both remain **0/48** on the original final ten-second quiet gate. The stronger quiet objective did not establish a quality improvement. Source/plan versions differ explicitly; this is one descriptive controlled-budget comparison, not statistical significance or a causal guarantee.

**Reset-window clarification:** quiet-priority env20 (forward_fast) terminates at25.12s, inside the22.02–32.00s quiet window. Its1.870m excursion includes the reset displacement and is not uninterrupted physical drift. CAPS env20/44 reset at18.82/20.86s before that window; quiet-priority env44 resets at17.52s. All trial resets still invalidate their replicas.

| Final quiet/stop measure | CAPS50 | Quiet-priority50 |
|---|---:|---:|
| Quiet passing replicas |0/48|0/48|
| Median maximum planar excursion |50.42mm|57.60mm|
| Maximum planar excursion (reset included) |75.11mm|1870.23mm|
| No-trial-reset subset median / max (46 rows each) |49.91 /75.11mm|56.47 /89.89mm|
| Median worst-joint requested saturation |22.2%|24.0%|
| Mean all-joint quiet requested saturation |10.545%|11.178%|
| Per-replica worst target-step p95 (all48) |0.040000021rad|0.040000021rad|
| Worst SDK joint RMS |1.29346rad/s|1.29457rad/s|
| Worst interval-angle joint RMS |1.50176rad/s|1.51592rad/s|
| Full32s stop-trial requested torque peak |8.0824Nm|9.9708Nm|
| Postsettle constant requested torque peak |7.1081Nm|11.8293Nm|

The no-reset subset is descriptive only; it does not replace the original gate or remove any failed replica. SDK endpoint rates and interval-angle averages remain separate measurements. All applied torque peaks retain the1.6Nm clamp (float32 readback1.600000024Nm).

Final constant-command tracking covers four replicas per case. Planar error is lower in9/12 cases and higher in3; yaw error is higher in10/12; requested saturation is higher in9/12. Lower/higher counts have no invented significance threshold.

| Case | Planar error CAPS→quiet (m/s) | Yaw error CAPS→quiet (rad/s) | Requested saturation CAPS→quiet |
|---|---:|---:|---:|
| stand | 0.03073→0.02955 | 0.07394→0.07413 | 10.658%→11.158% |
| forward | 0.03631→0.03707 | 0.09089→0.09278 | 10.933%→10.775% |
| reverse | 0.03476→0.03462 | 0.09021→0.09795 | 9.764%→10.008% |
| left | 0.03493→0.03441 | 0.08840→0.08888 | 9.489%→9.431% |
| right | 0.03207→0.03067 | 0.08288→0.08342 | 10.025%→10.492% |
| forward_fast | 0.06074→0.05554 | 0.13520→0.14459 | 8.533%→9.011% |
| turn_left | 0.02986→0.02821 | 0.08375→0.08671 | 9.519%→10.014% |
| turn_right | 0.03337→0.03239 | 0.08206→0.07655 | 10.936%→11.597% |
| arc_left | 0.03714→0.03711 | 0.09855→0.09711 | 11.086%→11.956% |
| arc_right | 0.03311→0.03390 | 0.10151→0.10229 | 10.072%→10.458% |
| strafe_arc | 0.03069→0.03164 | 0.10963→0.10982 | 9.722%→9.767% |
| diagonal | 0.03601→0.03559 | 0.09211→0.09342 | 10.686%→10.544% |

The initial constant report is exactly equal. The initial stop reports differ only in20 heading values, at most1.4305115e-5degrees; every other reported value and the failed gate outcomes match. This preserves the existing remote CAPS versus local new-analysis numerical distinction without rewriting either report. All32 inputs for each analyzer report were independently hashed; historical remote CAPS paths were mapped to already fetched exact bytes.

Recompute the descriptive summary from the included exact reports into a fresh path:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B compare.py --output /fresh/path/comparison.json
```

VERIFICATION.json preserves72 selected quiet-run files/224,646,566bytes versus122 files/443,634,816bytes in the complete remote inventory, with50 exact intermediate autosaves explicitly remote-only. It also binds both32-input maps and actual reset event rows. No raw source, scorer, gate, policy or GPU job was changed. Eventual adoption must follow docs/PROJECT_SITE.md and its bounded central progress/evidence update.
