# Moving PPO001: reference support outcome, then reset API failure

The campaign completed fresh standing and both exploration calibrations, then stopped during the untrained profile after **224 controls: 200 startup plus 24 active controls**. It performed **zero PPO updates**. Replica 6 lost a required retained RR support during the planned LF swing; the attempted reset then raised PyTorch's inference-tensor mutation error at `reset_buf.copy_`. These are two linked findings, not a trained policy's rejected evaluation.

| Actual phase | Result |
|---|---|
| Fresh standing | 32 × 1000; original physical and all quiet verdicts pass |
| Zero-mean calibration | All 32 quiet; zero residual offset |
| Sampled calibration, std 0.02 | All 32 quiet; largest residual offset 0.001395 rad |
| Profile | 224 controls, one reference outcome in replica 6, then reset API exception |
| PPO and later screens | Not entered; no updates or decision checkpoint |

Both calibration trial metrics reproduce exactly from raw data. Standing physical metrics and quiet verdicts reproduce; any tiny heading-number differences are retained in [REVIEW.json](REVIEW.json). All physics substeps and their control endpoints align. All 14 recorded contact clocks advance consistently in calibration and through the available profile; there was no completed physical reset or new sensor epoch. Post-startup 400 Hz requested torque stays below 1.6 N m in all three phases (maximum approximately 1.483903 N m). Initial startup torque exceedances remain recorded and excluded under the unchanged startup rule.

At control 224 (4.48 s), replica 6 has RR normal force **0.597675 N**, below the unchanged 1 N classifier, while LF still carries **5.457182 N**. Five total contacts conceal only **four retained supports when the currently planned LF foot is excluded**. RR had fallen from 3.625588 N at control 218. The reference correctly returns support failure code 4. There is no native termination, nonfoot contact or residual target lag. This is an unloading/load-distribution outcome; LF has not established a measured successful airborne step.

The raw event ledger records the attempted reset and its cause but lacks `reset_kind`, reset-target and reset-clock completion evidence. The unchanged full moving-audit primitive therefore rejects with `KeyError('reset_kind')`; that failure is preserved. A separate exact replay of the **223 pre-event controls** passes schedule, mask, torque, endpoint and clock checks. The final control's complete physical/substep/clock record is checked separately without inventing a successful reset. The traceback stops before `_reset_idx`; the recorded API failure and missing completion receipt are consistent.

The same initial checkpoint SHA is preserved in calibration, sidecar and actual profile reload receipt. It is a scratch zero-mean actor with no optimizer update, not a walking improvement. A reset-only API correction cannot by itself satisfy the existing no-event profile gate: the preceding support outcome still requires review. No reset fix, gate waiver or new physical run is included here.

The raw SDK rate discrepancy persists. For example, RM tibia in calibration accumulates about 1.012486 rad difference between integrated reported rate and actual angle change over the post-startup window. No native velocity-fidelity qualification is inferred from otherwise passing quiet checks. Actor packets, history-valid arrays and critic bootstrap tensors were not exported by this consumer; the raw episode audit does not directly prove those GPU tensors correct.

The read-only terminal audit matched **56 raw files**, **926 source009 files**, **35 consumer files**, **3 host files**, **2 guard files**, **18 bridge files**, **160 observation files** and **550 admitted assets**. All three actual owned container names and IDs are absent. Pause049 restoration is recorded at Unix `1789046082.2900512`. The user unit failed with exit 1 and invocation `53fa4c8afc7940e0931a77c18d310557`. Original inputs and outputs remain unchanged.

[analyze.py](analyze.py) accepts explicit raw/source/audit-primitives/output paths, refuses an existing output, and performs CPU-only replay. The initial local reader mistakenly included string joint-name metadata in numeric metrics; its error log is retained, and the final reader separates that metadata without changing any raw evidence or scoring rule. No services, GPU jobs, robot states, source files or production configuration were modified by this review.
