# Full-C terrain standing attempt005

**The full selected C robot passed standing on the original terrain fixture's
flat entry pad, with complete contact classification under the bounded smoke.**
Fresh exact-plan flat validation passed at 32 environments × 1000 controls,
then the terrain entry check passed at 1 × 1000 controls. Both completed logs
contain zero reported incomplete contact/friction data warnings.

| Settled terrain measurement | Result |
| --- | ---: |
| Peak requested joint torque | 0.4984625 N·m |
| Requested torque saturation | 0% |
| Minimum distal foot supports | 6 |
| Nonfoot / base contact environment steps | 0 / 0 |
| Terminations / truncations | 0 / 0 |
| Mean root height | 0.1295423 m |

- [Original terrain gate](run/terrain/state.json), [fresh flat admission](run/flat/admission.json), [campaign](run/campaign.json).
- [Terrain contact-data audit](run/jobs/terrain_contact_data_audit.json), [flat audit](run/jobs/flat_contact_data_audit.json), [full terrain log](run/logs/terrain.log), [flat log](run/logs/flat.log).
- [Comparison with rejected004](comparison_004_005.json), [exact source changes](source_comparison.json), [method](METHOD.md).
- [662-file frozen source manifest](campaign_source_hashes.json), SHA-256 `4fcaedf7a43020b18e964f1bc666dd6dc0a95e6180ac431e686b716959b968e9`; [provenance audit](provenance_audit.json).
- [Original 19-file run/restoration map](full_result_SHA256SUMS.json), [post-run verification](postrun_verification.json), [pause020 restoration](forecast_pause_020/restored.json).

The point/friction data budget increased from 8 to at least 128 per tracked
body; the host now rejects any reported truncation even when raw metrics pass.
Torque, height and reset metrics exactly match004, while support changed from
5 to 6 and nonfoot contacts from 800 to 0. Together with warnings falling from
2796 to 0, this supports the earlier diagnosis of incomplete contact data.
Attempt004 stays rejected with its original evidence intact.

All 662 source files and 550 admitted asset files were unchanged after exit.
Both exact owned containers were absent; the unit exited successfully and
pause020 restored both forecasting timers. At the later verification snapshot,
CUDA was empty and both locks were free; this is historical evidence, not a
claim about a subsequent job. The main coordinator retained the next GPU slot.

This admits this exact full-C stance at the flat entry of admitted
`train_ramp_1103` only. It does not qualify ramp traversal, derived fixtures,
terrain walking, perception, new velocity-policy architecture, or the physical
four-bar robot. No policy was loaded, trained or saved.
