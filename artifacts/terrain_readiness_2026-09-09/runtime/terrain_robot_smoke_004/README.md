# Full-C terrain standing attempt004

**Correcting the quaternion removed the immediate resets, but the standing
gate still rejected the run.** Fresh exact-plan32×1000 flat standing passed.
The corrected robot then completed1000 controls on the original ramp fixture's
flat entry pad, with no terminations or truncations.

Settled peak requested torque was0.4984625N·m, requested saturation zero,
minimum/mean root height0.1291335/0.1295423m, and base contacts zero. The raw gate
reported800 nonfoot contact environment steps and minimum5 distal supports.
PhysX also emitted2796 warnings that contact data exceeded capacity8; flat had
none. Those truncated records cannot establish complete distal-versus-shaft
contact classification. Preserve the rejection; do not treat the missing
support or nonfoot counts as fully resolved physical contact evidence.

- [Original rejected gate](run/terrain/state.json), [full raw terrain log](run/logs/terrain.log), [warning counts and log hashes](contact_data_completeness.json).
- [Fresh flat admission](run/flat/admission.json), [campaign](run/campaign.json), [exact original result map](full_result_SHA256SUMS.json).
- [661-file frozen source manifest](campaign_source_hashes.json), SHA-256 `9fc04076a6397cc5377f6a02c3d9c1ffd5b18967a6d3d586dd178ae0d96c7e95`; [source provenance](source_origin.json), [independent provenance audit](provenance_audit.json).
- [Post-run source/asset/ownership verification](post_run_verification.json), [pause017 restoration](forecast_pause_017/restored.json), [method](METHOD.md).

Only the adapter's XYZW yaw tuple and provenance changed from source003. The
original standing harness, force/torque/support gates, full-review plan, motor
limits, original30-fixture catalog and passed fixture003 evidence stayed
unchanged. All661 source and550 admitted asset files verified unchanged after
exit. Both owned containers were absent, CUDA empty and both locks free at
that check; both forecasting timers were restored before GPU ownership passed
back to Stage2's coordinator.

A separate CPU-reviewed change raises the contact-data budget and rejects any
reported truncation before campaign admission. It has not run in Isaac. Its
next test still needs fresh exact-plan flat admission and complete terrain
standing. No terrain standing/walking, derived fixtures, perception or physical
four-bar are admitted by this attempt.
