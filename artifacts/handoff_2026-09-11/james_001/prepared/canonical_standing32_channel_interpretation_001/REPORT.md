# Standing32: contact loss is coupled to measured motion

All 73 missing-foot events were exact zero force, each lasting one 2.5 ms step. None was a small nonzero load near the 1 N threshold. Each affected tibia reported exactly 128 inactive records: zero scalar force, zero normal and zero separation, all at the same recorded point. The next step returned to ordinary contact. This repeated local signature is more specific than a generic buffer-pressure explanation.

The independent getter comparison and measured motion agree with a real loss of normal support in the simulated dynamics. They do not establish why that loss occurred, or prove a shared backend's completeness. The original result remains 10/32 combined/physical and 24/32 quiet; PPO remains disabled.

| Across all 73 events | Previous step | Missing step | Following step |
|---|---:|---:|---:|
| Affected toe force, median | 11.717 N | 0 N | 16.949 N |
| Sum of floor normal Fz, median | 73.239 N | 61.513 N | 78.933 N |
| Whole-robot COM acceleration from native velocities, median | −0.00041 m/s² | −1.571 m/s² | +0.762 m/s² |
| Affected tibia COM vertical velocity, median | −0.00000056 m/s | −0.06822 m/s | −0.0000439 m/s |

Every event was bracketed by forces above 1 N, with no concurrent second missing foot elsewhere in the batch. The affected tibia origin fell 82.3–91.8 µm during the loss and rose 75.3–94.2 µm on the next step. The largest affected-leg joint position increment was 0.003312 rad; native joint velocity reached 2.564 rad/s at these brief events. The motion therefore cannot be dismissed as only a zero reported by the Python classifier. Nor does the record show a gradual commanded lift: the controller held the unchanged standing target.

At the missing step, vertical normal force agrees with the mass-weighted link-COM momentum calculation within 8.03 µN. The local point-position second difference also indicates downward acceleration (median −0.822 m/s²), though it is not numerically interchangeable with the native-velocity estimate. Integration timing and FP32 position differencing limit that comparison. The horizontal force residual is deliberately not used: the recorded matrix and patches cover normal contact, while the installed API exposes friction through a separate getter. `API_REVIEW.json` binds the exact installed source and method locations; no SDK source is republished here.

Observed contact-view occupancy reached only 2,641/32,768 slots (8.06%) over the full run and 2,271/32,768 (6.93%) at event rows. This argues against exhaustion of this exported buffer. It says nothing conclusive about unexported solver buffers, contact generation, or a per-pair internal limit. The value 128 is an observed signature, not a proven capacity setting; increasing the global view capacity has no supporting evidence here.

The losses occurred in 22 robots across every x column and every y row, without a monotonic distance pattern. Middle legs account for 60/73 events (RM32, LM28); RF7, LR3, LF2 and RR1 also fail. Source003's prior32/1 result had 74 events, all with the same 128 inactive zero-force signature. Switching to32/0 did not remove that signature or improve batch admission (11/32 to10/32). The affected quiet-failure environments changed, so these separate runs do not isolate a solver-setting effect. Eight original 50 Hz SDK quiet failures remain; 400 Hz rates and actual angle increments are diagnostics, not substitute acceptance scores.

The next bounded discriminator should compare one robot at the observed failing batch coordinate `(14, 4)` with one at the origin, using the same source005 dynamics, geometry, servo, clocks and original gates. Batch env23 at `(14,4)` had seven losses; its first was sequence1710, time4.2775 s. A placement-only, explicitly frozen diagnostic identity and matched fresh initialization would separate this coordinate from the 32-robot context more directly than another solver or capacity sweep. If the isolated translated robot reproduces the failure, spatial placement becomes a stronger explanation; if it does not, batch/contact-pair context becomes more plausible, but one non-reproduction cannot exclude a rare event. Exact event outcomes and the 128-slot signature should remain the comparison targets.

This is a recommendation only. No new native code, source edit, placement change, GPU launch, threshold change or admission was performed. Root retains the requested explicit source/identity review and Fable partner review before any new comparison.

`interpret.py` checks and rechecks the actual result SHA `304bccb643f1be8ab95a41d98c85c985e7248652a208e20ac51b71006ff60231`, source005 freeze and exact analyzer identity. `report.json` retains all73 compact event rows, all32 spatial rows, rate details and calculated summaries. `prior_signature_check.json` separately binds the published source003 result. The original 22.36 MB analysis and raw data are referenced, not duplicated. Root/terrain own publication and the central Markdown/site update required by `docs/PROJECT_SITE.md`.
