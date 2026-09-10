# Actual reference001 standing rejection

2026-09-10. The wave was **not launched**. The prior touchdown-timing prediction in README remains untested.

The standing trace identifies a specific initialization problem: the reference correctly preserves each exact randomized reset target, but the entrypoint holds that target for the entire20s instead of transitioning to the declared canonical stance. All32replicas have distinct targets within±0.03rad of nominal; the maximum target change across1000controls is **exactly0**. This is not actor noise or an oscillating target.

**17/32replicas** retain all six≥1N supports throughout the settled4.02–20s window. The15failed replicas are3,6,8,10,11,13,14,16,18,19,20,22,24,27,29. Thirteen failed replicas have one foot below the support threshold for all800settled samples. Replica19'sLF is below threshold795/800. Replica18 hasRF below threshold800/800 andLR760/800, producing the observed minimum of four supports.

Rejected-support samples have vertical force median **0N**,75th percentile0.385N andmaximum<1N. Thus the failure includes both unloaded feet and light touches, not merely dropped boolean telemetry. Total body load is carried by the remaining feet; no nonfoot contacts, terminations or post-settle torque saturation occurred. The maximum post-settle requested torque was1.3148N·m.

The exact nominal FK toe Z spread is only **0.000205mm**. Randomized held targets introduce millimetres of uneven toe height: for example replica18's target FK toe-height spread is3.693mm. In that replicaRF andLR are among the most retracted feet and carry approximately0 and0.902N mean vertical load, respectively. Actual pose and compliance redistribute load, so this geometric association is diagnostic evidence, not proof that canonical targets alone will make every replica pass.

The smallest justified successor is a **bounded startup transition from the exact initial emitted target to the named canonical stance**, preserving continuous position/velocity/acceleration. Hold the canonical target long enough to settle and run the same six-support/torque/contact gates. This preserves randomized initial state as a recovery test while correcting the unintentionally randomized long-term goal. Do not lower support thresholds or claim physical admission before the new screen passes.

The terrain agent owns a separate reference002 with a2sC2 transition followed by2ssettling. No frozen reference001 source, gate or model was modified by this review. Its rejection remains part of the record.

## Evidence

- `standing_review.py` reproduces the measured analysis and figure.
- `standing_review.json` records every replica/foot, exact traceSHA, mean/quantile forces, target offsets, measured toe heights, root pose and preload.
- `standing_review.png` shows support fractions, load distribution and virtual-minus-measured foot height.
- TraceSHA256:`2670cd064683f22c0ac5f601cc3c9708152899b161deb08654ffcf02edea3726`.
- Reference001 source manifest:`a13f1534f9f802871a2cacbc0be603eae2d80738905381da40691f7cd4beff7c`.

Measured toe height refers to the named tibia-local reference point, not full collision-mesh clearance. The earlier nominal static LP and touchdown prediction are explicitly separate analyses.
