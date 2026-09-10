# Standing32 failure analysis: transients, rate bias and open causes

The detailed-model standing32 screen remains rejected: **11/32 replicas pass; 21 lose six-toe support, and seven of those also exceed the original SDK-rate quiet bound**. All 8,000 steps were acquired. Its separate 600-second host timeout remains a failure. This diagnostic publication changes no threshold, source, controller or admission.

The [original rejection bundle](../native_standing32_rejected_002/README.md) contains the authenticated terminal report and curated raw metadata. This compact companion preserves the first analyzer's actual failure, the corrected analyzer, its complete 7.8 MB remote CPU result, independent physics review, interpreted result and Fable MAX review with explicit corrections.

## What the recorded events establish

- All **74** missing-support observations affect exactly one foot. The affected foot has zero resultant force and 128 inactive exact-zero patches; each other foot retains at least **10.74675 N**. Counts are RM 32, LM 25, LR 9, RF 6, RR 1 and LF 1. This is a measured support-classification transient; inactive records alone do not establish detached geometry or their native cause.
- Real angle motion accompanies some events. In env11/RM at 4.0100 → 4.0125 → 4.0150 s, force changes **12.3561 → 0 → 17.2886 N**. During the zero-force step the tibia angle changes by **0.00327935 rad**, with SDK rate 2.56532 rad/s and interval-angle rate 1.31174 rad/s. This instantaneous transient is distinct from the maximum **0.099565849 rad/s** original 50 Hz RMS score.
- A persistent SDK/angle discrepancy is also present. Origin LM tibia has 400 Hz SDK RMS **0.0229695 rad/s**, interval-angle RMS **0.00132590 rad/s**, and a **−0.367158 rad** SDK-minus-angle integral over 16 s. Env2 additionally has substantial transient angle motion: 400 Hz SDK/interval RMS **0.0480007 / 0.0240769 rad/s**, correlation 0.93579. The original 50 Hz SDK score remains 0.0864103 rad/s. These windows and channels are not interchangeable; none replaces the unchanged gate.
- The proposed all-body sleep signature is absent: no event has all six foot forces zero, and affected-leg SDK velocities are not all zero. This constrains that proposed signature without identifying the mechanism behind the inactive slots.

The [compact interpretation](interpretation/REPORT.json) gives exact events and rate comparisons. The [complete returned analysis](actual002/analysis.json) preserves targeted patches and neighboring steps, all replica rates and input hashes. It is an actual read-only CPU analysis performed against the original remote data; this publication did not run it again against the large raw traces.

## What was checked, and what remains unresolved

The exact 608 named contact paths, reset grid and native body/joint properties are checked. Translated reset positions differ by at most 0.894 µm. Env0 has an exact reset match to the accepted single robot, but later dynamics differ slightly even at that unchanged origin. Event counts across grid-y rows are 9, 29, 26 and 10. Translation and batch context are therefore not isolated causal interventions.

The selected ±1-step patch aggregates match NPZ exactly with source-equivalent arithmetic, and independent world-to-shape projections agree within 5.55×10⁻¹⁷ m with matching categories. Native contact start/count arrays were not exported in JSONL, so this is not an independent proof of the underlying contact-buffer ABI. The analyzer checks all 8,000 line-sequence prefixes but parses contact details only around the selected support events; it does not claim exhaustive semantic validation of every contact record.

The separate [physics review](physics_review/README.md) exactly reconstructs all 8,000 single-robot interval-angle rows and PD requested/applied torque. Its actual32 scope is the audited report, not a second full32 trace replay. A CPU translation/rounding test yields **zero classifier flips in 32,768 comparisons**, with sampled cap margin ≥977 µm. That rules out the tested host-classification rounding scenario, not native contact-generation or solver sensitivity. No concrete source row/indexing, held-target delay or torque-cap defect was established by that review.

## Preserve the precision failure

[Analyzer001](analyzer001/README.md) used float64 force×normal products and failed on the first selected aggregate; its original empty stdout and [actual traceback](actual001/analysis.stderr) are retained. The source actually multiplies float32 operands in float32, then accumulates foot forces in float64. [Analyzer002](analyzer002/README.md) reproduces that order, including source-equivalent point subtraction, without widening any tolerance.

At the first mismatch, the old Z aggregate was 12.356097655761346 N; the source-equivalent and stored value is exactly **12.356097221374512 N**. The 4.34387×10⁻⁷ N discrepancy was an analyzer precision defect. It neither erases the later zero-force events nor invalidates the original physical failure. The original failure, corrected source delta and actual single-robot product replay remain distinct immutable records.

## Partner review and bounded next hypothesis

The tools-disabled **claude-fable-5-1 / maximum-effort** review completed under session `40663e88-e12f-4797-9c3b-9964aabca2d0`. Only its prompt, final response and provenance are copied; no internal event stream is published. The [independent disposition](fable_review/DISPOSITION.json) corrects an ambiguous prompt range: Fable read `.07..10` as 10 rad/s rather than about 0.07–0.10. Its resulting 1.05 N·m derivative explanation was about 100× too large. The original prompt/response remain unchanged.

The review supports a bounded **32/1 → 32/4** velocity-iteration comparison only after observing the installed unauthored 32/1 public-schema precondition. Resolved USD attributes are not independent backend iteration instrumentation. Keep geometry, material, servo, captures and all physical/quiet gates; require fresh same-source one-robot then32 admission. Changed SDK feedback can also change PD torque, so any improved result would not isolate a contact-solver mechanism. No mandatory 255-iteration sweep, sleep API change or gate waiver was adopted. The 1,200-second operating budget is separate from physical acceptance. This bundle contains preparation/review of that hypothesis, **not its later native outcome**.

## Included evidence and reproducibility boundary

Every frozen byte is retained in analyzer001 (9 payloads), analyzer002 (16), physics review (6), interpretation (5), Fable review (10) and independent solver-source review (5). Both root execution snapshots preserve their three original stdout/stderr/transfer files. [PROVENANCE.json](PROVENANCE.json) records exact source paths, freezes, copied metadata and public parent bindings.

The original **36-file / 4,969,155,341-byte** run remains remote. This bundle downloads, decompresses and reconstructs **none** of it. Its copied [full inventory](inputs/FULL32_RAW_SHA256.json) and [curated selection](inputs/RAW32_SELECTION.json) preserve exact paths/sizes/hashes; the prior public bundle contains 24 small raw files, while the 4.62 GB contact stream, control trace and ten substep NPZ files remain remote-only. Independent one-robot raw inputs are hash-bound to [their prior publication](../native_standing_004/README.md), not duplicated here.

From the repository root:

```sh
python3 -B -S artifacts/mkii_updated_2026-09-10/standing32_failure_analysis_001/verify_bundle.py
```

The portable standard-library verifier checks every publication/component hash and actual analyzer/report/input binding, preserves the initial failed output, and exactly reproduces the compact interpretation from the included 7.8 MB result. It performs no native execution, network request or full raw replay. Frozen component scripts retain their original acquisition paths; use this wrapper verifier for portable checking. Root owns later shared status/plan/registry changes and publication under `docs/PROJECT_SITE.md`. No detailed-model PPO or hardware admission is implied.
