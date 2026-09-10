# Canonical robot: first supported-standing acquisition failed

The first one-robot standing run initialized the canonical 7.466088235 kg detailed direct-drive robot, then stopped at **14 actual physics steps, 13 complete recorded rows and one complete 20 ms control**. The contact parser rejected `Invalid patch normal`. The native state, host job and campaign all remain failed. **No standing, physical, learning or Stage 2 admission is granted.** The planned 1,000-control screen and quiet window were not reached.

All **32 raw files / 266,352 bytes** are retained in `terminal/`, including the partial row and original contact buffer. The independent terminal audit verified the source, host, executed guard, nine-file asset, prior actuation admission and ownership supervisor before and after the run; the exact owned container name and ID were absent, and pause006 restoration completed. This records that finished allocation, not current Spark status.

The failure buffer has 153 disjoint counted slots. Of these, **127 RF entries contain exactly zero force, a zero normal and zero separation**, all at one finite point inside the authored distal cap. All seven nonzero-force records pass the existing unit-normal tolerance; their maximum error is 3.58×10⁻⁷ against the unchanged 10⁻³ bound. The 127 entries add exactly zero force. There is no evidence here establishing why the native backend emitted them.

`contact_diagnosis/` preserves the independent original parser reproduction and proposed narrow interpretation: retain these exact tuples as inactive raw records that provide no support, while continuing to reject malformed or nonzero-force invalid normals. This wrapper does not apply that correction, rewrite the failure or claim that a corrected run will stand. The actual native rates and settling transients remain raw evidence; the short prefix is not a quiet or torque-safety result.

## What ran

`source/` is the exact 40-payload standing001 source (`f81f61d…`), using the accepted coordinate/effort diagnostic and the unchanged canonical nine-file asset. Native initialization read back the named 19 bodies, 18 direct joints, complete masses/inertias, joint limits, coordinate frames, 153 SDF shape identities, materials and offsets. The source applies explicit 400 Hz PD with 50 Hz held neutral targets, a 1.6 N·m software cap and a **provisional, uncalibrated 48 V** speed envelope. Battery voltage and hardware actuator behavior are not established.

The planned comparison retains the original quiet and motor thresholds. The plate-origin minimum of 55 mm, exact source-shape toe classification, and all-controlled-step non-toe floor clearance/contact checks are explicitly new geometry-bound standing criteria. Source geometry includes triangle-edge intersections at the 115 mm cap boundary; no sphere, collider simplification or hidden contact-force threshold was substituted. No historical actor, checkpoint or old physical task was loaded.

## Attempts and preserved provenance

1. `root_checks/actual_spark_setup.stderr` preserves a CPU setup rejection caused by supplying the parent actuation output directory. The corrected setup supplied its `actuation/` child and passed on Spark Python 3.12.3, without changing the frozen source or host.
2. `guard_prelaunch_failed/` and `root_checks/dispatch_001.json` preserve the erroneous NVIDIA option `--standing-compute-apps`. This failed before allocation. `first_guard_no_mutation.json` confirms no output, pause006 or owner existed afterward.
3. `guard_executed/` retains the separate correction to `--query-compute-apps` and its original freeze. `dispatch_002.json` records the actual launch, invocation `f3cd14c1078e47c891d97718643b22aa`. The native parser failure followed; host exit status was 1.
4. `terminal/`, `auditor/`, `independent_standing_review/` and `contact_diagnosis/` retain their original bytes and complete inventories. `RESULT.json` is the concise outcome; `PROVENANCE.json` maps every copied component to its original identity.

The exact asset, accepted actuation diagnostic and legacy ownership-only supervisor remain separately published dependencies. Their complete identities are retained here; their large trees are not duplicated. No raw file from this standing allocation is omitted.

Run `python3 -B -S verify_bundle.py` from this directory. The portable verifier checks every payload and nested freeze, all 32 raw hashes, authentic failure/cleanup/restoration receipts, and independently decodes the recorded NumPy contact arrays using the Python standard library. It does not launch Isaac or grant admission. Frozen producer tests and numerical diagnosis scripts retain their original dependency/path requirements.

The next bounded step is a separately versioned parser correction under unchanged physical gates, then successful one-robot and 32-robot standing screens before a fresh two-update PPO integration. Current execution and the central research poster are maintained separately by the publication owner; this evidence remains immutable.
