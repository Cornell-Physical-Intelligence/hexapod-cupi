# Canonical detailed robot: native import inspection completed

The approved motor-mass-corrected direct-drive robot completed its passive Isaac Sim6.0.1 import inspection. Actual native readback matches19 bodies,18 named joints, full mass/inertia/COM tensors and limits. All153 exact SDF shape paths exist and their native views are valid. Eight explicit2.5ms steps completed after SDK initialization, with finite state and consistent link/coordinate FK.

**This is an import and representation check. It does not admit joint actuation, supported standing, contact-distance accuracy, PPO, terrain or hardware deployment.** Stage2 and Stage3 remain incomplete. The prior two API failures remain separate unchanged records.

## Measured result

| Measurement | Result |
|---|---|
| Imported mass | 7.466088220 kg; selected model7.466088235 kg |
| Actual body / joint count | 19 /18 |
| Native SDF view | 153 exact paths, valid |
| Explicit samples | Initial readable sample plus8 steps at400Hz |
| Post-warmup joint range / root displacement | Zero over this short observation |
| Drive stiffness / damping | Zero; no controller or targets |
| Native PhysX error events | None, including post-exit validation |
| Initialization interval / native wall time | 2.281971s /15.181309s |

Zero gravity, no ground and authored self-collisions were preserved. SDK reset/play warmup precedes the first sample and is not observed by the explicit step counter. The zero-motion trace therefore does not prove quiet supported standing or absence of source-geometry interference. No simulator signed-distance values or ground contacts were queried.

The installed110.1.13 SDF binding accepts a single string pattern, despite its Python documentation advertising lists. The source now uses that supported argument and rejects any missing or extra returned path. Its initialization barrier returned and all native views are valid. The legacy cooking queue getter is actually unavailable; its count stays null, never a fabricated zero. Model bytes, SDF resolution, masses/inertias, limits and physical tolerances are unchanged from the failed attempts.

## Integrity and ownership

Root passed20 inspector and21 updated guard CPU tests, then the actual installed Spark Python3.12.3 host setup against source33/host15/guard38/supervisor926 and the exact nine-file asset. The producer also records12 host tests and strict preflights. These tests prepare execution; the raw native run is the evidence above.

Invocation`ab32f02fccd1400da1e25ff3b8e525d7` completed with exit0. The root terminal audit verifies the native post-exit receipt, all immutable inputs, all23 raw payloads/216,420bytes in two full hash passes, both owned identifiers absent and exact weather restoration. `terminal/audit.json` reports `audit_verified=true` and `inspection_completed=true`, while physical and training admission remain false. All original raw files are included.

Run `python3 -B -S verify_bundle.py` here to verify the complete published bundle. Original source, host, guard and terminal auditor freezes are retained. The actual source and geometry bindings are in the manifests; no old robot task, actor or checkpoint was loaded.

## Next admission steps

Use this result as the exact input for small coordinate/effort tests and separate SDF-distance/frame probes. Declare a provisional simulation actuator and explicit solver settings before gravity support or learning. Verify the real nonspherical distal+X toe contact region, a cold neutral hold and1-to32 replica throughput, then a fresh32×24×2 PPO integration with strict reload. Native import cannot substitute for those tests or the unchanged all-direction/quiet-standing quality gates. Hardware identification remains a separate transfer requirement.
