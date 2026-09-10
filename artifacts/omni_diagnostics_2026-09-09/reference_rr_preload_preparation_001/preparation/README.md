# RR preload diagnostic 001 — immutable CPU preparation

This bundle prepares one fresh 32-replica standing/quiet screen and one full-C **0.005 m/s left-strafe** trial. It contains **no physical trial result** and does not authorize a broader policy or production change. Root owns the next review/dispatch decision and its separate pause/restoration guard.

The [owner README](owner/README.md) explains the measured hypothesis and limits. The only reference change is a one-time **0.5 mm downward RR stance-anchor correction**, with C2 interpolation inside the existing **0.3 s hold after its first confirmed landing**. Existing support, 1 N, 1.6 N m, contact/flight, collision, geometry, motion and quiet gates remain. The arc's RM support loss is explicitly outside this trial.

- Owner: 26 payloads, freeze `a2d79f4a05e14eb37c90db6e86877136c8591c95d53b96b45fbdd81538130152`.
- New source: 931 payloads, manifest `9b71ad4e4be3e75ec34735725e0f815ac4ab0f4c40871150384658e2fdfc9268`.
- Guard: two payloads, freeze `727374d3f998fa03e22f9490a69804459fcc45d17eda6b5ae30ebe5a05a8c1d1`.
- [16 final CPU tests](owner/tests_final.log) passed, including actual-input prefix, exact zero-residual core, C2, one-shot, interruption, finite stop and named-case restrictions.
- [Final no-App preflight](owner/INTEGRATION_PREFLIGHT_FINAL.json) rejects old standing admission and failed quiet. The explicitly named draft preflight belongs to earlier metadata and is not final evidence.
- [Exact source delta](owner/DELTA_REVIEW.json): 925 parent payloads unchanged, one helper added, base wave AST unchanged after removal of the declared hooks, all solver functions and owned supervisor/phase allocation unchanged.
- [Target-only replay](owner/ACTUAL_PREFIX_REPLAY.json) retains the original rejected contact verdict; reused original forces do not predict the counterfactual physics.

The 846/849 observation lineage is retained as history. The added correction state is exported separately and is **not bound to that old actor/schema**. No actor is loaded or trained. The original CPU proposal and actual diagnosis remain immutable external references in [REFERENCED_INPUTS.json](REFERENCED_INPUTS.json).

The full source need not be duplicated in this preparation. The overlay includes all changed runtime bytes, the new manifest and origin metadata; the exact directional002 parent supplies the other 925 payloads. Both reconstruction paths were checked against all 931 output hashes during preparation.

```sh
python3 verify_payload.py --parent-source /path/to/exact/directional002/source
python3 reconstruct_source.py --parent-source /path/to/exact/directional002/source --output /fresh/path/source_rr_preload_001
```

`verify_payload.py` also works without `--parent-source` to verify every compact payload and overlay binding. Reconstruction refuses an existing output and verifies the original parent before and after copying. All commands above are CPU/filesystem operations. The host/guard files are preserved for root review; this bundle makes no claim that either was dispatched.
