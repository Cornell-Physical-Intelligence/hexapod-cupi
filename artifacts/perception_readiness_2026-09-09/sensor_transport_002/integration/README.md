# Sensor transport v2 working-tree integration

The current runtime matched reviewed original SHA101f6a59488f702117e93da87354b0d68c3d475e50421f445a12d9caf1491336 before replacement. The owner nine-payload map and independent six-payload map were verified. The three supplied runtime/test/old-fixture files were copied exactly; the phase3 README now specifies ordering, finite simulation clocks, full versus row reset, inference-safe persistent state, float32 public age and unchanged map lease.

All21 focused sensor tests pass:15 transport regressions and six existing hardware-contract tests. These include old-fail/new-pass counterexamples and401-read nominal bitwise parity. No new actor, noise setting, physics, map lease or qualification gate was introduced. No GPU or remote operation occurred.

INTEGRATION_RESULT.json records all four changed paths and final hashes. Root owns the required release identity update, STATUS/PLAN/central site record, site and lineage checks, and publication under docs/PROJECT_SITE.md. This receipt records the verified working tree; it does not claim those integration steps or any sensor hardware/terrain qualification are complete.
