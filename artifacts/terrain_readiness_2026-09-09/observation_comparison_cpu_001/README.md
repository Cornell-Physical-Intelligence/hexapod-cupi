# CPU terrain observation comparison adapters

The first two seams from the [comparison design](../observation_comparison_plan_001/README.md) now execute: placed-course surface queries and blind, ideal-height, or causally delivered map observations. Root and an independent reviewer each ran all 13 tests successfully. No robot traversal or Stage 3 acceptance is claimed.

The [implementation guide](frozen/README.md) explains the exact interfaces. Queries preserve translated/rotated terrain geometry, missing support and separate pit eligibility. Observations use 100 × 100 cell centres, explicit capture/receipt times, the unchanged 250 ms lease and selected-row reset epochs. Already transformed and masked `WorldPoints` are required; this adapter does not simulate sensors or establish calibration. Actor, source and physics selections remain explicitly unresolved.

`frozen/` preserves all ten original payloads and their manifest. `independent_review/` preserves three review payloads and its manifest. The original test code expects its directory immediately below repository `tmp/`; its frozen contents were not rewritten for publication. To replay from the repository root, copy `frozen/` to a fresh `tmp/terrain_observation_replay` directory, then run `PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s tmp/terrain_observation_replay -p 'test_*.py'`. Dependencies are bound in [ORACLE_SHA256.json](frozen/ORACLE_SHA256.json). The included [smoke](frozen/smoke_result.json) is one synthetic point, not a sensor rollout.

Run `python3 artifacts/terrain_readiness_2026-09-09/observation_comparison_cpu_001/verify_payload.py` to verify published bytes and current oracle dependencies. Historical failure logs remain intact.
