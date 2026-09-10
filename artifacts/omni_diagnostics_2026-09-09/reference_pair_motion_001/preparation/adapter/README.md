# Paired motion physics adapter 001 — preparation only

This source prepares one full-C forward/stop discriminator with a new, explicit four-support contract. It uses the reviewed paired CPU controller frozen in `reference_pair_motion_001` (owner SHA `636844a6032642014304408f1021e61478be3aed46642fd67686ed9d2f9a6f55`). No actual run is included in this preparation. The root agent retains GPU dispatch, coordination, forecasting pause/restoration and outer-guard ownership.

The source retains all 925 executable/asset parent009 payloads unchanged. Only `source_origin.json` changes in the parent; 20 new files add the paired runtime, contract, entrypoint, scorer and host plus the exact sensor-clock checker already exercised by device001. `SOURCE_BUILD.json` enumerates every difference and binds the 946-file source map. The original scalar wave and its five-support gates remain available unchanged. This new paired branch does not adopt the old 846/849 actor/critic packet, register a PPO task or infer any training admission.

## Exact phases

1. Fresh 32-replica, 1,000-control standing. The existing physical and per-replica quiet scorer must pass against this exact source, plan, full selected-C asset, actuators and solver settings.
2. One independently reset robot for 2,400 controls / 48 s. The unchanged 4 s canonical target startup comes first. The paired sequence then runs for the same 44 s used by its CPU evidence: 2 s reference hold, 24 s forward request `[0.01, 0, 0]`, and 18 s stop. There is no automatic repeat or speed search.

Commands remain body navigation forward = −body Y, left = +body X, yaw = +body Z, with measured SDK XYZW quaternions. The reference initializes from the last loaded executed target and retains actual PD preload. The robot receives only bounded named joint position targets and zero residual actions. No base/joint state overwrite, prescribed body motion, extra physics step or actor is introduced. The current 0.01 m/s candidate is twice the earlier reference's requested speed; that is a request to be tested, not a measured speed claim.

## Live and replay gates

The paired controller requires four retained feet and a measured articulated-COM support margin of at least 50 mm. Both moving feet independently require the unchanged 2 mm measured flight, apex/descent, bounded C2 landing, 12 mm correction/contact-region bounds, and three stable contact confirmations. The first confirmed foot must remain supported while its partner finishes. All six feet must be supported before a new pair starts. A missing foot cannot borrow its partner's qualification or be completed by schedule.

Every pre-reset row retains actual contact points, forces, named joint targets/positions/rates and complete paired state. The source-bound 14-sensor checker proves each clock advances through the eight normal 2.5 ms updates and has a current lazy data buffer. The existing distal classifier requires a finite point, normal-force norm greater than 1 N and the correct tibia-local distal region. Contact data remains at 50 Hz. Buffer-overflow warnings are separately fatal in the host; no unknown point is synthesized from an anchor.

The unchanged recorder captures all eight existing physics updates at 400 Hz, including root/link/COM motion, actual joint angles/rates and requested/applied torques. Each paired control checks every one of its eight torques before another target is emitted: requested torque ≤1.6 N·m and applied torque ≤1.60001 N·m. The initial reset peak is retained separately and excluded only by the existing 4 s settling policy; this is not hardware startup qualification. A fresh standing pass is mandatory even if an earlier source already passed.

`pair_motion_metrics.py` replays complete paired state from each preceding measured row and recomputes every recorded poststep support/drift/torque check from raw data. It verifies all substep indices/counters and exact control endpoint equality. Structures, counters and flags match exactly; floating-point state/check replay tolerance is declared as 1e−8 absolute. Final-row support is checked without advancing the controller. This catches missing contact and tampered recorded margins rather than trusting them.

Actual progress must cover at least half the 0.01 m/s requested forward motion, with corresponding reported forward velocity, over the complete 24 s interval. The original 50 Hz 5 mm position/velocity-integral consistency bound remains a gate. Existing quiet metrics must pass for at least 10 contiguous measured seconds after the reference becomes quiet and a further 2 s settle. Actual contacts, all six independently qualified legs, at least three complete pairs, no reset/nonfoot contact, unchanged target limits/cast/lag bounds and complete substep torque evidence are required. The unresolved SDK rate-integral bias remains visible; no finite-difference substitute or favorable alternate metric is adopted.

## Scope and evidence

The focused tests use explicitly synthetic contact/body/substep streams. They cover complete forward+stop/replay, a hidden intermediate torque spike, the original 5 mm consistency gate, final quiet/support rejection, changed per-foot state, tampered poststep margin, stale/unknown/underloaded contact, exact timing/counters, a foreign standing admission, strict failure finalization and the original ownership-cleanup AST. The first 12-test run passed. A subsequent review made independent poststep recomputation explicit and added a thirteenth regression; `tests_002.log` is authoritative for that source.

`history_before_independent_poststep_replay/` preserves the first unlaunched draft's metric code, source map, source build and preflight, reconstructible from parent009 and the unchanged additions. No deployed/frozen historical source or raw result was changed.

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s tmp/reference_pair_motion_adapter_001 -p 'test_pair_motion_adapter.py'
```

`check_integration.py` records pinned 16-file runtime verification, rejects the actual old009 standing receipt, checks the new synthetic phase identity and rejects failed quiet. Its synthetic receipt does not admit physics. Evidence/build scripts write beside their source; rerun in a fresh copy after freeze.

The root-owned outer guard must bind `source_pair_motion_001/campaign_source_hashes.json`, both GPU locks, current coordination hash, fresh forecasting pause/restoration and previous owned-container absence. The prepared host invocation is:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 SOURCE/tools/launch_pair_motion_spark.py --source SOURCE --output FRESH_OUTPUT
```

Each phase is bounded at 600 s, with a 90 s AppReady deadline, a 45 s repeating startup traceback and unbuffered output. The host is source/asset read-only, verifies exact container identity on cleanup even after client exit, fails closed on unknown Docker inspection, and audits source/assets on failure. Expected outputs are `standing/admission.json`, `paired_forward/state.json`, full/partial `trace.npz`, `reference_states.json`, `paired_poststep_checks.json`, `physics_substeps.npz`, sensor clock channels, named layout/asset/solver/startup records, phase logs/jobs, `campaign.json` and input asset maps. Any failed or incomplete campaign remains unqualified. No Stage 2, terrain, production or physical speed pass is claimed before those actual results exist.
