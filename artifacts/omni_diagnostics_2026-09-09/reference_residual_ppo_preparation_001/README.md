# Reference-residual PPO: verified preparation, physical result pending

This bundle preserves the exact new 846/849-observation PPO consumer, guarded
host, local and Spark preflights, and independent CPU reviews. It contains no
new physical policy result. Fresh standing, exploration calibration, two actual
PPO updates and matched retention still need to run under the guarded campaign.
Stage 2 remains incomplete.

The [consumer](consumer/README.md) uses finite position residuals around the
physically demonstrated slow wave reference. Its first optimization is strictly
stand-only: two updates of 24 controls on 32 replicas. Each of two 20-second
calibration trials must first pass every replica's unchanged quiet and motor
bounds. The initial Gaussian standard deviation of 0.02 remains a measured
proposal until that calibration passes. Zero command retains policy feedback.

A passed smoke is followed by separate cold initial/final checkpoint trials:
24 seconds at 0.005 m/s forward, then 20 seconds stopping. Existing support,
measured landing, progress and quiet criteria remain. This preparation does not
admit other bearings, faster motion, terrain, prompt stops, deployment or long
training. The original 0.03 diagnostic intervention is not treated as a motor
speed limit or pooled with the formal 0.04 action profile.

The [host](host/README.md) preserves the original source009 supervision and
requires fresh 32-by-1000 physical and quiet standing. A failed calibration,
physical step, checkpoint reload, quiet check or initial retention prevents
later allocation. The complete smoke output becomes immutable before retention;
the phase and exact selected checkpoint must match their allocation. Source
checks, both job locks, bounded container lifetimes and exact cleanup remain in
place. The outer forecasting-pause guard is root-owned and is not included here.

## Evidence and review boundaries

- [Local consumer preflight](preflight/local_consumer.json) and
  [Spark host preflight](preflight/spark_host.json) bind the exact source009,
  complete 550-file asset package, device001 proof, bridge and observation tree.
  Their fresh-standing field is deliberately null: preflight is not a new
  standing admission. The compact local raw-result download lacks a duplicated
  asset tree, so the full host preflight used Spark's complete package.
- [Independent session and RSL review](independent_review/session_RSL/README.md)
  preserves all 30 original payloads. Five command/history tests and actual
  RSL-RL 5.0.1 two-update/reload integration passed with synthetic measured
  physics. Actor, critic, normalizers and all 17 Adam entries restored exactly.
  These CPU checkpoints cannot be used as robot policies.
- [Final consumer and host review](independent_review/physical_host/README.md)
  independently passed four actual009 scoring/admission tests, five cleanup
  fault cases and ten host orchestration tests. Every retained scoring-fixture
  array matches its original actual009 trace exactly. These are source and
  CPU checks, not a new GPU calibration.
- [Review lineage](REVIEW_LINEAGE.json) records the earlier independent plan
  hash and final plan hash. The final plan only adds explicit declarations for
  the existing calibration, quiet/retention lengths and lack of extra actor
  input noise. Previously reviewed session, runner, helper and regression bytes
  are unchanged. The final-plan RSL rerun is retained as owner evidence in
  [CPU_RSL_REGRESSION.json](consumer/CPU_RSL_REGRESSION.json).

Raw simulator joint rates and separately valid interval-angle rates remain
observable and separately labelled. Their physical discrepancy is unresolved;
original scoring does not substitute one for the other. The frozen underlying
observation/controller prototype flags remain training-disallowed. The new
consumer grants only its explicit, contingent two-update scope.

## Reproduction without duplicating existing publications

[RECONSTRUCTION.json](RECONSTRUCTION.json) records every copied freeze and exact
published dependency path/hash. Reuse the already published source reconstruction
from `reference_physics_009`, the owner tree from
`reference_policy_observation_005_001`, and the device adapter/actual proof from
`reference_device_smoke_001`. The 926-file source, 160-file observations and full
device raw tree are referenced rather than duplicated here. The existing
admitted 550-file asset package is required by both preflight and runtime.

The copied consumer and host directories must retain their exact nested freeze
inventories. Do not add launch outputs, bytecode or runtime dependencies inside
them. Keep output directories fresh and separate. Only the consumer's top-level
bound `initial.pt` and `final.pt` sidecars are admitted by its strict loader;
framework logging autosaves and synthetic review checkpoints are not substitutes.

Verify this preparation without Isaac or third-party packages:

```
python verify_preparation.py --repository /path/to/HEXAPOD
```

The optional repository argument also checks the five referenced published
input identities. The verifier checks hashes and stated scope; it does not rerun
CPU learning, physics or any GPU job. Root can append a separately frozen
terminal wrapper after the actual campaign; preserve this preparation unchanged.
