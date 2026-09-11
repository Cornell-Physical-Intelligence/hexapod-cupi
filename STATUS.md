# Hexapod progress

<!-- Generated from site/project.json by tools/project_site.py status. Do not edit. -->

Evidence snapshot: 2026-09-11T01:47:58Z · source `a074f7927b4104d2916e73e1304bfb3d967495f1`.

Research is paused for James; no new experiment is queued. The approved detailed robot passes single-robot standing, but its completed batch screen passes 10/32 replicas. Canonical PPO has not started. Historical walking remains a comparison, with 0/48 quiet trials passing.

[Architecture](ARCHITECTURE.md) · [Visual roadmap](https://cornell-physical-intelligence.github.io/hexapod-cupi/#roadmap)

## Roadmap

### Walking — Accepted

The first forward walking policy is the accepted visual benchmark. This establishes the starting point for the new approach.

Scope: Historical C-study model. Owner: James.

Required proof: The accepted clip is a forward-gait benchmark; it does not qualify every direction or the physical hardware.

Current limitation: Historical forward visual comparison accepted; requested-torque failure remains on record.

[Evidence](artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300/README.md) · Architecture: R-04, R-11.

Accepted scope: Historical forward visual benchmark only; torque, omnidirectional and hardware qualification excluded.

Next step: **not defined**. What does the canonical model need to establish before this comparison can be used?

### Omnidirectional movement — Blocked

Translate in any direction and turn at the same time. Match the forward benchmark’s smoothness and stand still when commanded to stop.

Scope: Canonical detailed model. Owner: Platform lead.

Required proof: Numeric gates AND visual quality matching accepted forward Benchmark 1.

Current limitation: Canonical batch-standing admission fails; omnidirectional policy qualification remains open.

[Evidence](artifacts/mkii_updated_2026-09-10/native_standing32_rejected_005/README.md) · Architecture: R-04, R-10, R-11.

Next step: **not defined**. What is the smallest evidence-producing increment that addresses the current admission blocker?

### Terrain and perception — Needs definition

Cross increasingly difficult ground, then use timely sensor information to select safe movement. Retain the flat-ground controller’s capability.

Scope: Canonical model with causal sensing. Owner: Survey and platform leads.

Required proof: A frozen terrain envelope, untouched qualification cases and preserved motor/contact limits.

Current limitation: Terrain fixtures and sensing prototypes exist; canonical traversal is unqualified.

[Evidence](artifacts/terrain_readiness_2026-09-09/qualification_plan_001/QUALIFICATION_PLAN.md) · Architecture: R-02, R-03, R-04.

Next step: **not defined**. What is the smallest terrain and sensing envelope worth demonstrating?

### Autonomous surveying — Needs definition

Cover an operator-defined area on the accurate robot while maintaining a steady deck for data collection.

Scope: Calibrated robot and survey payload. Owner: James and both leads.

Required proof: Calibrated hardware and a complete bounded-area mission demonstration.

Current limitation: An integrated bounded survey with useful measurements has not been demonstrated.

[Evidence](ARCHITECTURE.md) · Architecture: R-01, R-03, R-05, R-06, R-07, R-08, R-09, R-10.

Next step: **not defined**. What smallest region, payload quality and operating envelope would make a useful first survey?

## Recorded attempts

- **Canonical standing (failed)** — Isaac simulation · detailed direct-drive model. One robot passes standing. The completed 32-robot screen accepts 10/32; 22 fail measured support, including eight that also fail quiet joint-rate limits. No canonical PPO has run. [Evidence](artifacts/mkii_updated_2026-09-10/native_standing32_rejected_005/README.md)
- **Recorded contact analysis (inconclusive)** — CPU replay · detailed model recording. Both recorded normal-force channels agree on 73 below-bound events. They may share a backend; this does not identify a physical cause or prove buffer completeness. [Evidence](artifacts/mkii_updated_2026-09-10/standing32_channel_analysis_001/README.md)
- **Historical walking policy (failed)** — Isaac simulation · historical C-study model. The 500-update policy retains partial motion but passes 0/48 quiet trials. It does not qualify the canonical robot. [Evidence](artifacts/omni_diagnostics_2026-09-09/direct_omni_extended_caps_002/cpu_analysis/remote/analysis/REPORT.md)
- **Research paused for handoff (passed)** — Recorded operations snapshot. The team lead recorded research agents and continuation stopped, with no native job queued. The external Spark reservation stays in place. Repository cleanup and publication do not resume research. [Evidence](artifacts/handoff_2026-09-11/james_001/README.md)

Define the smallest useful next increment with James and the two subleads after cleanup. No experiment or contributor assignment is implied by these markers.

Current compute rules and reservation records: [operations](docs/SPARK_COMPUTE_COORDINATION.md). This is not live GPU telemetry.

[Full execution snapshot before consolidation](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/a074f7927b4104d2916e73e1304bfb3d967495f1/STATUS.md). Historical results retain their original evidence and gates.
