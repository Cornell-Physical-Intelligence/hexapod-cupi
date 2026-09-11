# Hexapod progress

<!-- Generated from site/project.json by tools/project_site.py status. Do not edit. -->

Evidence snapshot: 2026-09-11T01:47:58Z · source `a074f7927b4104d2916e73e1304bfb3d967495f1`.

Research is paused. The approved robot model passes a single-robot standing test, but only 10 of 32 simulated robots pass the batch test. Walking training on this model has not started. Videos show earlier robot models; the latest earlier controller failed all 48 stop tests.

[Architecture](ARCHITECTURE.md) · [Visual roadmap](https://cornell-physical-intelligence.github.io/hexapod-cupi/#roadmap)

## Roadmap

### Forward walking reference — Accepted

Use the existing forward-walking video as the gait-quality reference. It shows an earlier, simplified robot model.

Scope: Earlier simplified robot model. Owner: James.

Required proof: Accepted as a visual reference for forward walking. Its motor-torque test failed.

Current limitation: The gait is accepted as a reference. It still exceeds the tested motor-torque limit.

[Evidence](artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300/README.md) · Architecture: R-04, R-11.

Accepted scope: Forward gait appearance on the earlier robot model. Motor-torque, sideways movement and physical-robot tests remain separate.

Next step: **not defined**. Compare future walking results with this video and their measured test results.

### Walk in any direction — Blocked

Walk forward, backward and sideways while turning. Match the reference gait’s smoothness and remain still after a stop command.

Scope: Approved detailed robot model. Owner: Platform lead.

Required proof: Pass direction, stopping, motor-torque and foot-contact tests, and match the accepted reference gait’s visual quality.

Current limitation: Only 10 of 32 simulated robots pass the standing test. Walking training on this model has not started.

[Evidence](artifacts/mkii_updated_2026-09-10/native_standing32_rejected_005/README.md) · Architecture: R-04, R-10, R-11.

Next step: **not defined**. Define and assign the next controller task, including its test cases and pass conditions.

### Use sensors to cross terrain — Needs definition

Use a map of nearby ground to guide walking across uneven surfaces. Keep the ability to walk and stop on level ground.

Scope: Detailed robot model with simulated sensors. Owner: Survey and platform leads.

Required proof: Agree which slopes and obstacles to test. Pass separate test cases while staying within motor and foot-contact limits.

Current limitation: Sensor and ground-map prototypes exist. Walking across terrain with the approved model has not passed testing.

[Evidence](artifacts/terrain_readiness_2026-09-09/qualification_plan_001/QUALIFICATION_PLAN.md) · Architecture: R-02, R-03, R-04.

Next step: **not defined**. Choose the first terrain test and its pass conditions.

### Survey a drawn area — Needs definition

Follow a route through the requested area, stop with a steady sensor platform to scan, and deliver a 3D terrain map. Start on a level, hard surface with no obstacles.

Scope: Physical robot and survey sensor. Owner: James and both leads.

Required proof: On the physical robot, cover the drawn area and deliver the map. Coverage, map-accuracy and platform-motion limits still need agreement.

Current limitation: The robot is not built. M1 is the agreed first simulation test; it needs an owner and reviewer.

[Evidence](ARCHITECTURE.md) · Architecture: R-01, R-03, R-05, R-06, R-07, R-08, R-09, R-10.

Defined increment: **M1 — Save and reload one simulated laser scan**. Scope approved by James.

Use a stationary Mid-360 laser scanner in simulation to measure a known floor and wall. Save the 3D points, reload them in a separate check, and compare them with the scene. This task can start before walking works.

Fixture: Sensor height: 0.25 m. Wall: 0.80 m wide × 0.08 m thick × 0.50 m high, centred 1 m in front of the sensor. Use 20,000 rays per scan. Fixed random seeds: 360 for the scan pattern and 20260824 for sensor noise and timing.

- Reload the points, validity flags and sensor position and orientation without changing their values.
- With noise disabled, every expected floor or wall hit must be within 1 mm of its expected position.
- With the existing sensor noise enabled, at least 95% of expected hits must be within 7 cm. Missing hits count as failures; report false hits separately.

Assignment pending. Assign an owner and reviewer to M1.

## Recorded attempts

- **Standing test: 10 of 32 pass (failed)** — Simulation · approved detailed model. 22 of 32 simulated robots fail the foot-support check. Eight of those also move their joints too quickly while asked to stand still. Walking training on this model has not started. [Evidence](artifacts/mkii_updated_2026-09-10/native_standing32_rejected_005/README.md)
- **Cause of standing failures unresolved (inconclusive)** — Analysis of saved simulation data. Two recorded foot-force signals show the same 73 events below the support threshold. They may come from the same internal source, so agreement between them does not establish the cause. [Evidence](artifacts/mkii_updated_2026-09-10/standing32_channel_analysis_001/README.md)
- **Earlier controller: 0 of 48 stop tests pass (failed)** — Simulation · earlier simplified model. The controller shown after 500 training updates moves in some requested directions, but fails every stop test. Its results apply to that earlier robot model. [Evidence](artifacts/omni_diagnostics_2026-09-09/direct_omni_extended_caps_002/cpu_analysis/remote/analysis/REPORT.md)
- **Experiments paused (passed)** — Team lead’s recorded handoff. The team lead stopped research automation and left no experiment queued. The shared training computer remains reserved. Current work is limited to repository and website updates. [Evidence](artifacts/handoff_2026-09-11/james_001/README.md)

M1 has an agreed test scene and pass conditions. Assign an owner and reviewer next. Define later work with James and the leads.

Current compute rules and reservation records: [operations](docs/SPARK_COMPUTE_COORDINATION.md). This is not live GPU telemetry.

[Full execution snapshot before consolidation](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/a074f7927b4104d2916e73e1304bfb3d967495f1/STATUS.md). Historical results retain their original evidence and gates.
