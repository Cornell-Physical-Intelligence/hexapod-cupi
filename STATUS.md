# Hexapod progress

<!-- Generated from site/project.json by tools/project_site.py status. Do not edit. -->

Evidence snapshot: 2026-09-17T18:19:39.672968+00:00 · source `94db110e9c362604ce1cf415bd2dfbcc1bca9f03`.

Researchers passed the optimized forward-trajectory screen, but the tested PPO policies fail walking acceptance. The paired initialization test found no benefit in its one seed. The consolidated kernel supports the proposed simulation-reproduction order in TRAINING; paper-method implementation and fresh-source native admission remain pending.

[Architecture](ARCHITECTURE.md) · [Visual roadmap](https://cornell-physical-intelligence.github.io/hexapod-cupi/#roadmap)

## Roadmap

### Historical forward-walking reference — Accepted

Preserve the accepted forward-walking video as a historical gait-quality reference. It shows an earlier, simplified model and does not qualify the confirmed mass-corrected robot.

Scope: Earlier simplified robot model. Owner: James.

Required proof: Accepted as a visual reference for forward walking. Its motor-torque test failed.

Current limitation: The gait is accepted as a reference. It still exceeds the tested motor-torque limit.

[Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300/README.md) · Architecture: R-04, R-11.

Accepted scope: Forward gait appearance on the earlier robot model. Motor-torque, sideways movement and physical-robot tests remain separate.

Next step: **not defined**. Compare future walking results with this video and their measured test results.

### Stand, walk and stop — Blocked

Qualify signed translation, turns, transitions and quiet stops on the approved model. Historical standing admissions apply to their frozen sources; obtain matching admission for a new allocation.

Scope: Approved detailed robot model. Owner: Platform lead.

Required proof: Pass standing admission on the exact confirmed model before walking training. Then pass direction, stopping, motor-torque and foot-contact tests, and match the accepted historical reference gait’s visual quality.

Current limitation: Both paired PPO arms fail forward and quiet-stop checks. The optimized trajectory passes one forward screen. A learned policy still needs the full direction, transition, stop, motor and contact suite and human gait acceptance.

[Evidence](docs/TRAINING.md) · Architecture: R-04, R-10, R-11.

Next step: **not defined**. Start the TRAINING reward audit under issue #18. Resolve paper formulas and the stationary tracking criterion before implementing a new reward schema.

### Use sensors to cross terrain — Blocked

After flat-ground walking and stopping qualify on the confirmed mass-corrected model, restart terrain qualification using a map of nearby ground. Preserve admitted flat-ground behavior.

Scope: Detailed robot model with simulated sensors. Owner: Survey and platform leads.

Required proof: Agree which slopes and obstacles to test. Pass separate test cases while staying within motor and foot-contact limits.

Current limitation: Canonical flat-ground walking and stopping have not qualified. Sensor and ground-map prototypes remain historical preparation; the first terrain envelope and pass conditions still need definition.

[Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/terrain_readiness_2026-09-09/qualification_plan_001/QUALIFICATION_PLAN.md) · Architecture: R-02, R-03, R-04.

Next step: **not defined**. After canonical walking and stopping qualify, define the first terrain test and its pass conditions.

### Survey a drawn area — Blocked

Follow a route through the requested area, stop with a steady sensor platform to scan, and deliver a 3D terrain map. Start on a level, hard surface with no obstacles.

Scope: Physical robot and survey sensor. Owner: James and both leads.

Required proof: On the physical robot, cover the drawn area and deliver the map. Coverage, map-accuracy and platform-motion limits still need agreement.

Current limitation: The robot remains unbuilt and walking remains unqualified. Issues #19 and #20 cover independent simulation work; mapping and mission software remain pending.

[Evidence](ARCHITECTURE.md) · Architecture: R-01, R-03, R-05, R-06, R-07, R-08, R-09, R-10.

Defined increment: **M1 — Save and reload one simulated laser scan**. Scope approved by James.

Use a stationary Mid-360 laser scanner in simulation to measure a known floor and wall. Save the 3D points, reload them in a separate check, and compare them with the scene. This task can start before walking works.

Fixture: Sensor height: 0.25 m. Wall: 0.80 m wide × 0.08 m thick × 0.50 m high, centred 1 m in front of the sensor. Use 20,000 rays per scan. Fixed random seeds: 360 for the scan pattern and 20260824 for sensor noise and timing.

- Reload the points, validity flags and sensor position and orientation without changing their values.
- With noise disabled, every expected floor or wall hit must be within 1 mm of its expected position.
- With the existing sensor noise enabled, at least 95% of expected hits must be within 7 cm. Missing hits count as failures; report false hits separately.

Assignment pending. Assign owners and reviewers for #19 mapping and #20 survey rehearsal. Preserve the M1 fixture; define combined-map error and mission-rehearsal limits before their tests.

## Recorded attempts

- **Exact one- and 128-robot standing admission adopted (passed)** — Lead-reviewed simulation standing admission002 · confirmed mass-corrected model · full independent raw audit. The lead adopted one002 and all 128 rescored batch128 replicas with matching model/controller/layout identity and contact checks. Raw captures remain on Spark with hash-bound metadata. This standing admission covers its frozen source; the consolidated source requires fresh admission before training. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/restart_2026-09-14/paper_walk_execution_001/admission_002/verification.json)
- **Improved optimized reference passes the forward screen (passed)** — One admitted Isaac Sim replica; 1000 controls and 8000 recorded physics steps; no PPO. The optimizer adds root-speed tracking and a target-curvature term. Native planar tracking error falls from 0.033271 to 0.003029 m/s against the unchanged 0.025 m/s limit. The robot moves 0.974348 m forward in 20 seconds. The audit verifies 36 transferred files, reproduces servo targets and reclassifies 130821 contact patches. Native motor, joint and contact checks pass. This one-command result does not qualify turning, stopping or Stage 2. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/ppo_reference_comparison_20260917/review_smooth_001/RESULT.json)
- **The saved PPO policy fails forward, quiet and stop probes (failed)** — One admitted Isaac Sim replica; final checkpoint1200; 3050 controls and 24400 physics samples. The audit reproduces the original verdicts and reclassifies 530534 contact patches. CPU inference from the saved weights matches recorded actions within 0.00000144. Forward tracking error is 0.051091 m/s against the unchanged 0.025 m/s bound; mean forward speed is 0.000423 m/s. Quiet includes three native joint-speed violations and stopping includes one nonfoot contact sample. All cases retain complete videos and force summaries. Stage 2 remains incomplete. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/ppo_reference_comparison_20260917/review_vanilla_001/RESULT.json)
- **Example initialization produces no passing forward checkpoint (failed)** — Native Isaac Sim; approved detailed model; one paired seed; equal PPO budgets; unchanged 0.040 rad per 20 ms limiter. Each arm completed 1200 updates and 3686400 transitions. Both fail forward tests at updates 0, 300, 600 and 1200; the copied actor fails before PPO. Final mean speeds are 0.004811 m/s from scratch and 0.000961 m/s after initialization against 0.05 m/s. Planar errors are 0.051045 and 0.053322 m/s against the 0.025 m/s bound. Both fail motor-demand and quiet-stop checks. The benefit rule fails for this seed; other seeds and AMP remain untested. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/forward_example_ppo_20260917/COMPARISON_001.json)
- **We record force and motor load for both final policies (passed)** — 400 Hz native captures; 18-second forward window after two seconds of settling. Mean vertical ground support is 73.246 N for scratch and 73.227 N for example initialization. Mean absolute applied motor torque is 0.488782 and 0.405900 N·m. Per-foot forces and per-joint requested and applied torque remain in the force summaries. These values describe failed, low-speed motions and do not establish efficiency at matched walking speed. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/forward_example_ppo_20260917/COMPARISON_001.json)

Use issue #18 for walking, #19 for simulated mapping and #20 for mission rehearsal. Preserve the approved model and numerical gates. Record contact force and motor torque in native evaluations; these loads add no acceptance limits. Follow TRAINING for the proposed reproduction order.

Current compute rules and reservation records: [operations](docs/SPARK_COMPUTE_COORDINATION.md). This is not live GPU telemetry.

[Full execution snapshot before consolidation](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/STATUS.md). Historical results retain their original evidence and gates.
