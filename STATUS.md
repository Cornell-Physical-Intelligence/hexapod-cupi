# Hexapod progress

<!-- Generated from site/project.json by tools/project_site.py status. Do not edit. -->

Evidence snapshot: 2026-09-17T18:19:39.672968+00:00 · source `94db110e9c362604ce1cf415bd2dfbcc1bca9f03`.

The optimized forward reference passes its screen. Standard PPO and the paired action-imitation initialization fail to produce an accepted walking policy. The paired run used 1,200 PPO updates per arm; the copied actor failed before PPO. We found no benefit from that initialization in the tested seed. Stage 2 remains incomplete. Use the locomotion kernel for current code and the linked records for measured results.

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

Restart qualification on the mass-corrected model confirmed on 14 September 2026. First pass standing admission, then train forward, backward and sideways motion with turning and quiet stops. Compare gait quality with the historical reference.

Scope: Approved detailed robot model. Owner: Platform lead.

Required proof: Pass standing admission on the exact confirmed model before walking training. Then pass direction, stopping, motor-torque and foot-contact tests, and match the accepted historical reference gait’s visual quality.

Current limitation: Both controlled PPO arms fail every scheduled forward test and both final stopping diagnostics. The improved prepared trajectory passes its forward screen. A learned policy still needs to pass direction, transition, quiet-stop, motor and contact tests, followed by human gait acceptance.

[Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/locomotion/README.md) · Architecture: R-04, R-10, R-11.

Next step: **not defined**. Review the failed example initialization before defining the next learning increment with James and the platform lead.

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

Current limitation: Fresh survey qualification depends on canonical movement and terrain qualification, and the physical robot is not built. The agreed independent M1 scan test still needs an owner and reviewer; it does not qualify robot surveying.

[Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/ARCHITECTURE.md) · Architecture: R-01, R-03, R-05, R-06, R-07, R-08, R-09, R-10.

Defined increment: **M1 — Save and reload one simulated laser scan**. Scope approved by James.

Use a stationary Mid-360 laser scanner in simulation to measure a known floor and wall. Save the 3D points, reload them in a separate check, and compare them with the scene. This task can start before walking works.

Fixture: Sensor height: 0.25 m. Wall: 0.80 m wide × 0.08 m thick × 0.50 m high, centred 1 m in front of the sensor. Use 20,000 rays per scan. Fixed random seeds: 360 for the scan pattern and 20260824 for sensor noise and timing.

- Reload the points, validity flags and sensor position and orientation without changing their values.
- With noise disabled, every expected floor or wall hit must be within 1 mm of its expected position.
- With the existing sensor noise enabled, at least 95% of expected hits must be within 7 cm. Missing hits count as failures; report false hits separately.

Assignment pending. Retain M1’s agreed fixture and thresholds; assign its owner and reviewer without treating it as robot survey qualification. Define the survey increment after its canonical prerequisites qualify.

## Recorded attempts

- **Exact one- and 128-robot standing admission adopted (passed)** — Lead-reviewed simulation standing admission002 · confirmed mass-corrected model · full independent raw audit. The lead adopted the unchanged one002 result and all 128 independently rescored batch128 replicas. Exact model/controller/layout identity, full contact classification, recorded lifecycle and cleanup pass. Full native raw traces remain on Spark with local hash-bound compact metadata and audit. This admits only the exact simulation standing layouts; admission001 remains unchanged, and walking, hardware and Stage 2 are unqualified. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/restart_2026-09-14/paper_walk_execution_001/admission_002/verification.json)
- **Improved optimized reference passes the forward screen (passed)** — One admitted Isaac Sim replica; 1000 controls and 8000 recorded physics steps; no PPO. The optimizer adds root-speed tracking and a target-curvature term. Native planar tracking error falls from 0.033271 to 0.003029 m/s against the unchanged 0.025 m/s limit. The robot moves 0.974348 m forward in 20 seconds. The audit verifies 36 transferred files, reproduces servo targets and reclassifies 130821 contact patches. Native motor, joint and contact checks pass. This one-command result does not qualify turning, stopping or Stage 2. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/ppo_reference_comparison_20260917/review_smooth_001/RESULT.json)
- **The saved PPO policy fails forward, quiet and stop probes (failed)** — One admitted Isaac Sim replica; final checkpoint1200; 3050 controls and 24400 physics samples. The audit reproduces the original verdicts and reclassifies 530534 contact patches. CPU inference from the saved weights matches recorded actions within 0.00000144. Forward tracking error is 0.051091 m/s against the unchanged 0.025 m/s bound; mean forward speed is 0.000423 m/s. Quiet includes three native joint-speed violations and stopping includes one nonfoot contact sample. All cases retain complete videos and force summaries. Stage 2 remains incomplete. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/ppo_reference_comparison_20260917/review_vanilla_001/RESULT.json)
- **We complete both controlled PPO training arms (passed)** — RSL-RL 5.0.1; one paired seed; fixed 0.05 m/s forward command; 128 admitted replicas. Each arm completes 1200 PPO updates and 3686400 transitions. Each final optimizer records 24000 steps per parameter, and each run records 29491200 force/torque samples. Both native jobs exit with code 0 and complete exact-container cleanup. The initial saved critic and actor normalization state match; both PPO optimizers start empty. These results establish matched training execution. Policy performance requires the scheduled native evaluations. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/forward_example_ppo_20260917/README.md)
- **Copying the example does not pass the initial walking test (failed)** — One admitted Isaac Sim replica; example-arm update zero; unchanged 0.040 rad per 20 ms limiter. Before PPO, the copied actor completes the forward trial with mean speed 0.005338 m/s and planar tracking error 0.056366 m/s against the 0.025 m/s bound. The quiet trial ends at control 149 with joint-limit and joint-speed violations. The forward-to-stop trial ends at control 467, before the stop command at control 500. The audit reproduces actions, actuator output and force summaries; videos preserve both failed prefixes. Low action-prediction error on the demonstration does not establish closed-loop walking. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/forward_example_ppo_20260917/review_001/example_update000000.json)
- **Example initialization produces no passing forward checkpoint (failed)** — Native Isaac Sim; approved detailed model; one paired seed; equal PPO budgets; unchanged 0.040 rad per 20 ms limiter. Both arms fail all four scheduled forward tests. Final mean speeds are 0.004811 m/s from scratch and 0.000961 m/s after example initialization, against 0.05 m/s. Final planar errors are 0.051045 and 0.053322 m/s, above the 0.025 m/s limit. Both final policies fail motor-demand and quiet-stop checks. The final example quiet trial terminates at control 490 of 1000. The declared benefit rule fails for this pair; other seeds and motion-prior methods remain untested. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/forward_example_ppo_20260917/COMPARISON_001.json)
- **We record force and motor load for both final policies (passed)** — 400 Hz native captures; 18-second forward window after two seconds of settling. Mean vertical ground support is 73.246 N for scratch and 73.227 N for example initialization. Mean absolute applied motor torque is 0.488782 and 0.405900 N·m. Per-foot forces and per-joint requested and applied torque remain in the force summaries. These values describe failed, low-speed motions and do not establish efficiency at matched walking speed. [Evidence](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/5e65918020dc9ef2d72da8dad0a346016b98728d/artifacts/forward_example_ppo_20260917/COMPARISON_001.json)

Use the locomotion kernel for canonical work. Preserve the approved model and numerical gates. Each native evaluation records ground-contact force and motor torque. Treat those loads as descriptive measurements. The Spark reservation remains retained and the historical heartbeat remains paused.

Current compute rules and reservation records: [operations](docs/SPARK_COMPUTE_COORDINATION.md). This is not live GPU telemetry.

[Full execution snapshot before consolidation](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/STATUS.md). Historical results retain their original evidence and gates.
