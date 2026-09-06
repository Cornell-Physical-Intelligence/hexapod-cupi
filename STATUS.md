# Current status

Reviewed 2026-09-06. **No corrected physical-model PPO run has started or completed. No learned-policy video exists for that model.** CPU checks, startup probes, standing results and historical serial/mock policies do not change that.

## Latest physical result

The coincident-origin campaign completed full nominal validation at **12:10 UTC**: 32 robots, 1,000 standing plus 2,400 driven controls, 54,400 sampled physics substeps per robot. It failed **one bound**: C-pin separation reached **0.111171 mm**, exceeding the unchanged **0.100 mm** limit. All 18 individual and all 18 grouped motor response checks passed. Driven raw/applied torque peaked at **4.423689 Nm**, within the modeled envelope; at least three feet remained loaded. No resets, non-foot contacts or nonfinite samples were recorded. Refined validation and PPO did not begin. [Original report and byte verification](artifacts/mkii_fourbar_2026-09-06/coincident_nominal_failure_001/README.md).

Frozen source: `ae4f38828181b33c9fc4fd2b8f9d68b03e6c31f1`; functional identity `8fb32bc3e39642338bfbcf37bcc3a7e58d1ab0b8c901174532234248644a93e9`. Remote campaign: `/home/orionh/HEXAPOD_runs/mkii_coincident_layout_v1/campaign_001/fourbar-campaign-20260906T114047Z-73b4f841/campaign.json`. The supervisor rejected the failed report and removed its exact container.

The preceding grid-layout campaign had closure, support and direction failures, reproduced by the [exact motion-prefix replay](artifacts/mkii_fourbar_2026-09-06/prefix_replay_001/README.md). Keeping each flat environment near the world origin substantially improved this comparison; it has not yet established a fully qualified numerical recipe. The separate full-body overlap controls passed their filtered/unfiltered tests. They verify tested collision isolation, not rough-terrain or camera isolation.

## Work and compute policy

The five-minute **Hexapod training progress** automation is removed. It has not been recreated. Local investigation and implementation continue.

The user explicitly resumed work with **“take full training priority”** after reviewing the prepared comparison. At **23:12:41 UTC**, supervisor **1959403** launched the complete refined 128/16 comparison using corrected source `5d476d42546bf4f5c84ca39b8c224c8f68f1454b`. Output is `/home/orionh/HEXAPOD_runs/mkii_rsl501_compat_v1/refined_diagnosis_001/hexapod-fourbar-validate-20260906T231242Z-6d88ed34/`. Its exact container is `43e61b33e0199341f9f4d1527bd23794717a1d364a89b40c074b1f2d7671ec65`; CUDA process1959662 was observed and standing reached100 controls at23:15UTC. This is active validation, not PPO. [Launch record](artifacts/mkii_fourbar_2026-09-06/rsl501_release/refined_launch_001.json).

The authoritative shared control is now `HEXAPOD_SHARE_STATUS=NONE`. The actual job holds normal per-job locks, including `/opt/wx/gpu.lock` through a no-fork wrapper; they release when its supervisor exits. No separate persistent reservation was created. The previous13:33UTC pause and reservation release remain historical evidence. Existing CPU-only weather work and the unrelated viewer continue while the resource gates pass; no competing CUDA workload was present at launch.

Local CPU integration review found an actual Isaac Lab 3 / RSL-RL 5.0.1 startup incompatibility: deprecated model fields were passed to constructors that reject them. The new trainer applies the installed SDK compatibility adapter. Testing the real runner additionally exposed an inference-buffer checkpoint-reload failure. Both fixes now pass actual CPU construction, learning, strict save/reload, further learning, fresh-runner resume and deterministic inference. [Reproduction and verification](artifacts/mkii_fourbar_2026-09-06/rsl501_startup_review/README.md). These are learner API issues, separate from the physical closure failure.

[Capture v2](artifacts/mkii_fourbar_2026-09-06/policy_capture_tools_v2/README.md) and its [one-shot campaign follower](artifacts/mkii_fourbar_2026-09-06/campaign_capture_followup/README.md) are prepared and CPU-tested only. Neither is deployed. They respect the sharing control, create no persistent lease and require an actual completed 512-robot, 1,000-update campaign before recording its learned policy.

## Model and redesign

The intended physical candidate is the **v5 USD: 31 rigid bodies, 30 articulation coordinates, 18 active motors, mass 8.260811322 kg**. It retains the moving pushlever/rod bodies and native bilateral constraints for the CAD parallelogram. The 19-link serial export welds those moving parts and is not a faithful physical substitute. The original serial USD inertia defects and stale closure frames have separate repaired, versioned assets and preserved historical evidence.

The [CAD/reference review](artifacts/mkii_fourbar_2026-09-06/cad_reference_review/README.md) compares export structure with official robot descriptions and separates software corrections from mechanical redesign choices. Reference arm URDFs are not measurement ground truth for this robot's foot contact or dynamics.

The current RS05 envelope permits **5.5 Nm peak**, with **1.2 Nm continuous stall** and a provisional speed/cooling/burst model. Actual battery voltage, braking and regeneration limits, hardware stops, friction, compliance and thermal behavior remain unmeasured. Metal heat conduction alone does not calibrate cooling. Only CAD is complete; single-leg tests precede remaining-parts ordering.

## Next gates

1. Local runner/checkpoint integration checks pass. The complete repository suite passes **931 tests in 75.778 seconds**, and the new 298-path compatibility release plus the archived 112-path lineage verify. Freeze and share this corrected source; older manifests and results remain preserved.
2. The complete 128/16 comparison is running to diagnose the remaining closure error. A failed nominal plus a refined pass cannot admit PPO. Use its result to select the next numerical/controller candidate and fresh complete qualification.
3. Establish full matching nominal/refined physics and convergence on the exact selected training source, without relaxing acceptance bounds. Then run the 64-robot/3-update startup check and separate 512-robot/1,000-update resume.
4. Verify checkpoint bytes, restored optimizer/normalizers and actual finite inference; capture and inspect the learned policy video. Completing PPO alone does not establish useful all-terrain walking or hardware readiness.

The mission remains no-RTK bounded-area zigzag coverage, learned omnidirectional locomotion, vision/LiDAR/IMU navigation and a generic survey-payload interface on Jetson Orin Nano. Mechanical, electrical and test-stand development run in parallel. Hardware readiness, survey accuracy and field dates remain open inputs.

[Living plan](docs/PLAN.md) · [Training design](docs/TRAINING.md) · [Leg test stand](artifacts/project_review_2026-09-04/LEG_TEST_STAND.md) · [Run history](HANDOFF.md) · [RS05 review](docs/RS05_SPEC_REVIEW.md).
