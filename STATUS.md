# Current status

Reviewed 2026-09-07 UTC. **No corrected physical-model PPO run has started or completed. No learned-policy video exists for that model.** CPU checks, startup probes, standing results and historical serial/mock policies do not change that.

## Latest physical result

The corrected-source **128/16 refined comparison failed at 23:48 UTC** during the right-middle tibia pushlever's positive 0.04 rad test. All 1,000 standing controls and the first 16 individual motor checks completed; all 32 robots then terminated as upside-down. Driven raw demand peaked at 83.598953 Nm, delivered torque remained capped at 5.5 Nm, and support fell to zero. C-pin separation stayed under its bound at 0.060937 mm, demonstrating that positional closure alone did not establish stable dynamics. The supervisor rejected the result and removed its exact container. [Original report and verified retrieval](artifacts/mkii_fourbar_2026-09-06/refined_rsl501_failure_001/README.md).

The preceding coincident-origin campaign completed full nominal validation at **12:10 UTC**: 32 robots, 1,000 standing plus 2,400 driven controls, 54,400 sampled physics substeps per robot. It failed **one bound**: C-pin separation reached **0.111171 mm**, exceeding the unchanged **0.100 mm** limit. All 18 individual and all 18 grouped motor response checks passed. Driven raw/applied torque peaked at **4.423689 Nm**, within the modeled envelope; at least three feet remained loaded. No resets, non-foot contacts or nonfinite samples were recorded. Refined validation and PPO did not begin. [Original report and byte verification](artifacts/mkii_fourbar_2026-09-06/coincident_nominal_failure_001/README.md).

Frozen source: `ae4f38828181b33c9fc4fd2b8f9d68b03e6c31f1`; functional identity `8fb32bc3e39642338bfbcf37bcc3a7e58d1ab0b8c901174532234248644a93e9`. Remote campaign: `/home/orionh/HEXAPOD_runs/mkii_coincident_layout_v1/campaign_001/fourbar-campaign-20260906T114047Z-73b4f841/campaign.json`. The supervisor rejected the failed report and removed its exact container.

The preceding grid-layout campaign had closure, support and direction failures, reproduced by the [exact motion-prefix replay](artifacts/mkii_fourbar_2026-09-06/prefix_replay_001/README.md). Keeping each flat environment near the world origin substantially improved this comparison; it has not yet established a fully qualified numerical recipe. The separate full-body overlap controls passed their filtered/unfiltered tests. They verify tested collision isolation, not rough-terrain or camera isolation.

## Work and compute policy

The five-minute **Hexapod training progress** automation is removed. It has not been recreated. Local investigation and implementation continue.

The user explicitly resumed work with **“take full training priority”**. The complete refined 128/16 attempt ran from 23:12 to 23:48 UTC on corrected source `5d476d42546bf4f5c84ca39b8c224c8f68f1454b` and failed as described above. Its frozen output remains `/home/orionh/HEXAPOD_runs/mkii_rsl501_compat_v1/refined_diagnosis_001/hexapod-fourbar-validate-20260906T231242Z-6d88ed34/`. No training was admitted.

The new **1,600 Hz / 32-substep** candidate is frozen at `c2af43ca0f384a4c2c7ab8f1d627f309dc78a683`, functional identity `c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc`. All **935 repository tests** pass and GitHub CI is green. Its 308-path manifest and every staged Spark file verify. The policy still runs at 50 Hz; geometry, gains, motor torque limits and acceptance bounds are unchanged. It requires entirely fresh nominal/refined physics admission. The bounded campaign launched at **00:45:23 UTC**, PID `2021559`: `/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/campaign_001/fourbar-campaign-20260907T004524Z-8592e14a/campaign.json`. The 100-control / 3,200-substep probe passed and the full 32-robot nominal phase is now running. [Probe evidence](artifacts/mkii_fourbar_2026-09-07/1600hz_probe_001/README.md). Its external, pinned host wrapper acquires the shared GPU lock separately for every actual phase; it changes no simulation source.

The installed measurement optimization preserves the original closure calculation and all native sensor reads. Complete-row CUDA comparison passed 400 rows with every field byte-identical; the new 32-substep CPU comparison passed 800 rows. Actual simulation profiling measured 55% of host time in physics stepping and 30% in metric capture; these percentages overlap with nested sensor acquisition. The earlier 10–11x synthetic arithmetic speedup was not adopted as a full simulation speed claim. [Profiler analysis](artifacts/mkii_fourbar_2026-09-06/profile_analysis_v2/README.md).

The authoritative shared control is now `HEXAPOD_SHARE_STATUS=NONE`. Actual jobs hold normal per-job locks; the completed jobs have released them. No separate persistent reservation was created. The previous13:33UTC pause and reservation release remain historical evidence. Existing CPU-only weather work and the unrelated viewer continue while the resource gates pass; no competing CUDA workload was present at launch.

Local CPU integration review found an actual Isaac Lab 3 / RSL-RL 5.0.1 startup incompatibility: deprecated model fields were passed to constructors that reject them. The new trainer applies the installed SDK compatibility adapter. Testing the real runner additionally exposed an inference-buffer checkpoint-reload failure. Both fixes now pass actual CPU construction, learning, strict save/reload, further learning, fresh-runner resume and deterministic inference. [Reproduction and verification](artifacts/mkii_fourbar_2026-09-06/rsl501_startup_review/README.md). These are learner API issues, separate from the physical closure failure.

[Capture v2](artifacts/mkii_fourbar_2026-09-06/policy_capture_tools_v2/README.md) and its [one-shot campaign follower](artifacts/mkii_fourbar_2026-09-06/campaign_capture_followup/README.md) are prepared and CPU-tested only. Neither is deployed. They respect the sharing control, create no persistent lease and require an actual completed 512-robot, 1,000-update campaign before recording its learned policy.

## Model and redesign

The intended physical candidate is the **v5 USD: 31 rigid bodies, 30 articulation coordinates, 18 active motors, mass 8.260811322 kg**. It retains the moving pushlever/rod bodies and native bilateral constraints for the CAD parallelogram. The 19-link serial export welds those moving parts and is not a faithful physical substitute. The original serial USD inertia defects and stale closure frames have separate repaired, versioned assets and preserved historical evidence.

The [CAD/reference review](artifacts/mkii_fourbar_2026-09-06/cad_reference_review/README.md) compares export structure with official robot descriptions and separates software corrections from mechanical redesign choices. Reference arm URDFs are not measurement ground truth for this robot's foot contact or dynamics.

The current RS05 envelope permits **5.5 Nm peak**, with **1.2 Nm continuous stall** and a provisional speed/cooling/burst model. Actual battery voltage, braking and regeneration limits, hardware stops, friction, compliance and thermal behavior remain unmeasured. Metal heat conduction alone does not calibrate cooling. Only CAD is complete; single-leg tests precede remaining-parts ordering.

## Next gates

1. Local runner/checkpoint integration and all 935 repository tests pass. The new 308-path release and archived 112-path lineage verify. Source is pushed and staged independently on Spark; older manifests and failures remain preserved.
2. Launch the frozen 1,600 Hz candidate through fresh probe and complete nominal/refined validation. The earlier 800 Hz 128/16 overturn remains a failure and supplies no training admission.
3. Establish full matching nominal/refined physics and convergence on the exact selected training source, without relaxing acceptance bounds. Then run the 64-robot/3-update startup check and separate 512-robot/1,000-update resume.
4. Verify checkpoint bytes, restored optimizer/normalizers and actual finite inference; capture and inspect the learned policy video. Completing PPO alone does not establish useful all-terrain walking or hardware readiness.

The mission remains no-RTK bounded-area zigzag coverage, learned omnidirectional locomotion, vision/LiDAR/IMU navigation and a generic survey-payload interface on Jetson Orin Nano. Mechanical, electrical and test-stand development run in parallel. Hardware readiness, survey accuracy and field dates remain open inputs.

[Living plan](docs/PLAN.md) · [Training design](docs/TRAINING.md) · [Leg test stand](artifacts/project_review_2026-09-04/LEG_TEST_STAND.md) · [Run history](HANDOFF.md) · [RS05 review](docs/RS05_SPEC_REVIEW.md).
