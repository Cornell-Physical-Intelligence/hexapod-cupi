# Current status

Last reviewed: 2026-09-04 locally, 5 September UTC for the later Spark runs. **Offline repairs and animated inspection are implemented; corrected serial standing validation now runs on Spark.** The user authorized live execution after the earlier pause. No training is queued. The plan remains open to continued context. [Offline repairs](docs/MKII_STEP1.md) · [Live evidence and limits](artifacts/mkii_step2_2026-09-04/README.md).

[Mission](dar.md) · [Living plan](docs/PLAN.md) · [Prepared future runs](docs/NEXT_RUNS.md) · [Dated audit](artifacts/project_review_2026-09-04/URDF_VALIDATION.md) · [Leg test plan](artifacts/project_review_2026-09-04/LEG_TEST_STAND.md)

## Current program

Build no-RTK bounded-area coverage with a learned omnidirectional gait, vision/LiDAR/IMU navigation, and a generic survey payload interface. The deployment target is Jetson Orin Nano and 18 RS05 motors. Only the CAD is complete; single-leg testing precedes remaining-parts ordering. There is no assembled robot ready for full-body powered testing.

The target is a four-week integrated simulation prototype and six-week candidate to begin hardware transfer, from the resumed development campaign. A first field demo is conditionally targeted H+2–3 weeks, where H is assembled-robot readiness and is currently unknown. Those targets depend on passing the plan's gates.

## Blocking asset findings

The 19-link/18-joint serial URDF passes the performed static checks. Its modeled mass is 8.26081134 kg; installed masses, motor behavior, stops, linkage mapping and payload configuration are not physically verified.

The original deployed USD **fails all 19 link inertia tensor round trips**. A new versioned serial-v2 bundle repairs them from the URDF and passes the all-link gate locally, in Spark's standalone USD library and in Kit's own USD library. It is staged separately on Spark. The new nominal reset removes the approximately 8 mm penetration; the opt-in v2 task/runtime pair has explicit anatomical frames and CAD-specific named action mapping. Historical assets and task defaults remain unchanged.

G0 remains open. The serial URDF welds pushrod/lever motion into the femur, so successful standing does not establish correct four-bar dynamics. The restored linkage viewer exposes the moving mechanism. The earlier approximately 0.5 mm cut-point residual came from stale reference frames: independent recovery of the current CAD pin lines finds transverse discrepancies below 0.00073 mm and a 30 / 77.5 mm planar parallelogram. [The recovered frames and physical implementation plan](artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/README.md) are ready; the physical-loop USD, 30-coordinate / 18-actuator adapter and solver qualification remain to implement.

Live startup, reset, named joint order, standing torque and contact checks now have [dated reports](artifacts/mkii_step2_2026-09-04/README.md). The hardened 32-environment / 1,000-control-step run passes with all 4,000 physical substeps sampled: settled computed peak 0.880025 N·m and six loaded pads throughout the settled window. Startup computed demand reaches 2.364431 N·m, above the imposed 1.6 N·m applied cap but below the manufacturer's 5.5 N·m peak. The [RS05 specification review](docs/RS05_SPEC_REVIEW.md) confirms **1.2 N·m continuous stall**, versus 1.6 or 1.8 N·m at 100 rpm under different cooling conditions. The inherited model already records 5.5 N·m but does not deliver that burst torque or model thermal derating. The 1.6 cap is not universally conservative. Driven joint/direction tests, motor-model qualification and per-collider contact checks remain open. A standing pass does not grant complete G0, training admission or hardware readiness. [The original audit](artifacts/project_review_2026-09-04/URDF_VALIDATION.md) and [step 1](docs/MKII_STEP1.md) remain historical evidence.

## Repository and compute evidence

- Reviewed `main` at `81d7c6f`, including James's six-package refactor, contracts, tests, reward structure, sensor prototypes and new CAD task. Current main has no operational coverage planner or camera/LiDAR/IMU fusion stack. Existing terrain and sensor work is preliminary.
- Main's pinned Python 3.12/uv suite passed **520 tests**; all 112 entries of the archived pipeline manifest matched. The older local branch passed 288 tests. Those checks do not certify the imported physics or real hardware.
- Stopped the identified hexapod queue, wrapper, training container and log followers. The interrupted scratch CAD run was `q_crouch_s52`, near iteration 418/500. Existing checkpoints and logs were preserved. Follow-up inspection found no remaining hexapod training process or loaded hexapod service.
- The Spark project mirror `/home/orionh/HEXAPOD` is a bind-mounted directory, not a Git repository. Live validation uses the separate `/home/orionh/HEXAPOD_runs/mkii_v2_step2_a0f0b39/source` mount and per-run source hashes. Each bounded run has its own supervisor/report/log directory. Unrelated workloads remain untouched; inspect live state before every launch.
- The initial review was documentation only. The subsequent authorized implementation adds a versioned task/runtime contract, verified importer and animated mechanism inspector. Historical checkpoints, tasks and all 112 archived manifest entries remain unchanged. [Current test evidence](artifacts/mkii_step1_2026-09-04/README.md).
- Current compute instruction: **use full available Spark compute until another agent requests sharing through `/home/orionh/SPARK_COMPUTE_COORDINATION.md`**. Read the shared file before each new launch and at checkpoint boundaries during future long runs; the [handoff note](docs/SPARK_COMPUTE_COORDINATION.md) records the procedure. The earlier 60/40 split is only a starting preference if sharing is requested. No quota, MPS partition or shared long-training launcher is enabled; current short validation already uses exclusive admission and full available compute.

## Historical policy evidence

There is **no admitted CAD policy ready for hardware**. Spark has scratch CAD runs, making the old “nobody has trained CAD yet” statement stale, but those runs used the now-flagged USD and are not a corrected-asset baseline.

The archived mock's recorded best checkpoint remains:

```text
artifacts/phase2_recovery_stage2c_stable_forward/checkpoints/current_best/model_2.pt
SHA-256: a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a
```

It misses the moving-command yaw gates and highest-command deck gate. It is neither hardware-ready nor the required parent of the new CAD campaign. Stage2G's improvement claim still lacks an admitted artifact bundle; its tripod-phase reward is outside the requested new gait design. The prior metrics and unresolved evidence questions are preserved verbatim in [the archived status](docs/archive/STATUS-before-autonomy-review-2026-09-04.md). Historical gates and artifacts remain unchanged.

## Next work

Implement the physical four-bar reference and actuator adapter from the recovered CAD evidence, qualify one mechanism, then complete full-body stance and driven joint/direction tests before a fresh CAD training lineage. The conversion, nominal reset and frame/runtime repairs are implemented; live serial standing is a characterized intermediate baseline. Perception, navigation, runtime emulation and leg-test preparation proceed as independent streams. Early policies are useful bootstrap candidates, not guaranteed transferable assets; measured dynamics updates require screening and sometimes retraining.

Open inputs are the hard deadline, accountable stream owners, full-robot delivery/assembly dates, sensor availability, survey footprint/stability and measured actuator/terrain bounds. The [decision record](docs/PLAN.md#9-new-context-and-decision-record) defines how new context changes the plan and which evidence must be re-established.
