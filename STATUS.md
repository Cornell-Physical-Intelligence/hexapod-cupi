# Current status

Last reviewed: 2026-09-04. **Planning only: hexapod jobs stopped; no new training, simulation validation, queue or automatic restart launched by this review.** The user requested that the reviewed goals and steps be pushed to GitHub and remain open to continued context.

[Mission](dar.md) · [Living plan](docs/PLAN.md) · [Prepared future runs](docs/NEXT_RUNS.md) · [Dated audit](artifacts/project_review_2026-09-04/URDF_VALIDATION.md) · [Leg test plan](artifacts/project_review_2026-09-04/LEG_TEST_STAND.md)

## Current program

Build no-RTK bounded-area coverage with a learned omnidirectional gait, vision/LiDAR/IMU navigation, and a generic survey payload interface. The deployment target is Jetson Orin Nano and 18 RS05 motors. Only the CAD is complete; single-leg testing precedes remaining-parts ordering. There is no assembled robot ready for full-body powered testing.

The target is a four-week integrated simulation prototype and six-week candidate to begin hardware transfer, after development resumes. A first field demo is conditionally targeted H+2–3 weeks, where H is assembled-robot readiness and is currently unknown. Those targets depend on passing the plan's gates.

## Blocking asset findings

The 19-link/18-joint serial URDF passes the performed static checks. Its modeled mass is 8.26081134 kg; installed masses, motor behavior, stops, linkage mapping and payload configuration are not physically verified.

The deployed USD **fails all 19 link inertia tensor round trips**. Its converter's principal-axis orientation is inverted relative to the URDF. A numerical inverse-axis diagnostic recovers the tensors, but no deployed asset or importer has been repaired. The local reset starts the foot spheres approximately 8 mm inside the floor. Main's CAD task needs explicit anatomical command-frame wiring, and the default runtime action pipeline still selects mock joint defaults.

These findings block a new training campaign or hardware-readiness claim. Standing logs that passed their implemented gate do not clear the inertia defect. [The audit](artifacts/project_review_2026-09-04/URDF_VALIDATION.md) records reproduction helpers, hashes, measured discrepancies and required fixes.

## Repository and compute evidence

- Reviewed `main` at `81d7c6f`, including James's six-package refactor, contracts, tests, reward structure, sensor prototypes and new CAD task. Current main has no operational coverage planner or camera/LiDAR/IMU fusion stack. Existing terrain and sensor work is preliminary.
- Main's pinned Python 3.12/uv suite passed **520 tests**; all 112 entries of the archived pipeline manifest matched. The older local branch passed 288 tests. Those checks do not certify the imported physics or real hardware.
- Stopped the identified hexapod queue, wrapper, training container and log followers. The interrupted scratch CAD run was `q_crouch_s52`, near iteration 418/500. Existing checkpoints and logs were preserved. Follow-up inspection found no remaining hexapod training process or loaded hexapod service.
- The Spark project mirror `/home/orionh/HEXAPOD` is a bind-mounted directory, not a Git repository. No new source synchronization or GPU run occurred. Unrelated workloads were left alone; inspect the live GPU state again before any later launch.
- No robot model, training config, importer, historical checkpoint or acceptance-gate implementation changed in this review. Publication changes are documentation, audit evidence and read-only comparison helpers.

## Historical policy evidence

There is **no admitted CAD policy ready for hardware**. Spark has scratch CAD runs, making the old “nobody has trained CAD yet” statement stale, but those runs used the now-flagged USD and are not a corrected-asset baseline.

The archived mock's recorded best checkpoint remains:

```text
artifacts/phase2_recovery_stage2c_stable_forward/checkpoints/current_best/model_2.pt
SHA-256: a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a
```

It misses the moving-command yaw gates and highest-command deck gate. It is neither hardware-ready nor the required parent of the new CAD campaign. Stage2G's improvement claim still lacks an admitted artifact bundle; its tripod-phase reward is outside the requested new gait design. The prior metrics and unresolved evidence questions are preserved verbatim in [the archived status](docs/archive/STATUS-before-autonomy-review-2026-09-04.md). Historical gates and artifacts remain unchanged.

## Next work when execution resumes

Repair the USD conversion and add an all-link numerical gate; reconcile nonpenetrating reset, anatomical frames and asset-specific runtime mapping; then pass short import/stance/direction tests before a fresh versioned CAD training lineage. Perception, navigation, runtime emulation and leg-test preparation proceed as independent streams. Early policies are useful bootstrap candidates, not guaranteed transferable assets; measured dynamics updates require screening and sometimes retraining.

Open inputs are the hard deadline, accountable stream owners, full-robot delivery/assembly dates, sensor availability, survey footprint/stability and measured actuator/terrain bounds. The [decision record](docs/PLAN.md#9-new-context-and-decision-record) defines how new context changes the plan and which evidence must be re-established.
