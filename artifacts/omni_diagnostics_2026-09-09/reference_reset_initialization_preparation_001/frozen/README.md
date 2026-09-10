# Bounded reset-initialization A/B diagnostic

Compare two explicit joint-reset distributions while leaving the actual C-study root reset, geometry, masses, gains, 1.6 N·m requested-torque cap, solver configuration and recovery controller unchanged. This is a zero-residual reliability experiment, not a policy test, a new gait, or a reason to postpone the direct omnidirectional branch.

The existing failure review attributes the terminal rejection to row 8, RM tibia `revolute_2_1`: the final eight physical substeps requested 1.62764847 N·m despite six contacts, zero residual and a canonical target. The preceding two seconds remained over the cap. This motivates a reset-pose comparison; it does not establish the underlying solver/contact mechanism.

## Exact experiment

Use three phases. Each has a 600 s wall bound; proposed outer bound is 1,920 s plus 180 s cleanup, with a 40-minute restoration fallback. Root assigns the actual unit/pause and decides whether to dispatch.

1. **Fresh standing:** unchanged source009 32 × 1,000 physical/quiet admission, with its original reset distribution and all original gates. Both diagnostic arms bind this fresh receipt.
2. **A — recorded failing pose:** a fresh 32-row scene; initialize every row to the exact recorded 18-joint reset vector from failed row 8. This includes the original world row 8 and repeats the same vector across other origins. Joint names are resolved from recorded runtime names, not assumed indices.
3. **B — canonical pose:** a second fresh, otherwise identical 32-row scene; initialize each row to its exact imported named canonical joint position. This is a point distribution, not a claim that randomized training resets are robust.

The independent variable is actual reset joint position **and its synchronized first executable target**. Original reset RNG draws still occur, so the custom joint-position replacement does not perturb later random-number consumption. Joint velocity, root pose/quaternion/velocity, ground, origin layout, actuator settings and sensor-reset operations remain inherited. No base or joint state is overwritten while a physical control is being executed.

Each arm has exactly these maximum five epochs:

| Start control | Selected rows | Following physical recovery |
|---|---|---|
| 0 | All 32 | 200 controls |
| 200 | All 32 | 200 controls |
| 400 | The original 26 timeout rows, including row 8 | 200 controls |
| 600 | Their six-row complement | 200 controls |
| 800 | All 32 | 200 controls |

A failure stops that arm immediately and preserves its prefix. The **other independent arm** may run once as predeclared; no failed row is quietly retried until it passes. Total maximum: 1,000 controls per arm, plus unchanged standing. No actor, optimizer, learned checkpoint, or policy evaluation is loaded.

This reproduces the recorded reset vector and the inherited root convention. It does **not** recreate the preceding 2,600-control learned trajectory, full articulation/contact warm-start state, or all native solver caches. If A does not reproduce the historical failure, that limitation remains part of the result.

## Existing gates and measurements

`diagnostic_driver.py` reuses frozen consumer003 `MovingSession`, `RecoveryTargets`, sensor freshness, raw pre-reset capture, history initialization and the 400 Hz recorder. The recovery remains the exact two-second quintic plus two-second hold. Canonical initialization makes this target trajectory constant rather than replacing it with a different controller.

At recovery completion, preserve the existing test: exactly 200 controls, six distal supports and no original terminal reason, including the last eight requested-torque substeps over 1.6 N·m, nonfoot/support flags or native termination. Native termination during recovery, nonfinite/stale measurements and an applied torque above float32(1.6) remain fatal. Hold rows must continue to satisfy the original active zero-command physics checks while other rows recover. Target joint reserve and derivative bounds remain unchanged.

Record all 400 Hz requested/applied torques throughout recovery, including excluded initial peaks. Do not introduce an all-recovery torque gate that the original admission did not have, and do not describe its existing settling exclusion as qualified hardware startup. Report the final 100 controls separately to detect persistent over-limit equilibria. Preserve raw joint angles and reported rates independently.

The current 200-control epoch is too short for a fresh ten-second quiet window. The unchanged fresh-standing phase supplies its original quiet admission. Per-epoch quiet measurements may be reported as diagnostics with their actual duration; do not manufacture a ten-second quiet pass or substitute it for the original recovery condition.

After each original reset and after the joint-only intervention, record named joint state/targets, full root position/XYZW quaternion/velocity and exact selected mask. Check all unselected buffers unchanged at both boundaries. Existing reset-clock rebasing and history initialization must prove selected epochs were cleared and unselected rows remained untouched. The first observation after recovery must retain the frozen one-frame/no-cross-reset interval-rate rules.

Planned resets are explicitly tagged `diagnostic_planned_reset_mask` and `diagnostic_reset_is_not_training_timeout`. The reused training-ledger truncation field marks that declared reset event solely for the existing reset/history verifier. Native termination/truncation fields remain preserved; no learner or value bootstrap consumes these records. Unplanned terminal events reject the arm before another state reset occurs.

## Implementation and remaining dispatch seam

- `plan.json` is the machine-readable protocol and exact recorded pose. `INPUT_BINDINGS.json` identifies the frozen consumer, failure report, raw trace/substep/episode hashes and installed API source receipts.
- `reset_pose_override.py` wraps the original `_reset_idx` only when the next declared counter/mask is explicitly armed. It runs the original reset once, changes selected joint position/position targets with the inspected SDK index methods, synchronizes controller state, checks root/unselected preservation and restores the method on exit. It never steps physics or writes a root pose.
- `diagnostic_driver.py:run_arm` accepts an already constructed source-verified environment and executes the bounded zero-command arm using the frozen recovery session. It is a small adapter, not a standalone launch script.
- `prepare.py` verifies all frozen consumer files and four original raw inputs before producing the exact plan. The joint-name source is `physics_substeps.npz`; the ordinary trace does not contain the name vector.

**Dispatch is not ready or authorized by this CPU bundle.** Root's final integration can reuse the exact reviewed `run_owned`/`owned_container`/lock cleanup from `tmp/reference_learning_ppo_launch_001/launch_learning_ppo_spark.py`. It needs only a separately bound, three-phase entrypoint/command/result adapter: source009 standing, A, B. Preserve the 90 s AppReady/45 s traceback/unbuffered startup and read-only source mounts. Phase A physical failure should not suppress the separately predeclared B arm; ownership/preflight, corruption or cleanup uncertainty must stop the campaign. Freeze that final source before a launch.

No shared supervisor has been rewritten here. The callable adapter and SDK assumptions have CPU tests; no Isaac application was launched. No old source, gate or production asset was edited.

## Interpretation before learning

| Actual outcome | Supported conclusion |
|---|---|
| A repeats the excessive demand; B passes every declared full/mixed recovery and hold | Evidence for bounded reset-distribution sensitivity. Canonical training initialization is a candidate, subject to a separate reviewed source/admission; not random-reset or hardware qualification. |
| Both fail | The joint-reset distribution alone is insufficient. Investigate force allocation/load transfer or stance feasibility without increasing the cap. |
| A does not reproduce; B passes | Pose alone does not reproduce the historical failure under this prehistory. Preserve the limitation; do not declare a causal fix. |
| Any scope/clock/state/source/cleanup check fails | Infrastructure rejection, not a motor-equilibrium verdict. |

## CPU validation

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover \
  -s tmp/reference_reset_initialization_001 -p 'test_*.py'
```

Ten tests passed. They execute the **actual frozen** `RecoveryTargets` through all 200 knots for both endpoints, check exact canonical final hold and derivative budgets, exercise full/mixed reset readback with a mocked SDK, test named reordering, unarmed/wrong-time/wrong-mask rejection, root/unselected corruption, cleanup and frozen source hashes. They do not reproduce contacts, motor loads or Isaac native dynamics. The full `MovingSession`/native entrypoint integration remains a separate preflight seam; the driver receives an AST-level no-actor check here.
