# Independent reference001 physical-feasibility review

2026-09-10. Read-only review of frozen `../reference_physics_adapter_001/source_001`; no source/controller/gate changes or GPU launches. This is a bounded prediction to compare with the actual screen, not a physics result or admission.

## Decisive risk: touchdown timing conflicts with preserved loaded toe plane

The reference preserves the initial executed joint target, correctly avoiding a preload-erasing target jump. Its virtual foot anchor consequently differs from the measured loaded foot. `tools/wave_reference.py:130` records that offset, and line292 preserves the virtual anchor's Z at the swing endpoint. The 2s swing adds a 5mm bump, but line262 rejects confirmed post-flight contact before the last0.20s of the swing.

For normalized swing time `u=t/2`, the bump is `h=0.005×64×u³×(1−u)³`. At the permitted-contact boundary `t=1.8s`, it is only **0.23328mm** above the virtual anchor. If this anchor is below the physical contact surface by more than0.23328mm, the descending nominal trajectory can meet the surface earlier than that gate permits. A1mm offset crosses at approximately **1.644s**, a2mm offset at **1.513s**. Additional airborne gravity deflection, actual body motion and the foot mesh can change the exact time; none is modeled by this calculation.

The prior pilot's **initial, stationary-target** diagnostic trace provides relevant empirical scale. From3–4s, the virtual-minus-measured toe Z has median **−0.854mm**, range **−2.334 to+0.462mm**, across12replicas and6feet. Executable target velocity in this interval is exactly0. At4s in replica0, the six Z offsets in `lf,lm,lr,rf,rm,rr` order are **−0.217,−1.059,−1.283,−1.522,−0.864,−0.310mm**. Five of those six downward offsets exceed0.23328mm. This is a prior-run measurement using the same frozen serial geometry, not the new reference screen's initialization.

This exposes a plausible **false rejection of a physically expected flat-ground touchdown**, rather than evidence that the robot cannot walk. It can also expose a real low-clearance scrape: the timing calculation alone cannot distinguish them. The existing measured2mm-lift, sustained-flight, five-support, torque and actual-progress gates must remain.

The original CPU motion fixtures assumed perfect target tracking and did not model loaded toe-plane deflection during a complete swing. Their success therefore did not exercise this physical interaction. The separate preload test verified only reset continuity.

## Support geometry was not shown infeasible

`static_load.py` independently solves ideal quasistatic whole-body force/moment balance at the exact nominal frozen stance. It includes each downstream link's gravity torque and named toe Jacobians. Minimizing peak actuator torque over nonnegative **vertical-only** contact forces gives0.6864N·m with six contacts and **1.1285–1.1288N·m** when each possible single foot is removed. This is below1.6N·m and demonstrates that five-foot support at this nominal geometry is not inherently torque-impossible under that ideal allocation.

Allowing an inscribed octagonal friction cone withμ0.5 yields a0.6219N·m worst five-foot optimum; do not expect the position controller to realize that optimal contact-force distribution. These are optimistic static feasibility calculations, not measured motor demands. They exclude transient loading, actual PD force distribution, settled-pose changes, collision/contact-patch geometry and dynamics. They neither waive nor predict passage of the actual1.6N·m gate.

## Minimal successor concept, contingent on the actual trace

Keep source001 and its result frozen. First compare each actual virtual-minus-measured offset, planned bump/apex, measured contact loss/return, measured toe clearance and requested torque. A rejection at the predicted plane crossing with sustained flight and measured lift is evidence for a geometry/contact reconciliation change; absence of real lift is evidence for inadequate unloading/clearance.

A separately versioned successor should distinguish the **measured physical contact surface** from the **loaded virtual motor target**. Keep the exact executed target at reset. Transition through a bounded unloading segment before a swing whose clearance is defined against the measured contact surface; then use a bounded touchdown/loading segment that restores the required virtual preload without a target jump. A zero-residual experiment must preserve the combined discrete P/V/A budget and all existing torque/support/failure checks.

A smaller contact-state correction may be appropriate if the actual trace already demonstrates adequate clearance: accept touchdown only after sustained measured flight, measured≥2mm lift, passage through the planned apex and actual descent, with bounded endpoint position/velocity and force/contact confirmation. Contact accepted before the planned end must switch to an endpoint-matched C2 landing/hold trajectory, with the entire new reference checked against P/V/A and no additional liftoff until stable support. Merely increasing the allowed time window while continuing the old swing through ground contact would be insufficient.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tmp/reference_physical_review_001/static_load.py
PYTHONDONTWRITEBYTECODE=1 python3 tmp/reference_physical_review_001/preload_review.py
```

- `static_load.json`: nominal stance, all force allocations, torque solutions and support margins.
- `preload_review.json`: prior trace SHA256, selected interval, measured offsets and analytic bump crossings.
- Source being reviewed remains unchanged. The actual reference001 screen decides whether this predicted mechanism occurs.
