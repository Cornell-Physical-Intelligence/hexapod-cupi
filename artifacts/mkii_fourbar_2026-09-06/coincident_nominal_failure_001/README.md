# Coincident-origin full nominal result

The complete 32-environment validation finished at **2026-09-06 12:10:03 UTC**
and failed one unchanged physical bound: driven C-pin separation reached
**0.111171183 mm**, exceeding **0.100 mm**. No refined phase or PPO started.
The supervisor correctly rejected the report despite Kit exiting normally.

Source is `ae4f38828181b33c9fc4fd2b8f9d68b03e6c31f1`, functional identity
`8fb32bc3e39642338bfbcf37bcc3a7e58d1ab0b8c901174532234248644a93e9`.
This run uses the unchanged physical v5 robot, RS05 model, Kp 30 / Kd 0.30,
800 Hz physics, 64/16 TGS iterations and the opt-in coincident flat layout.
It includes 1,000 standing plus 2,400 driven controls, or 54,400 sampled
physics substeps per environment. There were no resets or non-foot contacts.

| Measurement | Settled standing | Driven motion |
|---|---:|---:|
| Maximum raw/applied torque | 0.723802 Nm | 4.423689 Nm |
| Maximum C-pin separation | 0.000767 mm | 0.111171 mm |
| Mean plate height | 0.135706 m | 0.135393 m |
| Minimum loaded feet | 5 | 3 |
| Maximum passive-coordinate relation error | 0.0000160 rad | 0.0015144 rad |

All 18 individual and all 18 grouped motor response tests pass. Delivered torque
remains inside the modeled envelope, all samples are finite and every other
reported gate passes. The preceding grid-layout campaign had multiple failures;
this is a substantial improvement in this comparison, not training admission or
proof that changing placement removes every numerical problem.

The next useful physics comparison is the same complete trajectory at 128/16
iterations. A pass alone would not admit learning: both nominal/refined reports,
the convergence comparison and the exact subsequent training source must pass.
The current source additionally needs the separate RSL-RL startup compatibility
fix. No threshold was changed to accept this near miss.

## Provenance and scheduling

The six original files here were retrieved and checked against SHA-256 values
computed on Spark at 2026-09-06 22:55:43 UTC. The remote root is
`/home/orionh/HEXAPOD_runs/mkii_coincident_layout_v1/campaign_001/fourbar-campaign-20260906T114047Z-73b4f841`.
`source.tar.gz` and other unlisted remote files were not copied or independently
rehashed locally. `verify.py` verifies the six retrieved bytes and decisive result.

The newer shared Spark policy dated **13:33 UTC** says the user paused HEXAPOD
and prohibited a long-lived exclusive reservation. The canonical sharing control
is `REQUESTED`; guard 1845570 released at 13:33:47 UTC. A subsequent root preflight
at approximately 22:54 UTC was blocked before creating an output directory,
reservation or GPU process. No new GPU run was launched. Local fixes and the CAD
audit continued; this pause is separate from the failed physical result.
