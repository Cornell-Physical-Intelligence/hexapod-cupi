# Campaign 005: native physical coupling, 5 September 2026 UTC

The v5 probe passed. The full nominal run completed **32 environments × 1,000 standing + 2,400 driven control steps**, with all **54,400 physics substeps per environment** measured, but **failed**. No refined qualification or PPO phase started.

- Maximum driven pin separation: **0.108253 mm**, above the unchanged **0.100 mm** bound.
- Maximum passive-coordinate residual: **0.00125587 rad**, within **0.005 rad**.
- All 18 independent motor direction checks passed. The simultaneous knee group failed at the left-middle motor: its minimum positive-minus-negative response was **−0.00476402 rad**, below the required **+0.005 rad**.
- Driven support briefly reached zero. There were no non-foot ground contacts, terminations or truncations.
- Raw requested torque peaked at **16.6328 N·m**; delivered torque peaked at **5.5 N·m**, with zero envelope excess. Settled peak torque was **0.668155 N·m**, with six supporting feet.

This retains the CAD bodies, inertias and colliders. Native bilateral PhysX couplings enforce the exact parallelogram coordinate relations; they are not kinematic pose copies. Numerical settings are TGS 64 position / 1 velocity iteration, 1.25 ms physics and 50 Hz policy control. The source is frozen at `cfe0cf5`, functional identity `ee8785ffb1c50e89e06a78ae965a6d87f2faf3cda68b9eec99bb5795b247e6f1`. Test-only commit `ffa44f6` has the same functional identity and passed GitHub CI.

Raw reports, CPU audits, supervisor state and exact source hashes are preserved here. Full source archives and container logs remain on Spark under:

`/home/orionh/HEXAPOD_runs/mkii_fourbar_physical_mimic_v1/campaigns/fourbar-campaign-20260905T165013Z-e9c5e5a6/`

The `admitted` marker only releases the host resource barrier; it is not physical or training admission. The report explicitly records `simulation_training_admission=false`.
