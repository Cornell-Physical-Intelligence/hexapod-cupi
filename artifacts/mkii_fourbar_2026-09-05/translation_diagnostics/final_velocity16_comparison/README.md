# Final velocity iterations: matched 128/1, 128/4 and 128/16

Sixteen final velocity iterations reduce the largest settled constraint
velocity errors substantially. They do not establish uniform convergence:
some individual worlds regress, and the peak torque rises slightly relative
to four iterations. All three short diagnostics pass their existing physical
checks; none constitutes full physical qualification or PPO admission.

| Settled metric, 0.8–4.0 s | 128/1 | 128/4 | 128/16 |
| --- | ---: | ---: | ---: |
| Peak raw/applied torque, N·m | 0.968125 | 0.734734 | 0.768448 |
| Peak active joint speed, rad/s | 1.094548 | 0.532397 | 0.627615 |
| Peak passive velocity residual, rad/s | 4.425369 | 1.104570 | 0.097386 |
| Peak C-pin relative speed, m/s | 0.362515 | 0.020240 | 0.003487 |
| Largest per-world RMS passive velocity residual, rad/s | 0.032024 | 0.011862 | 0.001166 |
| Largest per-world RMS C-pin relative speed, m/s | 0.003046 | 0.000435 | 0.000096 |
| Peak passive position error, µrad | 33.7362 | 7.92742 | 10.4308 |
| Peak C-pin gap, µm | 0.861694 | 0.860147 | 0.864267 |
| Minimum primary-report support, feet | 5 | 5 | 5 |

The complete per-world table and event windows are in `result.json`. From
four to sixteen iterations, row 0 at (2, −2) improves from a 1.104570 rad/s
passive velocity residual to 0.004002 rad/s; row 7 at (−2, 0) improves from
0.770753 to 0.003762 rad/s. However, row 2 at (2, 2) worsens from 0.004378
to **0.097386 rad/s**. Row 4 at (0, 0) changes from peak torque 0.669921 to
0.708513 N·m and minimum support six to five feet. An aggregate maximum
alone would hide these regressions.

Startup is separate from the settled table. The sixteen-iteration run's
first 0.8 s contains a peak passive velocity residual of **0.837510 rad/s**
and C-pin relative speed **0.016048 m/s**. The improvement in the settled
window does not erase that landing transient.

## Controlled inputs and provenance

All three runs use eight environments, 200 controls and 3,200 physics steps
per environment: 1.25 ms physics, sixteen substeps per 20 ms command, TGS,
128 position iterations and external forces on every iteration. The
resolved runtime comparison permits only the numerical recipe ID and final
velocity count to differ. Kp30/Kd0.30, mass/inertia/armature, the physical
v5 asset, kinematics, motor bounds, collision-isolation rules and joint/name
mapping remain equal.

Actual origins, initial root positions, all thirty initial joint coordinates,
all delivered/processed targets, velocity targets and feedforward inputs
match exactly over the entire trace. Motor/backend readbacks also match.
Cached versus direct pre-step positions and velocities differ by zero; the
largest P + D + feedforward reconstruction error is 5.96e−8 N·m in each run.

The sources are `d863663` (one), `83a9bca` (four), and **`c804169`** (sixteen).
The last candidate's functional identity is
`d3442002687f4ff7b34bd2e24134a8e3265a87e05221d3cbf33f2630c0b52d36`.
Four functional files changed from four to sixteen iterations: the numerical
recipe, validator, diagnostic and lineage checker. The validator/diagnostic
changes add observational velocity telemetry and share its pin calculation;
the claim of matched physical inputs is established by the resolved runtime
and trace checks, not by claiming a one-line source difference. Native
internal solver iteration state was not recorded.

The genuine supervisors verify unchanged source, successful diagnostic
completion and exact owned-container removal. The sixteen-iteration source
manifest contains 303 files: 293 release files and ten Python 3.12 cache
files. This is separate from the 292-entry release pipeline manifest. Its
captured SHA is
`32c57b0e09d92ba9698dddde159727626528b8b66de1f5a89edeb5212436e436`.
The large source archive remains on Spark and was not downloaded or rehashed.

| Input | Report SHA-256 | Trace SHA-256 |
| --- | --- | --- |
| 128/1 | `3fe5a9e257d9419b9bda7c6b4413795a5f534ac5ab9be6b08cb0e7e2b2d40183` | `71042eff15c71a63c2e345795f57c6de7c860c5a80c86936ff6a18a9d04d9bef` |
| 128/4 | `6bd9ed7827c59f0ca6ad12e50ecbc4e2aabf4af32c9622902217dd2b5b07d84b` | `388691b7c542e3622924bef1d13e7d5d065b0c60c06b040b3cdf5be7a272acb5` |
| 128/16 | `1125e9c8ebb3da8fb0c5c81aa83e01c53732063151ce8d9b0fd875274c9a8b59` | `39146769e2716e4c92c8a060d1bc422e40206fc3c2d2abbe2f813d52bc07b667` |

Original candidate report, NPZ, supervisor, source manifest and logs are in
`../final_velocity16_refined/`. Remote run:

```text
/home/orionh/HEXAPOD_runs/mkii_final_velocity16_v1/standing_refined_20260905T213116Z/hexapod-fourbar-diagnose-20260905T213117Z-203b3597
```

## Independent telemetry verification

The new primary report's velocity peaks, squared sums, denominators and RMS
values are independently reconstructed from the raw trace for startup
samples [0, 640) and settled samples [640, 3200). Settled denominators are
245,760 passive-relation samples and 122,880 pin samples. Whole-population
settled RMS is 0.00104036324 rad/s for passive residuals and
0.00007545317 m/s for C-pin speed. These differ intentionally from the table's
largest per-world RMS values.

The reported settled peak passive residual matches exactly; reconstructed
pin peak differs by 2.23e−10 m/s. Both settled RMS values agree within
2.1e−12. The report squares float32 quantities and accumulates in float64;
the independent reconstruction uses float64 and stored hinge-local vectors
instead of report-world vectors. The script's stated float32 roundoff
allowance checks telemetry consistency only. It adds no physical acceptance
threshold.

## Next discriminating run

Hold the final velocity count at sixteen and compare a matched **32-world,
200-control standing pair at 64/16 and 128/16**. This covers the actual
32-world placement grid and the time window in which the earlier large-batch
instability already appeared. Compare existing physical/convergence checks
and observational velocity peaks/RMS, with startup separate from settling.
If consistent, continue the unchanged full nominal/refined standing and
driven campaign. The short pair cannot replace that campaign. The present
mixed per-world changes do not justify another iteration increase based
only on an aggregate maximum.

## Reproduction

From the repository root, with NumPy installed:

```sh
python3 artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/final_velocity16_comparison/analyze.py \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/filtered_batch8/hexapod-fourbar-diagnose-20260905T210111Z-96687d5d/report.json \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/final_velocity4_refined/hexapod-fourbar-diagnose-20260905T211904Z-59720fc9/report.json \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/final_velocity16_refined/hexapod-fourbar-diagnose-20260905T213117Z-203b3597/report.json \
  robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v5/kinematics.json \
  --out /tmp/hexapod_final_velocity16_comparison_new.json
```

The frozen reader verifies trace hashes, ranges and finiteness. Its dependency
identity is recorded in the result. This artifact changes no model, gate,
production source or previously published evidence.
