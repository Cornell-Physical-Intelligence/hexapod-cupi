# Sixteen final velocity iterations: 32-world standing comparison

The matched **64/16 versus 128/16** standing comparison passes the existing
torque and height consistency checks. Both primary validators also pass
their individual short physical checks. This supports running the full
standing/driven qualification; **it is not training or hardware admission**.

Each run completed 32 environments × 600 controls, sampling all 9,600 physics
substeps per environment. Both use the same 2.4–12.0 s settled window,
samples [1920, 9600). Startup samples [0, 1920) are preserved separately.

| Settled measurement | Nominal 64/16 | Refined 128/16 |
| --- | ---: | ---: |
| Peak raw / applied torque, N·m | 0.759236634 | 0.755706251 |
| Mean plate height, m | 0.135701924 | 0.135720036 |
| Peak C-pin gap, µm | 1.720294 | 1.509069 |
| Peak passive position residual, µrad | 19.073486 | 7.688999 |
| Minimum loaded feet | 5 | 5 |
| Peak passive velocity residual, rad/s | 0.692813993 | 0.714486241 |
| RMS passive velocity residual, rad/s | 0.002460947 | 0.001540201 |
| Peak C-pin relative speed, m/s | 0.012707432 | 0.012560686 |
| RMS C-pin relative speed, m/s | 0.000110026 | 0.000076001 |
| Non-foot contacts / episode resets | 0 / 0 | 0 / 0 |

Delivered-torque difference is **0.003530383 N·m**, below
`max(0.05, 0.05 × refined peak)` = **0.05 N·m**. Mean-height difference is
**18.11125 µm**, below **1 mm**. Raw demand equals delivered torque at the
peak in both runs and has the same passing difference. The explicit
raw-demand comparison applies the added metric in qualifier `fd34f66` with
the same torque error budget; it is not attributed to the frozen `c804169`
producer's original comparison checks.

Velocity residuals remain measurable. Refinement reduces whole-population
RMS but increases the peak passive velocity residual by approximately 3.1%.
No new velocity bounds are assigned here, and these aggregate reports cannot
locate the remaining events by environment or determine their cause. Their
effects must remain visible in the full driven qualification.

Startup peak raw/applied torque is 1.559165 / 1.616783 N·m for nominal/refined.
Startup peak passive velocity residual is 0.659075 / 0.765599 rad/s, and
C-pin relative speed is 0.011653 / 0.013588 m/s. Startup minimum support is
zero during release from the reset clearance. These startup observations are
not merged into the settled convergence comparison.

## Identity and placement checks

Both runs use source **`c804169de680aa51816f7c33bca29b6eaf96e062`**, functional
identity `d3442002687f4ff7b34bd2e24134a8e3265a87e05221d3cbf33f2630c0b52d36`.
The complete resolved runtime differs only in solver position iterations
64 → 128. The physical v5 asset, Kp30/Kd0.30, RS05 envelope, 1.25 ms physics,
sixteen-step command delivery, sixteen final velocity passes, TGS, external
forces every iteration and collision-isolation rules are equal.

All **32 actual reset-root XYZ positions match exactly**, covering the first
32 cells of the 6×6, 2 m grid: row 0 is (5, −5), row 5 is (5, 5), and row
31 is (−5, −3). The recorded first environment's thirty initial joint
coordinates and all observed joint/motor/body name mappings match. Both
reports' maximum initial joint errors across all environments satisfy the
existing 5e−6 rad reset bound.

The identical validator source supplies seed zero and zero standing actions.
Actual delivered-target histories were not traced in this validator, so this
artifact does not claim the stronger trace-level input comparison available
in the earlier eight-world diagnostic.

Both supervisors confirm unchanged source and removal of their exact owned
container. Their captured 304-file source manifests are byte-identical, SHA
`f78af62a1cdc3040249c4996623b87a56374427691b50016e702d541402f6267`.
Every functional-contract file matches that captured list. Source archives
remain on Spark; their large bytes were not downloaded or independently
rehashed. Genuine reports, logs, CPU audits, supervisors, hash lists and the
final pair orchestration record are preserved here.

| Primary report | SHA-256 |
| --- | --- |
| [Nominal](nominal/hexapod-fourbar-validate-20260905T215120Z-27e6ced2/report.json) | `7a7b7cd91fc864282faa0d2ce04201c5fd935c0ec6b76fb8bfad1e93137d4428` |
| [Refined](refined/hexapod-fourbar-validate-20260905T215725Z-23ef7e34/report.json) | `dc3a30c71f4873878f420d1d32d514b984c17ad3f01fddf52fe2db3e0cdedb79` |

Remote parent directory:

```text
/home/orionh/HEXAPOD_runs/mkii_final_velocity16_v1/standing_pair_20260905T215119Z
```

The pair orchestration finished at **2026-09-05 22:07:20.910700 UTC**, with
both phases complete and no simulation-training or hardware admission.

## Reproduction and scope

`comparison.json` contains the complete metric deltas, runtime and root
positions. The analyzer verifies source/cleanup identity, exact placement,
runtime equivalence and reported RMS arithmetic. Each settled RMS uses
**2,949,120 passive-relation samples** or **1,474,560 pin samples**. The
reported square sums and denominators reproduce all startup/settled RMS
values exactly. No NPZ state trace was produced for these validation runs;
this is arithmetic verification, not independent raw-state reconstruction.

From the repository root:

```sh
python3 artifacts/mkii_fourbar_2026-09-05/final_velocity16_standing_comparison/analyze.py \
  artifacts/mkii_fourbar_2026-09-05/final_velocity16_standing_comparison/nominal/hexapod-fourbar-validate-20260905T215120Z-27e6ced2/report.json \
  artifacts/mkii_fourbar_2026-09-05/final_velocity16_standing_comparison/refined/hexapod-fourbar-validate-20260905T215725Z-23ef7e34/report.json \
  --out /tmp/hexapod_final_velocity16_standing_new.json
python3 artifacts/mkii_fourbar_2026-09-05/final_velocity16_standing_comparison/test_analyze.py
```

Seven focused tests cover RMS population/arithmetic, changed placement or
runtime, torque convergence, raw-demand differences hidden by clipping, and
the prohibition on admitting a short run. Neither report includes the
required 1,000 standing + 2,400 driven controls. No production code, model,
gate, prior evidence or remote workload was changed by this analysis.
