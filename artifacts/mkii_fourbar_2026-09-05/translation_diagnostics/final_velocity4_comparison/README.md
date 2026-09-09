# Final velocity iterations: matched 128/1 versus 128/4

Four final velocity iterations substantially reduce the worst constraint
velocity residual in this eight-world standing diagnostic. They do **not**
establish uniform convergence or complete physical qualification: a previously
quiet location now has a transient, and the minimum support remains five feet.

| Settled metric, 0.8–4.0 s | 128/1 | 128/4 |
| --- | ---: | ---: |
| Peak raw/applied torque | 0.968125 N·m | 0.734734 N·m |
| Peak active joint speed | 1.094548 rad/s | 0.532397 rad/s |
| Peak passive velocity-relation residual | 4.425369 rad/s | 1.104570 rad/s |
| Peak C-pin relative speed | 0.362515 m/s | 0.020240 m/s |
| Largest per-world RMS passive velocity residual | 0.032024 rad/s | 0.011862 rad/s |
| Largest per-world RMS C-pin relative speed | 0.003046 m/s | 0.000435 m/s |
| Peak passive position-relation error | 33.7362 µrad | 7.92742 µrad |
| Peak C-pin gap | 0.861694 µm | 0.860147 µm |
| Minimum primary-report support | 5 feet | 5 feet |

The full per-environment table, RMS values, event windows and original physical
gate summaries are in `result.json`. Every original individual physical check
passes in both diagnostics. These short traces are not the full
standing/driven campaign or the nominal/refined numerical comparison.

The former worst world, row 7 at (−2, 0), improves: peak velocity-relation
residual 4.425369 → 0.770753 rad/s and minimum force-loaded feet 5 → 6.
However, row 0 at (2, −2) changes from a quiet 0.012266 rad/s residual to
**1.104570 rad/s**, with support 6 → 5 and peak torque 0.670462 → 0.732411 N·m.
Thus an aggregate maximum alone would hide a regression. The final-solve
experiment supports the relevance of velocity convergence but does not identify
a unique cause or justify accepting the remaining residual.

## What was controlled

The analyzer compares the complete runtime manifests and permits exactly two
differences: the numerical recipe ID and final velocity iterations **1 → 4**.
Both runs retain 128 position iterations, 1.25 ms physics, 16 substeps, TGS,
external forces on every iteration, Kp30/Kd0.30, the same mass/inertia/armature,
physical v5 asset, kinematics, limits and explicit collision-isolation rules.

All eight actual world origins, initial root positions, initial 30-joint
coordinates, delivered/processed targets, velocity targets and feedforward
inputs match exactly. Motor and backend readbacks also match. The source's
joint-coordinate/name mappings are identical. Native internal constraint
iteration traces were not recorded; the comparison uses the resolved recipe
reports and observed dynamics.

The baseline is **`d863663`**, functional identity
`a1eaf8e5411519c7b8fe70edff45acfc5e4f217c5cff8904c5cd0ca3e607b20f`.
The candidate is **`83a9bca`**, functional identity
`f825a1fb3cfcf33d27dae217cb29aa777a5dd5e0bed96819fa48175a97e7d694`.
The only functional-file differences are the core numerical recipe and
pipeline lineage checker; their hashes are recorded in the result.

Both genuine supervisors confirm unchanged source, completion and removal of
their exact owned container. The candidate's captured source manifest has
**302 files**: 292 release files plus ten Python 3.12 bytecode cache files.
The archive count is not conflated with the 291-entry release pipeline
manifest. The archived source manifest bytes match its supervisor SHA; the
large source archive remains on Spark and was not downloaded or rehashed here.

## Evidence identity

| Input | Report SHA-256 | Trace SHA-256 |
| --- | --- | --- |
| 128/1 baseline | `3fe5a9e257d9419b9bda7c6b4413795a5f534ac5ab9be6b08cb0e7e2b2d40183` | `71042eff15c71a63c2e345795f57c6de7c860c5a80c86936ff6a18a9d04d9bef` |
| 128/4 candidate | `6bd9ed7827c59f0ca6ad12e50ecbc4e2aabf4af32c9622902217dd2b5b07d84b` | `388691b7c542e3622924bef1d13e7d5d065b0c60c06b040b3cdf5be7a272acb5` |

Original candidate report, NPZ, source manifest, supervisor and logs are under
`../final_velocity4_refined/`. Remote run:

```text
/home/orionh/HEXAPOD_runs/mkii_final_velocity4_v1/standing_refined_20260905T211904Z/hexapod-fourbar-diagnose-20260905T211904Z-59720fc9
```

The measurement window is physics samples **[640, 3200)** in both runs,
2,560 settled samples per environment. Pin-relative speed uses actual link
velocities and the shared CAD hinge frames; passive velocity residual uses
`qdot_passive − multiplier*qdot_source`. RMS pin speed averages squared vector
norms over six legs and all window samples within each world. The aggregate
RMS rows above report the largest of those per-world RMS values.

## Reproduction

From the repository root with NumPy installed:

```sh
python3 artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/final_velocity4_comparison/analyze.py \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/filtered_batch8/hexapod-fourbar-diagnose-20260905T210111Z-96687d5d/report.json \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/final_velocity4_refined/hexapod-fourbar-diagnose-20260905T211904Z-59720fc9/report.json \
  robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v5/kinematics.json \
  --out /tmp/hexapod_final_velocity_comparison_new.json
```

The script reuses the frozen verified trace reader, checks hashes/ranges and
finiteness, and rejects changed physical inputs or runtime differences beyond
the intended numerical intervention. It introduces no new acceptance bounds.
No production source or previous evidence was modified by this analysis.
