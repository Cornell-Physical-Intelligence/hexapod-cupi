# Matched single-robot and eight-robot standing comparison

The robot at **(2, −2) m has the same physical trajectory** singly and as row 0
of the eight-robot batch. All 3,200 recorded samples are bit-identical for its
30 joint positions/velocities, 31 body poses/velocities, targets, PD terms,
raw/applied torques, motor headroom and six foot-force vectors. The independent
origin control also matches batch row 4 at (0, 0) exactly in those fields.
This experiment does **not** establish a generic batch-dependent instability.

Other batch locations have short contact/velocity transients. The largest is
row 7 at (−2, 0): 0.96812451 N·m and 1.09454834 rad/s, with one foot briefly
unloaded. Spatial position, native constraint/contact ordering and other
numerical causes are still candidates; coordinate precision is not confirmed.

## Matching and provenance

Both new diagnostics used frozen source **`d6d5863`**, the same complete
1,220-file source manifest (`19e6d2c3…a9c94`), identical functional contract
(`1f35d529…5736`), full runtime manifest, physical v5 asset and kinematic
contract. Each used Kp 30/Kd 0.30, 128/1 TGS iterations, external forces at
every iteration, 1.25 ms physics and 16 substeps per control update.

The analyzer verifies genuine report and NPZ hashes, complete trace ranges,
finite named fields, source-manifest hashes, exact initial 30-joint states,
all delivered/processed targets, and motor/backend readbacks in every row.
Both supervisors confirm unchanged source, diagnostic completion and removal
of their exact owned container. Source archives remain on Spark and were not
downloaded or independently rehashed here.

| Input | Report SHA-256 | Trace SHA-256 |
| --- | --- | --- |
| Single (2, −2) | `06d93c9162eb64d563ddf4f9801802ede718bfe48cafccbbe02249a81477b2c5` | `d4206081ed539d7080f17b45730a4b6e836c954a8c0d81438fb3ab0812af7f49` |
| Eight-robot batch | `1af7f85028294638a237c5fa8c9bdb157f154f9d99207883c792822e0b2de9d7` | `71042eff15c71a63c2e345795f57c6de7c860c5a80c86936ff6a18a9d04d9bef` |

The original reports, traces, source manifests, supervisor records and logs
are retained in sibling `translated_diagonal/` and `batch8/` directories.
The earlier `origin/` evidence is read without modification. Remote runs are:

```text
/home/orionh/HEXAPOD_runs/mkii_placement_diagnostics_v1/translated_x2_yminus2_20260905T204813Z/hexapod-fourbar-diagnose-20260905T204813Z-ae3fbdf2
/home/orionh/HEXAPOD_runs/mkii_placement_diagnostics_v1/batch8_20260905T205224Z/hexapod-fourbar-diagnose-20260905T205225Z-8faf9917
```

These are standing diagnostics, not admission or PPO runs. Native per-clone
PhysX mimic-reference paths were not recorded. The analyzer verifies the
shared authored 12-relation kinematic mapping and measures every passive
coordinate; it does not substitute that for runtime clone-reference proof.

## Exactly matched time windows

The main comparison uses physics samples **[640, 3200)**, after 0.8 s and
through the physics step ending at 4.0 s. `result.json` also includes the
narrower **[1920, 3200)** interval, 2.4–4.0 s, corresponding to the earlier
32-robot log interval where a 1.627909 N·m transient was already observed.
Thus the prior 32-robot discrepancy cannot be attributed solely to its longer
12-second observation period, although its full 12-second maximum is not a
matched statistic for these four-second traces.

| Environment / location (m) | Peak raw = applied (N·m) | Peak active speed (rad/s) | Minimum loaded feet |
| --- | ---: | ---: | ---: |
| Single / (2, −2) | 0.670462 | 0.055531 | 6 |
| Batch 0 / (2, −2) | 0.670462 | 0.055531 | 6 |
| Batch 1 / (2, 0) | 0.706318 | 0.407498 | 5 |
| Batch 2 / (2, 2) | 0.661526 | 0.054090 | 6 |
| Batch 3 / (0, −2) | 0.711751 | 0.445889 | 5 |
| Batch 4 / (0, 0) | 0.666658 | 0.066699 | 6 |
| Batch 5 / (0, 2) | 0.720267 | 0.519139 | 5 |
| Batch 6 / (−2, −2) | 0.732395 | 0.551999 | 5 |
| Batch 7 / (−2, 0) | 0.968125 | 1.094548 | 5 |

Here “loaded” means foot-force magnitude above 1 N. The primary report retains
the full pad/ground classifier. Batch settled maximum pin gap is 0.861694 µm,
maximum passive-coordinate error 33.7362 µrad; its individual physical checks
pass. This does not establish solver convergence or terrain/hardware fitness.

The matching row's derived hinge calculations differ by at most 5.69e−14 m
or axis chord and 1.87e−9 m/s. These are downstream calculation differences;
the physical state inputs themselves match exactly. Both traces have zero
cached-versus-direct joint-state discrepancy, zero target motion, zero
velocity target/feedforward, and PD reconstruction residual below 5.97e−8 N·m.

## What happened at the largest batch transient

For row 7's right-rear leg, sample 1928 has **76.8505 N** vertical pad force.
The next sample starts at 2.41125 s with femur speed 1.094548 rad/s. Its
computed and applied torque is −0.968125 N·m, reconstructed as P −0.639760
plus D −0.328365. The foot reports zero force on that following sample.
Actual joint-position change over that step is **1.069784 rad/s** and final
speed is 0.895306 rad/s. This particular event includes real recorded motion;
it cannot be described merely as noisy reported velocity with a stationary
joint. The trace establishes the sequence, not the underlying solver cause.

## Next discriminating control

A single robot at **(−2, 0)** with the same frozen source and four-second
window directly tests the batch's actually noisy location. If its trajectory
matches row 7, this transient follows location independently of batch size.
If it differs, a subsequent permutation of row-to-location assignments can
separate environment ordering from geometry without changing the set of
physical world positions. Avoid changing gains or solver settings during
that causal comparison.

The installed `grid_transforms` formula is also significant: eight worlds
form a 3×3 grid at even coordinates −2, 0, 2 m; 32 worlds use the first 32
positions of a 6×6 grid at odd coordinates −5, −3, −1, 1, 3, 5 m. In
particular, row 0 is **(5, −5)**, not (2, −2). A single robot at (5, −5)
is the appropriate far-corner spatial control for the 32-world setup.
`result.json` records the exact formula-derived layouts and confirms that
the eight-world layout matches its live origin readbacks. The 32-world
layout is source-derived, not a new live readback.

## Reproduce locally

From the repository root, with NumPy installed:

```sh
python3 artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/batch_analysis/analyze.py \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/translated_diagonal/hexapod-fourbar-diagnose-20260905T204813Z-ae3fbdf2/report.json \
  artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/batch8/hexapod-fourbar-diagnose-20260905T205225Z-8faf9917/report.json \
  --origin-report artifacts/mkii_fourbar_2026-09-05/translation_diagnostics/origin/hexapod-fourbar-diagnose-20260905T201519Z-26c8180e/report.json \
  --repo-root . --out /tmp/hexapod_batch_comparison_new.json
```

The shared verified trace reader is reused and its source hash is recorded.
No source/asset modifications, remote containers, or GPU launches were made
to produce this analysis.
