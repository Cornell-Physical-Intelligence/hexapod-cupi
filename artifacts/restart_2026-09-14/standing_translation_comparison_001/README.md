# Early standing trace comparison

The translated single robot reproducibly fails the unchanged standing gates. Both translated attempts have identical SHA-256 values for all 12 raw traces. The origin attempt matches all 12 hashes in the historical accepted single-run manifest. This narrows the failure to a reproducible placement-linked case without requiring a 32-robot batch; it does not identify a physical or numerical cause. Standing remains blocked before walking or training.

| Saved run | Post-settle missing-support steps | Maximum quiet joint RMS, rad/s | Original quality gates |
|---|---:|---:|---|
| Fresh origin | 0 | 0.0029493831 | Pass |
| Fresh (14,4) | 5 | 0.0746171027 | Fail |
| Fresh (14,4) repeat | 5 | 0.0746171027 | Fail |
| Historical batch environment 23 at (14,4) | 7 | 0.0712999627 | Fail |

Each translated attempt has 17 isolated exact-zero force events across 8,000 steps; five follow the 4-second settle period (LM two, RM three). Every event has 128 inactive zero tuples. The failure bounds are six-toe support and the original 0.03 rad/s quiet-rate limit. The diagnostic grants no standing or batch admission even when its quality gates pass.

## Where the trajectories first separate

Sequence is zero-based; each row is the state after its 2.5 ms step. These are exact stored-value differences, not threshold violations. Tiny early differences alone are not failures.

| First difference | Origin versus translated single | Translated single versus batch environment 23 |
|---|---|---|
| Joint position and velocity | Sequence 0, 2.5 ms | Sequence 14, 37.5 ms |
| Computed/applied torque | Sequence 1, 5 ms | Sequence 15, 40 ms |
| Toe force | Sequence 12, 32.5 ms | Sequence 14, 37.5 ms |
| Tibia mesh minimum Z | Sequence 12, 32.5 ms | Sequence 14, 37.5 ms |
| Global minimum mesh floor distance | Sequence 12, 32.5 ms | Sequence 16, 42.5 ms |
| Six-toe contact flags | Sequence 34, 87.5 ms: translated RF zero | Sequence 26, 67.5 ms: batch RF zero |

Origin and translated states already differ slightly on the first post-reset step: LF coxa position differs by 6.417e-12 rad, velocity by 4.978e-9 rad/s. The first torque difference is 2.277e-9 Nm. All toes have zero force through sequence 11 in both runs. Both first touch at sequence 12, where LF force is 52.3883581 N at origin and 52.3679495 N translated. The global lowest mesh point differs by 3.808e-7 m at that step.

At first touchdown, LF/LM/LR/RF/RM/RR tibia patch counts are **3/6/4/4/6/3** at origin and **3/3/3/3/3/3** for both translated single and batch environment 23. The translated single and historical batch have exactly equal stored joint states and forces through sequence 13, including two contact steps. At sequence 14 their first LF force difference is only 1.90735e-6 N, joint position difference 1.60071e-10 rad, and joint velocity difference 8.49832e-8 rad/s. Their first support losses occur at different steps. This shows that batch context affects the subsequent trajectory, while the isolated translated robot itself still exhibits the failure.

Inactive tuple padding also coexists with valid support: origin RF has 131 patch tuples at sequence 13. The 128 signature alone does not prove an internal capacity problem.

## Reset and frame checks

Requested origin and translated roots differ only in declared X/Y. Both use Z=0.08161108940839767 m and quaternion [0,0,0,1]. After reset, joint positions, joint velocities, root velocity and every link velocity match exactly. Subtracting only the declared translation leaves a maximum link-position residual of 8.9406967e-7 m, and a root-X residual of 7.2759576e-12 m. The original physics-view reset tolerance is unchanged.

The translated single and historical environment 23 have identical requested roots and identical selected numerical pre-reset and post-reset states: joint state, root/link pose and velocity. Actual native body/joint order is checked across all three inputs. The fresh arms' saved native scene, solver-readback and material records match exactly. USD solver attributes are not independent backend solver introspection.

Small origin/translated differences already exist in the pre-reset warmup state. Generalized coordinates and velocities are reset, but these records do not establish that internal warm-start caches were cleared. The floor is a finite 80 m square formed by two triangles, with diagonal y=x. Origin is on that seam; (14,4) is inside one triangle. Placement therefore changes the relationship to floor tessellation as well as world coordinates. No conclusion about float precision, distance, tessellation, warm caches or a solver defect follows yet. The next investigation should explain the saved contact onset and patch differences before proposing a physics change.

## Evidence and reproducibility

`ANALYSIS.json` contains exact first differences, first 45 per-leg rows, contact prefixes, reset comparisons and consumed-file hashes. `SUMMARY.json` ties the early analysis to terminal outcomes and repeat-hash receipts. `EXECUTION.json` records the successful read-only SSH CPU query, with no remote files written and no native/GPU calls.

Numeric arrays are bounded to the first 800 steps (2 seconds); per-tibia geometry and contact prefixes to the first 45 (112.5 ms). Post-settle counts and 20-second quality results come from separately pinned terminal receipts, not this short prefix. The geometry channel is the complete tibia mesh minimum Z, not a claim that every minimum is a toe-only vertex. Pose comparisons subtract declared X/Y only.

The analyzer rehashes each consumed first substep NPZ and selected metadata before and after use against its recorded state, and binds the exact source and state hashes. Contact streams are read only through the first 45 rows: the prefix hash and whole-file size/mtime are recorded, while the recorded full-stream hash is retained without rereading the full historical 4.5 GB file. This is not a self-contained full-raw replay. Raw evidence remains in the recorded Spark run directories. Origin contact repeatability uses the historical immutable manifest; its separate live comparison receipt has no prior contact hash. The repeat comparison uses root-collected receipts, not a new independent full-raw rehash by this analyzer.

`run_readonly.py` reruns the same bounded CPU query only when remote read access is authorized; it writes local analysis outputs and does not modify the remote evidence. `verify_bundle.py` checks this local bundle and the pinned sibling evidence without contacting Spark.
