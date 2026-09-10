# Full-C measured-contact reference: source and CPU evidence

**Runtime verdict pending.** This bundle records the exact source and CPU checks for reference physics screen 001. Root is supervising the Spark run separately and will append its raw terminal evidence. This preparation does not establish walking, Stage 2 completion, terrain qualification, or hardware readiness. `status.json` deliberately leaves the runtime verdict unset.

The experiment first admits 32 replicas for 1,000 standing controls. Only a fresh passed admission permits one robot to execute 4 seconds settling, 24 seconds of 0.005 m/s forward command, and 20 seconds requested stop. A measured-contact wave lifts one leg at a time. The residual is zero, and there is no policy, checkpoint transfer, prescribed body pose, or automatic training continuation. Existing physical limits and quiet gates remain in force.

## Exact contents and identities

- `source_overlays/` contains all **14** runtime files declared in the frozen source origin, with their original repository-relative paths.
- `source_identity/` contains the complete **923-file** source map and its exact origin. The map SHA-256 is `a13f1534f9f802871a2cacbc0be603eae2d80738905381da40691f7cd4beff7c`.
- `parent_identity/` binds the exact velocity003 parent rather than duplicating its 909 source files. The parent manifest is `00d4e07e0e31776e3d2faa0136a2b1386f79fd4940bf81d2f4a484155f81bb25`; executable identity is `fb5e472bb710daa50d1d7f16a6f3812d2ed845e2400fcc6769859b3fb0b9519e`. Its origin hash is `c3c0ddba7abd0ca55daf3f21c6987d684645aebf5cf3804743f1a1fd412f7e8f`.
- `adapter_owned/` preserves every file in the physics adapter's owner freeze, including its source builder, tests, installed SDK state contract, preflight receipt and original CPU log. The original nested source map is retained; the other 923 files are not duplicated.
- `residual002/` and `wave001/` preserve their complete owner freezes, source, tests and recorded CPU results. They include the inference-tensor reset regression and synthetic measured-contact fixtures.
- `outer_guard/` preserves the exact dispatched outer guard and its tests. Its review report describes tests already executed against the actual AST-extracted embedded restorer. It includes the dispatch-client ambiguity fix: stop the exact owner before cleanup and forecasting restoration.

`COPY_VERIFICATION.json` records the hash of each of the 57 copied files and its original local path. Every copied hash and all 923 frozen source hashes were checked during assembly. These provenance paths describe where the files came from; they are not new runtime dependencies. Original freezes remain byte-for-byte unchanged.

## Reproduce the source without copying a second baseline into this artifact

Obtain the exact root-deployed velocity003 source using the parent identities above. Its archived manifests also appear in `artifacts/omni_diagnostics_2026-09-09/velocity_candidate_003/results/` and `velocity_pilot_003/results/`. A generic checkout, a newer production shim, or a source with merely the same filenames is not an acceptable substitute.

From this bundle directory, verify the exact parent and all overlays without generating another source tree:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 reconstruct_source.py --parent /absolute/path/to/velocity003-source
```

To materialize the source for an explicitly authorized new run, add a fresh `--out` directory. The script verifies all parent bytes and the composed 923-file map before writing, then verifies every emitted byte. It refuses existing output and does not launch Isaac or change any prior run.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 reconstruct_source.py \
  --parent /absolute/path/to/velocity003-source \
  --out /absolute/path/to/fresh-reference-source
```

Use the frozen adapter README for the physical screen contract. A new launch still requires current workload/lock checks and a fresh output and forecasting pause record. The included outer guard binds this historical attempt's exact paths, pause 027 and prior-owner requirements; do not replay it blindly for a later experiment.

## CPU evidence and limits

Recorded checks are 20 adapter tests, 12 residual tests, 9 wave tests, 23 idealized wave fixture cases, and 8 outer-restorer tests. They establish local contracts and failure handling. They do not establish physical contact, motor margin, useful movement, or quiet stop. Packaging did not rerun these suites unnecessarily. The reconstruction verifier was run against the exact parent and compared its composed map to the frozen source.

The measured run must preserve pre-reset events and establish actual support, six confirmed foot flights and touchdowns, lift, forward displacement consistent with integrated link velocity, requested torque within 1.6 N·m, and ten contiguous seconds passing quiet gates after reference stop and settling. Planned contact and quiet labels alone cannot pass. Root will add the result and forecasting-restoration evidence after the attempt is terminal; source and CPU preparation must remain distinguishable from that verdict.
