# RR preload diagnostic 002 — host import correction only

This is the same bounded RR first-landing preload experiment as frozen001, with a standard-library-only host import path. **No controller equation, target, solver behavior, physical gate, observation lineage, actuator setting, CAD input, or case schedule changes.** Source932 manifest: `bfb0bbde27fdfae1f2ddc7797ff12adf184c7cd68a8e637f5fce87f7e0e3cfa3`.

Actual Spark host preflight for001 rejected before any pause/GPU because `directional_contract` imported `PROPOSAL` from a NumPy-dependent controller helper. NumPy is intentionally not installed in the host Python. Root preserved that actual rejection in the separate four-file `reference_rr_preload_host_failure_001` receipt (`5ea889ecb21639a11f4607819d209b50443d6231dd986c99739b6da5781d8bee`); frozen001/source/guard remain unchanged.

The correction extracts the exact same `PROPOSAL` dictionary to [rr_preload_contract.py](rr_preload_contract.py), which has no imports. The numerical helper imports that constant alongside its existing NumPy dependency inside the Isaac environment. The host contract imports only the standard-library metadata and pins both metadata/helper hashes. This adds one file and changes only the helper import, host metadata import/hash checks and source origin record. **928 of 931 parent payloads are byte-identical.** The full wave/controller file and host launcher are byte-identical to001; the helper class AST and proposal values are exactly equal. [DELTA_REVIEW.json](DELTA_REVIEW.json) records the hashes and scope.

[tests_final.log](tests_final.log): **19 CPU tests passed**. The three new tests reproduce the old failure under real Python `-S` with site packages disabled and NumPy explicitly absent, then run the actual new host import and full `host.check_source()` in the same restricted environment. They also verify exact proposal/class parity and the four-path source delta. The inherited 16 tests still cover C2 endpoints, one-shot RR scope, named strafe-only commands, actual recorded prefix, residual P/V/A, support interruption, finite stop, reset and unchanged physical/host functions.

[STDLIB_HOST_IMPORT.json](STDLIB_HOST_IMPORT.json) preserves old stderr and new stdout. [INTEGRATION_PREFLIGHT_FINAL.json](INTEGRATION_PREFLIGHT_FINAL.json) binds the new932 source, rejects the old actual directional002 standing receipt and a failed-quiet synthetic receipt, and permits only fresh standing followed by the one left-strafe case. These are CPU/no-App checks, **not physical admission**. `DRAFT_COUNT_PREFLIGHT.json` preserves an earlier receipt whose source-count display remained931 after copying the old checking script; its hash/source checks were already against the new source. The final script computes the count from the verified map and reports932.

The unchanged experimental behavior is: 0.5 mm downward RR world-anchor correction, C2 over the existing0.3 s hold after its first confirmed measured landing; fresh32×1000 standing/quiet then one1×2400 zero-residual left-strafe run. Existing1N/five-support/1.6N m/flight/contact/geometry/quiet gates remain. Other commands and actors are rejected. The added diagnostic state remains explicitly unbound to the old846/849 actor/schema. Actual arc003 lost RM support, so this is not an arc fix. Consult the frozen001 [owner README](../reference_rr_preload_diagnostic_001/README.md) and [actual-prefix evidence](../reference_rr_preload_diagnostic_001/ACTUAL_PREFIX_REPLAY.json) for the unchanged behavior and torque limitations.

Root owns the external guard, any fresh source002 admission/dispatch, and subsequent evidence. No host dependency was installed and no GPU action was performed by this agent. Read-only tests:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tmp/reference_rr_preload_diagnostic_002 -p 'test_*.py'
```

The source directory is `source_rr_preload_002`; its inherited host entrypoint remains `tools/launch_directional_physics_spark.py`. `build_source.py` requires the exact frozen001 parent and creates only a fresh source tree. Receipt-writing scripts must run in a new copy after freeze.
