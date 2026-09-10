# Reference physics screen002: canonical target startup

CPU-reviewed preparation only; no new physics result is claimed. Source001 and its rejected six-support result remain immutable. Independent trace review confirms 32 distinct targets with zero temporal change and up to 0.0299696 rad difference from the named canonical stance. Fifteen replicas persistently lost at least one distal support; original basic physics passed. See `failure001_target_audit.json` and the sensor agent's frozen `tmp/reference_physical_review_001` report.

This successor changes only the startup target trajectory and its explicit identity. The original randomized physical reset, root pose, motor gains, geometry and noise are unchanged. A two-second quintic C2 trajectory starts at the actually emitted reset target and reaches the exact nominal target. That named plan target is converted through float32 and must equal the SDK's default joint target exactly. The target is held for the next two seconds before the unchanged postsettle gates and wave initialization. There is no direct joint or body state overwrite, hidden clipping, retiming or gate relaxation.

Continuous endpoint velocity/acceleration are zero. Actual discrete knot velocity and acceleration, including the first knot and both first hold derivatives, are independently bounded and recorded by residual002. Startup analytic derivatives remain separately named telemetry. `startup_reference.json` records exact initial and final targets, runtime order and measured planned rate maxima. Trace `reference_source_code` is 0 during transition, 1 during canonical hold, 2 during wave; it does not assert physical tracking. A first failing physical sample is still retained before reset.

Twenty-four focused tests pass. Four new tests verify exact random-start/nominal-end values, input immutability, C2 endpoints, finite-difference continuity, no hidden retiming, soft-limit margin and identical zero-residual output through the actual residual002 core. The batch includes 32 independently randomized targets in reversed joint order, under inference mode. The worst tested startup speed is below 0.0282 rad/s and acceleration below 0.0434 rad/s², compared with the unchanged 1.75/6 reference budgets. These CPU checks do not establish that contact support will pass in Isaac.

The host, physical environment, contact sensors, metrics and residual/wave dependencies are byte-identical to source001. The host still runs fresh 32 × 1000 standing, then only upon admission 1 × 2400 controls: four seconds startup/settle, 24 seconds forward at 0.005 m/s, 20 seconds stop. Six-foot standing, five-foot wave support, torque/contact, pre-reset, progress, measured touchdown and quiet gates are unchanged. Root owns remote verification, pause028, guard, launch and restoration.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 python3 \
  <remote-source002>/tools/launch_reference_physics_spark.py \
  --source <remote-source002> --output <fresh-reference002-output>
```

Reproduce CPU tests using `PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s tmp/reference_physics_adapter_002 -p 'test_*.py' -v`. `build_source.py` refuses overwrite, verifies every source001 byte and copies only its manifest. `source002.patch` contains the complete runtime change. No main/Git or GPU action was taken by this preparation.
