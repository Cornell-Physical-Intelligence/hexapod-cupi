# Reference003 landing integration

This is a new source for a bounded physical screen, not a physics admission. The immutable reference002 source is the parent. Its canonical startup, physical environment, actuator settings, assets, host, measurements, support/torque/contact/quiet gates, residual core and observation schema remain unchanged. The only executable change is the sensor agent's independently reviewed `wave_reference.py` landing successor. Full fresh standing admission remains required before the wave.

Actual002 had a real LF flight and early returned contact. The new reference starts a bounded 0.5-second C2 landing trajectory from the current commanded point/velocity/acceleration, preserves the recorded preload and independently checks measured flight clearance/descent, returned-contact speed, original planned-endpoint error and contact-region consistency. It does not instantly snap a target to the measured foot or call one contact a completed landing. Three stable contact samples after the landing blend endpoint are required, with explicit contact-loss rejection. The complete landing state remains in reference telemetry; no body or joint state is prescribed.

The actual002-prefix CPU replay produces **1.6968 mm of planned horizontal overshoot and return** while bringing the current moving target to rest. It is bounded, remains explicit in the report and is not silently described as monotonic motion. Maximum planned point excursion is about 2.218 mm. Synthetic continuation stays within the existing reference joint P/V/A budgets, but cannot establish real friction, slip, contact force, quietness or physical target tracking. The unchanged physical proof must decide whether this trajectory is acceptable.

`build_source.py` checks every924parent source byte and the complete wave002 owner freeze, refuses overwrite, requires the same five runtime dependencies and unchanged geometry, and accepts only `tools/wave_reference.py` as an executable difference. `source_origin.json` adds exact lineage and the CPU overshoot report. All924files and no extra files are checked before dispatch. Runtime snapshots retain raw SDK XYZW, measured pre-reset contact/torque, actual progress and quiet-stop evidence.

Root owns remote hash verification, pause029, exact cleanup/restoration guard and dispatch. Use the unchanged host with a fresh output:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 python3 \
  <remote-source003>/tools/launch_reference_physics_spark.py \
  --source <remote-source003> --output <fresh-reference003-output>
```

The host first runs32×1000standing and only a passed exact-source admission allows1×2400controls (4sstartup/settle,24s0.005m/sforward,20sstop). No PPO or checkpoint is loaded. Every physical failure remains a failed admission; no GPU/main/Git action is taken by this integration preparation.
