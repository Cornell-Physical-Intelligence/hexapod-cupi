# Paired forward/stop trial — immutable preparation 001

The reviewed 0.01 m/s opposing-pair controller is ready for a bounded physical discriminator. This bundle contains CPU evidence, the tested adapter and exact source reconstruction; it contains no actual paired walking result or GPU launch.

Fresh 32 × 1,000 standing and all-replica quiet must pass first. One independently reset robot then runs the unchanged 4 s canonical startup plus the CPU candidate's 44 s sequence: 2 s reference hold, 24 s forward request at 0.01 m/s, 18 s stop. This is an explicitly new four-support branch; the original scalar five-support gate, source009 physics, full C asset, actuator and target limits remain unchanged. Both moving feet must independently qualify flight and landing. Complete 400 Hz torque checks and poststep required-support checks run before another target, and final scoring replays the full paired state and recomputes support from raw data.

`adapter/README.md` and its contract describe the precise physical gate and output requirements. Thirteen focused adapter tests pass; `paired_cpu_owner/` retains all 43 original CPU payloads, including higher-speed ideal-contact rejections and separate planned target-only comparisons. `independent_cpu_review/` retains the 17 passing independent CPU checks; its final frozen-owner binding is separate and unchanged. The new 1,014-field controller fragment is not a complete actor/critic packet, and the old scalar 846/849 packet is not adopted.

`BINDINGS.json` binds every owner and the original009/new946 source identities. `source_delta/` has only 21 additions/metadata changes; it does not duplicate the 925 unchanged parent executable/asset payloads. Build the exact source from an existing verified009 parent with:

```sh
python3 reconstruct_source.py --parent /absolute/path/to/source_009 --output /absolute/path/to/fresh_source_pair_motion_001
python3 verify_payload.py
```

The root dispatcher must supply a separately reviewed outer guard, current coordination/locks, workload ownership and forecasting pause/restoration. The prepared inner host is `tools/launch_pair_motion_spark.py --source SOURCE --output FRESH_OUTPUT`, bounded at 600 s per phase, with a 90 s AppReady deadline and 45 s traceback. Any terminal result, restorer receipt, exact source/asset audit or additional review belongs in a new wrapper; this preparation stays immutable. No higher physical speed, walking policy, Stage 2 or terrain qualification is claimed here.
