# Exact reference005 → reference006 standing-only iteration comparison

This immutable preparation contains three runtime overlays, source lineage/map, five CPU tests and exact reconstruction without duplicating the926-file parent. Terminal evidence belongs in a separate wrapper.

Parent926-file map: `c557b71636c8b1e10883ab8a6b2b49f40b661988f9b09faeaefe4821a72844b4`. Target926-file map: `90d7cf551ccf52f7cc2d224b81547c1bad0ad6191c2db56389bb2d54de0ff193`.

The sole physical intervention is articulation velocity iterations4→1. Position iterations16, TGS external-force timing enabled, stabilization disabled, randomized reset/C2 startup, targets/PD/torque, full-C geometry,400Hz observer and every original physical/quiet gate remain unchanged. Host and runtime permit only32×1000 standing, with no wave or PPO continuation. New readback is fail-closed instrumentation; owner docs preserve its runtime uncertainty about independent link metadata. CPU checks do not prove runtime admission.

Verify without writing a source tree:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B reconstruct_source.py --parent-source /path/to/reference_source005
```

Add `--output /fresh/reference_source006` only when materialization is needed. The recipe checks all parent/overlay/target bytes and rejects extras or overwrite. It launches no job. This preparation remains unchanged after any terminal result.
