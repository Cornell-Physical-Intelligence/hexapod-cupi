# Exact reference006 → reference007 readback correction

This immutable preparation contains the sole runtime overlay, source lineage/map, owner tests and a verified reconstruction recipe. It does not duplicate the 926-file parent. Terminal results, guard/restoration and independent receipts belong in a separate wrapper.

Parent map: `90d7cf551ccf52f7cc2d224b81547c1bad0ad6191c2db56389bb2d54de0ff193`. Target 926-file map: `666571ed6e37a73b857179323573ae20cdef0eee53aa25908108bd4e29465465`.

Only `tools/solver_comparison.py` and `source_origin.json` change. Physical TGS16/1 with external-force timing enabled remains exactly source006. Random reset, C2 startup, target/actuator settings, 400Hz observer, original metrics and every physical/quiet gate remain unchanged. The inherited host permits only32×1000 standing, with no wave or PPO continuation.

Source006 stopped before control0 because its readback wrongly required independent solver attributes on articulation links. The corrected readback requires exactly32 explicitly authored articulation roots at16/1,608 owned bodies and no conflicting/malformed present link metadata. Absent unauthored link fields are recorded explicitly. NVIDIA's articulation API assigns solver counts to the whole articulation; native tensor getter availability remains unverified, and no guessed getter/internal counter claim is made. The owner's README supplies primary documentation and exact semantics.

Verify without writing a source tree:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B reconstruct_source.py --parent-source /path/to/reference_source006
```

Add `--output /fresh/reference_source007` only when materialization is needed. The recipe verifies every parent/overlay/target byte and rejects extra files or overwrite. It launches no simulator or job. This bundle makes no physics or quiet-hold admission claim.
