# Full-body overlap controls

Artifact-only fixture; no GPU jobs were launched during implementation. It requires a frozen source checkout supporting `coincident_flat_origin_v1`, selected before scene construction. The fixture never moves terrain origins after initialization.

`live_full_body_overlap.py` creates 62 native contact views: each of31 exact bodies in each robot observes all31 exact foreign bodies, for1922 directed pairs. Native rigid-body views independently verify all62 exact source/target paths. Contact columns follow the explicitly ordered filter lists; this API has no separately exposed filter-path readback. The documented one-source/many-exact-target API and patch-count/index layout are described in [NVIDIA’s Omni Physics tensor reference](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html).

Every physics step records pair forces, lossless int64-converted native UInt32 counts and start indices, shared-buffer capacity status, both robots’ six independent foot-ground force channels and root positions. Loaded patch forces/points/normals/separations must be finite and their ranges must not overlap or overflow the allocation. The fixture also audits all24 cloned native-mimic USD references. Reports and NPZ traces precede Kit teardown.

The filtered case requires zero foreign counts, negligible foreign forces and ground support for both robots. The deliberately unfiltered control adds reciprocal foreign allowlinks before physics initialization. It requires contact count and force in both directions and lists every unexcited source body. `pair_control_success` means the bounded controls behaved as expected; universal negative-channel excitation requires the separate `all_negative_source_channels_excited` flag. Unexcited bodies are a limitation, never fabricated proof or training admission.

Root-operated host command (omit `--execute` for a command-only dry run):

```sh
python /separate/frozen_fixture/host_run_full_overlap.py \
  --source-dir /separate/frozen_source --source-commit ACTUAL_COMMIT \
  --fixture-dir /separate/frozen_fixture --output-root /separate/results \
  --physics-steps 256 --solver-multiplier 2 --timeout-seconds 600 --case all --execute
```

The host preserves the reviewed exclusive lock, GPU/memory/process gates, exact container ownership, read-only source/fixture mounts, admission barrier, source archives and cleanup. It uses canonical sharing control, so prose updates do not interrupt a run. Both cases run in separate Kit processes. Source, fixture and output directories must be distinct.

CPU tests: `uv run python -B -m unittest discover -s artifacts/mkii_fourbar_2026-09-06/full_body_overlap_tools -p 'test_*.py'`.
