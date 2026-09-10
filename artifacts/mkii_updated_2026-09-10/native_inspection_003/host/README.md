# Canonical native inspection host 003

This is a source-pin-only successor of frozen host002. Native003 contains 33 payloads and freeze `1d636f6b6909171e366be7b08b3590c0a65a8a9206e57b76792d3f4bcd166cd8`. The only host runtime change is its `SOURCE_FREEZE` constant; every function is byte-identical. The original source009 ownership supervisor remains unmodified, with a 600-second phase and actual AppReady required within 90 seconds. It supplies ownership infrastructure only, not historical robot physics or admission.

The native producer corrects the exact compiled SDF factory argument mismatch observed in inspection002. The previous run successfully read native identity, inertias, limits, frames and scene, then failed the SDF-view construction before the eight explicit steps. That failure and its raw evidence remain immutable. This host does not change the producer’s inspection logic or claim that the correction succeeds physically.

The same nine canonical asset files are mounted read-only. Native source, asset, exact original supervisor and this host’s full bundle are verified before launch and after exit. Only the native entry `/inspection/run_inspection.py --asset /asset --output /output/inspection --device cuda:0 --headless` runs. No checkpoint, controller, drive gain, joint target or physical admission is supplied. The completed producer receipt and post-exit seal must validate; original errors and shutdown logs remain failure evidence.

Twelve focused CPU host tests and actual local Python 3.12 `-S` setup checks pass on the new source binding. All external calls are prohibited in that setup test, which checks the unchanged supervisor code objects and actual native contract import. The full-input preflight is captured separately. These checks do not execute native physics.

Root-only preflight after transfer:

```sh
python3 -B /home/orionh/HEXAPOD_runs/canonical_direct_20260910/inspection_host_003/launch_inspection_spark.py --preflight-only \
  --source /home/orionh/HEXAPOD_runs/canonical_direct_20260910/inspection_source_003 \
  --asset /home/orionh/HEXAPOD_runs/canonical_direct_20260910/asset_001 \
  --supervisor-source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009 \
  --output /home/orionh/HEXAPOD_runs/canonical_direct_20260910/native_inspection_003
```

Root dispatches through guard003 only after actual setup preflight. The new guard binds the exact failed inspection002 owner and restoration. No tracked file or remote state was changed during preparation; any publication follows `docs/PROJECT_SITE.md` through root’s central update.
