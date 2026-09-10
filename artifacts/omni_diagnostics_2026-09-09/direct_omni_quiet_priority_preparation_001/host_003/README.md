# Native004 quiet-priority host003 — frozen preparation

This successor retains frozen host002 supervision and adds explicit `quiet_priority` branch selection for native004, schema `direct315_quiet_priority_native_v3`. **Exact source599 and native43 pins are now bound** in `FINAL_BINDINGS.json`; root supplies the external guard and actual remote preflight before dispatch. Historical draft receipts below remain unchanged. Root owns final binding, remote verification, coordination, pause/restore guard and GPU dispatch.

## Exact scope

- Smoke requires `--allocation smoke --branch quiet_priority`: fresh standing32×1000, training32×24×2, final constant48×12 s, and final stop48×32 s. It does not launch a pilot automatically.
- Pilot requires explicit `--allocation pilot --branch quiet_priority --smoke <completed same-source quiet-priority smoke>`: fresh standing, original constant/stop, training1024×24×50, final constant/stop. The native004 contract validates the entire same-source smoke and rejects historical CAPS smoke.
- Native004 still declares `caps` and `curriculum` pilot branches. They remain explicit choices requiring the same native004 quiet-priority smoke; retaining CLI availability does not schedule them.
- Same original checkpoint, direct315/318 observations, exact legacy16 runtime, source009 ownership supervisor, source/host read-only mounts, writable per-run output, per-phase checkpoint aliases,600 s phase bounds and inherited90 s AppReady requirement. No source009 physics substitution.
- No change to failure finalization, exact owned-container identity/cleanup, per-job locks, terminal source/previous-phase integrity, stop handling, or acquisition-versus-performance semantics. Training completion and evaluation acquisition are never a Stage2 pass.

Native004 `verify_inputs`, `runtime_arguments`, and `validate_result` signatures are unchanged. The new compact `optimizer_diagnostics` result is preserved by the existing accepted-result pass-through: smoke40 minibatches/2 sparse gradient rows, pilot1000/8. Raw minibatch diagnostics remain in the immutable producer training receipt; the host does not recalculate or weaken them.

## Reserved deployment names

```text
BASE=/home/orionh/HEXAPOD_runs/mock_length_study_20260909
source:     BASE/direct_omni_train_source_003
contract:   BASE/direct_omni_train_preparation_003
host:       BASE/direct_omni_train_host_003
smoke run:  BASE/direct_omni_train_smoke_003
supervisor: BASE/reference_physics_source_009
```

The concrete CLI remains:

```sh
python3 -B launch_train_spark.py \
  --source <exact-source003> --checkpoint <original.pt> \
  --contract <exact-native004-preparation003> \
  --supervisor-source <exact-reference-physics-source009> \
  --output <fresh-smoke003-output> --allocation smoke --branch quiet_priority \
  --preflight-only
```

Preflight verifies the host/source/native/supervisor manifests and original checkpoint. It neither pauses other work nor opens an Isaac process. Root supplies the separate external guard before any actual campaign.

## CPU validation and binding history

`tests_draft_002.log`: **28 tests pass**. Tests verify original host002 inventory, byte-identical metadata entry and legacy map, unchanged host function AST except branch selector/CLI, unchanged inherited `run_owned`/container cleanup AST, actual inherited validation before process creation, final-checkpoint and phase integrity, exact phase order, failed-standing blocking training, truthful failure/update metadata, new native branch argv, compact gradient-summary pass-through and standard-library-only host/contract import.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tmp/direct_omni_train_host_003 -p 'test_*.py' -v
```

The original final-binding checklist was: verify final native004 owner map and root's one-time source build; replace only `SOURCE_MAP` and `CONTRACT_FREEZE` with those exact SHA-256 values; copy the source-build receipt; verify unchanged interfaces against the final native freeze; rerun the focused suite as necessary; then create the host manifest. Do not reuse parent hashes or turn placeholders into invented values. `parent_evidence/` contains historical host002 receipts only, with its original freeze unchanged.

All work here is temporary and local. Any later repository publication must follow `docs/PROJECT_SITE.md`: corresponding evidence/progress/roadmap updates or checked no-impact declaration, a new bounded `site/updates/` record, and site validation/build. The first walking benchmark remains achieved, omni remains in progress, terrain/perception follows, and autonomous survey is the final mission; no broader physical milestone is inferred from this host preparation.

Final binding: source599 `ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62`; native43 `1ab678beeba6bf74978c210c4e38f13655e25728fe66e3007e3b1db5e19b6b20`. The exact native inventory was verified locally; source-build values came from root's completed remote build. `tests_final.log` records the final28 focused tests. No local or remote campaign was launched by this preparation.
