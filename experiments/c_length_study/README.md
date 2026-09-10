# Isolated C morphology study runtime

This directory preserves the runtime used by the user-selected C leg-length study and its PPO evidence. It is an explicit archived-study lineage, not the production physical four-bar task. The current production implementation remains under `packages/`; its `isaaclab/hexapod_rl` compatibility shims are unchanged.

The study selected 72.5 mm femurs and 126 mm tibias with fixed coxa length. Its mass/inertia and torque constraints were transferred into the mock geometry for a controlled study. CAD fit, four-bar behavior, measured actuator response and hardware deployment remain separate decisions. The accepted forward checkpoint remains frozen; the user's current Stage 2 requirement additionally demands quiet standing and smooth motion in every direction and path, alongside all numeric gates.

## Runtime identity

[`runtime/SHA256SUMS.json`](runtime/SHA256SUMS.json) records the 16 unchanged Python files copied from the validated runtime at base commit `36769da93f90e0602fb0e3591caeb98bbd7090ed`. Their canonical manifest-map SHA-256 is:

```text
abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280
```

[`tools/c_study_runtime.py`](../../tools/c_study_runtime.py) verifies that identity and every file before simulator startup. It pins only the `hexapod_rl` namespace in the current study process. It rejects missing, extra, changed or escaping Python files; a missing vendored runtime cannot silently select the production shim. Subsequent imports are checked again and compiled from the verified source rather than a stale bytecode cache. The binding remains fixed even if another library later changes `sys.path`.

If any `hexapod_rl` module was already imported from a different path, bootstrap fails and requests a fresh process. It never deletes or reloads modules to hide mixed runtime classes. This applies to direct imports of the study's omni and reference environment modules as well as the executable study and terrain-robot entry points.

The bootstrap itself imports only Python's standard library, so checking it requires neither Isaac Sim nor a GPU:

```sh
python3 tools/c_study_runtime.py
```

Normal study launch arguments remain unchanged. `train_length_study.py` and `validate_terrain_robot.py` bootstrap before importing `AppLauncher` and record the selected runtime binding in their state JSON. `omni_flat_env.py` and `length_reference_env.py` also bootstrap before importing their base environment. Importing production packages in another process does not invoke this bootstrap or change their behavior.

## Frozen-source publication

For every new published study source directory, the source builder must copy:

1. The chosen study tools, including `tools/c_study_runtime.py` and the appropriate study entry points.
2. The complete `experiments/c_length_study/runtime/` directory, including its hash manifest and all 16 Python source files.
3. The exact study robot package, training plan and any referenced input artifacts already required by the guarded launcher.

Add the bootstrap, runtime manifest and every vendored source file to `campaign_source_hashes.json`, alongside the existing source and input hashes. Keep production packages and compatibility shims intact if the published source also contains them. They cannot supply `hexapod_rl` to a bootstrapped study process.

Before publishing or freezing the directory, run:

```sh
python3 tools/c_study_runtime.py --repo-root /absolute/path/to/new/source
```

Then run the usual full source-hash verification, exact-plan standing admission and bounded job launcher. Bootstrap identity verification does not replace standing admission, checkpoint lineage, GPU ownership or evaluation gates. Replacing the vendored runtime requires a deliberately new validated lineage and corresponding pin; do not regenerate its manifest to accept a changed implementation.

## Legacy snapshots

Existing frozen Spark snapshots do not contain this bootstrap and retain their original bytes, source manifests and import behavior. Do not patch, move or retrofit them. They continue using their already-frozen `isaaclab/hexapod_rl` tree through their existing guarded launcher.

For a **newly packaged, deliberately legacy-layout** source only, an explicit compatibility mode is available:

```sh
HEXAPOD_C_STUDY_RUNTIME_MODE=legacy-frozen python3 tools/c_study_runtime.py --repo-root /absolute/path/to/new/legacy/source
```

The same environment variable must be passed to that new study process. This mode requires `campaign_source_hashes.json`, selects `isaaclab/hexapod_rl` from that exact source root, and still requires all runtime file hashes to match the pinned historical runtime. It therefore rejects the modern compatibility shims even if their hashes appear in a newly generated campaign manifest. There is no automatic legacy fallback.

## Evidence and tests

The study artifacts remain under [`artifacts/omni_flat_2026-09-09`](../../artifacts/omni_flat_2026-09-09) and [`artifacts/omni_diagnostics_2026-09-09`](../../artifacts/omni_diagnostics_2026-09-09). Runtime identity is only one part of their provenance; failed trials and frozen checkpoints remain preserved.

```sh
python3 -m unittest discover -s robot/tests -p 'test_c_study_runtime.py'
python3 -m unittest discover -s robot/tests -p 'test_omni*.py'
```

Focused tests exercise shim precedence, late path changes, already-imported wrong modules, late file tampering, missing and unexpected files, explicit legacy mode and entry-point bootstrap ordering. The omni slew test reads the vendored base implementation when it is present, so it does not accidentally test the production compatibility shim.

The geometry generator also resolves the historical `isaaclab/hexapod_rl/asset_cfg.py` input through this verified runtime when creating a new output. It records the actual vendored actuator path in new provenance. Existing URDFs and their historical manifests are preserved.

The locked workspace includes the CPU dependencies for the complete study suite: `uv run python -m unittest discover -s robot/tests`. CI runs it alongside the production suite and both source-lineage checks.
