# Cold baseline first: preserved direct PPO, declared 0.04 profile

This is the minimal two-phase preparation requested by root. It contains no training dispatch and does not authorize allocation. It preserves the original 315/318 direct-position checkpoint rather than loading the reference-residual policy.

Original checkpoint:

- Local: `tmp/ppo_repair_003_preparation/original/policy.pt`
- Spark: `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_repair_003/branch_a/inputs/original.pt`
- SHA256: `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`

Parent: `BASE/omni_repair_source_003_a`, historical 146-file manifest `f469abdad81cab7bf63719c72cf4468b8827b802499f85a66d6b1abe8e9bd5e3`. A fresh read-only Spark inventory verified all 146 and captured the complete current 588-file parent tree, including all 550 study assets. `remote_base_inventory.json` binds those bytes. Historical source manifests did not themselves cover every USD/mesh file; this complete contemporary capture is explicitly distinguished from the historical proof.

The builder copies only into a fresh destination, verifies the complete parent before/after, and makes three explicit changes:

1. The plan's target slew changes from 0.03 to **0.04 rad per 20 ms**. Everything else in that plan is identical.
2. The entrypoint imports the previously Isaac-tested read-only asset audit instead of the USD authoring repair helper. The rest of its AST is unchanged.
3. It adds that read-only helper and a source-origin receipt.

The parent has no explicit external-forces override: the reviewed installed default is false. Keep TGS position/velocity iterations **16/4**, the old action/reward/noise behavior, PD30/0.6, asset, material, stance, 20 ms control / 2.5 ms physics and 1.6 Nm applied cap. Resolve and retain `environment.yaml` before interpreting the result. Do not silently match reference009's True/16/1 solver configuration. This first preparation does not assert a newly added native scene readback; root should inspect the resolved configuration.

## Builder and supervisor seam

Copy this preparation directory to Spark, then root can run the CPU-only builder:

```sh
python3 build_source.py \
  --parent /home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_repair_source_003_a \
  --output /home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_cold_source_001
```

This writes a complete successor `campaign_source_hashes.json` and prints its exact identity. It does not launch a container or claim physical admission.

`cold_contract.py` imports only the standard library and exposes:

- `verify_inputs(args)` with `args.source` and `args.checkpoint`; returns exact source/plan/checkpoint and diagnostic-profile identity.
- `runtime_arguments('standing'|'baseline')`; returns the original entrypoint CLI with the correct late AppLauncher flags. There is no train/video mode.
- `validate_result(directory, phase, identity)`; standing requires the exact new-plan admission, baseline requires the exact checkpoint, .04 overrides, 315/318 history audit, all 12 cases and a raw trace. It preserves diagnostic failures as measurements and does not promote them to policy admission.

Reuse root's current `run_owned` lock, deadline, contact-overflow and exact-container cleanup supervisor with only its command callback adapted. The old source `tools/launch_length_training_spark.py:run_job` shows the historical CLI but has a writable source mount and older cleanup; it is not the recommended host implementation.

Mount successor source at `/source:ro`, output at `/output:rw`, and original checkpoint at `/checkpoint/original.pt:ro`. Baseline also mounts completed standing output at `/admission:ro`. Set `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=/workspace/isaaclab/source/isaaclab:/source/isaaclab:/source/tools`, preserving the exact legacy C runtime import order. Invoke the CLI returned by `runtime_arguments` with the installed Isaac Python launcher. No asset writer remains, so the source/assets can stay read-only in both phases.

Only **32 replicas × 1,000 standing controls**, then **48 replicas × 12 seconds / 12 diagnostic cases** are allocated by this proposed host. The second phase starts only after passed new-plan standing. Keep old diagnostic admission behavior: requested saturation and jitter can be observed as failures without suppressing them. `complete` means a complete measurement, not that formal motor/contact/smoothness gates passed. These inherited diagnostics record 50 Hz outcomes, so they do not establish 400 Hz peak safety.

This is a new .04-profile cold comparison, not a retroactive relabeling of repair003's .03 results or a full Stage 2 qualification. Once baseline evidence exists, root can decide whether to allocate a separate stand/stop-curriculum and CAPS ablation. No long training follows automatically.

## CPU checks

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tmp/direct_omni_recovery_001/baseline -p 'test_*.py'
```

Five focused tests cover the single plan delta, exact entrypoint AST preservation outside the audit import, two-phase CLI restriction, standing identity/rejection, actual diagnostic layout, .03/.04 label mismatch and nonfinite applied torque. The copied inputs are byte-exact old evidence. Source assembly and GPU behavior remain pending root's preflight/review.
