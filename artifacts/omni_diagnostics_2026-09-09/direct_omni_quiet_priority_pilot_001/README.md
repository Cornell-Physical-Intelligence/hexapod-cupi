# Quiet-priority50 pilot: completed experiment, quiet failure

All six bounded phases completed, with50 actual PPO updates and strict checkpoint reload. The original final ten-second quiet screen remains **0/48**. The stronger quiet objective did not establish smooth stops or improved overall omni quality; Stage2 remains incomplete.

**The1.870m worst quiet score includes a reset.** Forward_fast env20 terminates at25.12s, inside the22.02–32.00s scored quiet window. That excursion is not uninterrupted physical drift. In the matched CAPS50 run, env20/44 reset at18.82/20.86s before quiet; this run's other reset is env44 at17.52s. Every trial reset still invalidates the replica; no failed row is removed from admission.

The unchanged [analyzer003 report](analysis/REPORT.md) has `errors=[]`, `evidence_verified=true` and all consumed inputs unchanged. The [exact CAPS50 comparison](comparison/README.md) retains the following quantitative results:

| Final measurement | CAPS50 | Quiet-priority50 |
|---|---:|---:|
| Original quiet pass |0/48|0/48|
| Median max quiet excursion |50.42mm|57.60mm|
| Max excursion, reset included |75.11mm|1870.23mm|
| No-trial-reset subset median/max,46 rows |49.91/75.11mm|56.47/89.89mm|
| Median worst-joint requested saturation |22.2%|24.0%|
| Mean all-joint quiet requested saturation |10.545%|11.178%|
| Worst target-step p95, all48 |0.040000021rad|0.040000021rad|
| Worst SDK joint RMS |1.29346rad/s|1.29457rad/s|
| Worst interval-angle RMS |1.50176rad/s|1.51592rad/s|
| Full stop-trial requested peak |8.0824Nm|9.9708Nm|
| Postsettle constant requested peak |7.1081Nm|11.8293Nm|

The no-reset subset is descriptive and never replaces the all48 gate. Applied torque remains clamped at1.6Nm. SDK endpoint rates and angle-derived interval averages are reported independently, without substituting one for the other. Constant-command planar error improves in9/12 cases and worsens in3; yaw error worsens in10/12, and requested saturation worsens in9/12. All12 bearing/turn/arc rows and numerical deltas remain in the comparison. These are descriptive observations from one matched-budget comparison, not significance tests or a causal guarantee. Initial constant measurements match exactly; initial stop reports differ only in20 heading values by at most1.4305115e-5degrees, with no changed verdict.

Final checkpoint SHA256: `195593ca4d595184fefbc9161bad433f9fb9768d8fd16d514c7b755eee0a6173`. Source599/native004/host003 remain bound in [SOURCE_PINS.json](SOURCE_PINS.json); source preparation and the separate exact pilot guard/dispatch remain in their published parent bundles. This wrapper does not relabel those historical snapshots as terminal results.

The immutable [remote terminal audit](terminal/remote_terminal_audit.json) contains the full122-file/443,634,816byte inventory. All72 selected files/224,646,566bytes match their sizes and hashes. Exactly50 named intermediate `model_*.pt` autosaves remain remote-only and are explicitly listed; final/decision checkpoints, raw traces, logs and receipts are present. The five fetched auxiliary audit/transfer files are preserved too. The audit observed inactive exit0, all12 exact owned container identifiers absent and pause060 restored. This is a historical terminal observation, not a new live GPU-availability claim.

The full frozen analyzer003 and the compact frozen comparison are included. Original analysis path records remain byte-identical; PORTABLE_INPUTS.json maps every consumed new-run input. Only the consumed historical cold campaign metadata is copied. Large historical CAPS raw stays in its existing artifact; EXTERNAL_PARENT_INPUTS.json binds exact published maps and all32 consumed CAPS inputs using repository-relative paths.

Verify local/parent hashes and rerun the unchanged analyzer plus exact comparison:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B verify_bundle.py --repo-root /path/to/HEXAPOD --replay
```

No Torch, Isaac, network, GPU, source edits or gate changes are used for replay; NumPy is required. The verifier creates fresh temporary outputs and requires exact report equality. The publisher targets `artifacts/omni_diagnostics_2026-09-09/direct_omni_quiet_priority_pilot_001` and must follow docs/PROJECT_SITE.md: append a bounded site update, current STATUS/relevant Markdown changes and required site check/build. Public milestones remain first walking benchmark achieved, omni in progress, terrain/perception next, autonomous survey final.
