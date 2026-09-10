# Native005 smoke004: completed integration, quiet behavior unqualified

The actual source004/native005 smoke completed **two PPO updates**, strict actor/critic/normalizer/Adam reload and both final diagnostic acquisitions. All **48 stop/quiet replicas failed** the unchanged quiet gates. This establishes execution and preservation of the small-run behavior, not Stage2 completion or evidence that a longer run will improve it.

| Result | Measured evidence |
|---|---|
| Allocation | 32 replicas ×24 controls ×2 updates;1,536 transitions |
| Training wrapper time | 3.550371 seconds; collection/learning times remain separately recorded |
| Optimizer diagnostics | 40 minibatches;2 sparse gradient rows |
| Reload | Exact deterministic action and actor/critic/normalizer/optimizer state;17 Adam entries |
| Final checkpoint | `ffdb347eacea5d87172fdcd122eaa6836fc43ed1bd139f20493fdf3d9672b000` |
| Quiet qualification | 0/48 |
| Final ownership | Exact unit invocation inactive, exit0; four owned container names and four IDs absent |
| Forecast restoration | pause061 restored; existing deferred halo archive unchanged and retained remotely |

Read `analysis/REPORT.md` and `analysis/report.json` for unchanged-gate comparisons against the actual cold formal004 baseline. Cold004 has no moving-to-stop baseline; the smoke's stop result is an absolute screen, not a before/after improvement claim. All reported SDK rates remain separate from interval-angle evidence. No requested torque cap, applied cap, reset/contact rule or quiet threshold changed.

The runtime source004 contains599 files, including550 package files and16 legacy runtime files. Native00528, host004, guard005 and frozen supervisor926 were independently hash-verified on Spark using read-only Python/SSH. Each of the four accepted phase trees matches its immutable receipt. The terminal audit records exact input maps, raw output inventory, unit/journal identity, owned-container absence and pause restoration. Source/host preparation is separately identified in `PREPARATION_POINTERS.json`; no production asset was substituted.

`terminal/remote_terminal_audit_002.json` is the successful terminal audit. The earlier001 audit intentionally retains an **in-progress observation**, when final_stop was active and restoration did not yet exist; its nonterminal verdict is not a failed physics run and its temporary receipt pins must not be used for admission. `terminal/TERMINAL_PINS.json` provides the verified final seven campaign/jobs/pause/restored hashes for the separately guarded successor. Expected invocation: `5d7736253ce84660baefda6a672bf010`.

All52 remote raw files (76,234,247 bytes) were fetched and individually size/hash checked with **no raw omissions**, including the two ordinary autosaves, decision002 and final checkpoint. The49MB halo archive remains remote; its full unchanged inventory was verified without downloading it. Both local copying and this publication wrapper use immutable byte-preserving files. No GPU action or remote write was performed by this audit worker.

A separate comparison in `terminal/SMOKE_BEHAVIOR_PARITY.json` shows exact byte equality to source003/native004 smoke003 for the training trace, sampled joint trace, event ledger, final checkpoint, decision002, constant raw trace and stop raw trace. This is direct actual evidence that native005 preserved this old smoke behavior. It is not a claim of convergence or universal bitwise CUDA determinism.

## Portable verification

```sh
python3 -S verify_bundle.py
```

The verifier needs only the Python standard library. It checks the complete wrapper manifest, every raw file against the recorded remote audit, accepted phase states/immutable phase trees, seven terminal pins, all eight recorded absence outcomes, restoration, exact final checkpoint and analyzer integrity verdict. It does not contact Spark or reassert that the GPU is currently free. Remote source/cleanup attestations describe the audited terminal time; numerical recomputation uses the frozen analyzer and separately available cold baseline.

`analyzer/` contains the exact frozen analyzer00417 payloads and its own freeze; the generated analysis is separate. `analysis/INPUTS_SHA256.json` retains original analysis-time absolute paths for provenance. The portable verifier uses bundle-relative hashes and does not require those original paths.

Root owns integration and any next GPU dispatch. Eventual publication must update STATUS.md, docs/PLAN.md, docs/NEXT_RUNS.md, site/project.json and an append-only site/updates record under `docs/PROJECT_SITE.md`. The included framework proposal keeps omnidirectional Stage2 in progress and preserves the older benchmark/video/checkpoint distinctions.
