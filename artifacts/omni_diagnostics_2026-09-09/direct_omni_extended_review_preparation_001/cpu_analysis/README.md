# Extended500 evidence acquisition plan

Run the unchanged frozen analyzer004 on Spark's original terminal output tree, then fetch a curated local evidence bundle. Keep every original raw file and ordinary checkpoint on Spark. With about 1 GB free locally, the complete training trace and all 500 autosaves should not be downloaded speculatively. This is infrastructure preparation; no extended analysis or GPU job was executed here.

## Existing CPU route verified

A read-only SSH probe imported NumPy using `/opt/wx/venv-models/bin/python -B`: Python 3.12.3, NumPy 2.5.2 on aarch64. Torch, Isaac and CUDA were not imported. Child-only BLAS/OpenMP thread counts were one and `CUDA_VISIBLE_DEVICES` was empty. No installation, weather configuration, remote file, service or process ownership was changed. Original probe code, command, stdout, stderr and hashes are preserved.

At that observation Spark had 2,647,556,939,776 free disk bytes and approximately 124 GB MemAvailable. These are historical measurements; root should check current capacity before running the terminal analyzer. This existing weather environment is only reused as a CPU interpreter; its packages and configuration stay unchanged.

## Practical size bound

The completed 50-update trace is 79,338,381 compressed bytes and 126,579,168 uncompressed NPZ member bytes. A linear 500-update estimate is **793,383,810 bytes compressed / 1,265,791,680 bytes uncompressed**. This estimate is not an actual extended output size. Event counts and compression can change it. The four initial/final evaluation traces total 113,912,056 bytes in that pilot. Seven decisions plus final are roughly 35 MB; 500 ordinary autosaves are roughly 2.19 GB.

The unmodified analyzer materializes NPZ arrays, including temporary finite checks and per-row work. A few GB of peak CPU memory are plausible; NPZ compressed size is not its memory footprint. Record actual peak RSS and avoid concurrent CPU analysis if shared memory is constrained. Single-thread CPU analysis after the owned job is terminal avoids allocating another GPU workload. The proposed executor has a 900-second analysis timeout; it preserves logs and failure receipts rather than changing the analyzer.

## Exact analyzer inputs

Analyzer004 freeze is `be6625b977aa8ee333ecf1aa744bed99ba16d991dca0f3679c53414bbdc8bd94` (17 files), bound to source004/native005. Its runtime imports only NumPy, standard-library modules and the frozen sibling `optimizer_summary.py`. `DEPENDENCIES.json` also lists historical testing dependencies; Torch/RSL/Isaac are not runtime dependencies of this analyzer. No analyzer bytes were edited.

For an extended run it consumes the campaign and cold-baseline campaign identity; all six phase state files and matching jobs; training receipt, state, raw training trace, joint trace, event ledger, initialization; final and all seven explicit decisions; available initial/final constant JSON plus trace; and initial/final stop JSON plus trace. Training logs retain actual timing. Optional failure files are retained. The joint trace is required and hash-checked by the training audit; the existing analyzer does not independently recompute every training joint value from it. Constant traces cover one of four replicas per direction, while stop traces contain all 48 replicas. Do not widen those coverage claims.

The analyzer replays training event/torque/command summaries from `training_trace.npz`, checks required raw hashes, verifies selected checkpoint bytes, and separately replays the unchanged stop scores. It does not read ordinary `model_*.pt` files. All ordinary autosave cadence/names/sizes/hashes must therefore come from root's full terminal inventory. Root's source/assets/ownership/restoration audit remains separate from numerical analysis.

## Root execution sequence

1. Wait for the actual extended unit/campaign to be terminal and for root's exact job-name/ID absence, source/native/host/guard/assets and pause-restoration audit. Pin the actual campaign and terminal audit SHA values. Do not use a launch snapshot as this receipt.
2. Transfer the exact 17-file analyzer004 bundle if absent, with its unchanged freeze. Verify it on Spark. Run the prepared `terminal_cpu_analysis.py` only after explicit root dispatch. This executor is a CPU subprocess wrapper, not another systemd/GPU/forecast guard.
3. The executor checks the frozen analyzer and exact terminal extended CAPS/source004 campaign, hashes the complete original campaign and cold trees before and after, captures stdout/stderr and peak RSS, and keeps all three analyzer outputs. It pins root's audit bytes without claiming to perform that audit. It never resumes training or loads an actor.
4. Read both `execution_receipt.json` and `analysis/report.json`. Analyzer CLI exit zero only means a report was written; `evidence_verified=false` and every forensic error must remain visible. A successful execution receipt is not a physical pass. Failed or partial campaigns remain failed or partial.
5. Generate an acquisition inventory from actual post-terminal sizes. Fetch all evaluation traces, selected decisions/final, metadata, logs, audits and analysis outputs. Hash every received byte. Preserve the complete remote inventory and explicit remote-only disposition rows alongside the local map. Verify the remote full trees still match the inventory before publication.

Prepared command shape (paths and receipt hashes supplied only after the actual terminal audit):

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B terminal_cpu_analysis.py \
  --campaign /absolute/terminal/extended_run \
  --cold /absolute/direct_omni_cold_001 \
  --analyzer /absolute/frozen_analyzer004 \
  --output /absolute/fresh_cpu_analysis_receipt \
  --campaign-sha256 ACTUAL_TERMINAL_CAMPAIGN_SHA256 \
  --terminal-audit /absolute/root_terminal_audit.json \
  --terminal-audit-sha256 ACTUAL_ROOT_TERMINAL_AUDIT_SHA256
```

This command is prepared, not executed. Root selects paths; the executor rejects existing/nested output, a nonterminal/wrong-source/wrong-branch campaign, changed analyzer and changed pinned receipts before analysis. It then uses the observed NumPy interpreter with thread limits and no GPU visibility.

## Curated local bundle and omissions

`acquisition_plan.py` accepts a complete `{relative_path: {sha256, bytes}}` inventory, optionally with `run/` prefixes. It emits a disposition for every file and a local disk budget; it does not fetch, delete or analyze anything.

- Ordinary `train/policy/model_*.pt` autosaves remain remote, with every exact name, byte count and SHA retained. They are separate from the required `decision_001/010/025/050/100/250/500.pt` and `final.pt` files, which are selected locally.
- Required training trace, training joint trace or event-ledger raw files over 64 MiB remain remote by default. Each is explicitly labeled **remote-only required training raw**, not merely an omitted autosave. A complete remote analyzer report may verify those inputs remotely; the local subset cannot reproduce that training replay.
- All evaluation traces stay selected, even if larger than the raw-audit cap. Other files remain selected. Any selected file above the conservative 100,000,000-byte Git threshold requires lossless chunks at no more than 64 MiB each, with ordered offsets, chunk hashes, whole-file size/SHA and a portable reconstruction verifier. Chunking does not lower total local storage requirements.
- The plan reserves 256 MiB of local free space beyond selected payload bytes and refuses to call an over-budget selection feasible. If actual mandatory files exceed the budget, stop the copy plan and obtain more capacity; do not drop evaluations, decisions, diagnostics or failure evidence silently. APFS clones can avoid duplicate local work/publication copies, but are not compression.

The local wrapper should contain the exact analyzer and freeze; root audit; full remote inventory; local verified subset inventory; explicit omitted autosave list; explicit omitted required-raw list; CPU interpreter/command/environment/exit/RSS receipts; original stdout/stderr; `REPORT.md`, `report.json`, `INPUTS_SHA256.json`; and preserved phase/checkpoint/evaluation evidence. Keep original absolute input paths in analyzer output. Add a portable mapping from each consumed remote path to its hash and local payload or explicit remote-only location; do not rewrite the original report to conceal omission.

An outer portable verifier checks all included payload bytes and inventories. For remote-only rows it can only verify the recorded metadata locally. A claim that those remote bytes still exist needs a fresh read-only remote hash check. This partial public bundle is not a self-contained full training replay. Lossless chunks of omitted raw can be added in a new versioned evidence bundle later, once full storage and a portable reconstruction check are available. Never rewrite, truncate or recompress original NPZ bytes merely to fit a limit.

## CPU tests and framework handoff

Eight focused tests pass: size-based omissions, no false local replay claim, mandatory large evaluation chunks without storage discount, unsafe inventory rejection, active-campaign rejection, timeout/failure preservation, changed-input detection, and report success separated from evidence validity. These use synthetic temporary files and mocked subprocesses. They do not run the frozen analyzer or inspect an extended outcome.

All work is under this fresh temporary directory. Eventual publication needs a bounded `site/updates/` record under `docs/PROJECT_SITE.md`, describing evidence logistics rather than a new locomotion result. Root owns central-site/status changes, remote analysis authorization and GPU allocation. Stage 2 remains incomplete.
