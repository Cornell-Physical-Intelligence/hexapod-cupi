# Direct quiet-priority smoke003 terminal result

The bounded two-update integration smoke completed all four phases and verified strict checkpoint reload. **The unchanged quiet screen passed 0/48 replicas.** This establishes execution/reload evidence, not convergence, smooth walking, or Stage2 completion.

The frozen [analyzer003 report](analysis/REPORT.md) independently replayed all48 stop replicas, the training event ledger and checkpoint identities against the exact historical formal0.04 cold constant baseline. It reports `evidence_verified=true`, unchanged inputs and no evidence errors. The historical cold run has no moving-to-stop baseline: the final stop result is an absolute screen, not a claimed improvement.

Training completed 2 updates, 48 controls ×32 replicas (1,536 transitions). The actual same-runner strict reload proved actor/critic/normalizer/optimizer values and deterministic actions, with the previously reviewed value-preserving inference-buffer conversion. Training recorded zero terminations, one timeout, no nonfoot environment-steps, requested torque peak8.476Nm and applied peak1.6Nm. The unchanged torque/quiet failures remain visible; software integration completion is not physical admission.

The new observational optimizer report retains40 minibatches and two sparse gradient rows. The learning rate spans1e-5 to5e-5;31 retained minibatch rows are at the existing floor. Quiet temporal MSE is0.309061 then0.341619, and the two observed PPO–quiet gradient cosines are−0.01016 and−0.06740. These sparse optimizer measurements neither represent every update nor admit training quality. All directional comparisons and per-replica failed bounds are preserved in [report.json](analysis/report.json).

Final checkpoint SHA256: `ffdb347eacea5d87172fdcd122eaa6836fc43ed1bd139f20493fdf3d9672b000`. Campaign SHA256: `fc6e962d83920180c46445aee7fa5808caa7ea509f2f9455dfd72f304dbfb3b5`. Full source/checkpoint/analyzer lineage is in [SOURCE_PINS.json](SOURCE_PINS.json). Source599/native004/host003/guard004 and the actual preflight audit are preserved in the separately published [quiet-priority preparation](../direct_omni_quiet_priority_preparation_001/README.md), exact preparation manifest `b560a9018e781f0d24377cbe049bc72dfa6766a933974f764a8d14e3fb3b782f`.

All52 fetched run/pause files, totaling76,234,469bytes, independently match every size and SHA256 in the immutable [remote terminal audit](terminal/remote_terminal_audit.json). All five auxiliary audit/transfer files are also preserved. That audit observed the unit inactive with exit0, all eight exact container names/IDs absent, and pause059 restoration complete. These are historical terminal observations, not a new current GPU-availability claim. The complete raw checkpoint/autosave/event/log payloads remain included.

The frozen analyzer has22 payloads plus its original manifest. The three new analysis files remain unchanged, including original absolute input paths. [PORTABLE_INPUTS.json](PORTABLE_INPUTS.json) separately maps every consumed input to copied raw files and the three selected historical cold files. The selection is explicit in [PUBLICATION_SELECTION.json](PUBLICATION_SELECTION.json); there are no omissions from this run's52 raw files.

Verify hashes, all terminal and source bindings, and every relocated analysis input:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B verify_bundle.py
```

Recompute the exact numerical report with the unchanged NumPy-only analyzer in a temporary directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B verify_bundle.py --replay
```

Publication belongs at `artifacts/omni_diagnostics_2026-09-09/direct_omni_train_smoke_003`. The publisher must follow `docs/PROJECT_SITE.md`: append a bounded site/update record, update current STATUS and relevant Markdown, and validate the site check/build. This task changes only ignored tmp artifacts. Public milestones remain first walking benchmark achieved, omni in progress, terrain/perception next, autonomous survey final; no assertion that all Stage1 physical gates passed.
