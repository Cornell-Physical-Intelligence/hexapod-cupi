# Direct PPO smoke002: completed integration, quiet walking unqualified

The corrected smoke completed fresh standing, two actual RSL-RL PPO updates, strict checkpoint reload, constant-command evaluation and moving-to-stop acquisition. The finalizer alone changed from smoke001. Both runs produced the identical final checkpoint SHA `4aaf556613e72a80332381c09309cc0ab52a03c6d55527bc76e8a7000a29e1f2`. The reload now verifies actor, critic, normalizers, all17 Adam entries and deterministic actions; cloning the two inference-created normalizer buffers preserves their values.

This is integration evidence, not policy convergence. Independent raw replay scores **0/48 quiet-stop passes**. Constant-command changes are mixed after only two updates; no overall quality improvement or Stage 2 acceptance is claimed. The full analysis retains all direction-specific changes, reset events, requested versus applied torque and SDK versus interval-angle rates. Formal comparison retains0.040 rad/20ms and1.6 N·m applied cap.

All52 raw files (76,190,510 bytes) match the fresh remote audit. Four exact owned container names and four IDs are absent; pause055 restored ordinary forecasting timers. The separate halo deferral archive remained unchanged at audit time; its restoration is a subsequent operational action, not established here. Full598 native source files,926 supervisor files, original checkpoint, native00326-file contract, host00210-file and guard00316-file freezes, legacy16 and completed phase trees were checked remotely.

- [Measured comparison and quiet failures](analysis/REPORT.md)
- [Machine-readable metrics](analysis/report.json)
- [Independent terminal audit](audit/terminal_audit.json)
- [Training receipt](raw/run/train/training_receipt.json)
- [Prior failed smoke](../direct_omni_train_smoke_001/README.md)
- [Exact successor preparation](../direct_omni_native_preparation_002/README.md)

The preserved analyzer is source/native-bound and its six CPU tests include the actual earlier failed receipt. Its original README refers to workspace preparation paths; replay using this bundle's `raw/run`, the separate cold001 raw run and a fresh output. Run `python3 -B verify_payload.py` from this directory for a portable byte and terminal-summary check. Re-executing the remote audit is a new read-only observation, not a replacement for these frozen audit bytes.
