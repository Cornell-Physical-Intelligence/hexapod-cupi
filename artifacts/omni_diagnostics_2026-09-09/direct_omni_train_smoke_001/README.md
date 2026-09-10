# Direct-PPO CAPS smoke001: two updates, then reload failure

The fresh standing gate passed. **Two actual PPO updates completed**, after which strict checkpoint reload failed while copying the actor observation normalizer's `obs_normalizer._std` inference tensor. The host accepted **zero training updates** because that finalizer did not complete. Neither the final constant-command evaluation nor the moving-to-stop evaluation ran. This establishes **no policy quality gain**, qualified checkpoint or Stage 2 completion.

| Stage | Actual result |
|---|---|
| Fresh standing | 32 replicas × 1,000 controls; passed |
| CAPS training | 32 replicas × 24 controls × 2 updates completed |
| Final checkpoint reload | Failed on an inference-created normalizer buffer |
| Host-accepted training updates | 0; only standing was accepted |
| Final constant / stop evaluations | Neither ran |
| Cleanup | Both exact owned names and both recorded container IDs absent |
| Forecast pause054 | Restoration recorded at `1789054912.4481833` Unix seconds |

The saved final checkpoint is retained as **unqualified failure evidence**: [`final.pt`](raw/run/train/policy/final.pt), SHA256 `4aaf556613e72a80332381c09309cc0ab52a03c6d55527bc76e8a7000a29e1f2`. The original warm start was `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`. Original per-iteration checkpoints, optimizer records, traces, event ledger and logs are preserved without rewriting. Their historical runner iteration numbers are not new-update counts.

The exception and traceback are in [failure.json](raw/run/train/failure.json), [training_receipt.json](raw/run/train/training_receipt.json) and [train.log](raw/run/logs/train.log). The [campaign](raw/run/campaign.json) deliberately retains both its zero accepted-update count and the producer's two completed updates. Its generic inherited supervisor error includes the word “standing”; the explicit failed phase is training finalization. The unchanged raw training state is not relabeled as a completed phase.

## Provenance and preservation

This run used native002's 598-file source `37b4b6d5d75f6d697a122fc2a1903f8f625f6a67d1727e1257e1533300e44da6`, plan `9fb7d0b01480523972b4f1c17e717756d7dcf1c54b9caa436245671937de538c`, host001 and guard002. Source009 supplied the owned-process supervisor only; the native direct315/318 physics and controller lineage remained separate. Exact input manifest files are retained under [inputs](inputs). The new native003 reload correction and host002 preparation are intentionally separate from this historical failure bundle.

All **30 raw files / 13,549,925 bytes** are included under [raw](raw), copied using APFS clones and individually compared with both the local inventory and the root's remote audit. The [original terminal audit code](audit/terminal_audit.py), [recorded audit result](audit/terminal_audit.json) and [local raw map](audit/local_raw_map.json) are preserved verbatim. The audit binds exact invocation `1de80a43928f4d7497846dc762048d86`, complete input trees, admitted standing immutability, both owned-container removals and pause restoration. The separately deferred weather archive was recorded unchanged and not restarted; **no raw weather dataset is included**.

The input manifests are provenance references. Full input source trees are not duplicated here; their remote verification is preserved historical audit evidence, not a fresh network check by the portable verifier. All actual smoke outputs are present. No compression, truncation, checkpoint rewrite or recovery is applied to the originals.

Run from any location:

```sh
python3 -B verify_payload.py
```

The verifier is standard-library-only and read-only. It checks the complete bundle inventory, every raw hash and byte count against both maps, the checkpoint identity, the standing and update/failure distinctions, absent evaluations, recorded cleanup/restoration and input-manifest identities. It reproduces [RESULT.json](RESULT.json) without loading a checkpoint pickle, running the simulator, querying Spark or changing any file.
