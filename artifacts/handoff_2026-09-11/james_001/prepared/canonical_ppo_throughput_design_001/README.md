# CPU throughput review — unadopted preparation

[Two-page design](DESIGN.md) identifies a narrow lossless-recording/prefix-copy change and separates actual throughput, prior replicated CPU measurements and new limited codec timing. No runtime, geometry, gate, GPU job or tracked file was changed. The independently assigned classifier work is outside this proposal.

Reproduce from the repository root using already available local inputs:

```sh
python3 -B -S tmp/canonical_ppo_throughput_design_001/benchmark.py --output /tmp/canonical_contact_benchmark_replay.json
python3 -B -S tmp/canonical_ppo_throughput_design_001/collect_evidence.py --output /tmp/canonical_performance_evidence_replay.json
```

The benchmark reads a gzip stream in place for only 128 complete rows and a published 7.8 MB analysis containing 218 selected 32-env excerpts. It hashes those local files, never retrieves the 5 GB raw run, and never creates a full decompressed contact stream. Output timing is machine/load dependent; payload lengths, byte hashes and parity verdicts should repeat. `EVIDENCE.json` uses source and receipt files still in this shared workspace. This is not a standalone full-native replay bundle.

For eventual publication, root must follow `docs/PROJECT_SITE.md`: a new bounded site update and appropriate evidence/design links; STATUS changes only for an actual execution-state change. Nothing here admits standing, starts training or alters the four research milestones.
