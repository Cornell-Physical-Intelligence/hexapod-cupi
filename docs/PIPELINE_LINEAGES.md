# Source release verification

Source identity establishes which files were checked. Robot admission and policy
acceptance require matching physical/evaluation evidence under [ARCHITECTURE](../ARCHITECTURE.md).
The checker's historical `mkii_fourbar_v1` label does not qualify the canonical model.

## Verify a checkout

```sh
python3 tools/check_pipeline_lineages.py historical
python3 tools/check_pipeline_lineages.py current
python3 tools/c_study_runtime.py
```

The historical check verifies `stage2_pipeline.sha256` against the immutable Git
revision `81d7c6f2a43c7de99f32cd6bb1b7efb0f54874df`. Fetch repository history first.
The current check selects `repository_canonical_restart_20260915_v7_pipeline.sha256`.
It covers current runtime/source paths, tests and retained release inputs,
including the inventoried top-level `experiments/paper_walk/*.py` prototype
modules and `experiments/paper_walk/tests/*.py` CPU checks. Nested experiment
outputs and frozen attempt copies are not scanned as maintained source. CI runs
the prototype's CPU checks separately; its native Isaac allocations retain their
own exact source, model, standing-admission and result identities. The two explicitly
allowed historical packaging changes remain in the checker; frozen package guides
and all other historical contracts retain their original bytes.

## Publish a source change

1. Preserve all published manifests. Give the successor a new filename.
2. Update the checker's current manifest and the CI selection to that filename.
   Include the preceding release manifest in the retained source inputs.
3. After edits and focused checks, generate the new manifest with
   `python3 tools/check_pipeline_lineages.py generate --manifest <new-path>`.
   The command rejects overwriting an existing manifest.
4. Verify historical and current identities, run required suites, and add an
   append-only site update identifying the change and its qualification limits.
5. Commit the coherent release. Use its pinned checkout for reproduction;
   source relocation does not transfer old admission or checkpoint compatibility.

The canonical-restart v6 release retains the earlier canonical-restart
manifests and every preceding release unchanged. It adds explicitly labeled
cold and settled BC startup diagnostics, preserving separate whole/prefix/policy
physical evidence and all original evaluation cases. It also checks exact
evidence-path capitalization across macOS and Linux before paper publication.
Explicit hash-pinned correction records repair historical evidence links while
retaining original record bytes and visible provenance. The v3 rollback and actor
normalizer controls remain unchanged. Cross-schema checkpoints require an explicit recorded migration;
the runtime loader remains strict. Source inclusion does not admit a robot,
accept a policy or transfer historical checkpoint compatibility. The checker's
historical header remains a lineage identifier, not a native run's robot model.
The canonical-restart v7 release retains v6 and every preceding manifest
unchanged. It binds the prototype guard to the pause coordination hash, adds the
maintained `refit_bc.py` helper with its CPU tests, and records the executed
paired BC refit under protocol 002. Source inclusion of a fitted candidate does
not evaluate it natively.
[Archived release history](archive/README.md) preserves earlier explanations and
failed-attempt references. New release explanations belong in site updates.
