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
The current check selects `repository_visual_roadmap_20260911_pipeline.sha256`. It covers
current runtime/source paths, tests and retained release inputs. The two explicitly
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

The documentation cleanup archives historical prose with exact-byte checks and
keeps old site evidence links pinned to their original revision. It changes no
robot model, controller, acceptance gate, checkpoint or frozen runtime. The
preceding `repository_progress_tests_20260911_pipeline.sha256` stays unchanged.
[Archived release history](archive/README.md) preserves every earlier explanation
and failed-attempt reference. New release explanations belong in site updates.

The visual-roadmap release updates the page renderer and validates agreed test
definitions separately from assigned or completed work. The preceding
`repository_docs_20260911_pipeline.sha256` remains unchanged.
