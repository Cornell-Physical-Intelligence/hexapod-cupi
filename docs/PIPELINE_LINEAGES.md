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
The current check selects `locomotion_kernel_20260917_v12_pipeline.sha256`.
It hashes the kernel, its tests, optional optimizer code, repository tools and
package source, plus the exact model selected by `robot/active_model.json`.
It has no four-bar asset dependency and does not import an old training contract.
Nested experiment outputs and frozen attempt copies remain outside maintained
source coverage. Native allocations carry their own source and input manifests.

The [inventory](../configs/source_inventory.json) records each removed source
and test at revision `33ec6f16d70c8b0e74a9608d69be7c563c11bfbb` with a SHA-256.
The inventory check reads those Git objects and verifies their bytes. The Pages
builder resolves links to removed files against that revision. Existing frozen
package guides, published manifests and result artifacts retain their bytes.

## Publish a source change

1. Preserve all published manifests. Give the successor a new filename.
2. Update the checker's current manifest and the CI selection to that filename.
   Keep preceding manifests unchanged; historical verification has its own pinned scope.
3. After edits and focused checks, generate the new manifest with
   `python3 tools/check_pipeline_lineages.py generate --manifest <new-path>`.
   The command rejects overwriting an existing manifest.
4. Verify historical and current identities, run required suites, and add an
   append-only site update identifying the change and its qualification limits.
5. Commit the coherent release. Use its pinned checkout for reproduction;
   source relocation does not transfer old admission or checkpoint compatibility.

The [pre-cleanup release history](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/docs/PIPELINE_LINEAGES.md)
explains older manifests. Keep their bytes and original model labels. The v12
kernel release replaces active implementation paths and introduces no new
walking result or checkpoint compatibility claim.
