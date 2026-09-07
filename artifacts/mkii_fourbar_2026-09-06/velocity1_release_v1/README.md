# Local deployment release: TGS velocity-one fallback

**The fallback now has a complete deployment manifest and a verified exact Git
archive. It remains local and unqualified on the GPU.** No Spark sync, push,
GPU launch, physical admission or PPO result is claimed.

Source commit: `25e1e59fcd1bd1372386adaef6284292d15d0cca` on
`codex/mkii-velocity1-candidate`.
Functional identity: `b6258f6ccbd76e47abe765df1715dd12b0cf48853228f9e9986d88b58a8ab19b`.
This supersedes the initial CPU-only candidate identity
`328efb07dded31e871d0b6dad3c50d949a7af798bacd8f8e6f3451ccbb7ca340` at `efe4338`:
the lineage checker is part of functional source, so updating it changes the
contract even though the solver/motor implementation is unchanged.
The earlier candidate artifact remains frozen against its original commit.

## Release contract and checks

- `CURRENT_MANIFEST` and the explicit GitHub CI command now select
  `isaaclab/deploy/mkii_fourbar_v1_1600hz_velocity1_pipeline.sha256`.
- The new manifest covers **309 paths**, SHA-256
  `02b5e8feffbe316018b90db87b9d64e13d1a63aedde5a712b2512d64cdda37b3`. It includes the previous 308-path metrics manifest
  unchanged at `8d6ad0b053e2b2a73e6610a51871443efdeb1234d7c44165dbf27a67498e66fb`.
  All **21 earlier pipeline manifests** remain byte-identical to `c2af43c`.
- Local CI-equivalent checks passed: **939 CPU tests in 79.489 s**, all **10
  lineage tests**, the **112-file archived Git lineage**, and the **309-file
  current release**. No GitHub-hosted CI run is claimed because this branch has
  not been pushed.
- An exact committed-file archive contains **310 regular files**: every release
  manifest entry plus the release manifest itself. Every file's hash/path and
  the complete archive hash were verified. A fresh temporary extraction also
  ran its own standalone `tools/check_pipeline_lineages.py current` successfully
  against all 309 paths, without requiring a Git checkout in that extraction.

The numerical recipe remains
`mkii_fourbar_tgs_external_forces_1600hz_position64_128_velocity1_v7`:
64/1 nominal, 128/1 refined, 1600 Hz physics, 32 substeps per 20 ms policy update,
and external forces every iteration. Physical assets, control gains, motor
limits, all 31 native contact streams, every-substep capture, before-reset
monitor and admission thresholds remain unchanged. New complete paired
physical validation is still required before scratch or full PPO.

## Exact archive

Local file: `/Users/andreboufama/Documents/CUPI/HEXAPOD/tmp/mkii_velocity1_candidate/tmp/mkii_fourbar_velocity1_release_v1.tar.gz`.
Size: **58,481,591 bytes**.
SHA-256: `2328cfab5578671ff5772c415e9e31aa7fa82d77d4038a24876bc7d1f898b200`.

The archive was made with `git archive` from the exact source commit and only
the 310 declared paths, so it does not include mutable test logs, scratch
outputs, virtual environments or untracked local files. The archive stays in
the isolated worktree's ignored `tmp/` directory; its bytes are not committed.
`source_release.json` records the exact Git argv, every archived file hash/size,
source identity, recipes, preserved manifests and local checks.

From this worktree, run:

```sh
uv run python tools/check_pipeline_lineages.py historical
uv run python tools/check_pipeline_lineages.py current
uv run python artifacts/mkii_fourbar_2026-09-06/velocity1_release_v1/verify_release_v2.py
```

Pass `--archive /absolute/path/to/archive.tar.gz` to the verifier if the archive
has been moved. Run `shasum -a 256 -c SHA256SUMS` in this evidence directory to
verify its own files. The read-only verification imports no Isaac/GPU runtime.

## Preserved verifier attempt

The first local archive verifier rejected a macOS temporary path because it
compared a resolved `/private/var/...` target to an unresolved `/var/...` root.
`verification_attempt_001.json`, its empty original stdout and the original
`verify_release.py` preserve that failed verification attempt. The source,
pipeline manifest and archive were not changed. `verify_release_v2.py` resolves
the temporary root before applying the same containment checks;
`release_verification_002.json` records the successful complete repeat.

`build_release_archive.py` refuses to overwrite an existing archive or release
record. Future changes require a new release identity and artifact, not changes
to this manifest or replacement of the preserved evidence.
