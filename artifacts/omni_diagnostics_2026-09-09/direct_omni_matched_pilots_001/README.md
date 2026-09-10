# Matched direct PPO pilots: 50 updates each

Both independently initialized pilots completed **50 updates, 1,228,800 transitions** and exact actor/critic/normalizer/optimizer/action reload. Both failed **all 48 final quiet trials**. Stage 2 remains incomplete. [Read the matched report](REPORT.md) and [latest checkpoint identities](LATEST_CHECKPOINTS.json).

CAPS is the proposed unqualified progress-preview checkpoint: its worst final stop excursion is 75.1 mm, versus 1.887 m for curriculum. Both still exceed the existing 10 mm drift, .03 rad/s raw joint RMS, .002 rad target-step and .5% requested-saturation bounds. The target-step 95th percentile is .04 rad in every final quiet replica. This is not a smoothness success.

## Evidence layout

- `curriculum/raw/run` and `caps/raw/run`: 72 selected raw files per branch, including decisions 10/25/50 and final checkpoints, training traces/events, all matched evaluations, ownership jobs and restored timer receipts.
- Each branch's `audit/FETCH_PLAN.json`: every remote file, SHA-256 and size; exactly 50 intermediate native model autosaves per branch remain remote-only. No raw trace was dropped.
- Each branch's `analysis`: frozen analyzer002 outputs from the complete original remote tree, including every per-replica quiet failure and separate raw SDK/interval-angle evidence.
- `preparation`: exact small native003, host002, both guarded launches and analyzer002 bundles, all original manifests intact; complete 598-file source map and origin. Full 550-file asset package and legacy source are referenced by those identities rather than duplicated.
- `comparison`: paired branch calculations; initial constant and initial stop evidence compare exactly.

Acquisition completion is separate from quality admission. The raw audits verify six immutable phase aliases, original and saved checkpoint hashes, all 12 exact owned container names/IDs absent per branch, pause056/057 restoration and unchanged halo archive. They preserve the systemd historical invocation rather than treating an empty inactive InvocationID as the original identity. The existing halo archive stays remote-only.

## Reproducibility

`python3 verify_bundle.py` verifies the whole publication inventory, all 144 curated raw files against their original maps, every included preparation freeze, and final checkpoint/50-update identity.

Replay the compact paired calculation with NumPy using:

```sh
python3 comparison/compare.py --curriculum curriculum/analysis/report.json --caps caps/analysis/report.json --analyzer preparation/analyzer --output /tmp/hexapod_matched_comparison_replay
```

Full analyzer command (on the original complete remote campaign, where omitted autosaves still exist):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/wx/venv-models/bin/python -B /home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_train_analysis_002/analyze.py --campaign /home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_train_pilot_caps_001 --cold /home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_cold_001 --output /tmp/hexapod_caps_analysis_replay
```

Use the curriculum campaign name for its matched branch. This is CPU NumPy analysis, with no Isaac, Torch or CUDA import. A curated local tree is not the complete host-admission tree. Use the immutable full remote tree for host admission or the native recording adapter.

The selected C direct315/318 physics retain the old direct solver/actuator contract and formal .04 rad/20 ms software slew bound. Source009 contributes container ownership supervision only. The historical .03 diagnostic and separate slow reference branch remain distinct. Checkpoint, actual video and accepted benchmark remain separate; this bundle does not claim a new video exists.

`PROJECT_SITE_UPDATE_PROPOSAL.json` gives the exact registry links and bounded evidence paths for root publication under the required project-site contract. No tracked files were edited by this artifact preparation.
