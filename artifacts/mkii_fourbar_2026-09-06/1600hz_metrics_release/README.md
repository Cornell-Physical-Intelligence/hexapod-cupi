# Frozen 1,600 Hz release and actual campaign

Source `c2af43ca0f384a4c2c7ab8f1d627f309dc78a683`, functional identity `c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc`, passed all 935 repository tests in 75.487 seconds. GitHub CI [passed](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/actions/runs/34070188757). The 308-path current manifest and 112-path archived lineage verify. `source_release.json` records the exact archive and manifest hashes.

The source doubles outer physics frequency from 800 to 1,600 Hz while retaining 50 Hz control, motor dynamics in elapsed-time units, 0.04 rad maximum control endpoint increments, torque envelope, geometry and all physical acceptance bounds. It installs a conservative measurement optimization whose full 18-field CUDA rows match the original byte-for-byte; the permanent CPU regression covers the 32-substep timing. This is a fresh numerical recipe, so no older failed or successful physical report admits it.

`staging.json` proves the isolated Spark source matches every manifest path and full functional identity. The first staging script successfully extracted the verified archive but counted four manifest comment lines as file rows; its verifier assertion is preserved in `staging_attempt_001.json`. The separate corrected verifier checked the existing files without overwriting them. No GPU work was launched by either staging script.

The actual bounded campaign launched at **2026-09-07 00:45:23 UTC**, PID **2021559**, start ticks **105778155**. State lives at:

`/home/orionh/HEXAPOD_runs/mkii_1600hz_metrics_v1/campaign_001/fourbar-campaign-20260907T004524Z-8592e14a/campaign.json`

`launch_001.json` records the source, wrapper hash, process identity, resource preflight and shared-note hashes. The pinned external [host wrapper](../1600hz_campaign_host_v1/run_campaign.py) prefixes each actual phase with nonblocking, no-fork shared flock. The campaign itself holds no persistent reservation. It retains the frozen source's phase order, admission, exact-checkpoint resume, source checks and owned-container cleanup.

The campaign runs a fresh 1-robot/100-control probe; complete 32-robot/1,000-standing plus 2,400-driven nominal and refined comparisons; then 64-robot/3-update scratch PPO and separate 512-robot/1,000-update resume only on successful physics admission. Validation phases each have a 7,200-second bound and the full PPO phase has a 21,600-second bound. Actual 512-robot throughput is not yet measured; completion time is not guaranteed.

At publication this records launch, not physical admission or completed PPO. No learned-policy video exists yet. The user's full training priority remains active; the removed five-minute app automation remains removed.
