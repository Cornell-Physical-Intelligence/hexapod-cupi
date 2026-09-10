# C-study candidate 002: exploration passes, runner startup fails

This fresh experiment reduced initial action standard deviation from 0.05 to **0.005** and extended sampled calibration from 10 to **20 seconds**, matching a training episode. The other five candidate executable files, geometry, PD gains, target-rate/acceleration bounds, reward weights and every physical/quiet gate remain unchanged. [The minimal patch and original review](candidate/README.md) preserve the exact intervention.

Fresh flat standing passed 32 robots × 1,000 controls. All 32 zero-mean replicas and all 32 sampled replicas then passed calibration, with zero requested saturation, terminations, truncations or post-settle nonfoot contacts. The sampled trial's maximum per-joint velocity RMS was 0.008404 rad/s and its largest post-settle requested torque was 1.536477 N·m. These are initialization/exploration checks, not learned walking or sustained quiet-policy qualification. [Calibration and per-replica measurements](results/run/probe/calibration.json) retain the complete results.

The following two-update runner smoke failed **before PPO learning** while saving the initial checkpoint: the installed RSL logger had not created its `writer` attribute. [The failure](results/run/probe/failure.json), logs and partially published initial checkpoint bytes remain frozen. No successful runner-smoke receipt or walking admission exists. A new source must repair the logger lifecycle and repeat the guarded sequence; do not resume or promote this initial checkpoint.

Root and an independent reviewer passed 30 CPU tests and verified all 909 source payloads on Spark before dispatch. The tests exercised the candidate/controller and host guards, but their mock runner did not reproduce the real logger initialization failure. A real installed-RSL save/learn/reload regression is required for the correction. The frozen CPU suite can be replayed in its original temporary directory layout with:

```sh
uv run python -B artifacts/omni_diagnostics_2026-09-09/velocity_candidate_002/replay_cpu.py
```

Source manifest SHA: `c3db8ec79f690f68260383b80cb4f73335788b6ad862c7824538ba9f2ed9a453`. Executable source identity: `036d8c2dbb028f813e50c143e68c5e2473f9d6ff2b9517246f6f0054bb0d0b95`. Plan SHA: `356b72f0e82b207ba6105cb42057643d21a5b911d0517661465dcfe57745d00f`. The 16-file C-study runtime remains pinned to `abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280`.

The user unit was `hexapod-omni-velocity-probe-002-20260910.service`, with remote output `/home/orionh/HEXAPOD_runs/mock_length_study_20260909/omni_velocity_probe_002`. Root matched all 20 result payload hashes against Spark. [Post-run verification](results/run/post_run_verification.json) records source and 550 asset files unchanged, both owned containers exited, both GPU locks free and both forecasting timers restored under pause 019. Subsequent execution belongs in [STATUS](../../../STATUS.md); Stage 2 remains incomplete.
