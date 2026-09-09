# Prefix replay analyzer

This offline tool reads the completed `validation_prefix` diagnostic and the archived campaign008 full-validation report. It makes no training or hardware admission decision and changes no runtime source.

Run from the repository with its existing NumPy/Torch environment:

```sh
uv run python artifacts/mkii_fourbar_2026-09-06/motion_prefix_analysis/analyze.py \
  /absolute/path/to/prefix/report.json \
  artifacts/mkii_fourbar_2026-09-06/campaign_008_motion_failure/nominal/hexapod-fourbar-validate-20260906T021740Z-96e3036c/report.json \
  --output-dir /absolute/path/to/new_analysis_directory
```

The replay report must remain beside its hashed `trace_*.npz` and `control_*.npz` files. The analyzer verifies every file, column layout, environment count and ordered interval. It requires all 2500 control-boundary rows and the detailed physics interval `[35200,40000)` for the current 16-substep recipe. The local kinematic mapping must match the runtime SHA256; `--kinematics` can point at another exact archived copy.

`analysis.json` records all input/file hashes, 15 recomputed motor responses, every robot's positive/negative end-hold means, exact runtime/reset-position comparisons, independent chronological events, and small neighborhoods around each event. `README.md` in the output directory summarizes the response comparison and event times. Existing output directories are refused.

The five-sample means use float32 Torch arithmetic on CPU. Any difference from recorded GPU-reduced means is reported explicitly. Campaign008 retained global response minima, so it cannot establish the earliest exact cross-run trajectory divergence. The replay's full-control event timeline and detailed LF/LM/LR timeline locate observed events without treating separately timed global peaks as one event.

Instantaneous joint speed, finite-difference interval-average speed, raw PD demand, applied torque, recorded motor limit and remaining burst headroom retain separate meanings. A raw command opposing motion expresses braking intent; it does not establish delivered braking torque or identify a unique cause. Event selectors do not modify any physical gate.

Tests:

```sh
uv run python -m unittest discover \
  -s artifacts/mkii_fourbar_2026-09-06/motion_prefix_analysis -p 'test_*.py'
```
