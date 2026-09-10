# Low-speed reward scope audit

This read-only audit executes the exact source009 tracking and quiet-term functions
without importing Isaac. All five nonzero reference commands, as well as zero
command, enable the inherited standing penalties. Perfect tracking at these
commands receives no linear or yaw progress bonus because the inherited motion
thresholds are 0.03 m/s and 0.05 rad/s.

This mismatch does not establish that the total reward prefers stillness. For
0.005 m/s forward, perfect rather than zero velocity increases the weighted
tracking term by about 0.0576 per second. On the actual009 requested-motion rows,
the separately replayed standing joint-rate and target-rate components average
about -0.00285 and -0.00131 per second. These calculations exclude other terms;
they are not a counterfactual full-physics comparison or an exact full reward
replay. Internal reward command slew timing is not reconstructed, and reported
SDK joint-rate bias is retained explicitly.

The current two-update zero-command PPO integration remains unchanged. Before
moving-policy training, define a separate reward contract that distinguishes
requested motion, a finite supported stop, and settled quiet hold. Use the
admitted command envelope for tracking/progress semantics and log raw terms.
Keep all acceptance gates, measurement channels and frozen historical sources.
The audit does not admit a new reward, a checkpoint or either project stage.

`report.json` binds the two frozen source files, published009 trace and actual
resolved environment weights. Reproduce it with the workspace Python:

```sh
uv run python analyze.py --repository /absolute/path/to/HEXAPOD --source /absolute/path/to/source009 --output /new/path/report.json
```

Source009 can be reconstructed from its existing frozen publication. The script
writes only the explicitly selected new report; it does not launch a GPU job.

The independently reviewed decision also covers airtime and posture penalties,
which share the inherited moving/standing mask. The reviewer receipt binds the
exact analysis and report bytes under `independent_review/`.
