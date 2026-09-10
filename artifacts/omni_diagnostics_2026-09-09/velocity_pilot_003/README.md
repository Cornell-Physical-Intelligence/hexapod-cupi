# C-study velocity pilot003: walking and quiet acceptance failed

Exactly 50 scratch PPO updates completed, but the policy did not learn useful
locomotion and lost its initially quiet stance. The final policy fails all
48 quiet-standing and 48 stop-to-stand windows; the initial zero-mean policy
passed all 96. Do not continue this checkpoint unchanged or present the run's
completion as Stage 2 qualification.

Both checkpoints received the same 12 direction cases with four replicas each,
followed by matched quiet/stop reviews. The longer final quiet/stop review
recorded 42 terminations and peak requested torque 5.6253 N·m; applied torque
remained capped near 1.6 N·m. The short directional cases had no terminations,
but failed to produce useful requested translation or turning. All four
observation/history/executable-state audit differences were zero.

- [Original campaign](results/run/campaign.json), [full matched comparison](results/run/comparison.json), [concise independently verified summary](verified_summary.json).
- [Initial diagnostic report](results/run/initial/diagnostics.json), [final report](results/run/final/diagnostics.json), [initial quiet/stop report](results/run/initial/quiet_review/quiet_stand.json), [final quiet/stop report](results/run/final/quiet_review/quiet_stand.json).
- [Independent measured failure review](independent_review/README.md), [frozen review map](independent_review/FREEZE_SHA256.json).
- [Initial checkpoint](results/run/train/policy/initial.pt), [final checkpoint](results/run/train/policy/final.pt), [exact sidecar/embedded-contract audit](checkpoint_audit.json).
- [Original 51-file result/restoration/preparation map](full_result_SHA256SUMS.json), [post-run integrity and cleanup](results/postrun_verification.json), [pause022 restoration](results/forecast_pause_022/restored.json).

The independent trace review measures small deterministic action biases
accumulating into joint target displacement, with increasing PD target error
and requested torque while the body barely moves. This explains the observed
failure in this pilot; one seed and 50 updates do not establish that every
target-velocity architecture must fail. The next bounded experiment should
verify actual full-robot contact and swing behavior before allocating long PPO
training.

Initial checkpoint SHA-256:
`bae4f415ad7c2e63b2dbe8cfb8a95d348cdfc0da89d3d79e64bdd9fac4377eaa`.
Final checkpoint SHA-256:
`88f8e60a7b1536229f51524cdc646f06d53e4f33cabb74aa89ebde73d06f7247`.
Source manifest SHA-256:
`00d4e07e0e31776e3d2faa0136a2b1386f79fd4940bf81d2f4a484155f81bb25`.
Executable source identity:
`fb5e472bb710daa50d1d7f16a6f3812d2ed845e2400fcc6769859b3fb0b9519e`.

All 909 source files and 553 admitted inputs (550 asset files plus three
admission/calibration/smoke receipts) remained unchanged. Every downloaded raw
result matched its remote hash; both checkpoints also match their sidecars
and embedded contracts. All three exact owned containers were absent after
terminal completion and pause022 restored its forecasting timers. Later
recording or training jobs have separate ownership and evidence.

The [preceding successful probe](../velocity_candidate_003/README.md) preserves
all six executed candidate files, the actual RSL CPU regression and CPU tests.
This pilot used the same formal 0.04 rad/20 ms, 8 rad/s², 495/498 contract and
started from scratch rather than the probe's stand-only weights. The guarded
[pilot launcher](results/preparation/launch_velocity_train_spark.py) admits
exactly 50 updates and two evaluations, with no automatic continuation.
The full 154-scenario review, path following, visual smoothness, terrain,
perception and physical four-bar qualification remain outstanding.
