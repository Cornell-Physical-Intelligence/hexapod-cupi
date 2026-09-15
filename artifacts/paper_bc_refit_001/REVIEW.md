# Paired BC refit 001: descriptive fit result, no native evaluation

Both arms completed 1,000 additional CPU BC steps from the migrated fit004 parent
`dfb6d3ec...` under [protocol 002](../restart_2026-09-14/paper_bc_refit_protocol_002/PROTOCOL.json)
with identical minibatch draws (`c28cd12b95ced302...`),
identical final sampling RNG and exact strict reload. Zero PPO updates, zero
simulated transitions and unchanged normalizers, critic, AMP and optimizer states.

| Group | Rows | Parent target RMSE (rad) | Uniform after | Onset20 after | Parent velocity RMSE (m/s) | Uniform after | Onset20 after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3760 | 0.002929 | 0.001467 | 0.001677 | 0.005843 | 0.003398 | 0.003398 |
| steady | 1860 | 0.001980 | 0.001262 | 0.001653 | 0.005186 | 0.002445 | 0.002445 |
| first_moving_onset | 23 | 0.026681 | 0.008535 | 0.001857 | 0.000643 | 0.000519 | 0.000519 |
| first_five_moving | 115 | 0.013154 | 0.004850 | 0.003576 | 0.008849 | 0.003522 | 0.003522 |
| zero_prefix | 400 | 0.001260 | 0.000750 | 0.001051 | 0.008208 | 0.007400 | 0.007400 |

Candidates:

- `uniform/candidate_checkpoint_update000000.pt` SHA-256 `d1ecb388a2b6f642df886ee6585a3346beeb965ffa95638300b7d38385c22a35`
- `onset_weight20/candidate_checkpoint_update000000.pt` SHA-256 `98bcefd13174961958a1c191ba1baeb3ee32e79a5c23e30e2c420a8e479b329c`

Reading: the onset weighting reduces first-moving-onset target error by a factor
of about 14 against the parent and about 4.6 against the equal-budget uniform arm,
while the steady-row error stays above the uniform arm (0.001653 versus 0.001262 rad).
Velocity estimator errors are identical between arms. These are fitting errors on
the training rows; they establish no walking, quiet standing or stopping behavior.

Native cold and settled startup evaluation of both candidates under a fresh
`source_020` guard binding remains required before any interpretation of gait
onset. The Spark host was unreachable from the executing machine at fitting time,
so no native evaluation is recorded here. Stage 2 gates, the 13 learning probes
and the 96 required cases are unchanged and unrun for these candidates.
