# Direct PPO comparison

This campaign has incomplete or inconsistent evidence; preserved partial results are diagnostic only.

Acquisition completion is not physical acceptance. Every result retains the original 1.6 Nm cap and quiet bounds; no threshold or statistical significance is invented.

Training completed 2 updates, 48 controls × 32 replicas (1,536 transitions). Wrapper time 3.44 s; 447 transitions/s. Strict reload reported False.

Training events: 0 terminations, 1 timeouts, 0 nonfoot environment-steps. Requested peak 8.476 Nm; applied peak 1.600000 Nm.

Receipt-only evidence: producer reports 2 updates, complete=False, strict reload=None. Raw/checkpoint verification remains separate.

Optimizer diagnostics (measurement, not a qualification):

Retained 2 update rows / 0 minibatches; 0 sparse gradient rows.
Minibatch LR range unavailable–unavailable; 0 retained rows at the existing 1e-5 floor. Mean recorded KL unavailable.

| Loss | Initial-window mean | Final-window mean |
|---|---:|---:|
| caps_temporal | 0.310879 | 0.339205 |
| caps_quiet_temporal_mean | unavailable | unavailable |
| caps_moving_temporal_mean | unavailable | unavailable |
| caps_spatial | 0.00224995 | 0.00165433 |
| caps_weighted | 0.0313129 | 0.0340859 |

Windows contain 1 disjoint updates each; exact IDs and all minibatch LR/KL values are in report.json.
Conditional quiet/moving means use their own pair counts. These optimizer presentations repeat PPO epochs; they are not unique environment steps.

| Update / minibatch | PPO actor norm | Quiet norm | Moving norm | Spatial norm | PPO–quiet cosine | Combined before → after clip |
|---|---:|---:|---:|---:|---:|---:|

Optimizer diagnostics only. Aggregate temporal losses are not quiet target-step p95; sparse gradients do not describe every minibatch or the Adam parameter update.

Optimizer evidence gaps:
- Optimizer diagnostics schema is missing or differs
- update 1.caps_quiet_temporal_mean: missing/invalid finite number
- update 1.caps_moving_temporal_mean: missing/invalid finite number
- update 1: missing minibatches
- update 1: expected 20 retained minibatches
- update 2.caps_quiet_temporal_mean: missing/invalid finite number
- update 2.caps_moving_temporal_mean: missing/invalid finite number
- update 2: missing minibatches
- update 2: expected 20 retained minibatches


No historical moving-to-stop baseline exists for the cold constant run. The final stop result is an absolute screen, not an improvement claim.

Evidence problems:
- campaign identity: ValueError('Wrong/missing native campaign schema')
- standing identity/ownership receipt: ValueError('Phase asset/plan identity differs')
- train identity/ownership receipt: ValueError('Phase asset/plan identity differs')
- final_constant identity/ownership receipt: FileNotFoundError(2, 'No such file or directory')
- final_stop identity/ownership receipt: FileNotFoundError(2, 'No such file or directory')
- Optimizer diagnostic inventory has 9 evidence gaps; see optimizer_diagnostics.errors
- Campaign did not complete terminal input integrity; no continuation admission
- Matched final constant and stop diagnostics are incomplete or unverified

Next decision: Resolve the listed infrastructure/evidence failure before allocating a successor; no automatic training or promotion.

Input paths and SHA256 values are recorded in INPUTS_SHA256.json and rechecked after analysis. No source, checkpoint, input or GPU job is modified.
