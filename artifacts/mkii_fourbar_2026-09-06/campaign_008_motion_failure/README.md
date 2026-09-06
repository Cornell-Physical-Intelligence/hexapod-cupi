# Campaign 008: completed nominal motion test failed

The genuine nominal report completed **32 environments × (1,000 standing +
2,400 driven controls)** and failed its physical admission checks. All
**54,400 physics substeps per environment** were sampled, totaling
**1,740,800 environment-substep samples**. There were zero episode
terminations or truncations. The campaign stopped before refined validation
or either PPO phase.

The three reported failures are:

- Driven C-pin gap **0.213820967 mm**, exceeding the existing **0.1 mm** bound.
- Minimum force-loaded support **zero** during the driven window. Non-foot
  contact count remained zero; the generic contact/support error therefore
  reflects missing support in these aggregate metrics.
- Four motor-response measurements failed the existing requirement of
  **positive-minus-negative response > 0.005 rad**.

| Failed response | Value, rad |
| --- | ---: |
| Individual `lm_tibia_lever_pivot` | −0.005873203 |
| Individual `lr_tibia_lever_pivot` | −0.190767854 |
| Group `lf_tibia_lever_pivot` | 0.001421630 |
| Group `rm_tibia_lever_pivot` | −0.012264192 |

Sixteen of eighteen individual and sixteen of eighteen group response
measurements exceed the requirement. The LF group response is positive but
insufficient; it is not classified as a reversed response.

| Measurement | Settled standing | Driven |
| --- | ---: | ---: |
| Physics substeps per environment | 12,800 | 38,400 |
| Peak raw torque demand, N·m | 0.764960706 | **85.207611084** |
| Peak applied torque, N·m | 0.764960706 | **5.5** |
| Maximum C-pin gap, µm | 1.720294 | **213.820967** |
| Maximum passive position residual, rad | 0.000019073 | 0.003277421 |
| Minimum loaded feet | 5 | **0** |
| Non-foot contacts / invalid samples | 0 / 0 | 0 / 0 |
| Minimum non-foot clearance, m | 0.025156781 | 0.002492130 |
| Minimum plate height, m | 0.135634035 | 0.115751244 |
| Peak passive velocity residual, rad/s | 0.692813993 | **221.326507568** |
| RMS passive velocity residual, rad/s | 0.002320152 | 0.083983946 |
| Peak C-pin relative speed, m/s | 0.012707432 | **7.051413536** |
| RMS C-pin relative speed, m/s | 0.000107583 | 0.004135030 |

The first 3,200 substeps are startup; its peak raw/applied torque is
1.559164524 N·m. Applied motor-envelope excess remained zero in every
window. Clipped applied torque does not erase the much larger raw demand.
All windows and all 36 response values are retained in the primary report;
the derived summary retains every window and identifies the failed checks.

These aggregate results do not identify the worst environment, ordering of
events or physical/numerical cause. Zero force-loaded support alone does
not distinguish geometric flight from near-ground unloaded contacts. No
environment-specific cause or remedy is inferred here.

## Terminal state and identity

The timestamped log records the failing validation result at
**2026-09-06T03:09:38.210712561Z**. The final supervisor record has remote
modification time **03:09:40.548805 UTC**; the terminal campaign record has
modification time **03:09:40.563399 UTC**. The report exists and is preserved
verbatim. Container exit was **0**, but the supervisor correctly rejected
the failed primary report and returned **1**. This is a completed physical
gate failure, not a missing-report or coordination-yield incident.

Source is **`1239159c185cd504c359bb20e98bde9986acbbb4`**, functional identity
`c06e56ed68508f944363bfe6564a13d0760066e4a338164b664214edb695fb5a`.
The report uses the physical v5 model, **64/16** solver iterations and
1.25 ms physics. The supervisor verifies unchanged source and exact
owned-container removal. The 309-file captured source manifest has SHA
`739c16dcd0d1170b175e5ae04d4b0d1efebdbd4c29e38dd610690df293b03686`;
every functional-contract file matches it.

[Nominal primary report](nominal/hexapod-fourbar-validate-20260906T021740Z-96e3036c/report.json)
SHA-256: `c57a2f565d11b62213cca043aea4b26bdaa18d56cc2ad71848cf322793de751e`.
The already-preserved startup probe and launch are in
`../coordination_control_release/`; they are not copied again.

Remote campaign:

```text
/home/orionh/HEXAPOD_runs/mkii_coordination_control_v1/campaigns/fourbar-campaign-20260906T021507Z-9d668815
```

The later authorized CPU takeover is preserved separately in
`../cpu_takeover_20260906T031357Z/`. It occurred after this result and cannot
be treated as a demonstrated cause or correction of the motion failure.

## Reproduction

```sh
python3 artifacts/mkii_fourbar_2026-09-06/campaign_008_motion_failure/verify.py \
  artifacts/mkii_fourbar_2026-09-06/campaign_008_motion_failure \
  --out /tmp/hexapod_campaign008_failure_new.json
```

The verifier checks downloaded bytes against remote hashes, exact sample
counts, source/cleanup identity, all motor-response names and reported
velocity RMS arithmetic. It reproduces `summary.json` without changing the
primary grade. Raw state traces were not produced by this validator. The
large source archive remains remote and was not independently rehashed.
No model, acceptance bound, production source, prior evidence or workload
was changed by this audit.
