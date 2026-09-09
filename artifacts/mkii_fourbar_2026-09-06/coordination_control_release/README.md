# Coordination-control release and campaign 008 startup evidence

Release **`1239159c185cd504c359bb20e98bde9986acbbb4`** passed 906 local CPU tests
and GitHub CI. Campaign 008's completed startup probe also passes: one
environment, 100 controls and all 1,600 physics substeps. **This is startup
evidence only; full nominal/refined qualification and PPO are not established.**

The captured campaign state at **2026-09-06 02:23:03 UTC** shows the probe
passed and nominal validation running, with no PPO phase started. The
nominal run was created as `20260906T021740Z` (02:17:40 UTC); its first
recorded resource gate is 02:17:43 UTC. These are historical capture facts,
not a continuing live-status claim.

## Release and CI

| Evidence | Result |
| --- | --- |
| Local test log | 906 tests passed in 70.128 s |
| [GitHub tests run 34005889807](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/actions/runs/34005889807) | Completed successfully at the same commit; updated 02:15:31 UTC |
| Release pipeline | 294 manifest entries verified; 295 release files including the manifest |
| Functional identity | `c06e56ed68508f944363bfe6564a13d0760066e4a338164b664214edb695fb5a` |
| Pipeline manifest SHA-256 | `b75f8ca4d3045e058c9e2eb27deebea9e08aa789b56f50fbf7fd215f6960a1f8` |

`local_release/` preserves the original local release JSON and test-log
bytes, with their original paths, modification times and hashes recorded in
`local_capture.json`. `ci_run_34005889807.json` is the compact structured
GitHub CLI response; the query and response hash are recorded separately.
Remote staging fields match the local release record.

The original campaign launch record is dated **02:15:07 UTC**, campaign PID
1712641. Its bootstrap script is preserved verbatim and matches the launch
record's SHA `7414dbc6a09cff05f286a8569db2692c13a3022a570457b831767ca4a5c15682`.
No launcher or GPU job was executed by this evidence audit.

## Completed probe

The [primary report](probe/report.json), SHA
`2a85e74824a8fc28f8f74907e1324e94db5fa2728681d6685115dfe9bb35b9b2`,
records **PASS**, with no errors and no simulation-training or hardware
admission. It uses the physical v5 model with solver settings **64/16**,
1.25 ms physics and sixteen substeps per control.

| Probe measurement | Settled window |
| --- | ---: |
| Peak raw / applied torque | 0.701226115 N·m |
| Mean plate height | 0.135715164 m |
| Maximum C-pin gap | 0.674391 µm |
| Minimum loaded feet | 6 |
| Non-foot contacts / invalid samples | 0 / 0 |
| Peak / RMS passive velocity residual | 0.019164562 / 0.001018377 rad/s |
| Peak / RMS C-pin relative speed | 0.001623128 / 0.000082924 m/s |

The settled window contains 1,280 substeps (0.4–2.0 s); the first 320
substeps are startup. Startup peak raw/applied torque is **1.553987503 N·m**
and maximum C-pin gap **1.237066 µm**. Startup is not omitted or presented as
settled performance.

The genuine supervisor records successful exit, exact owned-container
removal and unchanged source identity. Its captured source manifest has
309 files, including runtime-generated files, SHA
`739c16dcd0d1170b175e5ae04d4b0d1efebdbd4c29e38dd610690df293b03686`.
Every functional-contract file matches that captured source list. This count
is separate from the 294-entry release pipeline manifest.

The probe supervisor reports coordination protocol
`canonical_share_status_v2`, with canonical status `NONE` at startup,
before admission and at its latest check. This artifact **does not test a
live edit** of coordination prose or control status; that experiment has
separate evidence. The shared file was not read or modified by this audit.

## Preservation and reproduction

The complete probe directory is preserved except its 61,971,161-byte source
archive, which remains on Spark and was not independently rehashed. Raw
remote bytes, SHA-256 values, sizes and modification times for the selected
original files are recorded in `remote_inventory.json`. Campaign/nominal
progress captures contain explicitly selected fields and hashes of their
source JSON; they avoid duplicating the complete embedded source contract.

Remote release and probe:

```text
/home/orionh/HEXAPOD_runs/mkii_coordination_control_v1
/home/orionh/HEXAPOD_runs/mkii_coordination_control_v1/campaigns/fourbar-campaign-20260906T021507Z-9d668815/probe/hexapod-fourbar-validate-20260906T021508Z-e5cb9280
```

From the repository root:

```sh
python3 artifacts/mkii_fourbar_2026-09-06/coordination_control_release/verify.py \
  artifacts/mkii_fourbar_2026-09-06/coordination_control_release \
  --out /tmp/hexapod_coordination_control_verification_new.json
```

The verifier checks captured original hashes, local/remote release agreement,
CI commit/conclusion, bootstrap identity, probe coverage/scope,
source-manifest agreement and exact cleanup. `verification.json` is derived
evidence, not an admission report. No production source, shared note, remote
workload, prior evidence or repository status document was changed.
