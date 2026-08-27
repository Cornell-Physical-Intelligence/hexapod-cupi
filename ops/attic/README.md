# Attic — retired deploy scripts

These scripts are kept verbatim for provenance. **They are not entry points.**
Nothing in the repository calls them, no test covers them, and they are not
listed in `isaaclab/deploy/stage2_pipeline.sha256`, so they are not part of the
verified release and are not synced to the Spark as launchers. Do not run one.

They were moved here from `isaaclab/deploy/` unchanged. Every one of them
belongs to a superseded phase: Phase 1 acceptance, recording, and screening; the
Stage1 / Stage2 / Stage2B batch screens; the pre-hardening Stage2C probe
drivers; and the smoke and stance-sweep helpers from before the current probe
launcher existed. The behaviour that replaced them lives in the three hardened
Stage2C launchers that stayed in `isaaclab/deploy/`, and the operator wrapper
around those lives in `ops/hexctl`.

A script qualified for the attic only if all four held: it is absent from
`stage2_pipeline.sha256`; it is not one of the three live Stage2C launchers
(`probe-stage2c-single-current-best`,
`calibrate-stage2c-bilateral-current-best`, `screen-stage2c-probe-sharded`); it
is not a systemd unit, the `hexapod-rl` operator CLI, or
`preflight-stage2-spark`; and no test under `isaaclab/tests/` references it.

Contents:

```text
accept-phase1-v4                              Phase 1 v4 acceptance run
accept-phase1-v5                              Phase 1 v5 acceptance run
probe-phase2-recovery-stage2c-parallel        Pre-hardening parallel Stage2C probe driver
probe-phase2-recovery-stage2c-rapid-sequential Pre-hardening sequential Stage2C probe driver
record-phase1-v4                              Phase 1 v4 video capture
record-phase1-v5                              Phase 1 v5 video capture
record-phase1-v5-army                         Phase 1 v5 multi-robot video capture
screen-phase1-v2                              Phase 1 v2 screen
screen-phase1-v3                              Phase 1 v3 screen
screen-phase1-v4                              Phase 1 v4 screen
screen-phase1-v4-batch                        Phase 1 v4 batch screen
screen-phase1-v5-batch                        Phase 1 v5 batch screen
screen-phase2-recovery-stage1-batch           Stage1 recovery batch screen
screen-phase2-recovery-stage2-batch           Stage2 recovery batch screen
screen-phase2-recovery-stage2b-lateral-batch  Stage2B lateral batch screen
screen-phase2-warmup-batch                    Phase 2 warmup batch screen
smoke-phase2-recovery-stage2                  Stage2 smoke run
smoke-phase2-recovery-stage2b-lateral         Stage2B lateral smoke run
validate-stance-sweep                         Stance-sweep validation driver
```

If one of these is ever needed again, treat it as historical source: read it,
port what is still true into a hardened launcher that holds the shared GPU lock
and cleans up by exact container ID, and leave the original here untouched. The
legacy smoke scripts in particular call `docker rm -f` and have no shared-lock
contract, which is exactly why `docs/OPERATIONS.md` §6 item 8 forbids using them
unchanged.
