# ADR-0001: Rebuild the training asset from leg v3 and retire the mock asset from training

Status: proposed (2026-08-27)

## Context

Every policy in `artifacts/` was trained on
`robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf`: an Onshape
mock assembly with user-specified mass targets (1.5 kg body, 0.8 kg legs),
estimated inertias, collision geometry taken from visual meshes, no foot pads,
and self-collision disabled (`docs/TRAINING.md` §7). Since then the team has
produced `robot/hexapod_leg_v3/`, a leg registered to the onshape-to-robot
export at 0.02 mm RMS with per-body inertials and a 0.7888 kg fused mass, and
a sensor payload of at least 0.381 kg is about to be added to the body.

No checkpoint passes the Stage2C gates and none is hardware-ready. The
Stage2C research loop is spending its effort on infrastructure (four attempts
lost to pre-AppReady stalls) and on a yaw-RMSE gate for a robot whose mass
distribution is known to be wrong.

## Decision

Build `robot/hexapod_mkii_v1/` from six leg v3 instances on the real body with
the sensor payload at the chosen mount, primitive collisions, explicit foot
pads, and self-collision enabled; generate its USD by script; register new task
IDs for it; retrain from scratch. Stop launching Stage2C probes on the mock
asset. Keep every existing artifact, gate, test, and launcher; they are the
tooling and the Phase-0 reference lineage.

## Consequences

- All existing checkpoints become historical. Their hashes and evidence do not
  change; their status becomes "Phase-0 lineage, mock asset".
- The 66-dim observation and 18-dim action contracts survive unchanged; the
  stance values, deck targets, and reward scales must be re-derived for the
  new mass distribution (repeat the stance sweep).
- `STATUS.md` open contradictions 1 and 2 close as moot; the record of them
  stays in the archive and the probe ledger.
- The first weeks of RL work produce no new gait; they produce an asset that a
  gait can transfer from. That is the intended trade.
