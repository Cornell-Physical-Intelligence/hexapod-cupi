# Current status

Last updated: 2026-08-26

This is the single current-state page for the project. **Rewrite this file in
place; never append to it.** Anything that must survive as a dated record goes
to `docs/incidents/` or to the probe ledger under `artifacts/`, both of which
are append-only. Durable design lives in `docs/TRAINING.md`; the operating
procedure lives in `docs/OPERATIONS.md`.

## Current best checkpoint

```text
local:  artifacts/phase2_recovery_stage2c_stable_forward/checkpoints/current_best/model_2.pt
remote: /home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/
          hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/
          2026-08-25_23-44-14_accel_scale4_yaw160_trackguard_20260825T234500Z_seed86/
          model_2.pt
SHA-256: a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a
```

Formal seed-60 nominal screen at the `0.040 rad / 20 ms` playback limiter:

| Command | Achieved forward | Yaw RMSE | Deck composite | Worst-joint duty | Falls |
| --- | ---: | ---: | ---: | ---: | ---: |
| stand | 0.0003 m/s | 0.0002 rad/s | 0.013 | 0.000 | 0 |
| 0.16 | 0.1599 m/s | 0.1070 rad/s | 0.765 | 0.206 | 0 |
| 0.20 | 0.1962 m/s | 0.1117 rad/s | 0.862 | 0.238 | 0 |
| 0.30 | 0.2420 m/s | 0.1378 rad/s | 1.032 | 0.223 | 0 |

Worst burst is `0.08 s`; peak raw demand is `2.691 Nm`.

This checkpoint is the immutable parent for every current probe. It is
operationally safe in the nominal screen and visually walks anatomical forward,
but it misses the yaw gate at every moving command and the deck gate at
`0.30 m/s`. It is not hardware-ready. Do not promote anything over it without
formal absolute-gate checks.

## Current target

Mission of record (2026-09-03): survey a bounded area an operator draws on a
map, steadily, as a stable platform for data collection. See `dar.md` and
ADR-0004. The training target below serves that mission: a steady deck is the
primary grade, and the Phase-0 checkpoints above are the reference lineage
while asset v1 (the CAD assembly, pull request #9) is brought up under its own
task ID.

Stage2C: stable anatomical-forward walking. A candidate is admitted only by
passing all of the gates below in the canonical formal screen (seed 60,
10 s / 475 samples, `0.040 rad / 20 ms` limiter, commands stand / 0.16 / 0.20 /
0.30 m/s):

- Yaw-rate RMSE at most `0.080 rad/s` at every moving command.
- Moving normalized deck composite at most `1.000`.
- Zero falls and zero timeouts.
- At least `0.240 m/s` achieved at the `0.30 m/s` command.
- RS05 limits held: bounded continuous-duty fraction, burst length, raw peak
  demand, and worst-joint duty; never exceeding the `5.5 Nm` raw-demand safety
  termination.

There is no admitted Stage2C breakthrough. Stage2D and Stage2E are configured
but not admitted. Phase 3 is paused.

## Open contradictions

These are unresolved as of this writing. Do not treat either side as settled.

### 1. Declared next action versus what actually happened next

`docs/archive/HANDOFF-2026-08-26.md` (snapshot `2026-08-26T17:46:47Z`) declares
that the next scientific action is completing exactly the same Probe21 causal
arm from the immutable parent, under a new unique attempt label, with nothing
changed:

```text
seed=99
num_envs=12288
rollout_steps=24
updates=12
learning_rate=2e-5
clip_param=0.06
inactive_bilateral_longitudinal_contact_moment_reward_scale=-2.0
inactive_bilateral_longitudinal_contact_moment_reference_nm=1.8
processed_joint_target_slew_limit_rad_per_20ms=0.040
```

That snapshot never mentions Stage2G. Commit `1f95f52`, made later the same
day, adds a full Stage2G insect-gait rework. Whether Probe21 was superseded on
purpose or simply set aside is not recorded anywhere in the repository.

### 2. Unverified Stage2G adapt result

The commit message of `1f95f52` claims that the first Stage2G adapt arm
"reaches 0.291 m/s at the 0.30 command (parent: 0.242) with zero falls and
equal-or-better deck motion."

By this repository's own evidence standards that claim is unverified:

- No Stage2G evaluation artifacts exist. `find artifacts -iname '*stage2g*'`
  returns nothing.
- No Stage2G checkpoint is mirrored, and no SHA-256 is recorded for one.
- No screen configuration is recorded, so it is not known whether the number
  came from the canonical 10 s / 475-sample formal screen, from a diagnostic
  6 s / 275-sample screen, or from an ad hoc rollout.
- The Stage2G adapt task trains at a `0.06 rad / 20 ms` slew limiter, while
  every recorded formal measurement, including the parent's `0.242 m/s`, uses
  `0.040 rad / 20 ms`. A speed comparison across different limiters is not a
  like-for-like comparison.

Resolution requires one of two things: import the Stage2G evaluation artifacts,
checkpoint, and hashes into `artifacts/` and re-screen at the common `0.040`
limiter, or withdraw the claim. Until then, `0.291 m/s` must not be cited as a
result, and the current best above does not change.

## Next actions

1. **Reconcile the Stage2G evidence.** Either import the adapt arm's
   checkpoint, evaluation JSON, resolved config, and SHA-256 manifest entries
   and re-screen at `0.040 rad / 20 ms`, or record that the claim is withdrawn.
   This comes first because it decides whether the research target has already
   moved.
2. **Validate the attempt-aware startup supervisor in vivo.** The supervisor
   specified in `docs/OPERATIONS.md` §6 is implemented in
   `packages/hexapod_train` and driven by `ops/hexctl`
   (`compose`/`doctor`/`probe`/`screen`), unit-tested against fakes, but it has
   never been run against the Spark. Four attempts have been lost to the
   pre-AppReady stall class (`docs/incidents/2026-08-26-preappready-stall.md`),
   and until one real attempt is supervised end to end that class and the
   post-launch GPU-contention race remain open in practice. Items 1, 8, and 10
   of the specification are knowingly partial.
3. **Then run one arm, not both.** Either retry the exact Probe21 bilateral arm
   under a new unique label with the settings quoted above, or put Stage2G
   through a formal screen under the `0.040 rad / 20 ms` limiter. Screen all 12
   children before interpreting either one.

## Environment note

The Spark mirror `/home/orionh/HEXAPOD` is not a Git repository. The dirty-work
transfer caveat in the archived handoff is resolved: the Probe15-21 evidence,
launchers, tests, and Stage2G configuration are committed on `main` through
`3d1500f`.
