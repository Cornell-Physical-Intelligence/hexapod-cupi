# Review: origin versus (14,4) single-robot standing diagnostic

## Context

James authorized the canonical qualification restart on 14 September 2026 on the detailed 19-body, 18-joint direct-drive model. URDF SHA-256 `9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78`. The 32-robot source005 standing attempt gave 10 of 32 combined passes, 22 support failures, and 8 joint-rate failures among those. The saved analysis attributes support failures to 73 single-step zero-normal-force toe events. The proposed next discriminator is two fresh one-robot runs, origin versus the env 23 world position (14,4), with source005 dynamics, timing and gates unchanged.

This review assesses whether that comparison isolates placement, which checks it needs, what it can conclude, and which CPU fixtures should guard it. No repository file was read or modified. Items marked "verify in repo" are assumptions to confirm before dispatch.

If this plan is approved, my implementation scope is the fixtures and record schema in the last two sections. Dispatch stays root-only and outside my scope.

## Verdict

Approve with conditions, as a cheap placement rule-out pilot. Reject any inference beyond "reproduced" or "not reproduced" at the event level. Reject any use of a pass as a standing baseline, as stage qualification, or as training admission evidence.

Smallest corrections:
1. Pre-register the outcome table below and its allowed conclusions before dispatch.
2. Read out per-toe events over all 8000 steps with the saved detector, not only pass or fail.
3. Feed both placement authoring points from one declared record, verified by physics-view readback.
4. Record the floor collider, live scene parameters, body and joint order, and what the 128 slot figure denotes.
5. If a third run fits capacity, repeat origin rather than add a position.
6. Add the five fixtures. Mark the diagnostic identity non-admitting.

## Evidence versus hypothesis

Recorded evidence:
- 73 events, each exactly one 2.5 ms step, exact zero normal force, 128 inactive patch slots.
- Contact-view occupancy 2641 of 32768. Global buffer exhaustion is not supported.
- Events coincide with small tibia motion and joint-rate transients.
- Middle legs carry 60 of 73 events.
- Events span all grid rows and columns with no monotonic distance trend.
- One single-robot run passed, once. Env 23 at (14,4) had 7 events, the first at 4.2775 s.

Hypotheses, none established:
- H1 placement. World XY offset causes contact loss. The recorded evidence weighs against it. There is no distance trend, and a featureless floor cannot select middle legs. If the floor is tessellated, the tessellation is the variable, not distance.
- H2 multi-env interaction. Physics replication, pair ordering or GPU partitioning in the cloned scene behaves differently from a single articulation. Untested. The occupancy figure does not exclude it.
- H3 per-robot contact marginality. Toes rest at the contact-offset threshold and lose the patch for one step under actuator transients, regardless of placement or env count. Consistent with every recorded item, including the one-step duration and full patch inactivity.
- H4 reporting artifact. Not established.

A point the proposal does not state: the 32 envs are nominally identical apart from world position and index. If source005 applies no per-env randomization, verify in repo, then the spread from zero to seven events across envs shows the standing state amplifies numerical-level differences. Placement, env-index effects and GPU nondeterminism are candidate seeds. Marginality is the amplifier. A single-robot arm samples that marginality, so one discordant pair cannot be attributed to placement without replication.

Statistics that govern the design:

| Quantity | Value |
| --- | --- |
| 32-env per-robot pass rate | 10 of 32, about 0.31 |
| Mean events per env, versus envs with none and env 23 | 2.3 mean, 10 envs with none, 7 in env 23 |
| Probability a lone single-robot run passes if single-robot and 32-env do not differ | about 0.31 |
| Probability of "origin passes, (14,4) fails" under that null, one run per arm | about 0.21 |

The premise that single-robot standing differs from 32-env standing is therefore not yet evidence. It is one draw. The event counts cluster by env, which is why the readout must be events, not gates.

## Does the comparison isolate placement?

Only inside the single-robot topology, and weakly at one run per arm.

- It separates world XY from everything else held equal in one articulation. It cannot separate placement from multi-env interaction, because both arms drop the cloned scene.
- Pass or fail is too coarse a readout. The comparable signal is the per-toe count of single-step zero-force events over all 8000 steps, produced by the same detector and schema as the saved analysis. Env 23's seven events in its quiet window are the reference.
- Confirm what the 128 figure denotes. If it is the full per-body slot capacity, the event is "no patch generated", a geometric separation. If patches existed with zero impulse, it is a solver-inactive contact. These point to different causes and the diagnostic record must distinguish them.

Outcome table, recorded before dispatch and not revised after:

| Origin events | (14,4) events | Allowed conclusion |
| --- | --- | --- |
| 0 | several | Placement-linked effect suggested. A replicate pair is required before any claim. |
| 0 | 0 | Placement alone does not reproduce in a single robot. H2 and H3 remain. |
| nonzero | nonzero | Single-robot baseline is itself marginal. H3 favoured. The earlier pass was a draw. |
| nonzero | 0 | Same as the row above. H1 disfavoured. |

Smallest power correction: if capacity allows a third run, repeat origin. The inference rests on single-robot repeatability, which has one sample. If the pipeline is deterministic and the repeat reproduces the ledger exactly, GPU nondeterminism is excluded as a seed. If it does not, no two-arm comparison can attribute a difference to placement.

## Pre-warmup, reset and readback checks

Use one declared placement record consumed by both authoring points. Two hand-typed offsets can diverge silently.

Pre-warmup, `run_standing.py`:
1. Record the declared placement: x 14, y 4, z equal to source005's default root z, yaw 0, exact prim path. Read the robot prim's computed world transform before SDK warmup and assert it matches. Same xform op order in both arms, no scale, no rotation.
2. Assert the declared placement equals env 23's recorded initial root world position from the 32-env run, to recorded precision. Verify in repo that env 23 had zero initial yaw.
3. Record the floor collider as instantiated: prim type, extent, position, material, contact offset, rest offset. Assert byte-identical to source005's. A finite or tiled floor breaks the placement equivalence. Verify in repo.
4. Read live PhysX scene parameters after warmup, not the config: 32 position and 0 velocity iterations, 2.5 ms step, solver type, GPU pipeline and device, broadphase, CCD, friction model, GPU buffer capacities. Assert equal to source005's recorded values. A new config identity must not inherit different defaults.
5. Assert the 19 body names and 18 joint names, in order, match source005's recorded lists. Toe bodies must map to the same leg labels the saved analysis used.

Reset, `standing_session.py`:
6. Record every field the reset writes: root pose and its frame convention, root velocity, joint positions, joint velocities, joint targets. Assert root XY equals the declared placement, z and quaternion equal source005's, quaternion unit and in the expected component order. Verify the order convention in repo.
7. Diff the two arms' reset arguments. The XY term must be the only difference. Confirm the offset is applied exactly once. A reset that re-adds a zero env origin, or that overrides the pre-warmup placement, silently defeats the comparison.
8. Confirm the root write uses the same link-frame or centre-of-mass convention as source005. A mismatch injects an initial displacement. Verify in repo.

Readback, from the physics view, never from USD attributes:
9. After the first physics step: root XY within 1e-3 m of declared, z within source005's tolerance, joint positions equal targets, joint velocities near zero.
10. At the end of the 4 s settle: six toe world heights and normal forces equal the origin arm's within recorded tolerance, all six positive.
11. Continuous per-step record for all 8000 steps: per-toe normal force, active patch count, tibia link velocity, joint velocities, root pose. Same schema as the saved analysis, so one detector runs on both arms and on env 23's ledger. Include the settle window.
12. Record PhysX and Isaac builds, determinism flags, seed, GPU memory before and after, and the guard identity bound to the diagnostic source hash and the current coordination hash.

## Frame and floor assumptions to check

- The support gate must read world-frame normal force, not positions relative to an env origin. Verify in repo.
- Any analysis quantity computed relative to env origin must use the declared placement for the single robot, or it is silently wrong.
- Floor geometry type selects the SDF narrowphase partner path. Plane and triangle mesh are different collision paths and must be recorded.
- Contact sensor mode, net force versus filtered against the floor, and its per-body slot capacity must match the 32-env run, so "128 inactive slots" means the same thing in both.
- Same GPU pipeline as the 32-env run. A CPU PhysX run changes narrowphase and solver behaviour and is not comparable.

## Why a pass cannot promote 32-env standing

- Configuration differs: one articulation at a declared position versus 32 cloned envs. Formal comparisons require the common recorded limiter and matching configuration.
- Identity differs: the diagnostic source and guard identities are new and non-admitting by construction. The canonical stage identity is untouched.
- One run per arm cannot establish a rate. A pass is a roughly 3-in-10 draw under the observed per-robot rate.
- Admission needs the canonical standing stage to pass its unchanged gates in its own configuration. At best this diagnostic localises a cause. Any fix then needs a new identity and a fresh 32-env attempt.

## Minimal independent CPU rejection fixtures

Stdlib unittest with hand-authored JSON records. No expected value may be generated by the code under test.

1. Identity. A diagnostic record with offset (14,4) yields an ID different from source005 and carries the offset. A record claiming a frozen source ID with any offset is rejected. Frozen source005 bytes are unchanged.
2. Placement consistency. Accept a record where pre-warmup, reset and physics readback XY agree within tolerance. Reject reset-at-origin with pre-warmup at (14,4). Reject readback drift beyond tolerance. Reject readback sourced from USD.
3. Non-promotion. A diagnostic result with every gate passing still fails admission and stage-qualification checks, and status generation leaves the standing stage unchanged.
4. Detector comparability. On a hand-authored ledger with one single-step zero-force sample, the detector returns one event with the correct toe and leg label. A clean ledger returns none. Gate constants read from the frozen record equal the expected values: six toes, 0.03 rad/s, 4 s settle, 16 s window, 2.5 ms step.
5. Guard. Extend the existing guard check with one case that rejects a guard bound to the old source hash when the diagnostic source hash is present.

## Dispatch conditions

Root only, after shared GPU locks and verified live capacity. Compare free memory against the recorded peak of the prior one-robot run plus margin. Do not interrupt the DeepSeek inference service at 107 GB. Run the arms sequentially under one reservation. Preserve reservation controls and run post-exit checks per OPERATIONS.

## Verification

- Before dispatch: the five fixtures pass, the required suites in CLAUDE.md pass, source inventory check passes, frozen source005 hash is unchanged.
- After the runs: readback assertions 9 and 10 pass for both arms, event ledgers exist for both, the outcome table row is filled with its pre-registered conclusion only, and an append-only change record is added.
