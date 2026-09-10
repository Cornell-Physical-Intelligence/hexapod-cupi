# Continuous-reference feasibility review

The first omnidirectional extension of Benchmark 1 is **not ready for a robot or PPO pilot**. It reproduces the forward reference geometry within 0.263 µm, but geometry agreement alone does not establish executable turns or stops. This negative result motivates a support-aware reference and a separately versioned, stateful target-velocity action comparison.

| CPU screen | Finding | Decision |
|---|---|---|
| 100 mm stance travel, 20 mm lift | 98 of 245 command cases contain invalid workspace/IK samples | Reject this extension |
| 60 mm travel, 20 mm lift | All sampled constant-cycle cases reachable; independent transitions request up to 0.14508 rad/20 ms and 0.09864 m/s predicted stance slip | Reachability does not admit transitions |
| Coherent global slowing, 60 mm travel | The two screened choices admit only 0.018–0.030 m/s and take 14–22 s to settle their reference; slip remains | Retain as a diagnostic; do not adopt as navigation control |

The reported 0.03 rad/20 ms setting is the recent **diagnostic target limiter**, not a measured motor speed limit. Archived formal Stage2C comparisons use 0.04 rad/20 ms. Neither the 0.03 results nor the unrestricted reference may be pooled with that formal comparison. Future candidates must report their own actual limiter and requested versus admitted commands. No acceptance threshold changes here.

The independent source review also found that Benchmark 1's `ReferenceGaitEnv` replaces `_processed_actions` after the inherited target limiter and supplies velocity feedforward. Its reference derivative reaches about 7.70 rad/s at 1.3 Hz. Therefore reproducing that visually accepted waveform does not demonstrate compliance with the direct-omni limiter. The benchmark remains immutable and useful as a visual reference; it is not retrospectively relabeled as a qualified controller.

## What changes next

1. Test a reference that latches planted-foot anchors, matches swing endpoint position/velocity, suppresses new lift-offs during a stop and finishes current swings into support. Expose estimated body-motion dependency and requested/admitted twist. Reject invalid IK rather than clipping it into a claimed success.
2. Test an acceleration-bounded joint-target-velocity action path with executable position/velocity state visible to PPO, consistent resets and joint-bound anti-windup. This changes action meaning and observations, requiring a new controller/checkpoint lineage. It cannot silently resume the old absolute-position actor.
3. Compare feasible rates at 0.03 and 0.04 separately, retaining the 1.6 N·m applied cap and all motor/contact/tracking gates. Run reference-only or zero-action full-robot checks before allocating a short PPO pilot. Quiet standing and stop response remain measured behavior, not guaranteed by a smooth target.

The navigation command remains `(forward, left, yaw rate)`. Terrain-aware footholds, clearance and support can extend the reference; a fixed tripod is only a flat-ground test prior. The direct velocity-action alternative preserves freedom to learn different contact timing.

CAPS is a separate possible ablation: it regularizes temporal and nearby-state changes in the policy mapping. That differs from the standing action penalty already tested here, and it does not guarantee this robot's stability or motor compliance. [Authors' CAPS project and paper](https://ai.bu.edu/caps/). Behavior-conditioned locomotion is another relevant precedent for retaining adjustable gait choices rather than assuming one flat gait generalizes everywhere. Its quadruped results do not establish hexapod capability. [Walk These Ways paper](https://arxiv.org/abs/2212.03238).

## Evidence and reproduction

- [First prototype](first/README.md), [parameter screen](first/parameter_screen.json), [transition trace](first/transition_trace.npz).
- [Global-clock continuation](governor/README.md) and [measured report](governor/report.json).
- [Independent review](independent_review/README.md) and [per-case measurements](independent_review/cpu_review.json).
- [Frozen payload hashes](FROZEN_SHA256SUMS.json) preserve all original bytes, including historical temporary paths. The original 100 mm transition results must not be attributed to the separate 60 mm review.

![CPU reference paths and transition limitations](first/prototype.png)

The plot depicts the first, unqualified 100 mm prototype. It is not an Isaac recording. All results are point-foot CPU kinematics: they do not measure torque, dynamics, collision clearance, real support, sensors or terrain success.

The replay helper verifies frozen hashes, copies code and the two benchmark inputs into a fresh output tree, and reruns the 13 prototype/governor tests plus the independent review without touching frozen outputs:

```sh
uv run python artifacts/omni_diagnostics_2026-09-09/reference_feasibility_001/replay.py --output tmp/reference_replay_new
```

Original study scripts write beside themselves; only execute them in a new copied workspace. Do not run them in these frozen evidence folders.
