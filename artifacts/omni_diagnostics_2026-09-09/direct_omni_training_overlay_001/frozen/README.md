# Preserved direct PPO: stand/stop curriculum and separate CAPS ablation

This is a **reviewable training overlay, not a launch bundle or training approval**. The separate cold baseline must run first. The two proposed branches independently load original checkpoint `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`; neither starts from repair003 A/B or the reference-residual controller. Actor/critic widths remain 315/318, with exactly the same five-frame, 63-value history.

## The controlled comparison

Both branches use the newly declared .04 target-slew baseline, old 16/4 solver and external-forces default false, original C asset/stance/PD30/.6, noise1, filter off, original repair-A reward weights, and the 1.6 Nm applied cap. The old source's native training terminations, episode reset and reward timing remain intact. This does not import reference009's solver, five-support reference, recovery procedure, rewards, sensor observations or .005 m/s ceiling.

| Setting | Curriculum branch | CAPS branch |
|---|---|---|
| Initialization | Exact original actor, critic and both normalizers | Same |
| Exploration / optimizer | Explicit std0.10, fresh Adam, LR5e-5, entropy0, old adaptive schedule | Same |
| Command curriculum | 25% dedicated quiet rows; other rows alternate 8s movement /8s zero targets, staggered by row | Same |
| Motion target distribution | Bearing uniform; planar speed .06–.20 m/s; pure yaw .12–.40 rad/s with both signs; 25% pure yaw, 50% pure translation, 25% combined among moving targets | Same |
| Temporal mean regularization | 0 | 0.1 |
| Spatial mean regularization | 0 | 0.1 |
| Proposed initial size | 128 environments; 256 controls per update | Same |
| Explicit modes permitted | 2-update integration smoke or 10-update bounded comparison | Same |

The .1 coefficients are **initial test values, not claimed optimal**. The independent CAPS noise generator is seeded1157, so auxiliary sampling does not consume the policy/exploration RNG stream. Baseline exploration and optimizer choices match repair003; command schedule, batch organization and the .04 profile differ and must be reported. Comparisons between the two new branches isolate the loss because every other setting matches.

A ten-update run supplies 51.2 simulated seconds per continuously running replica, not a completion guarantee. Native reset/timeout shortens individual segments. Evaluation must use complete fixed episodes and report each direction, starts/stops and quiet separately. No automatic 25-update or longer continuation exists in the overlay.

The old direct environment terminates sustained extreme demand using its historical threshold and penalizes requested saturation while applying the 1.6 Nm actuator cap. This preparation does not substitute the reference branch's per-step requested-torque termination or call the two training protocols equivalent. Existing formal evaluation saturation/contact gates remain unchanged; neither a completed optimizer run nor clipping applied torque certifies them.

## What changes, and what does not

`curriculum.make_environment` subclasses the exact old environment. Its `evaluation=True` parent flag disables only the old random-command scheduler; the subclass supplies scheduled targets. The archived source has exactly three reads of that flag: `_sample_commands`, the guard in `set_evaluation_targets`, and the resampling block in `_get_rewards`. No physical reward, action, termination or reset branch depends on it. Tests execute the actual old reward method: it scores the current `_commands`, accumulates the same reward components, then slews toward the next target before the next actor observation. Reset makes the selected command zero and resets only its schedule.

The eight-second stopping segment requests zero velocity through the original bounded command ramp. It does not freeze joints or suppress feedback. Dedicated quiet rows supply sustained zero commands. The actor sees the same current command/history fields as before; future exogenous command schedule is not a new actuator state or terrain observation.

`CapsPairWrapper` adds two auxiliary **storage-only** keys, `caps_previous_policy` and `caps_pair_valid`. Neither enters actor or critic observation groups. It pairs successive real observations from the same episode and masks a pair if reset occurs or the currently observed command changes. This protects command ramps from an artificial demand to keep exactly the same action. Repeated observation reads return cloned cached packets, and caller mutation cannot alter the next pair.

`caps_ppo.CapsPPO` derives from the exact RSL5.0.1 PPO. The copied `update` method is AST-identical except for one added loss. It rejects a different installed PPO source hash and excludes recurrent, RND, symmetry and distributed configurations from this first bounded implementation. Zero weights perform no extra actor calls or random draws and give bit-exact two-update PPO results in the real RSL regression.

The loss compares **deterministic policy means**, not sampled actions or targets after clipping/slew. Both sides of a temporal pair receive gradients. Spatial loss compares the same mean with a neighboring noisy observation. Actor Gaussian standard deviation receives no gradient from this loss. The loss uses mean-square normalized action distance; before downstream clipping/slew, joint target offsets scale those actions by0.5, so it is not a directly measured physical joint-velocity penalty.

## Spatial neighborhood and units

Neighbors perturb each of the five history frames independently using a Gaussian clipped at±3 standard deviations. They preserve command entries and previous-action entries exactly. They are a regularization neighborhood, not a claim that every perturbed history is an executable trajectory.

| Observed field | Existing observation multiplier | Neighbor sigma in observation units | Corresponding physical sigma |
|---|---:|---:|---:|
| Body angular rate | .25 | .00375 | .015 rad/s |
| Projected gravity | 1 | .01 | .01 dimensionless |
| Joint position offset | 1 | .005 | .005 rad |
| Joint rate | .05 | .0025 | .05 rad/s |
| Command / prior action | existing scales | 0 | Unchanged |

Synthetic neighbors do not call `update_normalization`. An actual RSL model regression verifies all normalization buffers remain byte-identical across the regularization call; deterministic forward calls leave the global RNG unchanged. Raw SDK joint rates retain their existing meaning and unresolved fidelity limitation. No rate substitution or power-efficiency claim is made.

## Source assembly and bounded host recipe

`make_overlay.py` accepts the exact cold-baseline successor source and writes a **fresh overlay directory** containing the modified entry and two plan JSON files. It does not change that source or launch work. Entry changes are limited to selecting the curriculum subclass for training, adding storage pairing, and selecting the declared CAPS algorithm/config. The physical configuration assignments are AST-identical. Evaluation stays the original environment and original 315/318 interface.

```sh
python3 make_overlay.py --baseline-source <exact cold source> --output <fresh overlay>
```

For each independently built source, root must overlay `caps.py`, `caps_ppo.py`, `curriculum.py`, `training_config.py`, the generated entry, its one branch plan and the RSL license; produce a new complete source map and identity. The original source already contains the unchanged strict checkpoint loader. Preserve any extra baseline readiness/runtime metadata statements when applying the entry patch; exact seam matching fails on incompatible code rather than guessing.

Proposed orchestration:

1. Complete the cold .04 baseline and root's result review.
2. Fresh source-bound standing admission for each new plan; original checkpoint and assets read-only.
3. Explicit two-update real simulator smoke for one branch first, with finite per-row outcomes, reset/history and checkpoint readback review. The CPU result below is not a substitute.
4. Root may then separately authorize ten updates per independent ablation from the original checkpoint, followed by matched cold diagnostics and uninterrupted stop/quiet evaluation. No automatic second-stage or long allocation.

The modified runner saves every update so a late failure can leave unqualified diagnostic weights. Its outer campaign must bind those files to source/plan and iteration before any subsequent use; there is no automatic resume. Existing final state/checkpoint hash reporting remains. Before physical dispatch, root must finish the guarded host, raw-event retention and evaluation coverage checks; this overlay itself is not a complete host or an all-substep training audit.

## CPU evidence

Fifteen focused tests pass. They cover actual archived reward timing, evaluation-flag scope, reset/command pair exclusion, spatial protected fields and units, normalizer stability, mean-vs-sample semantics, exact zero ablation, PPO method delta, complete command-cycle distribution, selected-row schedule reset, plan-pair isolation, unchanged physical assignments, bounded allocation and wrong-checkpoint rejection before loader mutation.

The real RSL5.0.1 regression uses the actual original315/318 checkpoint and a small synthetic stepping fixture. Three branches each execute two updates /48 controls and retain17 Adam state entries. Ordinary PPO and zero-weight CAPS end with bit-exact actor/critic tensors. Positive CAPS produces a different trained actor. Every final actor, critic and deterministic action reloads exactly. The first attempt caught RSL observation storage's bool-to-float cast; the code now requires an exact finite binary mask before conversion, with a corruption regression. The failed attempt is preserved locally, not counted as a pass.

The successful positive branch's last minibatch recorded temporal loss0.227529, spatial loss0.001940 and weighted loss0.022947. These synthetic values prove the integration is active, not that its weights are suitable for physical locomotion.

```sh
PYTHONPATH=tmp/reference_residual_ppo_001/_deps:tmp/direct_omni_recovery_001/training \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 -m unittest discover \
  -s tmp/direct_omni_recovery_001/training -p 'test_*.py'

PYTHONPATH=tmp/reference_residual_ppo_001/_deps:tmp/direct_omni_recovery_001/training \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 \
  tmp/direct_omni_recovery_001/training/real_rsl_regression.py \
  --checkpoint tmp/ppo_repair_003_preparation/original/policy.pt \
  --output <fresh CPU output directory>
```

The isolated local RSL5.0.1 wheel has source parity with inspected Spark files; `inputs/RSL_SOURCE_PARITY.json` records that provenance. These are CPU implementation checks, not a promised GPU duration, smoother walking result, physical admission or Stage2 completion. No production package, frozen baseline or GPU workload was modified.
