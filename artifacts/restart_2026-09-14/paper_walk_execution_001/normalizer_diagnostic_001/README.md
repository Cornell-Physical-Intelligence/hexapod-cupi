# Same-observation normalizer diagnostic

This is an offline diagnostic on exactly24 actual pre-policy observations from
evaluation012's forward case, indices0–23, times0–0.46s. No observations were
duplicated, synthesized or reweighted. These are **not** train006's missing first
3072-sample rollout or its128 current probe states. The recorded update2
normalizer-policy KL5.797131 is preserved and is not reproduced here.

The saved BCfit004 actor reproduces these evaluation012 actions. For each
experiment, observations and all network parameters stay identical; only the
observation normalizer's buffers change in disposable CPU copies. No saved model,
optimizer, statistics, source or configuration was modified.

| Same24-observation experiment |Mean KL(old‖new) |Action-mean RMS change |
|---|---:|---:|
| BC004 weights; update BC statistics with24 actual rows |0.0369895 |0.00641089 |
| BC004 weights; replace statistics with saved final200 statistics |24.9701 |0.166567 |
| Final200 weights; update final statistics with the same24 rows |1.54269e−7 |1.31288e−5 |
| Final200 weights; replace statistics with saved BC statistics |185.189 |0.454906 |

KL sums the18 action coordinates and averages over24 observations. The maximum
sample KL in the first experiment is0.108896. The endpoint-statistics swaps span
200training updates and are counterfactual sensitivity measurements, not native
policy runs or a reconstruction of update2.

The BC statistics count is3760.0001. Adding24 rows gives them0.63425% of the updated
count. The actual training code adds128 current states before rollout1, then3072
last-rollout states before rollout2: the second batch contributes44.13793% of the
new count. Its raw distribution is unavailable, so this count calculation cannot
predict its action KL. Final200's count is615216.0001, explaining why the same small
diagnostic batch has much lower weight there.

## Which inputs move

For the24-row BC update, the following groups partition the231 inputs. The last
column changes only that group's statistics; effects are nonlinear and not additive.

| Feature group |RMS normalized-input change after clamp |Isolated-group mean action KL |
|---|---:|---:|
| Four older42-feature history frames |0.1293 |0.004077 |
| Current42-feature proprioception |0.1351 |0.012074 |
| Three command values |0.0174 |0.000688 |
| Previous held target,18coordinates |0.0507 |0.006903 |

The largest individual changes are LF tibia joint-velocity features across the
five history frames. Across all history, joint-velocity inputs have0.1857RMS change;
projected gravity0.1243, gyro0.0768 and joint offsets0.0411. `RESULT.json` preserves
the leading20features with raw subset mean/std, before/after statistics, clipping
counts and action/estimator changes.

Clamping is material. Before this BC update,24 of5544 transformed elements exceed
±10:19joint-velocity history elements and5gyro elements. Afterward20exceed it.
The actual clamped model gives KL0.03699; an explicitly diagnostic unclamped
calculation gives0.15121. No unclamped model ran in physics. For actual final200
statistics this subset has no clipped values before or after its24-row update.

## Implication and limits

The code freezes statistics during each PPO ratio update, but changes them between
rollouts. The current target-KL stopping rule constrains optimizer steps and only
reports the separate normalization-induced distribution change. The results support
reviewing a future acceptance/backoff rule for statistics updates using unchanged
weights and the same probe observations. They do not justify clearing statistics or
restoring BC statistics into the final trained network: that reverse swap produced
large changes here. Any successor must preserve the existing training lineage and
declare its continuation contract; no such change is implemented by this artifact.

The CPU script verifies exact learner-source identity and strict checkpoint loading,
holds parameters fixed within every comparison, verifies recorded-input actor parity,
and checks all input-file hashes before and after. It uses one thread and performs
no native or remote operation. This diagnostic grants no Stage2 qualification.
