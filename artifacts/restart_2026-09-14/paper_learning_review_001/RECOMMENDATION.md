# Learning adjustment review, 14 September 2026

This is an engineering recommendation for root review, not an adopted experiment,
a task queue, or Stage 2 acceptance. No implementation or native run was changed.
The current 200-update source_004 pilot remains its own immutable attempt.

## Actual consultation

Claude Code completed a substantive one-turn review using `claude-fable-5-1`
with `--effort max` and all tools disabled, session
`0bdf4b41-f075-47a3-b331-976c5d708446`. `REQUEST.json` preserves the invocation,
prompt hash and source hashes; `INPUT.txt` contains the readable complete prompt;
`RESPONSE.json` is the raw response; `REVIEW.md` is its verbatim result.
`RECEIPT.json` verifies model/session and immutable source inputs. The CLI also
reports a 16-token auxiliary Haiku output; the substantive review is attributed
to Fable, with no fallback model requested. The supplied evidence was the
update-37 interim snapshot, not a completed pilot or deterministic evaluation.

Fable favors softening both quiet penalties, with BC kept off, if actual reward
decomposition confirms their dominance and the completed pilot does not track
commands. I agree with that isolated experiment, with the corrections below.

## Preferred minimal successor

First retain the completed pilot and evaluate its deterministic policy. If it
still lacks command tracking, confirm the two quiet components explain the
large negative reward using time-weighted command-conditioned measurements.
Large aggregate reward or value loss alone cannot prove the mechanism.

For that condition, change only the tails of the two quiet costs. Preserve the
current weights and scales and define, separately for each cost:

```text
u_rate = mean_j((joint_rate_j / 0.03)^2)
u_step = mean_j((target_delta_j / 0.002)^2)
phi(u) = u                  when 0 <= u <= 1
         1 + ln(u)          when u > 1
quiet_rate_cost = -0.05 * quiet * phi(u_rate)
quiet_step_cost = -0.02 * quiet * phi(u_step)
```

This preserves the existing quadratic and its slope through the configured
scales, remains strictly increasing without a flat clipping plateau, and
preserves the original mean-square ranking across joint distributions. It is
continuously differentiable at 1, although its second derivative changes there.
For uniform 1 rad/s joint rates, the quiet cost falls from 55.56 to 0.401 per
sample; for uniform 0.03 rad target motion it falls from 4.5 to 0.128. These are
analytic scale comparisons, not measured performance.

Keep BC off, the command sampler, reward weights, PPO/AMP architecture and
hyperparameters unchanged for this first isolated successor. Use a new source,
task definition and output identity. Prefer fresh initialization to importing
old-scale critic/Adam state. Reporting corrections can accompany it if recorded
as such; they must not silently bypass the existing learner-byte checkpoint
identity check. A changed learner cannot claim exact resume from source_004.

## Independent corrections to the raw Fable review

- **Critic clipping is not a strict 0.2-per-update prediction cap.** The clipped
  value objective can give an individual sample a flat gradient after improvement
  beyond the clip range; other samples, shared network parameters, repeated
  minibatches and Adam momentum can still move that prediction. Fable's claimed
  hard bound of roughly 7 units after 37 updates is unsupported. Inspect actual
  old/new values, return scale, explained variance and the clipped-branch share.
  Do not combine a value-clip change with the first reward-tail experiment.
- **Global clipping does not establish actor starvation.** Actor and critic
  networks have disjoint parameters, but share the clipping coefficient.
  Adam approximately cancels a constant gradient rescaling; time-varying scaling
  can matter. The reported all-parameter norm and value loss cannot settle this.
  Log true actor/critic norms, clipping coefficient, KL, PPO clip fraction and
  parameter-update sizes before diagnosing suppression.
- **Plain log1p is not unchanged at the stated scale.** At normalized square 1,
  it retains only `ln(2) = 0.693` of the old cost. The proposed piecewise tail
  preserves the existing near-scale objective exactly.
- **Per-joint log followed by mean changes concentration incentives.** Equal
  mean-square vectors `[1,1]` and `[0,2]` cost 0.693 and 0.549 respectively under
  mean-log1p. Applying the monotone tail after the existing mean avoids adding
  that preference. `ANALYTIC_CHECK.json` records the arithmetic.
- **Quiet dominance remains a hypothesis.** Other reward components include
  unbounded height, angular-rate, vertical-rate and contact terms. Their being
  small in a plausible regime is not proof they were small in this run. If
  measured contact or another term dominates, diagnose that evidence instead.

## Evidence to judge the successor

Record reward components by zero, each speed class, yaw and arc commands;
return distributions and critic explained variance; correctly named PPO and
discriminator losses; signed speed conditioned on moving commands; quiet joint
RMS and worst-joint rates; saturation, contacts and falls. The current projection
metric divides by all rows, so it dilutes moving-command performance. Aggregate
unsigned speed is not tracking evidence.

Use matched deterministic held-out commands and the existing evaluation gates
to compare behavior. A larger shaped reward alone does not establish success.
If the softened-cost policy settles into standing, inspect tracking signal and
closed-loop prior quality before a separate BC warm-start experiment. BC must
use validated real observation/action history, undergo native closed-loop
evaluation, and record initialization and optimizer provenance explicitly.

The exact corrected URDF, native timing, limiter, torque/contact model and every
original Stage 2 numerical gate remain unchanged. Only full required native
qualification and the actual trained-policy Isaac video can fulfill the goal.
