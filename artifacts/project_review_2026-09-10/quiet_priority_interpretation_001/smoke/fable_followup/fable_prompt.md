Brief verification follow-up for the same actual native004 quiet-priority smoke. Read-only, no tools, answer at most 300 words with a final correction/disposition only, no internal thinking stream. Root has now separately dispatched the 50-update pilot; this review must not infer convergence or change gates.

Independent exact arithmetic: weighted quiet/PPO gradient norms 0.02300572033 and 0.05124415641. Quiet is 943.763/967.046 times spatial and 190.237/168.345 times moving. These are pre-Adam norm ratios, not actual Adam update shares. Common actor norm clipping preserves ratios; actor and critic clip separately. Let Q,P,M,S denote the recorded weighted gradient components. Q dot(P+Q) is 0.2401656423 and -0.1432637086. Cauchy-Schwarz bounds abs Q dot(M+S) by 0.0027174486 and 0.0031698118. Therefore Q dot total is positive [0.2374482,0.2428831] first and negative [-0.1464335,-0.1400939] last. This determines local plain-SGD quiet direction only; actual Adam preconditioning/moments are not decomposed.

Verified raw initialization below proves original optimizer moments were NOT loaded: optimizer reset, load_cfg optimizer false, zero entries. Internal iteration1847 is retained metadata only. Your prior suspicion of inherited Adam is refuted; do not infer that the first KL spike was or was not caused by quiet regularization from norm ratios alone. Similarly, neither PPO clipping nor noise is proven to cause the observed KL/loss evolution.

All40 LR transitions exactly follow source. 35/40 KL values exceed0.02 (35/38 nonzero). Floor-after rows are31 by exact equality or32 within1e-12 relative roundoff; among38 nonzero KL rows they are30 exact/31 with roundoff, not35. Your shared-term inversion is useful and verified: mean quiet contribution0.3022927417→0.2669426510, moving0.0007876942→0.0070314654, spatial0.0002248339→0.0001650793. It recovers aggregate component losses, not individual minibatch conditional losses. Don't claim the entire loss change is exclusively sample mix or the spatial loss trend is proven noise. Conditional quiet loss rises10.53%, weighted aggregate falls9.62%, quiet share97.70%→78.13%.

Please confirm or correct this bounded interpretation and state the most useful existing pilot10/25/50 evidence. No weight/budget recommendation before it arrives.

Exact initialization:
{
  "checkpoint_sha256": "1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8",
  "exploration_std": 0.1,
  "entropy_coef": 0.0,
  "optimizer": "reset",
  "learning_rate": 5e-05,
  "actor_and_critic_preserved_except_std": true,
  "observation_normalizers_preserved": true,
  "optimizer_state_entries": 0,
  "checkpoint_iteration": 1847,
  "runner_iteration": 1847,
  "exploration_std_before": [
    0.4032049775123596,
    0.38398510217666626,
    0.2871536314487457,
    0.45611193776130676,
    0.2955004572868347,
    0.23039953410625458,
    0.4657600224018097,
    0.44896653294563293,
    0.4871585965156555,
    0.39336958527565,
    0.5246296525001526,
    0.49469467997550964,
    0.4852926731109619,
    0.47244736552238464,
    0.4494502544403076,
    0.4971517026424408,
    0.45584002137184143,
    0.4284435212612152
  ],
  "exploration_std_after": [
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612,
    0.10000000149011612
  ],
  "load_cfg": {
    "actor": true,
    "critic": true,
    "optimizer": false,
    "iteration": true,
    "rnd": false
  },
  "learning_rate_schedule": "adaptive",
  "stage2_complete": false
}


Your prior final response:
## Verdict

The mode-separated losses and the two gradient rows show the quiet term is live and dominates the other two regularizers, but they do not show meaningful authority over PPO. Its pre-clip norm is a few percent of the PPO actor gradient, its cosine with PPO is near zero and slightly negative, and the global clip rescales every component uniformly, so its share of the applied vector is the same few percent. The quiet term's near-total share of the weighted regularizer loss comes from its coefficient and from quiet pairs being most of the valid pairs under the shared denominator, not from gradient leverage. Nothing here supports convergence, gate changes, or a proven weight change. The 0/48 endpoint and the audit's requested-torque rows describe the inherited policy after two small updates.

The telemetry is internally consistent. Valid-pair fractions, the weighted regularizer loss, the entropy of 18 Gaussians at std 0.10, and the component-sum norms all reconstruct from the raw rows. The component sum equals the assigned pre-clip actor norm to every printed digit in both rows, which confirms the decomposition covers every actor-affecting term and the value loss does not leak into the actor.

## Calculations

Gradient rows, both from update 1:

| minibatch | PPO | quiet | moving | spatial | quiet/PPO | cosine | clip scale | quiet share of applied vector |
|---|---|---|---|---|---|---|---|---|
| 1 | 28.51 | 0.656 | 0.0034 | 0.00070 | 0.023 | -0.010 | 0.035 | 0.023 |
| 20 | 13.16 | 0.674 | 0.0040 | 0.00070 | 0.051 | -0.067 | 0.076 | 0.051 |

Quiet is 168 to 190 times the moving gradient and 944 to 967 times the spatial gradient. PPO's norm halved between the rows while the quiet norm held, which is expected once ratios leave the clip band. The quiet gradient alone would pass the clip unclipped; its absolute step is set by PPO's norm.

Sample mix and mode losses:

| | update 1 | update 2 |
|---|---|---|
| valid pairs | 565 of 768 | 736 of 768 |
| quiet / moving pairs | 552 / 13 | 575 / 161 |
| quiet share of valid | 0.977 | 0.781 |
| effective quiet coefficient, 1.0 x share | 0.977 | 0.781 |
| effective moving coefficient, 0.1 x share | 0.0023 | 0.0219 |
| quiet per-pair mean | 0.309 | 0.342 |
| moving per-pair mean | 0.352 | 0.322 |
| spatial mean | 0.00225 | 0.00165 |
| share of caps_weighted, quiet / moving / spatial | 99.7 / 0.26 / 0.07 % | 97.4 / 2.6 / 0.06 % |

```
valid   u1: 138+144+131+152 = 565 ; 565/768 = 0.73568   (reported 0.73568)
        u2: 187+183+183+183 = 736 ; 736/768 = 0.95833   (reported 0.95833)
quiet_shared = (caps_weighted - 0.1*caps_temporal - 0.1*caps_spatial) / 0.9
        u1: (0.303305 - 0.031017 - 0.000225)/0.9 = 0.30229
        u2: (0.274139 - 0.033726 - 0.000165)/0.9 = 0.26694
check   quiet_mean * share: 0.3091*0.977 = 0.3020 ; 0.3416*0.781 = 0.2669
sum     u1 mb1: sqrt(28.511^2 + 0.656^2 + 2*28.511*0.656*(-0.0102)) = 28.512
entropy 18 * 0.5*ln(2*pi*e*0.01) = -15.91   (reported -15.90, std 0.0996..0.1003)
chatter rms per action dim per 20 ms step, raw policy-mean units:
        quiet sqrt(0.309) = 0.56 ; sqrt(0.342) = 0.58 ; spatial sqrt(0.00225) = 0.047
invalid u1: 768-565 = 203 = 32 wrapper-init rows + 171 other
        u2: 768-736 = 32 = 1 truncation + 31 other   (terminations: 0)
```

The weighted loss fell about ten percent between updates while the quiet per-pair mean rose about ten percent. The fall is entirely the quiet share dropping under the shared denominator, so the headline loss tracks sample mix, not smoothness. Under the stated pilot schedule the steady-state quiet share is roughly 0.625, so update 1 overstates the quiet term's effective coefficient by about 1.6 times and understates the moving term's. Chatter under zero command equals chatter under motion commands in raw mean units, about 5.6 times the learned std. These are raw policy-mean differences, not executed target steps under the slew cap and not joint speeds. The update-1 moving mean rests on 13 pairs reused across five epochs.

KL and learning-rate rows. All 40 transitions follow the rule: divide by 1.5 above 0.02, multiply by 1.5 between zero and 0.005, floor 1e-5. Update 1 reaches the floor at minibatch 5 and all 19 nonzero KLs exceed twice the target, peaking at minibatch 3 and decaying to the end. Update 2 climbs to 2.25e-5, hits the floor at minibatch 6, and holds KL near three to four times the target from minibatch 7 onward.

```
mean shift from KL, equal diagonal Gaussians, sigma 0.10:  sum_d dmu^2 = 0.02*KL
u1 mb2, one step at 5e-5    KL 0.134    rms/dim = sqrt(0.02*0.134/18) = 0.012
u1 mb3 peak                 KL 0.176    0.014
u1 mb20                     KL 0.052    0.0076
u2 mb2, one step at 1e-5    KL 0.00076  0.0009
u2 mb20                     KL 0.039    0.0066
one-step KL ratio u1/u2 = 176 ; lr^2 ratio = 25 ; excess about 7x
```

- **The update-1 spike is not the quiet term.** One Adam step at 5e-5 moved the policy 13 times the KL target while the quiet component was 2.3 percent of the pre-clip norm and near-orthogonal. Attribution would require it to concentrate in coordinates with negligible Adam second moments, which is unrecorded.
- **The within-update KL decay is PPO dynamics, not schedule control.** KL fell from 0.176 to 0.052 while the rate sat at its floor, which is the clipped surrogate pulling overshot samples back. The net update-1 change is dominated by the first two steps.
- **Realized drift is 40 to 85 times smaller than the chatter it targets.** This is context for why two updates are uninformative, not a bound, since a sensitivity change could reduce chatter without a large mean shift.

## Limits and implementation concerns

What the telemetry cannot establish:

- **Parameter-space authority after Adam.** The norm share is a global proxy; per-coordinate normalization can amplify the quiet direction where PPO is small. A same-batch comparison against the zero-coefficient branch, or a projection of the realized parameter delta onto the quiet gradient, would establish it.
- **Any effect on executed motion or the 10-second quiet result.** Raw mean chatter is not a target step under the 0.04 rad per 20 ms slew and not joint speed.
- **Direction of change.** Quiet rose and moving fell on different rollouts, with rows in unknown schedule phases; the spatial loss fell 27 percent with a gradient near zero, which shows loss trends here are noise.
- **What drives the chatter.** Observation noise at scale 1.0, exploration noise entering through the action-history slots if the stored action is the sampled one, and genuine state change are inseparable. Deterministic evaluation lacks the exploration component.
- **Transfer to pilot batch size.** PPO's norm at 192-sample minibatches may be variance-dominated; at 6,144-sample minibatches the quiet share could rise or fall.

Concrete concerns:

- **Gradient coverage misses update 2.** The sparse schedule yields two rows in a two-update smoke, both from the update with a 2.3 percent moving share and the anomalous first step. No decomposition exists for the 22 percent moving mix. An unchanged pilot adds six rows at updates 10, 25 and 50 but still records neither Adam-normalized authority nor executed steps.
- **The quiet coefficient sits at the validated ceiling.** validated_options bounds every coefficient and noise_scale to at most 1.0, so the quiet term's pre-clip norm cannot rise by configuration under the current gate, and the shared denominator lowers its effective coefficient toward about 0.63 under the pilot schedule. This is structure, not a prescription.
- **Inherited optimizer state.** The one-step KL excess of about seven times over quadratic rate scaling, the 12.6-fold value-loss drop between updates, and the initial runner iteration of 1847 point to loaded Adam moments and a critic fitted to the parent's reward scale. Verify whether checkpoint 1971's optimizer state is loaded, because the pilot starts from the same checkpoint and would repeat update 1.
- **Adaptive schedule saturated.** The rate is pinned at 1e-5 for 35 of 38 nonzero rows while KL stays 2 to 18 times the target, so the configured trust-region proxy exerts no further restraint.
- **Quiet detection is exact-zero float equality on the latest command slot.** Correct classification requires the generator to write exact zeros in stop phases and hold commands between changes. The 171 and 31 unexplained invalid pairs show command-slot changes, and the generator is not in the supplied source. For four steps after a stop the history frames still hold the motion command while the pair counts as quiet at full weight, penalizing the transition response, about one percent of steps per phase.
- **Spatial term inert as configured.** Its residual is 138 to 204 times smaller than the temporal residual and its gradient is 0.002 percent of PPO. The noise table is hard-coded in caps.py and noise_scale is capped at 1.0, so it cannot become material by configuration.