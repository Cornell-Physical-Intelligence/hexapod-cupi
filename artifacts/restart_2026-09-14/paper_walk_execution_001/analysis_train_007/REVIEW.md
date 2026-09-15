# Train007: optimizer repair is effective; forward learning remains unresolved

This analysis compares train006 updates 181–200 with train007 updates 201–220,
61,440 environment transitions each. It verifies 18 exact local inputs before
and after reading them, checks every recorded backtracking trial, and restores
the frozen source018 actor and discriminator for CPU inference on all 3,050 real
evaluation013 recorded observations. `REPORT.json` contains complete aggregates,
named-joint target-slew occupancy, command-class denominators and input hashes.

The source018 repair works on its stated optimization statistic: 253 model
steps were accepted, 379 trial steps rejected, and the largest accepted mean
whole-actor old||new KL was 0.0199954315 against 0.02. There were 263 proposals
and 632 trials; 10 proposals exhausted the allowed retries. All 400 scheduled
discriminator batches ran. No update accepted zero model steps. Actor observation
normalizer state remained bitwise unchanged, and every recorded statistics-only
policy KL was zero. The checkpoint retains the base model learning rate 0.0001.

| Measured quantity | Train006 last 20 | Train007 new 20 |
|---|---:|---:|
| Mean final actor KL per update | 0.163267 | 0.0185883 |
| Retained model steps | 20 | 253 |
| Moving-only signed command projection, m/s | 0.000175196 | 0.00124235 |
| Sampled raw-action clipping | 52.2070% | 45.1109% |
| Requested motor saturation fraction | 0.140765% | 1.75761% |
| Joint-limit terminations | 0 | 14 |
| Height / tilt terminations | 0 / 0 | 0 / 0 |
| Nonfoot event rows | 0 | 0 |
| Mean joint target-slew occupancy | 16.4623% | 21.3625% |
| Value loss, average update metric | 2.87326 | 8.44954 |
| Explained variance, average update metric | 0.137364 | -0.130162 |
| Style reward | 0.0304484 | 0.0389743 |
| Total reward | 0.482499 | 0.458287 |

The within-allocation trend is more cautionary than the overall comparison.
Between train007's first and last ten updates, joint-limit terminations fell
13→1, requested saturation fell 2.3855%→1.1297%, value loss fell 14.9558→1.94328,
and explained variance rose -0.442537→0.182212. However, moving projection fell
0.00223637→0.000318371 m/s while raw clipping rose 36.1361%→54.0856% and total
reward rose 0.423724→0.492850. This is evidence of settling optimization and
reduced physical events, not evidence of increasing forward speed. Even the
overall moving-only projection remains far below requested 0.025/0.05 m/s
linear commands. Class-specific projection is in the machine-readable report.

The accepted learning-rate counts were 30 at 0.0001, 122 at 0.00005, 86 at
0.000025 and 15 at 0.0000125. Trial counts including rejection were respectively
263, 233, 111 and 25. All rates, retry indices, accepted/rejected KL predicates,
model-step counts and full discriminator schedules were independently checked.

On the identical real evaluation013 forward observations, checkpoint220's raw
deterministic actor mean is outside [-1,1] on 60.1889% of joint values, versus
54.7667% for checkpoint200. Its estimator RMSE against recorded body-origin
velocity improves 0.00846033→0.00766255 m/s. Quiet and stop recorded states show
the same estimator improvement and slightly greater raw mean clipping. The
accumulated old200||new220 mean KL is 1.04–1.24 on these three recorded sets.
This off-rollout, twenty-update comparison does not violate the per-update
fixed-rollout safeguard and does not predict checkpoint220's closed-loop motion.
Style scores also change because the discriminator was trained; they are not
an independent gait-quality measurement.

**Decision scope:** these training records alone do not justify an unconditional
additional 200 updates or a walking-success claim. They do justify treating the
KL repair as functioning and considering a bounded unchanged-source continuation
if the matched native checkpoint220 probes show healthy behavior and no
regression. Root owns evaluation014, its native acceptance flags and the next
allocation decision; that evaluation is outside this artifact's input set.
Persistent near-zero forward tracking, increasing actor clipping and eventual
joint-limit events remain concrete quantities to assess after a continuation.
No architecture, physics, reward or acceptance-gate change is proposed here.

Train007 begins with fresh physical resets and command histories; train006's
last twenty updates are mid-run. Their distributions are not causal replicas.
Training contact summaries are compact endpoint/control maxima, not the full
Stage2 raw audit. Training zero-command joint-rate statistics include reset and
exploration transients and cannot replace a deterministic quiet-hold gate.
Slew occupancy correctly excludes first/reset samples. Raw-action clipping,
PPO ratio clipping, motor saturation and target-slew occupancy are separate
quantities. The correct allocation throughput is 61,440 / 67.36798 = 912.006
new transitions/s; the legacy cumulative `transitions_per_s` denominator makes
that older field misleading after resume. Stage2 remains unqualified by this
analysis.
