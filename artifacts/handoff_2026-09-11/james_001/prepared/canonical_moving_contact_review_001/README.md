# Moving contact boundary: independent review, not runtime adoption

The smallest next step is an explicitly labelled **moving bridge probe**, conditional on authentic same-source standing1/32 and the unchanged zero-command PPO smoke. It may end on its first finite failure. Do not turn that diagnostic policy into the architecture for longer moving PPO: ordinary exploration failures need selected-environment termination and tested recovery, while data-integrity failures still abort the run.

I recommend an explicit provisional minimum of **three accepted toe contacts** during every moving/stopping sample for that first probe. It permits tripod and sequential swing and avoids silently retaining six-contact standing semantics. It is a conservative experiment restriction, not an existing Stage 2 gate, a stability proof, or a demonstrated optimum. Its zero margin for tripod force dropouts is a known limitation. Neither the two-contact alternative recommended by Fable nor a one-contact floor has been validated on this model. Root must explicitly select and version the experiment rule before native adoption.

## What the actual contracts say

`docs/TRAINING.md` §6 specifies the historical Stage2C formal screen, target limiter, tracking/deck metrics, zero falls/timeouts and motor criteria. `packages/hexapod_eval/hexapod_eval/gates.py` mirrors those graders; it has no moving support-count gate. Stage2G's historical three-support reward bias is not authority for a new hard termination. The full-direction/quiet steady-deck mission is broader than that forward-only historical screen. New canonical model evaluation still needs its own explicit model-bound composition; this review does not admit it through the old task.

Source005 standing support is scored over its original post-settle window. Its current PPO003 bridge additionally requires six toes at the already admitted actor boundary and throughout the zero-command smoke. Neither is silently changed. In particular, `neutral_actor_boundary` in this CPU prototype is not a replacement for original initial settling or the original scorer.

## Precise proposed moving probe boundary

1. Preserve the source-bound distal classifier, resultant force `>1 N`, raw patch identities, exact inactive zero tuples and every nonfoot, clearance, joint, effort, finite-state and counter check. Do not normalize invalid raw normals or count inactive slots as support. Require all eight copied numeric rows and their matching accepted raw-patch packets.
2. Declare the experiment phase explicitly. `moving_or_stopping` uses the selected floor even when requested twist is exactly zero; a returning foot does not become an invalid swing because a stop command arrived. Formal quiet remains its separately declared full-window evaluation with unchanged thresholds. Requested command `c[t]` belongs to the whole current hold and its reward; advance to `c[t+1]` only after valid completion.
3. For each of eight samples, compute the six-element accepted contact mask and require count ≥3. Preserve every count, mask and force off/on event. The first violation is ordered by sample then environment, with its exact physical counter. Enforce the result before another action or successful history/command/storage commit. The existing `contact_facts` is intentionally a facts helper: a returned false result is not an exception. Ignoring that result during integration would make the proposed rule advisory.
4. Existing `step_control` returns after eight substeps. The added moving count check therefore detects a violation after an attempted 20 ms hold; it must not claim to have stopped physics at the first 2.5 ms bad sample. Earlier evidence remains intact, and existing native hard checks retain their earlier per-substep behavior. A truly immediate count callback would be another explicit source seam.
5. A force departure/return is not qualified geometric lift/landing. Record it without inventing airtime, separation or landing thresholds. Use actual distal geometry/time/force/slip evidence to qualify later motion claims. No separately prequalified all-leg gait is needed before this bounded diagnostic. Fable's suggested scripted lift tests are possible diagnostics, not prerequisites or proof that a reported dropout is harmless.

The CPU prototype is runnable, but it is not a native bridge or authorization to launch. Its seven contact tests cover a valid tripod and stopping return, endpoint masking of an earlier failure, invalid packets/counters, force-only event limitations, a count-three state with COM outside its support hull, a single-sample tripod dropout, and the ignored-false integration hazard. No geometry, thresholds or frozen source were edited.

## Genuine training requires a second, explicit contract

| Event | Moving probe | Later recoverable PPO | Formal evaluation |
|---|---|---|---|
| Valid finite count shortfall, fall or other selected physical episode criterion | Retain attempted prefix and stop the probe | Keep the failed transition, mark only that environment truly terminated, penalty exactly once, bootstrap zero, recover that row | Record the original failure; no automatic reset can turn it into a pass |
| Artificial episode time limit | Preserve explicit truncation | Use the same episode's final pre-reset critic observation, bootstrap once, end the GAE chain | Original timeout criteria remain unchanged |
| Missing rows, bad identities/counters, invalid/nonfinite tensors, overflow, inconsistent actuator enforcement or unrecoverable reset | Abort; retain partial evidence | Abort unoptimized collection; preserve the prior checkpoint and exact event ledger | Reject |

Source005 presently raises for any row's height/joint-speed/joint-limit/nonfoot/clearance termination inside the native step. Changing only RSL `dones` cannot implement the middle column. A new session must capture the first event and pre-reset observations, isolate selected reset IDs/origins/targets and contact/history clocks, and preserve other rows. Recovery controls must be excluded from PPO storage, advantages, minibatches and normalization until their new-episode actor packet is valid. Reinitialize all five history frames and previous held/action/command state explicitly. Count actual valid transitions and recovery time separately; never present configured capacity as realized learning data. Validate recovery under the canonical model rather than importing the historical mock reset controller.

The exact source-bound RSL 5.0.1 `process_env_step` updates normalizers with every supplied next observation, and its `time_outs` correction uses the value stored before the action. The two CPU tests execute that unchanged method body from its exact SHA-bound source using CPU tensors and transparent model/storage doubles. With reward1, gamma.99, pre-action value2 and final value7, its timeout reward is2.98; the project final-observation convention needs7.93. This is an integration incompatibility for the proposed recovery contract, not a claim that stock RSL is generally broken. No optimizer, simulator or CUDA call runs in these tests.

The authors' `legged_gym` code computes termination/reward, resets selected environment IDs, and resets their command and episode histories. It supplies a useful recipe precedent for learning through finite episodes, not a drop-in reset API for native Isaac 6.0.1 or this hexapod. [Official code](https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot.py). Rudin, Hoeller, Reist and Hutter (2022) study massively parallel locomotion and curriculum training on ANYmal; its reported runtime is not an ETA for our geometry and recorder. [Primary paper](https://proceedings.mlr.press/v164/rudin22a.html).

## Exploration and next decision

PPO003 starts with a zero final actor layer, Gaussian standard deviation `.02`, and target scale `.1 rad`: nominal initial target standard deviation is `.002 rad` before clipping, limits and the slew governor. Nonzero commands alone do not make that random policy command-responsive or prove enough exploration to lift a leg. Root should freeze an explicit moving exploration width, then measure actor/requested/emitted variation, clipping, effort duty and real motion during the bounded probe. Do not assume the smoke width is adequate, silently enlarge it, or promise convergence from a fixed update count.

Keep a short moving packing/probe allocation distinct from a later continuous learning pilot. The proposed 32×24×50 whole-run-abort draft is a possible diagnostic budget, not evidence of a scalable reset architecture. The fastest useful next implementation is the native eight-packet moving boundary plus the selected-row terminal/recovery collector; the latter requires meaningful isolation, timeout and normalization tests and actual bounded canonical recovery evidence before longer training. No blind long allocation or required gait-prequalification sweep is recommended.

Reproduce locally:

```sh
.venv/bin/python3 -B -m unittest discover -s tmp/canonical_moving_contact_review_001 -p 'test_*.py' -v
python3 -B -S tmp/canonical_moving_contact_review_001/verify.py
```

The final freeze and `REPORT.json` bind the exact inspected inputs. `FABLE_DISPOSITION.md` distinguishes the actual MAX response from our corrections. Root owns any later publication and the required STATUS/plan/NEXT_RUNS/central site update under `docs/PROJECT_SITE.md`; this review changes none of those living files and grants no native or training admission.
