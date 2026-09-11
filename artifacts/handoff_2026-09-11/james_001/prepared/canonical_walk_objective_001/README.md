# Canonical all-direction objective: bounded CPU proposal

This package implements a command curriculum, reward calculation and causal input checks for the **new 7.466088235kg detailed direct-drive model**. It is ready for CPU review. It has not been wired into the native learner, run on the Spark, or admitted as walking. The current approved405/408 learner remains its separate, frozen, zero-command two-update integration smoke.

The next useful integration is a single declared moving allocation after matching native 1/32 standing, the405/408 smoke, and a real foot lift/return contact diagnostic pass. A proposed first discriminator is32 replicas×24 controls×50 updates at level0 with no automatic physical resets. That is a proposal for root review, not an approved allocation. This avoids expanding reset machinery before obtaining moving evidence. It cannot run through the current smoke bridge unchanged: that bridge deliberately requires six planted feet and zero commands.

## Concrete interfaces and timing

- `commands.py`: `CommandBank(n,level,seed).observe()/advance()/reset_rows()/state()/from_state()`. Each row has its own12s command-program epoch. Reaching control600 requires explicit program restart. Restarting a program never resets physical state or policy history. Selected rows leave every other row's clock/RNG sequence unchanged.
- `causal_command.py`: `begin_control(c_t,frame81,u_previous,q_t,index)` checks the newest frame contains the newly requested command and the **previous actual held target minus current q**. It returns a private command for reward attribution. It neither shifts history nor invokes the actor. Reward for hold t uses c_t, not a command sampled after that hold.
- `objective.py`: `reward(...)` consumes all 8 native substeps for body motion, gravity, torque and contact costs, plus actual target increments and actual20ms joint-angle rates. All motion is expressed in forward/left/up at the **body-root origin**, after explicit world-to-native-body rotation and native(-y,x,z) conversion. It never substitutes COM velocity or actor-selected command scaling. Contact and linear velocity remain privileged reward/critic data; no actor feature is added.
- Material slip is a finite-interval estimate of the **same rigid material point** between adjacent link poses, projected into the current contact tangent plane. Patch-location changes on a stationary link do not count as slip. The native classifier first applies its existing raw-normal acceptance tolerance1e-3; the reward packet then normalizes accepted normals. The unit-vector helper’s1e-6 check is an internal mathematical precondition, not a tighter physical gate. Rejected raw normals must not be made acceptable by normalizing them first. Rigid pose matrices likewise come from normalized accepted native XYZW quaternions. A causal native patch accessor and pre/post400Hz pose pairing remain integration work. Source003's complete raw contact evidence remains authoritative.

The405/408 dimensions, named18-joint order, SI-unit inputs, empirical normalizers,0.10rad target scale,50Hz hold,400Hz servo and actual0.040rad/20ms target limiter are unchanged proposals from PPO001. No gait clock, action filter, standing freeze, fake velocity, old actor or old task is introduced. Requested-target excess beyond the executed target is explicitly penalized so the limiter is not a free hidden action filter.

## Commands and held-out comparison

| Proposed level | Translation request norm | Absolute yaw request |
|---|---:|---:|
|0|0.02–0.04m/s|0.08–0.16rad/s|
|1|0.04–0.08m/s|0.16–0.32rad/s|
|2|0.08–0.16m/s|0.32–0.50rad/s|

These are **request bands, not physically admitted speeds**. Level selection is explicit; no promotion is implemented. Twenty-five equal-duration strata comprise1quiet,8cardinal,6diagonal,4pure-yaw and6arc programs. Each moving program has 500moving and 100zero controls; the quiet program has 600zero controls. Thus balanced strata give exactly 20%zero time. A finite32-row first epoch is not exactly balanced (see `MATH_REPORT.json`);25epochs per row visit every stratum. Yaw sign is independently sampled for arcs and does not encode a fixed heading. Even epochs include stop-separated reversals, odd epochs include direct reversals, then stopping. Full negation reverses both translation and yaw in an arc.

The proposed held-out diagnostic is32 cases:8 translation headings,2 pure turns,16 offset-bearing arcs crossing both yaw signs, and6 quiet rows. Each is22 s: 2 sinitial zero,8 s requested motion,12 s stop, with the last 10 s scored by the existing quiet gates. Pure yaw requires yaw tracking and bounded translation drift, never nonzero translation. Training reversals have CPU coverage; a separate actual reversal evaluation sequence remains to be declared before claiming reversal quality. Evaluation seeds/magnitudes and complete reset-free raw traces must be bound by a future runtime. These case definitions do not create a replacement admission scorer.

## Reward proposal and safety separation

The reward is dt×(tracking benefits−bounded costs), with dt=0.02s. Planar and yaw tracking weights are5 and 2; tracking uses **mean squared error across all 8substeps before the exponential**, preventing opposite substep rates from canceling. Bandwidths are max(0.02m/s,half the requested translation norm) and max(0.05rad/s,half absolute yaw request).

`CONTRACT.json` records every weight and normalization. Costs cover tilt (including upside-down orientation), roll/pitch and vertical motion, applied effort, requested effort above1.6Nm, actual target first/second differences, requested-versus-executed target discrepancy, and material slip. At exactly zero command, actual angle rate and actual target movement also cost reward. There is no q-neutral pose reward or standing clamp. All costs have smooth bounded form x²/(1+x²); individual coefficients total 0.65persecond. This bounds worst negative discounted continuation at−1.3for gamma0.99. A true-terminal penalty 20 exceeds the conservative 1.44bound including the largest immediate positive credit. This algebra assumes this exact reward and discount; it does not prove a physical policy optimum or correct native reset handling. Timeouts are eligible to bootstrap; true terminals are not. Actual RSL timeout storage and mixed physical reset behavior still need separate tests before adoption.

Safety and quality do not come from reward. Preserve1.6Nm applied cap, 400 Hz requested-saturation limit 0.5%,actual target limit0.040rad/20ms,native joint/rate bounds,no prohibited contact/no reset and existing quiet thresholds. `INPUTS.json` pins source003's unchanged score/servo/quiet files. Six-toe standing remains an unchanged **standing** requirement; it cannot be silently relaxed inside that scorer to permit walking. A new moving-support contract must preserve per-foot observed force/point validity and qualify real separation/return. Merely allowing an empty contact set or reporting a support polygon is insufficient. No minimum moving-support count or gait schedule is adopted by this proposal.

Reward slip masking retains the admitted caller's1Ncontact threshold; Fable's suggested1.2Nheuristic is not adopted. Inactive feet contribute zero slip cost but earn no support admission. This can favor unloading absent a correct moving safety contract, another reason the reward cannot launch alone. Angle-derived slip and quiet diagnostics remain distinct from rawSDKrates; neither replaces the original gates.

## Review, evidence and next code changes

Actual Claude Fable5.1 with **MAX** reasoning supplied an independent critique. `fable/` contains exact prompt/input hashes, launch metadata, exit status and final response only. `FABLE_DISPOSITION.md` records accepted ideas and independently rejected mathematical/causal suggestions. No internal thinking text or event streams are included. The model is a reviewer, not an authority or native-data source.

The 25 CPUtests cover causal command/target timing, all signs and frame covariance,400Hz cancellation/torque pulses, actual versus raw state, upside-down tilt, terminal algebra, material-point slip,20%balanced command time, selective command restart and checkpoint continuation. Run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s /absolute/path/to/this/bundle -p 'test_*.py' -v
python3 -B /absolute/path/to/this/bundle/analyze_proposal.py --output /tmp/canonical_math_replay.json
python3 -S /absolute/path/to/this/bundle/verify_bundle.py
```

Priority code seams:1) reviewed native contact/8sample reward packet;2) a separate moving bridge and independent moving-support/lift proof;3) command c_t installation before the actor's next frame and history save/reload;4) fixed level0 no-auto-reset allocation with complete failure prefixes;5) held-out scorer composition retaining original quiet/physical rules. Generalized physical reset,timeoutbootstrap,curriculum promotion and larger budgets follow measured need. No robot gains, meshes, dynamics or actor architecture should change while making this comparison.

Command and behavior curricula are supported design patterns in [Rudin et al.](https://proceedings.mlr.press/v164/rudin22a.html), [Margolis et al.](https://proceedings.mlr.press/v205/margolis23a.html), and [Rapid Locomotion](https://arxiv.org/abs/2205.02824). These quadruped sources motivate an explicit experiment; they do not validate hexapod speeds, torque margins or our new native asset. HistoricalC500 noise/filter/4.4Hz jitter evidence is preserved as a lesson, not a new-model diagnosis.

Root owns eventual publication and any integration. `docs/PROJECT_SITE.md` requires a new bounded site update plus relevant Markdown/status and validated build; runtime adoption also requires integrated-lineage handling. No tracked file, GPU state or frozen parent was modified by this package.
