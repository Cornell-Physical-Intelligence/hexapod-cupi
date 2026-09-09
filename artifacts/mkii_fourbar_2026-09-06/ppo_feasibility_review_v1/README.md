# Learning feasibility review of the frozen 1600 Hz PPO campaign

Source commit `c2af43ca0f384a4c2c7ab8f1d627f309dc78a683`, functional identity `c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc`. Read-only source review plus CPU evaluation of the actual reward function. No simulation, policy evaluation, runtime edits or new admission claims. Individual source hashes and reproducible arithmetic are in `reward_arithmetic.json`; rerun `analyze_rewards.py` with the repository CPU environment.

**Assessment:** this is a coherent first learned flat-ground velocity controller. It provides signed multidirectional commands, explicit physical motor coordinates and smooth bounded target motion without imposing a gait. Keep this baseline intact long enough to obtain interpretable evidence. Its campaign success criteria prove functioning finite PPO and checkpoint continuation, not successful walking. Neither all-terrain capability nor accurate survey tracking is trained here.

## Actual task and learning budget

| Element | Frozen behavior |
| --- | --- |
| Commands | Anatomical forward = body -Y; lateral = body +X. Moving samples independently draw forward/lateral from ±0.15 m/s and yaw from ±0.30 rad/s; 20% of draws are exactly zero in all axes. Draw held for 4 s. Maximum diagonal speed is 0.2121 m/s. |
| Curriculum | None in this task. Full signed command box is present from the first rollout. Archived homotopies, phases and helper samplers elsewhere are not called. `axis_command_active_threshold` is not used by this reward or sampler. |
| Terrain and disturbances | Fixed plane, configured material friction 1/1, restitution 0, no self-collision. No terrain generation, sensor/action noise, pushes, friction/mass/inertia/payload/voltage/motor randomization configured. Coincident origins are collision-isolated independent robots, not additional terrain diversity. |
| Reset | Same closed 30-coordinate CAD-derived stance, root height 0.142970 m, nominal reward height 0.137970 m; zero joint jitter and root velocity. Motor budget resets to 0.5. RSL randomizes initial episode counters, not physical poses. Time limit 20 s. |
| Actions | 18 physical motor offsets, clipped to ±1 then scaled by 0.30 rad about the stance; six coxa, six femur, six pushlevers by name. Endpoint slew 0.040 rad/20 ms, linearly applied in 32 substeps at 1600 Hz. Target-rate ceiling 2 rad/s; this is not an actual joint-velocity bound. |
| Observations | 84 values: base linear/angular velocity and gravity in anatomical coordinates, command, 18 motor positions relative to stance, 18 velocities, previous clipped action, 18 budget headrooms. Base motion/gravity are simulator state, not noisy estimated navigation measurements. No terrain map, camera, LiDAR, foot-contact state or recurrent memory. |
| PPO | Separate actor/critic MLPs 256/256/128, ELU, empirical observation normalization; Gaussian initial std 0.15; 24 controls/rollout, 5 epochs, 4 minibatches; clip 0.2, value coefficient 1, entropy 0.01, adaptive LR 0.001, desired KL 0.01, gamma 0.99, lambda 0.95, gradient norm 1. Seed 42. Exact installed RSL 5.0.1 adapter and checkpoint path were separately CPU-reviewed. |
| Run sizes | Scratch: 64 environments × 3 iterations = 4,608 transitions, only 1.44 simulated seconds per environment. Full: exact scratch state resumes into 512 environments × 1,000 additional iterations = 12,288,000 transitions, 480 s per environment, 68.27 aggregate robot-hours, 20,000 minibatch updates of 3,072 samples. Final finite inference is 100 controls = 2 s. |

A rollout spans 0.48 s; bootstrapping carries value across rollouts. Gamma's time constant is 1.99 s; the gamma-lambda trace time constant is 0.326 s. These are reasonable quantities to monitor against gait learning, not proof that 1,000 iterations will converge. The aggregate 68.27 hours includes many parallel copies of the same flat dynamics; it is not 68 hours of continuous thermal or terrain testing. There is no defensible wall-clock ETA from these sample counts alone.

All 18 stance ±0.30 rad endpoint ranges fit inside the soft limits; the smallest margin is 0.03250 rad. Thus soft clipping does not secretly shrink the nominal action envelope. The full range remains unqualified for self-collision/hardware stops. Exploration's unslewed offset std is 0.045 rad. For a hypothetical centered initial Gaussian at the reset target, 37.4% of motor samples would encounter the first endpoint slew limit; almost every 18-motor action would limit at least one coordinate. Actual actor means and later std must be read from the run. This limiter is intentional, but latent action variance is not the same as physically delivered excitation.

## What the reward actually favors

All entries below are summed and multiplied by 0.02 per control, except the separate -5 termination penalty. Norms/sums use the relevant active coordinates; passive reaction torques are not charged as motor effort.

| Component | Reward rate |
| --- | --- |
| Linear tracking | `3 exp(-||v_xy-command_xy||² / 0.0225)` |
| Yaw tracking | `0.6 exp(-(yaw_rate-command_yaw)² / 0.09)` |
| Vertical velocity | `-1.5 vz²` |
| Roll/pitch rate | `-0.1 ||omega_xy||²` |
| Orientation | `-2 ||gravity_xy||²` |
| Height | `-10 (height-0.137970)²` |
| Applied motor torque | `-0.0001 sum(torque²)` |
| Motor work | `-0.001 sum(abs(torque*velocity))` |
| Motor acceleration | `-0.00000025 sum(acceleration²)` |
| Clipping | `-0.01 sum(abs(raw_demand-applied)²)` |
| Continuous overload | `-0.01 sum(relu(abs(raw_demand)-continuous_limit)²)`; this uses raw demand, not delivered overload |
| Action rate | `-0.02 sum((clipped_action-previous_clipped_action)²)` |
| Motor soft-limit violation | `-0.25 sum(violation_rad)` |
| Nonfoot contact | `-1 per loaded nonfoot body/shaft count` |
| Support shortfall | `-0.15 max(3-pad_support_count,0)` |
| Foot slip | `-0.1 sum(loaded_pad_horizontal_contact_speed²)` |

There is no forced tripod phase, airtime objective, foot-clearance target or prescribed swing. Nothing in the source makes every stable solution a walking gait: learning chooses whatever physical behavior maximizes these terms.

## Concrete interpretation risks

1. **Standing can look deceptively good in aggregate reward.** With ideal upright stationary state and no other penalties, exact command-draw expectation is 2.41707/s versus 3.6/s for perfect tracking: 67.14%. For a pure 0.05 m/s command, stationary tracking alone is 3.28452/s, or 91.24% of the ideal. The computation uses the command draw distribution, not measured episode occupancy, which resets can change. Tracking has a directional gradient and movement can improve reward; this does not prove standing is a local optimum. It proves a high total reward or long episodes cannot certify locomotion.
2. **Pure-axis behaviors are not deliberately rehearsed.** Except for standing, all three components are independently continuous; exact pure forward/lateral/yaw samples have probability zero. Near-axis examples exist, but isolated yaw, straight translation and immediate stop/reversal need explicit evaluation. There is no explicit cross-track, heading-hold, polygon coverage or route-completion objective here.
3. **Feet may shuffle or slide and still score well.** Six loaded pads sliding at 0.15 m/s cost only 0.0135/s. A 20 mm height offset costs 0.004/s; all 18 motors at 1.2 N·m cost 0.002592/s in the torque term alone. These comparisons exclude work, acceleration, overload and other simultaneous costs; they demonstrate relative scales, not a dynamically feasible exploit. Inspect actual pad trajectories, contact duty and slip, not gait appearance alone.
4. **Acceleration can dominate under numerical velocity noise.** Eighteen active motors at 100/300/1,000 rad/s² RMS incur 0.045/0.405/4.5 per second, respectively; the last exceeds all positive tracking reward. The captured PhysX articulation backend updates `joint_acc` by finite differencing at every scene update, and reward reads the final substep. For identical per-step velocity jitter, halving dt quadruples this squared penalty. Physical smooth acceleration need not worsen with smaller dt. Monitor the actual logged component before changing anything; no current PPO acceleration distribution was available to this review.
5. **The learner does not observe its full controller state.** The MLP sees previous clipped action, not the accumulated slew-limited endpoint or current PD target. Different older action histories can therefore leave different target states with the same observed previous action; positions/velocities may help infer it but do not remove the hidden state. If response lags or action saturation dominate, compare issued actions with delivered targets before proposing a versioned observation/history change. This is a genuine partial-observability property, not a proven training failure.
6. **Short episodes hide long-duration limits.** Each reset replenishes budget to 0.5. Twenty-second episodes and a two-second inference probe cannot establish a continuous survey duty cycle. The nominal 48 V and thermal recovery remain declared unmeasured assumptions. Self-collision off and perfect base state also preclude a hardware-readiness conclusion.

## What should decide the next step

The existing campaign should proceed without opportunistic reward edits. During PPO, read reward components, episode length, std, clipping/overload, closure and actual collection/learning time together. `progress.json` alone reports finite losses and physical windows, not command-conditioned tracking errors. TensorBoard episode components are accumulated and divided by the fixed 20 s maximum; short failed episodes are therefore not instantaneous reward-rate samples. Falling legitimately remains part of exploration; model closure/envelope failures invalidate the run.

After a checkpoint, evaluate deterministic behavior with the same 0.040 rad/20 ms limiter and physical capture: stand, ±forward, ±lateral, diagonals, ±in-place yaw, combined translation/yaw, and stop/reversal transitions. Use more than one reset/command seed. Record signed achieved velocity, per-axis error, lateral drift during straight commands, yaw drift during translations, stop displacement/settling time, survival/termination reasons, support/slip, motor demands/budget, and physical constraint bounds. Preserve raw traces and the source/checkpoint hashes. This is a proposed diagnostic screen, not a new acceptance gate or a replacement for any existing threshold.

Interpret results in order: physical-model failure → diagnose solver/actuator evidence; physical pass with near-zero motion → compare actual component balance and delivered excitation; commanded motion with sliding → contact/slip analysis; robust flat motion → longer duration, self-collision/workspace qualification, measured dynamics randomization and terrain/estimator integration as separately versioned work. The final terrain policy and survey path planner remain separate development stages. A finished 1,000-iteration PPO is a substantial baseline experiment; only measured command response demonstrates a successful controller.

## Source anchors and limits

Primary repository sources: `isaaclab/train_mkii_fourbar.py` (`main`, physical guard, checkpoint inference); `isaaclab/deploy/run-mkii-fourbar-campaign` (`phases`, `verify_phase`); `packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/{config,env,math}.py`; `packages/hexapod_env/hexapod_env/ppo_cfg.py`; `packages/hexapod_core/hexapod_core/fourbar_v1.py`; `configs/mkii_fourbar_v3_kinematics.json`; actuator `rs05_v2_model.py`. Hashes are in the adjacent JSON.

Exact previously captured installed RSL/Isaac sources are preserved under `rsl501_startup_review/installed_sources/`; no new SDK source copies are redistributed here. The PhysX backend source capture `/tmp/hexapod_profiling_physx_sources/articulation_data.py`, originating from the Spark SDK source readback, has SHA-256 `43a31992353ad3a500024d0ec908d5b253ada7a0aac9a1cd2261466e10bda4ff`: `update` at line 121 forces `joint_acc`, whose finite difference is at lines 964–985. See `profiling_plan_v1/README.md` and `profile_analysis_v2` for SDK capture provenance and limits. Actual joint acceleration or reward behavior of the new run is not asserted from source alone.

`docs/MKII_FOURBAR_TRAINING.md` still describes 800 Hz/16 substeps in its overview; the frozen executable contract is 1600 Hz/32. This documentation mismatch should be corrected in a subsequent documentation update, not confused with the actual launched recipe. No existing document or source was edited by this review.
