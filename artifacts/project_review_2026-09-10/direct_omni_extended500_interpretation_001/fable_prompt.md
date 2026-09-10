You are an independent engineering/scientific partner reviewing actual completed hexapod PPO evidence. You are not an authority. No tools, code execution, file changes or GPU actions. Use only supplied evidence. Give a final actionable critique in at most900 words, no internal reasoning transcript. First state what the data establish, then name the next bounded experiment/admission and decisive measurements. Explicitly reject unsupported causal claims and identify any inconsistency requiring review.

LATEST USER RULE OVERRIDES HISTORICAL NEXT-EXPERIMENT ADVICE: All future training uses the visually approved detailed direct-drive 19-body/18-joint model, mass7.466088235kg, canonical URDF9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78. It is NOT natively admitted and has no task ID. No new or resumed C/mock/fourbar training. This completed CAPS500 study is historical evidence. The next real work is model-bound runtime and native admission (USD/SDF processing, named joints/axes/zeros/limits, suspended sweeps, settled contact/torque/dynamics) before any new policy. Existing18-action checkpoints do not become compatible just because dimension matches. The 1.6Nm and .04rad/20ms values below are historical comparison rules, not a measured torque-speed/thermal identification of the newrobot. Newmotorplan requires declared calibrated/sourced envelope; do not invent controller gains or inherit study geometry. Current evidence: provisional corrected mass cylinder distribution; rounded-square +X distal tip, no sphere; perleg yawlimits; pitchzeros/bounds distinct.

Your choice is how to carry historical lessons into the NEW model's admission and later control experiments, not whether to schedule more C training. Rank further budget, sensing/action/physics diagnostic, and objective/architecture work conditionally. We currently favor native physical admission first with meaningful constant-target/controlled-excitation rate/torque/contact traces; do not automatically adopt the old policy afterward.

Independent preliminary PPO-agent frequency evidence (final frozen receipt pending, label provisional here): 47 reset-free CAPS500 quietrows have exactlyzero command; raw action, executedtarget and actualq share4.4Hz peak, not25Hz alternation. Rawaction75.9%power2–5Hz, target64.2%,q54.4%; actionclipping/slew recurrence residual0. Earlier ORIGINAL-policy no-observation-noise test had negligible effect and80ms targetfilter worsened saturation. Those results neither identify the cause nor prove every policy/newmodel will behave likewise; avoid repeating noiseoff/filter as an established cure.

Previously rejected explanations: diagnostic episode limit90s covers32s stop; rawXYZW explicitlyconverted where needed; actor deterministicmean evaluation and last10s commandexactzero; freshoptimizer reset proven (no inheritedAdam blame); minibatchLR notzero; spatial andtemporal losses active; SDKratefloor alone cannot explain actualqrange/intervalmotion. Sparse preAdamgradients are not actualAdamparameterupdates; learningloss reduction alone is not quiet admission. Don't claim a matched-budget intervention from50vs500, repeatedseeds significance, or guaranteedimprovement from newarchitecture.

All eventual repo edits require docs/PROJECT_SITE.md: bounded central siteupdate+STATUS/plan if affected, sitevalidation/build; only root dispatches/publishes. This review makes no changes.

ACTUAL DESCRIPTIVE SUMMARY:
{
  "scope": "Different optimization budgets and source/host versions, same declared physical/objective/evaluation contract; not a matched-budget intervention, significance test or causal attribution.",
  "caps50": {
    "updates": 50,
    "transitions": 1228800,
    "checkpoint_sha256": "ca0545760f81b8b8fb0cafde4b1ef89bfa5d8765a8b43b578e451f648a954415",
    "quiet_passed_replicas": 0,
    "quiet_metrics": {
      "max_planar_excursion_m": {
        "count": 48,
        "min": 0.025290245190262794,
        "median": 0.05042181722819805,
        "mean": 0.05062253955596437,
        "max": 0.07510803639888763
      },
      "max_heading_excursion_deg": {
        "count": 48,
        "min": 1.9549503326416016,
        "median": 7.852171182632446,
        "mean": 7.6116717060407,
        "max": 11.355448722839355
      },
      "max_joint_velocity_rms_rad_s": {
        "count": 48,
        "min": 1.2464606761932373,
        "median": 1.2735448479652405,
        "mean": 1.272754043340683,
        "max": 1.2934621572494507
      },
      "max_joint_position_range_rad": {
        "count": 48,
        "min": 0.2716671824455261,
        "median": 0.31820186972618103,
        "mean": 0.3165392813583215,
        "max": 0.35550475120544434
      },
      "max_target_step_abs_p95_rad_per_20ms": {
        "count": 48,
        "min": 0.04000002145767212,
        "median": 0.04000002145767212,
        "mean": 0.04000002145767212,
        "max": 0.04000002145767212
      },
      "max_requested_torque_saturation_fraction": {
        "count": 48,
        "min": 0.202,
        "median": 0.222,
        "mean": 0.22466666666666665,
        "max": 0.264
      },
      "max_applied_torque_nm": {
        "count": 48,
        "min": 1.600000023841858,
        "median": 1.600000023841858,
        "mean": 1.600000023841858,
        "max": 1.600000023841858
      }
    },
    "no_trial_reset_descriptive_subset": {
      "count": 46,
      "min": 0.025290245190262794,
      "median": 0.04991213604807854,
      "mean": 0.05005764819519675,
      "max": 0.07510803639888763
    },
    "no_quiet_reset_descriptive_subset": {
      "count": 48,
      "min": 0.025290245190262794,
      "median": 0.05042181722819805,
      "mean": 0.05062253955596437,
      "max": 0.07510803639888763
    },
    "quiet_all_joint_requested_saturation": {
      "count": 864,
      "min": 0.0,
      "median": 0.112,
      "mean": 0.1054537037037037,
      "max": 0.264
    },
    "quiet_per_env_mean_joint_saturation": {
      "count": 48,
      "min": 0.10177777777777777,
      "median": 0.10527777777777778,
      "mean": 0.1054537037037037,
      "max": 0.11077777777777778
    },
    "quiet_worst_interval_angle_joint_RMS_rad_s": 1.5017630828522333,
    "constant_peak_requested_nm": 7.108089923858643,
    "constant_equal_case_mean_requested_saturation": 0.1011875,
    "all_stop32s_peak_requested_nm": 8.082405090332031
  },
  "caps500": {
    "updates": 500,
    "transitions": 12288000,
    "checkpoint_sha256": "c376a0a4eb04d54396b4fd6171fe173167767463245213cce7c2cc1d3a2877cf",
    "quiet_passed_replicas": 0,
    "quiet_metrics": {
      "max_planar_excursion_m": {
        "count": 48,
        "min": 0.017827726900577545,
        "median": 0.03427206724882126,
        "mean": 0.07583574706222862,
        "max": 1.971535086631775
      },
      "max_heading_excursion_deg": {
        "count": 48,
        "min": 1.7371492385864258,
        "median": 6.245135307312012,
        "mean": 7.361561581492424,
        "max": 52.21211624145508
      },
      "max_joint_velocity_rms_rad_s": {
        "count": 48,
        "min": 1.2431389093399048,
        "median": 1.262895405292511,
        "mean": 1.2634193375706673,
        "max": 1.2893660068511963
      },
      "max_joint_position_range_rad": {
        "count": 48,
        "min": 0.2566477656364441,
        "median": 0.29662075638771057,
        "mean": 0.30048150196671486,
        "max": 0.5079135298728943
      },
      "max_target_step_abs_p95_rad_per_20ms": {
        "count": 48,
        "min": 0.04000002145767212,
        "median": 0.04000002145767212,
        "mean": 0.04000002145767212,
        "max": 0.04000002145767212
      },
      "max_requested_torque_saturation_fraction": {
        "count": 48,
        "min": 0.21,
        "median": 0.227,
        "mean": 0.22770833333333335,
        "max": 0.25
      },
      "max_applied_torque_nm": {
        "count": 48,
        "min": 1.600000023841858,
        "median": 1.600000023841858,
        "mean": 1.600000023841858,
        "max": 1.600000023841858
      }
    },
    "no_trial_reset_descriptive_subset": {
      "count": 47,
      "min": 0.017827726900577545,
      "median": 0.03415351361036301,
      "mean": 0.035501718560748914,
      "max": 0.054580722004175186
    },
    "no_quiet_reset_descriptive_subset": {
      "count": 47,
      "min": 0.017827726900577545,
      "median": 0.03415351361036301,
      "mean": 0.035501718560748914,
      "max": 0.054580722004175186
    },
    "quiet_all_joint_requested_saturation": {
      "count": 864,
      "min": 0.0,
      "median": 0.126,
      "mean": 0.11122685185185185,
      "max": 0.25
    },
    "quiet_per_env_mean_joint_saturation": {
      "count": 48,
      "min": 0.10377777777777777,
      "median": 0.1113888888888889,
      "mean": 0.11122685185185185,
      "max": 0.11533333333333333
    },
    "quiet_worst_interval_angle_joint_RMS_rad_s": 1.4782071617089974,
    "constant_peak_requested_nm": 6.723892688751221,
    "constant_equal_case_mean_requested_saturation": 0.10468518518518519,
    "all_stop32s_peak_requested_nm": 7.255267143249512
  },
  "common_no_trial_reset_subset": {
    "excluded_env_ids": [
      20,
      44
    ],
    "summaries": {
      "caps50": {
        "count": 46,
        "min": 0.025290245190262794,
        "median": 0.04991213604807854,
        "mean": 0.05005764819519675,
        "max": 0.07510803639888763
      },
      "caps500": {
        "count": 46,
        "min": 0.017827726900577545,
        "median": 0.03427206724882126,
        "mean": 0.03567732898923366,
        "max": 0.054580722004175186
      }
    },
    "gate_override": false
  },
  "raw_summary": {
    "caps50": {
      "scored_window_first_last_s": [
        22.02,
        32.0
      ],
      "reset_events": [
        {
          "env_id": 20,
          "name": "forward_fast",
          "time_s": 18.82,
          "in_scored_quiet": false,
          "terminated": true,
          "truncated": false,
          "next_sample_planar_jump_m": 1.8299697637557983,
          "reasons": [
            "reason_base_contact"
          ]
        },
        {
          "env_id": 44,
          "name": "diagonal",
          "time_s": 20.86,
          "in_scored_quiet": false,
          "terminated": true,
          "truncated": false,
          "next_sample_planar_jump_m": 1.157080888748169,
          "reasons": [
            "reason_base_contact"
          ]
        }
      ],
      "quiet_command_abs_max": 0.0,
      "quiet_requested_command_abs_max": 0.0,
      "quiet_limiter_active_fraction": {
        "count": 48,
        "min": 0.8526667952537537,
        "median": 0.8631671965122223,
        "mean": 0.8622875387469927,
        "max": 0.8692227005958557
      },
      "quiet_requested_peak_nm": 6.99406623840332,
      "all32s_requested_peak_nm": 8.082405090332031
    },
    "caps500": {
      "scored_window_first_last_s": [
        22.02,
        32.0
      ],
      "reset_events": [
        {
          "env_id": 20,
          "name": "forward_fast",
          "time_s": 30.240000000000002,
          "in_scored_quiet": true,
          "terminated": true,
          "truncated": false,
          "next_sample_planar_jump_m": 1.981821894645691,
          "reasons": [
            "reason_base_contact"
          ]
        }
      ],
      "quiet_command_abs_max": 0.0,
      "quiet_requested_command_abs_max": 0.0,
      "quiet_limiter_active_fraction": {
        "count": 48,
        "min": 0.8606671690940857,
        "median": 0.8682226836681366,
        "mean": 0.8681278228759766,
        "max": 0.8752231597900391
      },
      "quiet_requested_peak_nm": 5.816514015197754,
      "all32s_requested_peak_nm": 7.255267143249512
    }
  },
  "initial_constant_report_exact": true
}

PER-DIRECTION TABLE (4 replicas each; rate trace only1 each):
| Direction | Planar50→500 m/s | Δplanar m/s | Yaw50→500 rad/s | Δyaw rad/s | Sat50→500 % | Δsat pp | Peak50→500 Nm |
|---|---:|---:|---:|---:|---:|---:|---:|
| stand | 0.030732 → 0.023286 | -0.007446 | 0.073935 → 0.061926 | -0.012009 | 10.6583 → 11.0972 | +0.4389 | 6.67412 → 5.18546 |
| forward | 0.036312 → 0.034933 | -0.001379 | 0.090889 → 0.102604 | +0.011715 | 10.9333 → 10.3361 | -0.5972 | 6.69151 → 6.72389 |
| reverse | 0.034763 → 0.030179 | -0.004584 | 0.090212 → 0.088312 | -0.001901 | 9.7639 → 9.9889 | +0.2250 | 6.39275 → 5.21728 |
| left | 0.034926 → 0.025590 | -0.009337 | 0.088403 → 0.084654 | -0.003749 | 9.4889 → 10.0861 | +0.5972 | 6.57436 → 5.80310 |
| right | 0.032071 → 0.029659 | -0.002412 | 0.082883 → 0.082563 | -0.000320 | 10.0250 → 10.7611 | +0.7361 | 5.40143 → 6.32825 |
| forward_fast | 0.060739 → 0.049926 | -0.010813 | 0.135204 → 0.154089 | +0.018885 | 8.5333 → 9.4111 | +0.8778 | 5.72361 → 5.82604 |
| turn_left | 0.029860 → 0.023612 | -0.006248 | 0.083747 → 0.084451 | +0.000703 | 9.5194 → 10.3500 | +0.8306 | 6.48685 → 5.89549 |
| turn_right | 0.033367 → 0.026245 | -0.007122 | 0.082062 → 0.075736 | -0.006326 | 10.9361 → 11.7528 | +0.8167 | 6.40797 → 5.89814 |
| arc_left | 0.037139 → 0.033363 | -0.003776 | 0.098552 → 0.102068 | +0.003516 | 11.0861 → 10.1139 | -0.9722 | 6.65124 → 6.50047 |
| arc_right | 0.033108 → 0.035012 | +0.001904 | 0.101507 → 0.115592 | +0.014086 | 10.0722 → 10.7833 | +0.7111 | 7.10809 → 6.34196 |
| strafe_arc | 0.030687 → 0.026425 | -0.004262 | 0.109632 → 0.122563 | +0.012930 | 9.7222 → 9.6361 | -0.0861 | 6.88525 → 5.83435 |
| diagonal | 0.036013 → 0.029658 | -0.006355 | 0.092112 → 0.089259 | -0.002853 | 10.6861 → 11.3056 | +0.6194 | 6.23654 → 5.66711 |


ACTUAL500 ANALYZER HUMAN REPORT:
# Direct PPO comparison

The bounded allocation completed. Physical changes and regressions remain separate from optimizer loss and sparse gradients.

Acquisition completion is not physical acceptance. Every result retains the original 1.6 Nm cap and quiet bounds; no threshold or statistical significance is invented.

Training completed 500 updates, 12000 controls × 1024 replicas (12,288,000 transitions). Wrapper time 1033.24 s; 11,893 transitions/s. Strict reload reported True.

Training events: 683 terminations, 12129 timeouts, 13549 nonfoot environment-steps. Requested peak 39.707 Nm; applied peak 1.600000 Nm.

Receipt-only evidence: producer reports 500 updates, complete=True, strict reload=True. Raw/checkpoint verification remains separate.

Optimizer diagnostics (measurement, not a qualification):

Retained 500 update rows / 10000 minibatches; 14 sparse gradient rows.
Minibatch LR range 1e-05–7.59375e-05; 1278 retained rows at the existing 1e-5 floor. Mean recorded KL 0.0132849.

| Loss | Initial-window mean | Final-window mean |
|---|---:|---:|
| caps_temporal | 0.316994 | 0.24156 |
| caps_quiet_temporal_mean | 0.315038 | 0.235273 |
| caps_moving_temporal_mean | 0.325355 | 0.254237 |
| caps_spatial | 0.00167116 | 0.00126961 |
| caps_weighted | 0.0318665 | 0.024283 |

Windows contain 10 disjoint updates each; exact IDs and all minibatch LR/KL values are in report.json.
Conditional quiet/moving means use their own pair counts. These optimizer presentations repeat PPO epochs; they are not unique environment steps.

| Update / minibatch | PPO actor norm | Quiet norm | Moving norm | Spatial norm | PPO–quiet cosine | Combined before → after clip |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 1 | 4.3688 | 0.064278 | 0.00327946 | 0.000675904 | 0.117987 | 4.37727 → 1 |
| 1 / 20 | 4.12138 | 0.0646027 | 0.00307183 | 0.000664294 | 0.0021745 | 4.12217 → 1 |
| 10 / 1 | 7.31101 | 0.0688592 | 0.0162601 | 0.000469962 | 0.0755356 | 7.31845 → 1 |
| 10 / 20 | 7.32448 | 0.0699111 | 0.0148594 | 0.000467735 | -0.113014 | 7.31538 → 1 |
| 25 / 1 | 8.39001 | 0.0515578 | 0.0306256 | 0.000453691 | 0.0335108 | 8.39433 → 1 |
| 25 / 20 | 5.41185 | 0.0530641 | 0.029629 | 0.00046037 | -0.0535604 | 5.40832 → 1 |
| 50 / 1 | 9.81241 | 0.0564308 | 0.0264312 | 0.000467315 | 0.0333417 | 9.81662 → 1 |
| 50 / 20 | 6.37873 | 0.0543302 | 0.0280276 | 0.000457609 | 0.0195479 | 6.3813 → 1 |
| 100 / 1 | 7.96628 | 0.0558514 | 0.0247789 | 0.000448809 | 0.0506095 | 7.9703 → 1 |
| 100 / 20 | 6.96712 | 0.0556984 | 0.0245945 | 0.000453713 | -0.118528 | 6.95817 → 1 |
| 250 / 1 | 8.03077 | 0.0500505 | 0.0248658 | 0.000418602 | -0.108322 | 8.02283 → 1 |
| 250 / 20 | 5.8902 | 0.0498949 | 0.0254532 | 0.000417106 | 0.0120648 | 5.89219 → 1 |
| 500 / 1 | 7.71731 | 0.0455389 | 0.0225276 | 0.000350207 | -0.0174272 | 7.71726 → 1 |
| 500 / 20 | 7.30991 | 0.046487 | 0.0218235 | 0.000359505 | -0.20059 | 7.29631 → 1 |

Optimizer diagnostics only. Aggregate temporal losses are not quiet target-step p95; sparse gradients do not describe every minibatch or the Adam parameter update.

Ordinary model_*.pt files are not read by this curated analyzer. Count/cadence is producer and host receipt evidence; verify all remote files with the separate terminal inventory.

Extended decision milestones: 1, 10, 25, 50, 100, 250, 500. completed declared cadence and seven local decisions verified; ordinary autosaves require separate full remote inventory

Constant directions use four-replica summaries. Joint-rate traces cover one replica per direction; those trace values are never represented as all four.

| Direction | Planar error before → after (m/s) | Yaw error before → after (rad/s) | Requested saturation before → after | Requested peak after (Nm) | Δterm |
|---|---:|---:|---:|---:|---:|
| stand | 0.0379 → 0.0233 | 0.0765 → 0.0619 | 10.70% → 11.10% | 5.185 | +0 |
| forward | 0.0406 → 0.0349 | 0.0950 → 0.1026 | 10.96% → 10.34% | 6.724 | +0 |
| reverse | 0.0354 → 0.0302 | 0.0981 → 0.0883 | 9.71% → 9.99% | 5.217 | +0 |
| left | 0.0354 → 0.0256 | 0.0888 → 0.0847 | 9.52% → 10.09% | 5.803 | +0 |
| right | 0.0347 → 0.0297 | 0.0848 → 0.0826 | 10.04% → 10.76% | 6.328 | +0 |
| forward_fast | 0.0622 → 0.0499 | 0.1395 → 0.1541 | 8.91% → 9.41% | 5.826 | +0 |
| turn_left | 0.0317 → 0.0236 | 0.0867 → 0.0845 | 9.96% → 10.35% | 5.895 | +0 |
| turn_right | 0.0401 → 0.0262 | 0.0800 → 0.0757 | 11.00% → 11.75% | 5.898 | +0 |
| arc_left | 0.0420 → 0.0334 | 0.1009 → 0.1021 | 11.41% → 10.11% | 6.500 | +0 |
| arc_right | 0.0362 → 0.0350 | 0.1048 → 0.1156 | 10.55% → 10.78% | 6.342 | +0 |
| strafe_arc | 0.0330 → 0.0264 | 0.1099 → 0.1226 | 9.72% → 9.64% | 5.834 | +0 |
| diagonal | 0.0355 → 0.0297 | 0.0943 → 0.0893 | 10.51% → 11.31% | 5.667 | +0 |

Observed regressions (any positive difference, not a significance test):
- planar_error_mps: none.
- yaw_error_rad_s: forward, forward_fast, arc_left, arc_right, strafe_arc.
- torque_saturation_fraction: stand, reverse, left, right, forward_fast, turn_left, turn_right, arc_right, diagonal.

Final stop/quiet: 0/48 replicas passed the unchanged ten-second quiet checks.

| Direction | Passing replicas | Failed bounds across replicas | Worst raw joint RMS (rad/s) | Worst drift (mm) |
|---|---:|---|---:|---:|
| stand | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2626 | 34.15 |
| forward | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2695 | 34.39 |
| reverse | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2779 | 37.50 |
| left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2894 | 53.65 |
| right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2743 | 54.58 |
| forward_fast | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.2654 | 1971.54 |
| turn_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2733 | 37.32 |
| turn_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2761 | 44.09 |
| arc_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2792 | 48.41 |
| arc_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2706 | 31.90 |
| strafe_arc | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2718 | 50.06 |
| diagonal | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2872 | 49.41 |

All 48 stop replicas, named-joint values and before/after deltas are in report.json. Raw SDK rates and adjacent-angle interval averages remain separate; neither is substituted into the other’s score.

Next decision: Compare the matched constant and stop evidence with the preserved pilots. Losses, gradient alignment and learning-rate trends alone cannot promote or extend training.

Input paths and SHA256 values are recorded in INPUTS_SHA256.json and rechecked after analysis. No source, checkpoint, input or GPU job is modified.


PRIOR INDEPENDENT DECISION AND CORRECTIONS (historical newmodelrule now supersedes proposed C training):
# Physical counterevidence and next bounded experiment

The improving raw quiet loss did not produce an accepted quiet controller. Both quiet-priority50 and normal CAPS50 pass **0/48** final quiet trials. Every replica's target-step p95 remains about **0.04000002 rad per 20 ms**. The recorded raw SDK rate channel also remains far above its unchanged bound; it is not replaced by an angle-difference estimate.

Independent read-only comparison of the raw final-stop JSON and traces gives:

| Result | Normal CAPS50 | Quiet-priority50 |
| --- | ---: | ---: |
| Median quiet planar excursion | 50.42 mm | 57.60 mm |
| Maximum scored excursion, retaining resets | 75.11 mm | 1,870.23 mm |
| Median mean requested saturation in quiet | 10.53% | 11.16% |
| Constant scenarios with saturation worse than original | 2/12 | 9/12 |
| Trial terminations | 2 | 2 |

The maximum needs a crucial qualification. Quiet-priority environment20 terminates at **25.12 s**, inside the scored quiet window, and its next-row position jumps **1.86541 m** after reset. Normal CAPS environment20 terminates at **18.82 s**, before quiet scoring, with a **1.82997 m** reset jump. Their environment44 terminations are also before quiet scoring: 17.52 s for quiet-priority and 20.86 s for normal CAPS. Consequently, 1.870 m versus 75.11 mm does **not** show 25-fold uninterrupted walking drift or a reset-free CAPS branch. Both failed trials and both original scored maxima remain failures. No row or unfavorable window is removed. Medians and saturation provide a less misleading comparison, while still not establishing the quiet coefficient as the cause.

The next useful experiment is a **fixed, fresh-original normal CAPS500 budget baseline**, subject to root's normal source/standing/launch review. Preserve the original checkpoint and initialize the optimizer explicitly as before. Keep geometry, physics, command curriculum, observation/action contract, gates and inference evaluations matched. Retain intermediate checkpoints, including a midpoint such as250, so a cold constant/stop screen can assess whether extra budget helps. A checkpoint is diagnostic evidence, not a promoted controller or permission to continue automatically.

This experiment answers a missing question: does substantially more optimization improve the existing normal-CAPS branch's measured behavior under the same gates? It does not isolate a weight effect from a budget effect against quiet-priority50, prove normal CAPS superior, or establish that a quiet objective cannot work. Neither current branch is qualified. A longer quiet-priority run is a plausible later budget comparison, but its smooth loss decrease alone is insufficient reason to prefer it now. A stronger quiet weight or architecture change adds a new intervention before the available budget response is measured; the sparse pre-Adam gradients do not yet establish that such a change will improve executed target steps.

This is an experimental-priority recommendation, not a GPU dispatch or new acceptance rule. Root owns the bounded allocation and midpoint decision. The Fable partner received the training diagnostics and physical reports; its final response is independently checked against the reset-timing correction above before any claim is adopted.

Fable independently chose the normal CAPS500 budget baseline, with a midpoint diagnostic. Its proposed literal native003 execution is not adopted: that frozen CLI admits only50 pilot updates, so a separately reviewed500-update source/contract is required. Its speculative sensor-floor concern is not a verified launch blocker. The exact stop evaluator and pinned RSL source use deterministic mean outputs. A nonreset quiet-priority row also has actual joint range0.300rad, adjacent-angle RMS1.492rad/s and targetp95.04, so the failure cannot be explained solely by an SDK reporting floor. These separate measurements do not relax the SDK-rate gate or establish a native simulator cause.

`physical_comparison.json` preserves the exact reset times, next-row jumps, same-gate/override equality, per-branch summaries and source-file hashes. The raw final-stop checkpoint identities are `ca0545760f81b8b8fb0cafde4b1ef89bfa5d8765a8b43b578e451f648a954415` for normal CAPS50 and `195593ca4d595184fefbc9161bad433f9fb9768d8fd16d514c7b755eee0a6173` for quiet-priority50. The latter training receipt SHA remains `c41560dbe34d29bbce40d1060bd1c8b0155cdcc27173dc2316806f8368ad9b11`. This review does not repeat the complete terminal source/cleanup audit or modify any source, gate or GPU state.


CURRENT CANONICAL MODEL SELECTOR:
{
  "schema_version": 1,
  "id": "hexapod_mkii_updated_v1",
  "role": "canonical_ground_truth_training_urdf",
  "topology": "six_legs_three_direct_drive_joints_no_four_bar",
  "joint_review_revision": "20260910_user_travel_coxa_midpoint_v2",
  "urdf": {
    "path": "robot/hexapod_mkii_updated_v1/urdf/hexapod_updated_rs05_mass_corrected.urdf",
    "sha256": "9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78"
  },
  "model": {
    "path": "robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json",
    "sha256": "7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881"
  },
  "raw_cad_urdf": {
    "path": "robot/hexapod_mkii_updated_v1/urdf/hexapod_updated_rs05_raw_cad.urdf",
    "sha256": "7d16cfe101ee1b07c77ee5d453f8313e0858354d87cbf940a8951f9a4b3110a7"
  },
  "preview": "robot/hexapod_mkii_updated_v1/preview/",
  "usd": "artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected/robot.usda",
  "mass_kg": 7.466088235225788,
  "mass_basis": "Original CAD plus separately recorded nominal RS05 missing-mass cylinder estimate; not measured inertial identification.",
  "native_physics_admitted": false,
  "runtime_task_id": null,
  "selection_scope": "Default design, URDF and inspection for new robot work. Historical task IDs/checkpoints retain their own assets. New dynamics require a model-bound runtime adapter and native admission.",
  "visual_review": "User approved full-range movement and grouping on 10 September 2026.",
  "training_policy": "Every new training run must use this exact motor-mass-corrected physical model through a new model-bound runtime and native admission. Simplified C-study/mock and historical four-bar assets are retained solely for historical reproduction; do not launch new training on them.",
  "usd_sha256": "3be98420adc7923923757d6557d8492d5930f84128c1b1366b9c6db1bb03207c"
}
