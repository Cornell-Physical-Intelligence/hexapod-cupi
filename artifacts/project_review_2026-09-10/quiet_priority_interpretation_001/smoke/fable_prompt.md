Independent interpretation of an actual two-update direct locomotion PPO smoke. Use only supplied primary raw output and executable source. Read-only, no tool use or edits. Give a concise final answer with calculations and uncertainties, no internal thinking stream.

Question: Do the mode-separated losses and two sparse actor-gradient rows show the new quiet objective has meaningful actor authority compared with PPO and spatial regularization? Check the 40 KL/LR rows, clipping semantics, shared temporal denominator, quiet/moving sample mix, and what the telemetry cannot establish. Identify concrete implementation concerns if any. Do not infer convergence from two updates, change any gates, or prescribe a weight change as proven. Actual final ten-second quiet result was 0/48; this does not qualify a policy. Root is separately considering a 50-update pilot.

Actor is 315 observations, critic318, separate parameter sets. Fresh initialization uses original checkpoint1971, quiet objective1.0/moving0.1/spatial0.1, inherited dynamics. Stored paired observations use command equality and no done boundaries. Do not equate raw policy-mean differences with executed target-step or actual joint speed.

Raw campaign identity:
{
  "schema": "direct315_quiet_priority_native_v3",
  "source_manifest_sha256": "ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62",
  "plan_sha256": "eee25089a65ec112f5eeed9c70f6310ef019ca5f077618c736fcfaa859739fc2",
  "checkpoint_sha256": "1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8",
  "actor_width": 315,
  "critic_width": 318,
  "selection": {
    "schema": "direct315_quiet_priority_native_v3",
    "allocation": "smoke",
    "branch": "quiet_priority",
    "replicas": 32,
    "controls_per_update": 24,
    "updates": 2,
    "caps": {
      "temporal_weight": 0.1,
      "spatial_weight": 0.1,
      "noise_scale": 1.0,
      "noise_seed": 1157,
      "quiet_temporal_weight": 1.0
    }
  },
  "overrides": {
    "target_slew_rad_per_20ms": 0.04,
    "reward_weights": {
      "stand_joint_velocity": -0.5,
      "stand_target_velocity": -0.15,
      "stand_posture": -2.0,
      "action_rate": -0.075,
      "saturation": -0.75,
      "torque_excess": -0.6,
      "worst_torque_excess": -0.2,
      "stand_raw_action": 0.0
    },
    "observation_noise_scale": 1.0,
    "target_filter_time_constant_s": 0.0
  },
  "diagnostic_options": {
    "duration_s": 12,
    "settle_s": 2,
    "seed": 7057,
    "trace_envs_per_scenario": 1,
    "controller": "policy"
  },
  "Stage2_complete": false
}

Raw full training receipt:
{
  "selection": {
    "schema": "direct315_quiet_priority_native_v3",
    "allocation": "smoke",
    "branch": "quiet_priority",
    "replicas": 32,
    "controls_per_update": 24,
    "updates": 2,
    "caps": {
      "temporal_weight": 0.1,
      "spatial_weight": 0.1,
      "noise_scale": 1.0,
      "noise_seed": 1157,
      "quiet_temporal_weight": 1.0
    }
  },
  "initial_runner_iteration": 1847,
  "complete": true,
  "Stage2_complete": false,
  "updates_completed": 2,
  "optimizer_updates": [
    {
      "completed_update": 1,
      "optimizer_wall_seconds": 0.23738934798166156,
      "losses": {
        "value": 0.3323597706854343,
        "surrogate": -0.03224741187877953,
        "entropy": -15.903185319900512,
        "caps_temporal": 0.3101696833968163,
        "caps_spatial": 0.002248338842764497,
        "caps_weighted": 0.30330526977777483,
        "caps_valid_pair_fraction": 0.7356770932674408,
        "caps_quiet_temporal_mean": 0.3090614191107992,
        "caps_moving_temporal_mean": 0.35212958042438214
      },
      "learning_rate": 1e-05,
      "minibatches": [
        {
          "update": 1,
          "minibatch": 1,
          "kl_mean": 0.0,
          "learning_rate_before": 5e-05,
          "learning_rate_after": 5e-05,
          "gradient": {
            "norms": {
              "ppo_actor": 28.511287689208984,
              "quiet_temporal": 0.6559227108955383,
              "moving_temporal": 0.0034479324240237474,
              "spatial": 0.0006950078532099724
            },
            "ppo_quiet_cosine": -0.010163462720811367,
            "component_sum_norm": 28.512155532836914,
            "combined_actor_before_clip": 28.512155532836914,
            "combined_actor_after_clip": 0.9999999403953552
          },
          "pair_counts": {
            "valid_pairs": 138,
            "quiet_pairs": 136,
            "moving_pairs": 2
          }
        },
        {
          "update": 1,
          "minibatch": 2,
          "kl_mean": 0.13351528346538544,
          "learning_rate_before": 5e-05,
          "learning_rate_after": 3.3333333333333335e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 144,
            "quiet_pairs": 137,
            "moving_pairs": 7
          }
        },
        {
          "update": 1,
          "minibatch": 3,
          "kl_mean": 0.17612898349761963,
          "learning_rate_before": 3.3333333333333335e-05,
          "learning_rate_after": 2.2222222222222223e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 131,
            "quiet_pairs": 131,
            "moving_pairs": 0
          }
        },
        {
          "update": 1,
          "minibatch": 4,
          "kl_mean": 0.16240866482257843,
          "learning_rate_before": 2.2222222222222223e-05,
          "learning_rate_after": 1.4814814814814815e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 152,
            "quiet_pairs": 148,
            "moving_pairs": 4
          }
        },
        {
          "update": 1,
          "minibatch": 5,
          "kl_mean": 0.16783276200294495,
          "learning_rate_before": 1.4814814814814815e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 138,
            "quiet_pairs": 136,
            "moving_pairs": 2
          }
        },
        {
          "update": 1,
          "minibatch": 6,
          "kl_mean": 0.14639106392860413,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 144,
            "quiet_pairs": 137,
            "moving_pairs": 7
          }
        },
        {
          "update": 1,
          "minibatch": 7,
          "kl_mean": 0.14336872100830078,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 131,
            "quiet_pairs": 131,
            "moving_pairs": 0
          }
        },
        {
          "update": 1,
          "minibatch": 8,
          "kl_mean": 0.12911301851272583,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 152,
            "quiet_pairs": 148,
            "moving_pairs": 4
          }
        },
        {
          "update": 1,
          "minibatch": 9,
          "kl_mean": 0.12604880332946777,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 138,
            "quiet_pairs": 136,
            "moving_pairs": 2
          }
        },
        {
          "update": 1,
          "minibatch": 10,
          "kl_mean": 0.10397046059370041,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 144,
            "quiet_pairs": 137,
            "moving_pairs": 7
          }
        },
        {
          "update": 1,
          "minibatch": 11,
          "kl_mean": 0.10115230083465576,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 131,
            "quiet_pairs": 131,
            "moving_pairs": 0
          }
        },
        {
          "update": 1,
          "minibatch": 12,
          "kl_mean": 0.09217815101146698,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 152,
            "quiet_pairs": 148,
            "moving_pairs": 4
          }
        },
        {
          "update": 1,
          "minibatch": 13,
          "kl_mean": 0.08796652406454086,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 138,
            "quiet_pairs": 136,
            "moving_pairs": 2
          }
        },
        {
          "update": 1,
          "minibatch": 14,
          "kl_mean": 0.07134826481342316,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 144,
            "quiet_pairs": 137,
            "moving_pairs": 7
          }
        },
        {
          "update": 1,
          "minibatch": 15,
          "kl_mean": 0.06994026899337769,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 131,
            "quiet_pairs": 131,
            "moving_pairs": 0
          }
        },
        {
          "update": 1,
          "minibatch": 16,
          "kl_mean": 0.0659773200750351,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 152,
            "quiet_pairs": 148,
            "moving_pairs": 4
          }
        },
        {
          "update": 1,
          "minibatch": 17,
          "kl_mean": 0.06370258331298828,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 138,
            "quiet_pairs": 136,
            "moving_pairs": 2
          }
        },
        {
          "update": 1,
          "minibatch": 18,
          "kl_mean": 0.0529172345995903,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 144,
            "quiet_pairs": 137,
            "moving_pairs": 7
          }
        },
        {
          "update": 1,
          "minibatch": 19,
          "kl_mean": 0.052018292248249054,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 131,
            "quiet_pairs": 131,
            "moving_pairs": 0
          }
        },
        {
          "update": 1,
          "minibatch": 20,
          "kl_mean": 0.051845770329236984,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": {
            "norms": {
              "ppo_actor": 13.155975341796875,
              "quiet_temporal": 0.6741668581962585,
              "moving_temporal": 0.004004680085927248,
              "spatial": 0.0006971407565288246
            },
            "ppo_quiet_cosine": -0.06739687919616699,
            "component_sum_norm": 13.12768840789795,
            "combined_actor_before_clip": 13.12768840789795,
            "combined_actor_after_clip": 1.0
          },
          "pair_counts": {
            "valid_pairs": 152,
            "quiet_pairs": 148,
            "moving_pairs": 4
          }
        }
      ]
    },
    {
      "completed_update": 2,
      "optimizer_wall_seconds": 0.09457921609282494,
      "losses": {
        "value": 0.02630788115784526,
        "surrogate": -0.04494894801173359,
        "entropy": -15.899678087234497,
        "caps_temporal": 0.3372573047876358,
        "caps_spatial": 0.0016507927852217109,
        "caps_weighted": 0.2741391956806183,
        "caps_valid_pair_fraction": 0.9583333432674408,
        "caps_quiet_temporal_mean": 0.341618668411089,
        "caps_moving_temporal_mean": 0.321576967631808
      },
      "learning_rate": 1e-05,
      "minibatches": [
        {
          "update": 2,
          "minibatch": 1,
          "kl_mean": 0.0,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 187,
            "quiet_pairs": 142,
            "moving_pairs": 45
          }
        },
        {
          "update": 2,
          "minibatch": 2,
          "kl_mean": 0.0007597969379276037,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1.5000000000000002e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 140,
            "moving_pairs": 43
          }
        },
        {
          "update": 2,
          "minibatch": 3,
          "kl_mean": 0.003919667564332485,
          "learning_rate_before": 1.5000000000000002e-05,
          "learning_rate_after": 2.2500000000000005e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 150,
            "moving_pairs": 33
          }
        },
        {
          "update": 2,
          "minibatch": 4,
          "kl_mean": 0.011716465465724468,
          "learning_rate_before": 2.2500000000000005e-05,
          "learning_rate_after": 2.2500000000000005e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 143,
            "moving_pairs": 40
          }
        },
        {
          "update": 2,
          "minibatch": 5,
          "kl_mean": 0.02219279482960701,
          "learning_rate_before": 2.2500000000000005e-05,
          "learning_rate_after": 1.5000000000000004e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 187,
            "quiet_pairs": 142,
            "moving_pairs": 45
          }
        },
        {
          "update": 2,
          "minibatch": 6,
          "kl_mean": 0.028226863592863083,
          "learning_rate_before": 1.5000000000000004e-05,
          "learning_rate_after": 1.0000000000000003e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 140,
            "moving_pairs": 43
          }
        },
        {
          "update": 2,
          "minibatch": 7,
          "kl_mean": 0.031216349452733994,
          "learning_rate_before": 1.0000000000000003e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 150,
            "moving_pairs": 33
          }
        },
        {
          "update": 2,
          "minibatch": 8,
          "kl_mean": 0.03224186599254608,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 143,
            "moving_pairs": 40
          }
        },
        {
          "update": 2,
          "minibatch": 9,
          "kl_mean": 0.03389187902212143,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 187,
            "quiet_pairs": 142,
            "moving_pairs": 45
          }
        },
        {
          "update": 2,
          "minibatch": 10,
          "kl_mean": 0.034096550196409225,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 140,
            "moving_pairs": 43
          }
        },
        {
          "update": 2,
          "minibatch": 11,
          "kl_mean": 0.032707735896110535,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 150,
            "moving_pairs": 33
          }
        },
        {
          "update": 2,
          "minibatch": 12,
          "kl_mean": 0.032677046954631805,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 143,
            "moving_pairs": 40
          }
        },
        {
          "update": 2,
          "minibatch": 13,
          "kl_mean": 0.03292191028594971,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 187,
            "quiet_pairs": 142,
            "moving_pairs": 45
          }
        },
        {
          "update": 2,
          "minibatch": 14,
          "kl_mean": 0.03477400913834572,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 140,
            "moving_pairs": 43
          }
        },
        {
          "update": 2,
          "minibatch": 15,
          "kl_mean": 0.03315236419439316,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 150,
            "moving_pairs": 33
          }
        },
        {
          "update": 2,
          "minibatch": 16,
          "kl_mean": 0.03535676375031471,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 143,
            "moving_pairs": 40
          }
        },
        {
          "update": 2,
          "minibatch": 17,
          "kl_mean": 0.035131461918354034,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 187,
            "quiet_pairs": 142,
            "moving_pairs": 45
          }
        },
        {
          "update": 2,
          "minibatch": 18,
          "kl_mean": 0.03945241868495941,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 140,
            "moving_pairs": 43
          }
        },
        {
          "update": 2,
          "minibatch": 19,
          "kl_mean": 0.03677516430616379,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 150,
            "moving_pairs": 33
          }
        },
        {
          "update": 2,
          "minibatch": 20,
          "kl_mean": 0.03922561556100845,
          "learning_rate_before": 1e-05,
          "learning_rate_after": 1e-05,
          "gradient": null,
          "pair_counts": {
            "valid_pairs": 183,
            "quiet_pairs": 143,
            "moving_pairs": 40
          }
        }
      ]
    }
  ],
  "optimizer_diagnostics_schema": "direct315_actor_gradients_v1",
  "reload": {
    "passed": true,
    "exact_actor_critic_normalizer_optimizer": true,
    "exact_deterministic_action": true,
    "optimizer_entries": 17,
    "checkpoint_sha256": "ffdb347eacea5d87172fdcd122eaa6836fc43ed1bd139f20493fdf3d9672b000",
    "inference_buffers_cloned_after_learning": [
      "actor.obs_normalizer._std",
      "critic.obs_normalizer._std"
    ],
    "reload_method": "Strict same-runner load after value-preserving inference-buffer normalization"
  },
  "decision_checkpoints": {
    "2": {
      "file": "decision_002.pt",
      "sha256": "d93c371a1a6d83fcaebe1c73c2316a1449c2fbc428c37bea8a4d13ea040ef5d2",
      "native_iteration": 1848,
      "completed_updates": 2
    }
  },
  "final_checkpoint_sha256": "ffdb347eacea5d87172fdcd122eaa6836fc43ed1bd139f20493fdf3d9672b000",
  "final_runner_iteration": 1848,
  "learned_std": [
    0.10003285855054855,
    0.10026709735393524,
    0.10011585801839828,
    0.10003078728914261,
    0.09974411875009537,
    0.10025224089622498,
    0.10027354955673218,
    0.10013251006603241,
    0.09997212886810303,
    0.0997607409954071,
    0.09985163062810898,
    0.10002418607473373,
    0.10015317797660828,
    0.10005658119916916,
    0.09963622689247131,
    0.10017789900302887,
    0.10017314553260803,
    0.0999891385436058
  ],
  "wall_seconds": 3.6445635880809277,
  "audit": {
    "controls": 48,
    "replicas": 32,
    "terminations_per_row": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    "truncations_per_row": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    "requested_torque_max_per_row_nm": [
      4.940145969390869,
      5.309484481811523,
      4.779942512512207,
      5.608872890472412,
      4.63109016418457,
      5.095945358276367,
      6.860121250152588,
      5.167466163635254,
      5.475960731506348,
      5.109999179840088,
      4.553528308868408,
      4.54315710067749,
      5.18281364440918,
      4.883824825286865,
      4.812701225280762,
      4.897871971130371,
      4.76260232925415,
      5.35559606552124,
      5.531614780426025,
      4.6297078132629395,
      6.640103816986084,
      5.423450469970703,
      4.551339626312256,
      4.6847310066223145,
      5.210599899291992,
      4.988088130950928,
      5.8479132652282715,
      8.476327896118164,
      4.873516082763672,
      4.539839267730713,
      5.005599498748779,
      6.9754180908203125
    ],
    "applied_torque_max_per_row_nm": [
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858,
      1.600000023841858
    ],
    "requested_saturation_fraction_per_row": [
      0.08912038803100586,
      0.10995370149612427,
      0.11574075371026993,
      0.09837963432073593,
      0.10879629105329514,
      0.10185185819864273,
      0.1145833432674408,
      0.10532408952713013,
      0.11226852983236313,
      0.09375002235174179,
      0.1180555447936058,
      0.10185187309980392,
      0.10648149251937866,
      0.10300926119089127,
      0.10300926119089127,
      0.09375002235174179,
      0.10416669398546219,
      0.097222238779068,
      0.1145833432674408,
      0.10300926119089127,
      0.09606482833623886,
      0.09490742534399033,
      0.09837964177131653,
      0.10879629850387573,
      0.0937500074505806,
      0.09837964177131653,
      0.1076388880610466,
      0.09259261935949326,
      0.12615741789340973,
      0.097222238779068,
      0.097222238779068,
      0.09837961941957474
    ],
    "trace_sha256": "8841a65b1fa69303d5c88ac7c322dedfbce6360a1633cc43e73f1ff47ed08938",
    "joint_trace_sha256": "30d936faee28dfca0dd3316e12ad28a652f5289f8be74a9a4ab7ad15ffac8415",
    "event_ledger_sha256": "da6643bdcfdc48f59961b46911bb0e0fe8e278d7eac1d3ccc09c3b549ab63e71",
    "scope": "50 Hz pre-reset endpoint evidence; no 400 Hz motor qualification or SDK-rate fidelity claim"
  },
  "cuda_peak_allocated_bytes": 41497088,
  "cuda_peak_reserved_bytes": 48234496
}

Exact native004 caps.py:
"""CAPS-style policy-mean regularization; preserves the 315/318 model interface.

Only auxiliary rollout storage gains keys. Command/action history is never
perturbed. Temporal pairs exclude resets and command changes. Normalizers are
not updated by synthetic neighbors. This is a proposal, not admitted physics.
"""
import math
import torch
from tensordict import TensorDict
WIDTH=315

def validated_options(o):
 base={'temporal_weight','spatial_weight','noise_scale','noise_seed'}
 if not isinstance(o,dict) or set(o) not in (base,base|{'quiet_temporal_weight'}):raise ValueError('Exact CAPS options required')
 o=dict(o);o.setdefault('quiet_temporal_weight',o['temporal_weight'])
 for k in ['temporal_weight','quiet_temporal_weight','spatial_weight','noise_scale']:
  if isinstance(o[k],bool) or not isinstance(o[k],(float,int)) or not math.isfinite(o[k]) or not 0<=o[k]<=1:raise ValueError('Invalid CAPS coefficient')
 if not isinstance(o['noise_seed'],int) or isinstance(o['noise_seed'],bool):raise ValueError('Integer independent noise seed required')
 return dict(o)

def noise_scales(device,dtype):
 s=torch.zeros(5,63,device=device,dtype=dtype)
 s[:,:3]=.00375;s[:,3:6]=.01;s[:,9:27]=.005;s[:,27:45]=.0025
 return s.flatten()

def latest_commands(obs):return obs[:,4*63+6:4*63+9]

class PairState:
 def __init__(self,policy):
  self._check(policy);self.current=policy.detach().clone();self.previous=torch.zeros_like(policy);self.valid=torch.zeros(len(policy),1,device=policy.device,dtype=torch.bool)
 @staticmethod
 def _check(p):
  if p.ndim!=2 or p.shape[1]!=WIDTH or not torch.isfinite(p).all():raise ValueError('Finite Nx315 actor observations required')
 def advance(self,next_policy,dones):
  self._check(next_policy)
  if next_policy.shape!=self.current.shape or dones.shape!=(len(next_policy),):raise ValueError('Pair shape mismatch')
  unchanged=(latest_commands(next_policy)==latest_commands(self.current)).all(-1)
  self.valid=(~dones.bool() & unchanged)[:,None]
  self.previous=torch.where(self.valid,self.current,0.).detach().clone()
  self.current=next_policy.detach().clone()
 def decorate(self,obs):
  if not torch.equal(obs['policy'],self.current):raise ValueError('Cached observation changed without a step')
  out=obs.clone();out['caps_previous_policy']=self.previous.clone();out['caps_pair_valid']=self.valid.clone();return out

class CapsPairWrapper:
 def __init__(self,env):
  self.env=env;self.cached=env.get_observations().clone();self.pairs=PairState(self.cached['policy']);self.failure=None
 def __getattr__(self,key):return getattr(self.env,key)
 @property
 def episode_length_buf(self):return self.env.episode_length_buf
 @episode_length_buf.setter
 def episode_length_buf(self,value):self.env.episode_length_buf=value
 def get_observations(self):
  if self.failure is not None:raise RuntimeError(self.failure)
  return self.pairs.decorate(self.cached)
 def step(self,action):
  if self.failure is not None:raise RuntimeError(self.failure)
  try:
   obs,reward,done,extras=self.env.step(action)
   self.pairs.advance(obs['policy'],done);self.cached=obs.clone()
   return self.get_observations(),reward,done,extras
  except BaseException as exc:self.failure=repr(exc);raise

class MeanRegularizer:
 def __init__(self,options,device):
  self.options=validated_options(options);self.generator=torch.Generator(device=device);self.generator.manual_seed(options['noise_seed']);self.stats=[];self.last_terms=None
 def __call__(self,actor,obs):
  o=self.options
  if actor.obs_groups!=['policy'] or actor.is_recurrent:raise ValueError('Exact feedforward policy-only315 actor required')
  if not o['temporal_weight'] and not o['quiet_temporal_weight'] and not o['spatial_weight']:
   # No actor calls or random draws: the zero branch is an exact PPO ablation.
   self.last_terms=None
   return obs['policy'].new_zeros(())
  p=obs['policy'];previous=obs['caps_previous_policy'];valid=obs['caps_pair_valid']
  PairState._check(p);PairState._check(previous)
  if previous.shape!=p.shape or valid.shape!=(len(p),1) or not torch.isfinite(valid).all() or not ((valid==0)|(valid==1)).all():raise ValueError('Malformed temporal pairing evidence')
  # RSL5.0.1 RolloutStorage stores observation keys as float tensors.
  # Require exact binary values before interpreting this auxiliary mask.
  valid=valid.bool()
  mu=actor(obs)
  prev=obs.clone();prev['policy']=previous
  per_pair=(mu-actor(prev)).square().mean(-1)
  count=valid.sum().clamp_min(1)
  quiet=(latest_commands(p)==0).all(-1)&(latest_commands(previous)==0).all(-1)&valid[:,0]
  moving=valid[:,0]&~quiet
  temporal=(per_pair*valid[:,0]).sum()/count
  quiet_shared=(per_pair*quiet).sum()/count
  moving_shared=(per_pair*moving).sum()/count
  neighbor=obs.clone();noise=torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=self.generator).clamp(-3,3)*noise_scales(p.device,p.dtype)*o['noise_scale'];neighbor['policy']=p+noise
  spatial=(mu-actor(neighbor)).square().mean()
  # Equal coefficients preserve the parent's exact operation ordering.
  loss=o['temporal_weight']*temporal+o['spatial_weight']*spatial
  if o['quiet_temporal_weight']!=o['temporal_weight']:
   loss=loss+(o['quiet_temporal_weight']-o['temporal_weight'])*quiet_shared
  self.last_terms={'quiet_temporal':o['quiet_temporal_weight']*quiet_shared,
                   'moving_temporal':o['temporal_weight']*moving_shared,
                   'spatial':o['spatial_weight']*spatial}
  if not torch.isfinite(loss):raise ValueError('Nonfinite regularization loss')
  self.stats.append({'temporal':float(temporal.detach()),'spatial':float(spatial.detach()),'weighted':float(loss.detach()),'valid_pair_fraction':float(valid.float().mean()),'valid_pairs':int(valid.sum()),'quiet_pairs':int(quiet.sum()),'moving_pairs':int(moving.sum()),'quiet_temporal_mean':float((per_pair*quiet).sum().detach()/quiet.sum().clamp_min(1)),'moving_temporal_mean':float((per_pair*moving).sum().detach()/moving.sum().clamp_min(1))})
  return loss


Exact native004 gradient_diagnostics.py:
"""Read-only actor-gradient summaries on an already built graph; no model forward."""
import torch

def gradient_vector(loss,parameters):
    parameters=tuple(parameters)
    if not parameters:raise ValueError('Actor parameters required')
    grads=(torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
           if loss.requires_grad else (None,)*len(parameters))
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1)
                      for p,g in zip(parameters,grads)])

def gradient_decomposition(ppo_actor,regularizer_terms,parameters):
    parameters=tuple(parameters)
    terms={'ppo_actor':ppo_actor}
    terms.update({k:regularizer_terms[k] for k in ('quiet_temporal','moving_temporal','spatial')})
    vectors={k:gradient_vector(v,parameters) for k,v in terms.items()}
    norms={k:torch.linalg.vector_norm(v) for k,v in vectors.items()}
    a,b=vectors['ppo_actor'],vectors['quiet_temporal']
    denom=norms['ppo_actor']*norms['quiet_temporal']
    cosine=None if float(denom)==0. else float(torch.dot(a,b)/denom)
    result={'norms':{k:float(v) for k,v in norms.items()},'ppo_quiet_cosine':cosine,
            'component_sum_norm':float(torch.linalg.vector_norm(sum(vectors.values())))}
    if any(not torch.isfinite(v).all() for v in vectors.values()):raise ValueError('Nonfinite diagnostic gradients')
    return result

def assigned_gradient_norm(parameters):
    values=[p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(values))) if values else 0.


Exact native004 caps_ppo.py:
# Derived from RSL-RL5.0.1 PPO.update, BSD-3-Clause.
# Copyright (c)2021-2026 ETH Zurich and NVIDIA CORPORATION. All rights reserved.
# Full license preserved in inputs/RSL_LICENSE.
import hashlib, inspect
from pathlib import Path
import torch
from torch import nn
from rsl_rl.algorithms import PPO
from caps import MeanRegularizer
from gradient_diagnostics import gradient_decomposition, assigned_gradient_norm
BASE_SHA='a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c'
class CapsPPO(PPO):
    def __init__(self,*args,caps_options,gradient_diagnostics=True,**kwargs):
        if hashlib.sha256(Path(inspect.getfile(PPO)).read_bytes()).hexdigest()!=BASE_SHA:
            raise RuntimeError('Installed RSL PPO differs from reviewed source')
        super().__init__(*args,**kwargs)
        if self.rnd or self.symmetry or self.is_multi_gpu or self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError('This bounded CAPS consumer excludes RND/symmetry/distributed/recurrent modes')
        self.regularizer=MeanRegularizer(caps_options,self.device)
        if type(gradient_diagnostics) is not bool:raise ValueError('Explicit boolean diagnostics required')
        self.gradient_diagnostics=gradient_diagnostics
        self.completed_updates=0;self.last_update_diagnostics=[]
        if {id(p) for p in self.actor.parameters()}&{id(p) for p in self.critic.parameters()}:
            raise ValueError('Gradient decomposition requires separate actor/critic parameters')
    def update(self):
        self.regularizer.stats=[];self.last_update_diagnostics=[]
        self.current_update=self.completed_updates+1
        result=self._caps_update()
        self.completed_updates+=1
        for key in ['temporal','spatial','weighted','valid_pair_fraction']:
            rows=self.regularizer.stats
            result['caps_'+key]=sum(r[key] for r in rows)/len(rows) if rows else 0.
        for mode in ('quiet','moving'):
            rows=self.regularizer.stats;count=sum(r[mode+'_pairs'] for r in rows)
            result['caps_'+mode+'_temporal_mean']=sum(r[mode+'_temporal_mean']*r[mode+'_pairs'] for r in rows)/max(count,1)
        self.regularizer.last_terms=None
        return result
    def _caps_update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for minibatch_index,batch in enumerate(generator,1):
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Snapshot scalar state without changing the existing adaptive rule.
            lr_before=float(self.learning_rate);kl_value=None
            sparse=self.gradient_diagnostics and self.current_update in (1,10,25,50) and minibatch_index in (1,self.num_learning_epochs*self.num_mini_batches)
            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.gradient_diagnostics:
                if self.desired_kl is not None and self.schedule=='adaptive':kl_value=float(kl_mean)
                diagnostic={'update':self.current_update,'minibatch':minibatch_index,'kl_mean':kl_value,
                            'learning_rate_before':lr_before,'learning_rate_after':float(self.learning_rate),
                            'gradient':None}
            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            loss = loss + self.regularizer(self.actor, batch.observations)

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            if sparse:
                terms=self.regularizer.last_terms
                if terms is None:
                    zero=surrogate_loss.new_zeros(())
                    terms={k:zero for k in ('quiet_temporal','moving_temporal','spatial')}
                diagnostic['gradient']=gradient_decomposition(
                    surrogate_loss-self.entropy_coef*entropy.mean(),terms,self.actor.parameters())
            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            if sparse:diagnostic['gradient']['combined_actor_before_clip']=assigned_gradient_norm(self.actor.parameters())
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            if sparse:diagnostic['gradient']['combined_actor_after_clip']=assigned_gradient_norm(self.actor.parameters())
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            if self.gradient_diagnostics:
                if self.regularizer.stats:
                    stats=self.regularizer.stats[-1]
                    diagnostic['pair_counts']={k:stats[k] for k in ('valid_pairs','quiet_pairs','moving_pairs')}
                else:diagnostic['pair_counts']=None
                self.last_update_diagnostics.append(diagnostic)
            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict


Exact native004 direct_config.py:
"""Explicit launch selection for one immutable direct315 experiment source."""
import copy

SCHEMA = 'direct315_quiet_priority_native_v3'
SMOKE_BRANCH = 'quiet_priority'
CHECKPOINT = '1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
ALLOCATIONS = {'smoke': {'replicas': 32, 'controls_per_update': 24, 'updates': 2},
               'pilot': {'replicas': 1024, 'controls_per_update': 24, 'updates': 50}}

def protocol():
    return {'schema': SCHEMA, 'allocations': copy.deepcopy(ALLOCATIONS),
            'branches': ['curriculum', 'caps', 'quiet_priority'], 'smoke_branch': SMOKE_BRANCH, 'quiet_priority_objective': {'quiet_temporal_weight':1.0,'moving_temporal_weight':0.1,'spatial_weight':0.1,'denominator':'all_valid_pairs'}, 'optimizer_diagnostics': {'minibatches_per_update':20,'sparse_gradient_updates':[1,10,25,50],'sparse_minibatches':[1,20]}, 'checkpoint_sha256': CHECKPOINT,
            'command_schedule': '25percent_quiet_other_rows_8s_motion_8s_stop',
            'evaluations': ['constant', 'stop'], 'automatic_continuation': False}

def selection(mode, allocation, branch, evaluation, iterations):
    if mode == 'train':
        if allocation not in ALLOCATIONS or branch not in ('curriculum', 'caps', 'quiet_priority') or evaluation is not None:
            raise ValueError('Training requires explicit allocation and branch only')
        result = {'schema': SCHEMA, 'allocation': allocation, 'branch': branch, **ALLOCATIONS[allocation]}
        if iterations is not None and (type(iterations) is not int or iterations != result['updates']):
            raise ValueError('Iteration override differs from bounded allocation')
        result['caps'] = {'temporal_weight': .1 if branch != 'curriculum' else 0.,
                          'spatial_weight': .1 if branch != 'curriculum' else 0., 'noise_scale': 1., 'noise_seed': 1157, 'quiet_temporal_weight':1.0 if branch=='quiet_priority' else (.1 if branch=='caps' else 0.)}
        return result
    if allocation is not None or branch is not None or iterations is not None:
        raise ValueError('Nontraining phase cannot request allocation/branch/iterations')
    if mode == 'validate' and evaluation is None:
        return {'schema': SCHEMA, 'evaluation': None, 'replicas': 32}
    if mode == 'evaluate' and evaluation in ('constant', 'stop'):
        return {'schema': SCHEMA, 'evaluation': evaluation, 'replicas': 48}
    raise ValueError('Only explicit standing, train, constant or stop diagnostic phases exist')

def configure(cfg, selected):
    from caps import validated_options
    expected = selection('train', selected.get('allocation'), selected.get('branch'), None, None)
    if selected != expected:
        raise ValueError('Training selection changed')
    result = copy.deepcopy(cfg)
    if result['obs_groups'] != {'actor': ['policy'], 'critic': ['critic']} or result.get('clip_actions') is not None:
        raise ValueError('Exact 315/318 observation groups and unclipped wrapper required')
    result.update(num_steps_per_env=24, save_interval=1, max_iterations=selected['updates'])
    result['algorithm'].update(class_name='caps_ppo:CapsPPO', caps_options=validated_options(selected['caps']), gradient_diagnostics=True)
    if (result['algorithm']['num_learning_epochs'],result['algorithm']['num_mini_batches'],result['algorithm']['schedule'],result['algorithm']['desired_kl'])!=(5,4,'adaptive',.01):
        raise ValueError('Exact optimizer cadence/adaptive KL required')
    return result
