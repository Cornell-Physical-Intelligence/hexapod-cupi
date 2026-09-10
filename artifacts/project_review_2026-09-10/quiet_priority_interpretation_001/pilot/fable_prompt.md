Requested Fable5.1 MAX partnership: the actual 50-update native004 quiet-priority pilot training has now completed. Give a bounded final diagnostic interpretation (500 words maximum), no internal thinking stream. No tools/edits/GPU. Do NOT infer a physical quiet pass; final score/terminal audit is outside the supplied data. No automatic continuation or gate changes. Focus whether actual1/10/25/50 gradients and mode-specific trends support consistent quiet actor authority and whether a weight versus budget decision is yet evidenced. Distinguish sampled pre-Adam gradient ratios/dot products from Adam updates, and changing on-policy batch losses from causal improvement.

Exact receiptSHA c41560dbe34d29bbce40d1060bd1c8b0155cdcc27173dc2316806f8368ad9b11, checkpoint195593ca4d595184fefbc9161bad433f9fb9768d8fd16d514c7b755eee0a6173,1024 replicas×24 controls×50 updates. The independent stdlib analyzer replayed all1000 adaptiveLR rows against source, including sequential before/after LR, and verified rawreceipt text equals root remote observation and acceptedphase hash. The summaries below preserve all50 mode loss rows and8 raw sparse gradient rows. All gradients are weighted objective components, actor andcritic clip separately. Optimizer starts RESET with zero entries; retained iteration1847 is not inherited Adam state. Source excludes RND/symmetry/sharedactorcritic.

Source objective: mean[(mu(s)-mu(prev))²] overactiondim, valid adjacentpairs exclude dones and command changes; quiet requires both latestcommand==exactzero. Shared denominator is all validpairs; quietweight1.0,movingweight.1, spatialweight.1. Gradienttelemetry uses autograd on existinggraph, first/lastmb atupdates1/10/25/50, commonactorclip1.0, thenAdam. Quiet-dottotalinterval uses Qdot(P+Q) and Cauchy-Schwarz bound |Qdot(M+S)|≤||Q||(||M||+||S||); signs characterize only plain negativegradient direction. SourceadaptiveLR desiredKL.01, divide1.5 above.02, multiply1.5 for0<KL<.005, floor1e-5/ceiling1e-2.

Earlier smoke2update findings: Q/Ponly2.30/5.12%; bothfromupdate1. Quietconditionalmeanrose10.53% whileweightedtotalfell9.62%, confoundedbyquietshare97.70→78.13%. Those smoke ratios do not replace actualpilot. In priorpartnerresponse we rejected causes inferredfromKLshape, claims of inheritedAdam, and confusing35overshootrows with31/32floorrows. Preserve these caveats.

Actualpilot full numerical metadata and8gradientrows:
{
  "sparse_gradients": [
    {
      "update": 1,
      "minibatch": 1,
      "raw_gradient": {
        "norms": {
          "ppo_actor": 4.368802547454834,
          "quiet_temporal": 0.6427795886993408,
          "moving_temporal": 0.003279456403106451,
          "spatial": 0.0006759038078598678
        },
        "ppo_quiet_cosine": 0.11798661202192307,
        "component_sum_norm": 4.491004943847656,
        "combined_actor_before_clip": 4.491004943847656,
        "combined_actor_after_clip": 0.9999997615814209
      },
      "quiet_norm_over_ppo_actor_norm": 0.14712946664843202,
      "quiet_norm_over_spatial_norm": 950.9927022523973,
      "quiet_norm_over_moving_temporal_norm": 196.00187033755674,
      "common_actor_clip_scale": 0.22266725912901664,
      "quiet_component_norm_after_common_clip": 0.14312596923975884,
      "quiet_dot_total_gradient_interval": [
        0.7419504773636938,
        0.7470353269828192
      ],
      "dot_bound_method": "Q dot (P+Q) from measured norms/cosine; absolute Q dot (M+S) bounded by ||Q|| (||M||+||S||). Applies before Adam, not its parameter update.",
      "interpretation": "Weighted loss-gradient components, not measured Adam parameter-update shares"
    },
    {
      "update": 1,
      "minibatch": 20,
      "raw_gradient": {
        "norms": {
          "ppo_actor": 4.42371940612793,
          "quiet_temporal": 0.6439428925514221,
          "moving_temporal": 0.003066213568672538,
          "spatial": 0.0006622704677283764
        },
        "ppo_quiet_cosine": -0.128989577293396,
        "component_sum_norm": 4.3874921798706055,
        "combined_actor_before_clip": 4.3874921798706055,
        "combined_actor_after_clip": 0.9999998211860657
      },
      "quiet_norm_over_ppo_actor_norm": 0.14556594427291303,
      "quiet_norm_over_spatial_norm": 972.3261475937182,
      "quiet_norm_over_moving_temporal_norm": 210.01240720169588,
      "common_actor_clip_scale": 0.22792059340275733,
      "quiet_component_norm_after_common_clip": 0.14676784618780814,
      "quiet_dot_total_gradient_interval": [
        0.044818883972471454,
        0.04962074556293507
      ],
      "dot_bound_method": "Q dot (P+Q) from measured norms/cosine; absolute Q dot (M+S) bounded by ||Q|| (||M||+||S||). Applies before Adam, not its parameter update.",
      "interpretation": "Weighted loss-gradient components, not measured Adam parameter-update shares"
    },
    {
      "update": 10,
      "minibatch": 1,
      "raw_gradient": {
        "norms": {
          "ppo_actor": 7.5193772315979,
          "quiet_temporal": 0.6685775518417358,
          "moving_temporal": 0.01572003774344921,
          "spatial": 0.00045796166523359716
        },
        "ppo_quiet_cosine": -0.040818121284246445,
        "component_sum_norm": 7.522863388061523,
        "combined_actor_before_clip": 7.522863388061523,
        "combined_actor_after_clip": 0.9999998211860657
      },
      "quiet_norm_over_ppo_actor_norm": 0.08891395274494829,
      "quiet_norm_over_spatial_norm": 1459.898508100471,
      "quiet_norm_over_moving_temporal_norm": 42.53027650142524,
      "common_actor_clip_scale": 0.13292808464035444,
      "quiet_component_norm_after_common_clip": 0.08887273339985922,
      "quiet_dot_total_gradient_interval": [
        0.23097529240311576,
        0.25260778687982416
      ],
      "dot_bound_method": "Q dot (P+Q) from measured norms/cosine; absolute Q dot (M+S) bounded by ||Q|| (||M||+||S||). Applies before Adam, not its parameter update.",
      "interpretation": "Weighted loss-gradient components, not measured Adam parameter-update shares"
    },
    {
      "update": 10,
      "minibatch": 20,
      "raw_gradient": {
        "norms": {
          "ppo_actor": 8.99681568145752,
          "quiet_temporal": 0.6823315024375916,
          "moving_temporal": 0.014587539248168468,
          "spatial": 0.0004510105645749718
        },
        "ppo_quiet_cosine": -0.1942276656627655,
        "component_sum_norm": 8.887471199035645,
        "combined_actor_before_clip": 8.887471199035645,
        "combined_actor_after_clip": 0.9999998211860657
      },
      "quiet_norm_over_ppo_actor_norm": 0.07584144508416239,
      "quiet_norm_over_spatial_norm": 1512.8947213922017,
      "quiet_norm_over_moving_temporal_norm": 46.774955722793436,
      "common_actor_clip_scale": 0.11251792537955824,
      "quiet_component_norm_after_common_clip": 0.07677452507539478,
      "quiet_dot_total_gradient_interval": [
        -0.7370118811400855,
        -0.7164893285636619
      ],
      "dot_bound_method": "Q dot (P+Q) from measured norms/cosine; absolute Q dot (M+S) bounded by ||Q|| (||M||+||S||). Applies before Adam, not its parameter update.",
      "interpretation": "Weighted loss-gradient components, not measured Adam parameter-update shares"
    },
    {
      "update": 25,
      "minibatch": 1,
      "raw_gradient": {
        "norms": {
          "ppo_actor": 6.610074996948242,
          "quiet_temporal": 0.5003265142440796,
          "moving_temporal": 0.02923409640789032,
          "spatial": 0.00042903985013253987
        },
        "ppo_quiet_cosine": 0.032222211360931396,
        "component_sum_norm": 6.649167537689209,
        "combined_actor_before_clip": 6.649167537689209,
        "combined_actor_after_clip": 0.9999997615814209
      },
      "quiet_norm_over_ppo_actor_norm": 0.07569150342092514,
      "quiet_norm_over_spatial_norm": 1166.1539460484096,
      "quiet_norm_over_moving_temporal_norm": 17.114485334632775,
      "common_actor_clip_scale": 0.15039473075586718,
      "quiet_component_norm_after_common_clip": 0.0752464713997599,
      "quiet_dot_total_gradient_interval": [
        0.3420505287933997,
        0.3717330359244472
      ],
      "dot_bound_method": "Q dot (P+Q) from measured norms/cosine; absolute Q dot (M+S) bounded by ||Q|| (||M||+||S||). Applies before Adam, not its parameter update.",
      "interpretation": "Weighted loss-gradient components, not measured Adam parameter-update shares"
    },
    {
      "update": 25,
      "minibatch": 20,
      "raw_gradient": {
        "norms": {
          "ppo_actor": 6.622532844543457,
          "quiet_temporal": 0.4958641529083252,
          "moving_temporal": 0.02852238155901432,
          "spatial": 0.0004309850337449461
        },
        "ppo_quiet_cosine": -0.1280277967453003,
        "component_sum_norm": 6.575809001922607,
        "combined_actor_before_clip": 6.575809001922607,
        "combined_actor_after_clip": 0.9999998807907104
      },
      "quiet_norm_over_ppo_actor_norm": 0.07487530293139814,
      "quiet_norm_over_spatial_norm": 1150.5368263014304,
      "quiet_norm_over_moving_temporal_norm": 17.38508938611441,
      "common_actor_clip_scale": 0.15207252529663418,
      "quiet_component_norm_after_common_clip": 0.07540731393684535,
      "quiet_dot_total_gradient_interval": [
        -0.18890316934299473,
        -0.1601892961442692
      ],
      "dot_bound_method": "Q dot (P+Q) from measured norms/cosine; absolute Q dot (M+S) bounded by ||Q|| (||M||+||S||). Applies before Adam, not its parameter update.",
      "interpretation": "Weighted loss-gradient components, not measured Adam parameter-update shares"
    },
    {
      "update": 50,
      "minibatch": 1,
      "raw_gradient": {
        "norms": {
          "ppo_actor": 7.596764087677002,
          "quiet_temporal": 0.5010349750518799,
          "moving_temporal": 0.024538783356547356,
          "spatial": 0.000423260877141729
        },
        "ppo_quiet_cosine": -0.0950477123260498,
        "component_sum_norm": 7.565184593200684,
        "combined_actor_before_clip": 7.565184593200684,
        "combined_actor_after_clip": 0.9999998807907104
      },
      "quiet_norm_over_ppo_actor_norm": 0.06595373625786638,
      "quiet_norm_over_spatial_norm": 1183.749791465154,
      "quiet_norm_over_moving_temporal_norm": 20.418085435282816,
      "common_actor_clip_scale": 0.13218446535851544,
      "quiet_component_norm_after_common_clip": 0.06622904030314987,
      "quiet_dot_total_gradient_interval": [
        -0.12324564375219221,
        -0.09823192933245156
      ],
      "dot_bound_method": "Q dot (P+Q) from measured norms/cosine; absolute Q dot (M+S) bounded by ||Q|| (||M||+||S||). Applies before Adam, not its parameter update.",
      "interpretation": "Weighted loss-gradient components, not measured Adam parameter-update shares"
    },
    {
      "update": 50,
      "minibatch": 20,
      "raw_gradient": {
        "norms": {
          "ppo_actor": 8.178229331970215,
          "quiet_temporal": 0.48637551069259644,
          "moving_temporal": 0.026208356022834778,
          "spatial": 0.00041318315197713673
        },
        "ppo_quiet_cosine": 0.03508802130818367,
        "component_sum_norm": 8.2115478515625,
        "combined_actor_before_clip": 8.2115478515625,
        "combined_actor_after_clip": 0.9999998807907104
      },
      "quiet_norm_over_ppo_actor_norm": 0.05947198237536142,
      "quiet_norm_over_spatial_norm": 1177.1426505781383,
      "quiet_norm_over_moving_temporal_norm": 18.558032036379082,
      "common_actor_clip_scale": 0.12177970571046842,
      "quiet_component_norm_after_common_clip": 0.05923066655692318,
      "quiet_dot_total_gradient_interval": [
        0.36318236058488246,
        0.3890784900080267
      ],
      "dot_bound_method": "Q dot (P+Q) from measured norms/cosine; absolute Q dot (M+S) bounded by ||Q|| (||M||+||S||). Applies before Adam, not its parameter update.",
      "interpretation": "Weighted loss-gradient components, not measured Adam parameter-update shares"
    }
  ],
  "loss_change_percent": {
    "caps_quiet_temporal_mean": -14.277451862473322,
    "caps_moving_temporal_mean": -11.645110243249112,
    "caps_temporal": -11.389297407762689,
    "caps_spatial": -38.617074542157404,
    "caps_weighted": -38.85552679494152
  },
  "kl_mean_all1000": 0.0139089923203054,
  "kl_above_0_02_all1000": 108,
  "all1000_adaptive_learning_rate_transitions_replayed": true,
  "final_checkpoint_sha256": "195593ca4d595184fefbc9161bad433f9fb9768d8fd16d514c7b755eee0a6173",
  "strict_reload_reported": true,
  "checkpoint_reload_independently_reexecuted_here": false,
  "disjoint_window_summary": {
    "first5": {
      "update_ids": [
        1,
        2,
        3,
        4,
        5
      ],
      "loss_means": {
        "value": 0.078128971606493,
        "surrogate": -0.004607262530480512,
        "entropy": -15.914464139938355,
        "caps_temporal": 0.315807569026947,
        "caps_spatial": 0.0017404619429726154,
        "caps_weighted": 0.25857460021972656,
        "caps_valid_pair_fraction": 0.9336100369691849,
        "caps_quiet_temporal_mean": 0.31392232999578673,
        "caps_moving_temporal_mean": 0.3251960021959132
      }
    },
    "last5": {
      "update_ids": [
        46,
        47,
        48,
        49,
        50
      ],
      "loss_means": {
        "value": 0.1357662282139063,
        "surrogate": -0.00762246471032995,
        "entropy": -16.291588916778565,
        "caps_temporal": 0.2703834292292595,
        "caps_spatial": 0.001410242358688265,
        "caps_weighted": 0.17962782889604567,
        "caps_valid_pair_fraction": 0.9560709893703461,
        "caps_quiet_temporal_mean": 0.2607343629279105,
        "caps_moving_temporal_mean": 0.2882734611899009
      }
    }
  },
  "last5_over_first5_change_percent": {
    "caps_quiet_temporal_mean": -16.943033988244814,
    "caps_moving_temporal_mean": -11.353934475420912,
    "caps_temporal": -14.38348673454739,
    "caps_weighted": -30.531526010905562,
    "caps_spatial": -18.973099964504424
  },
  "lr_summary": {
    "min": 1e-05,
    "max": 7.593750000000004e-05,
    "floor_exact_count": 74,
    "floor_roundoff_count": 186,
    "final": 1.5000000000000004e-05
  },
  "schema": "direct_quiet_pilot50_training_only_review_v1",
  "training_receipt_sha256": "c41560dbe34d29bbce40d1060bd1c8b0155cdcc27173dc2316806f8368ad9b11",
  "source_manifest_sha256": "ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62",
  "checkpoint_sha256_verified_as_receipt_binding_only": "195593ca4d595184fefbc9161bad433f9fb9768d8fd16d514c7b755eee0a6173",
  "checkpoint_bytes_locally_verified": false,
  "campaign_status_at_supplied_observation": "completed",
  "accepted_train_receipt": {
    "phase": "train",
    "state_sha256": "681fa31e039e1f5930e31331646f13aa49aba82ca282f584cb25177fb7b0699d",
    "source_manifest_sha256": "ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62",
    "Stage2_complete": false,
    "complete": true,
    "optimizer_diagnostics": {
      "schema": "direct315_actor_gradients_v1",
      "updates": 50,
      "minibatches": 1000,
      "sparse_actor_gradient_rows": 8
    },
    "updates_completed": 50,
    "checkpoint_sha256": "195593ca4d595184fefbc9161bad433f9fb9768d8fd16d514c7b755eee0a6173",
    "receipt_sha256": "c41560dbe34d29bbce40d1060bd1c8b0155cdcc27173dc2316806f8368ad9b11",
    "scope": "Bounded optimization/integrity, not physical acceptance"
  },
  "full_terminal_source_cleanup_and_physical_scores_reviewed_here": false,
  "scope": "Actual pilot training receipt and retained optimizer diagnostics only; no physical qualification",
  "Stage2_complete": false
}

All50update summaries (paircountsarepresentations including5epochs):
[
  {
    "update": 1,
    "losses": {
      "value": 0.28123817294836045,
      "surrogate": 0.010855457978323102,
      "entropy": -15.904798364639282,
      "caps_temporal": 0.30196508169174197,
      "caps_spatial": 0.002294726495165378,
      "caps_weighted": 0.29119162559509276,
      "caps_valid_pair_fraction": 0.7461344599723816,
      "caps_quiet_temporal_mean": 0.3011479266976948,
      "caps_moving_temporal_mean": 0.3226886489494241
    },
    "pair_presentations": {
      "valid_pairs": 91685,
      "quiet_pairs": 88210,
      "moving_pairs": 3475
    },
    "quiet_fraction_of_valid_pair_presentations": 0.9620984893930304,
    "kl_min": 1.7158648688564426e-06,
    "kl_max": 0.2620846629142761,
    "kl_mean": 0.04955477567370394,
    "kl_above_adaptation_threshold_0_02": 12,
    "lr_after_exact_literal_floor": 15,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 15,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.28973960530711335,
      "moving_temporal": 0.001222547638462862,
      "spatial": 0.0002294726495165378
    }
  },
  {
    "update": 2,
    "losses": {
      "value": 0.026323551964014768,
      "surrogate": -0.009672414882516023,
      "entropy": -15.905768013000488,
      "caps_temporal": 0.3376327857375145,
      "caps_spatial": 0.0016411419841460884,
      "caps_weighted": 0.27431140244007113,
      "caps_valid_pair_fraction": 0.9486897885799408,
      "caps_quiet_temporal_mean": 0.33821946974781425,
      "caps_moving_temporal_mean": 0.3354168889487541
    },
    "pair_presentations": {
      "valid_pairs": 116575,
      "quiet_pairs": 92060,
      "moving_pairs": 24515
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7897061977267853,
    "kl_min": 6.169536845845869e-06,
    "kl_max": 0.014346424490213394,
    "kl_mean": 0.010161578107613423,
    "kl_above_adaptation_threshold_0_02": 0,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.2670933440754501,
      "moving_temporal": 0.007053944166206439,
      "spatial": 0.00016411419841460886
    }
  },
  {
    "update": 3,
    "losses": {
      "value": 0.028899954911321402,
      "surrogate": -0.007380327774444595,
      "entropy": -15.914419507980346,
      "caps_temporal": 0.3191551610827446,
      "caps_spatial": 0.0016055080515798182,
      "caps_weighted": 0.24854319915175438,
      "caps_valid_pair_fraction": 0.9865315854549408,
      "caps_quiet_temporal_mean": 0.3167330718818749,
      "caps_moving_temporal_mean": 0.32679634685585013
    },
    "pair_presentations": {
      "valid_pairs": 121225,
      "quiet_pairs": 92055,
      "moving_pairs": 29170
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7593730666116725,
    "kl_min": 8.991590220830403e-06,
    "kl_max": 0.02524697780609131,
    "kl_mean": 0.010778887448668683,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.2405190358203577,
      "moving_temporal": 0.007863612526238692,
      "spatial": 0.00016055080515798182
    }
  },
  {
    "update": 4,
    "losses": {
      "value": 0.028981720376759768,
      "surrogate": -0.009424467333883513,
      "entropy": -15.921150588989258,
      "caps_temporal": 0.3117829293012619,
      "caps_spatial": 0.0015934749622829258,
      "caps_weighted": 0.24113751202821732,
      "caps_valid_pair_fraction": 0.9912923276424408,
      "caps_quiet_temporal_mean": 0.30847528758416803,
      "caps_moving_temporal_mean": 0.32201723767624746
    },
    "pair_presentations": {
      "valid_pairs": 121810,
      "quiet_pairs": 92050,
      "moving_pairs": 29760
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7556850833264921,
    "kl_min": 1.0587488759483676e-05,
    "kl_max": 0.018256789073348045,
    "kl_mean": 0.010989618114172118,
    "kl_above_adaptation_threshold_0_02": 0,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.23311096844651424,
      "moving_temporal": 0.007867196085474764,
      "spatial": 0.0001593474962282926
    }
  },
  {
    "update": 5,
    "losses": {
      "value": 0.0252014578320086,
      "surrogate": -0.007414560639881529,
      "entropy": -15.926184225082398,
      "caps_temporal": 0.30850188732147216,
      "caps_spatial": 0.0015674582216888666,
      "caps_weighted": 0.23768926188349723,
      "caps_valid_pair_fraction": 0.9954020231962204,
      "caps_quiet_temporal_mean": 0.3050358940673818,
      "caps_moving_temporal_mean": 0.31906088854929016
    },
    "pair_presentations": {
      "valid_pairs": 122315,
      "quiet_pairs": 92085,
      "moving_pairs": 30230
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7528512447369496,
    "kl_min": 1.1641319360933267e-05,
    "kl_max": 0.019716039299964905,
    "kl_mean": 0.013880157553194295,
    "kl_above_adaptation_threshold_0_02": 0,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.22964703036575684,
      "moving_temporal": 0.007885485695571532,
      "spatial": 0.00015674582216888667
    }
  },
  {
    "update": 6,
    "losses": {
      "value": 0.02201360324397683,
      "surrogate": -0.0070016078709159045,
      "entropy": -15.936582565307617,
      "caps_temporal": 0.30931996256113053,
      "caps_spatial": 0.0015609133522957563,
      "caps_weighted": 0.23842867612838745,
      "caps_valid_pair_fraction": 0.9944254606962204,
      "caps_quiet_temporal_mean": 0.30582423204478204,
      "caps_moving_temporal_mean": 0.3199952283226291
    },
    "pair_presentations": {
      "valid_pairs": 122195,
      "quiet_pairs": 92050,
      "moving_pairs": 30145
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7533041450141168,
    "kl_min": 1.1713690582837444e-05,
    "kl_max": 0.025990262627601624,
    "kl_mean": 0.015052695315307573,
    "kl_above_adaptation_threshold_0_02": 1,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.23037843170782757,
      "moving_temporal": 0.007894153085330298,
      "spatial": 0.00015609133522957565
    }
  },
  {
    "update": 7,
    "losses": {
      "value": 0.022533556539565326,
      "surrogate": -0.004823547851992771,
      "entropy": -15.953128147125245,
      "caps_temporal": 0.30890984237194063,
      "caps_spatial": 0.0015672316134441645,
      "caps_weighted": 0.23827045261859894,
      "caps_valid_pair_fraction": 0.9938558042049408,
      "caps_quiet_temporal_mean": 0.30544191003687454,
      "caps_moving_temporal_mean": 0.31952815643864385
    },
    "pair_presentations": {
      "valid_pairs": 122125,
      "quiet_pairs": 92060,
      "moving_pairs": 30065
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7538178096212896,
    "kl_min": 1.1192906640644651e-05,
    "kl_max": 0.04168939217925072,
    "kl_mean": 0.014202234861068063,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.23024749468895606,
      "moving_temporal": 0.007866234768298458,
      "spatial": 0.00015672316134441645
    }
  },
  {
    "update": 8,
    "losses": {
      "value": 0.020671247132122518,
      "surrogate": -0.009427471106755547,
      "entropy": -15.974708795547485,
      "caps_temporal": 0.3083159029483795,
      "caps_spatial": 0.0015627060842234642,
      "caps_weighted": 0.2378319062292576,
      "caps_valid_pair_fraction": 0.9941813349723816,
      "caps_quiet_temporal_mean": 0.30486519008177526,
      "caps_moving_temporal_mean": 0.31887733608850977
    },
    "pair_presentations": {
      "valid_pairs": 122165,
      "quiet_pairs": 92095,
      "moving_pairs": 30070
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7538574878238448,
    "kl_min": 1.0984300388372503e-05,
    "kl_max": 0.019935201853513718,
    "kl_mean": 0.014080885066778136,
    "kl_above_adaptation_threshold_0_02": 0,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.22982671702888588,
      "moving_temporal": 0.007848918591949364,
      "spatial": 0.00015627060842234643
    }
  },
  {
    "update": 9,
    "losses": {
      "value": 0.8342456132173538,
      "surrogate": -0.007914079970214516,
      "entropy": -15.99676423072815,
      "caps_temporal": 0.306143245100975,
      "caps_spatial": 0.0015586945693939925,
      "caps_weighted": 0.2618439868092537,
      "caps_valid_pair_fraction": 0.7399902641773224,
      "caps_quiet_temporal_mean": 0.3037299292275833,
      "caps_moving_temporal_mean": 0.31936150410825404
    },
    "pair_presentations": {
      "valid_pairs": 90930,
      "quiet_pairs": 76865,
      "moving_pairs": 14065
    },
    "quiet_fraction_of_valid_pair_presentations": 0.8453205762674585,
    "kl_min": 1.0685520464903675e-05,
    "kl_max": 0.02512200176715851,
    "kl_mean": 0.013841825839608645,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.2567486587135742,
      "moving_temporal": 0.00493945863874008,
      "spatial": 0.00015586945693939926
    }
  },
  {
    "update": 10,
    "losses": {
      "value": 0.16837586909532548,
      "surrogate": -0.0098424136871472,
      "entropy": -16.006941890716554,
      "caps_temporal": 0.3028184324502945,
      "caps_spatial": 0.0015517494815867395,
      "caps_weighted": 0.24903305023908615,
      "caps_valid_pair_fraction": 0.8331706076860428,
      "caps_quiet_temporal_mean": 0.3004096192281724,
      "caps_moving_temporal_mean": 0.31297845862496115
    },
    "pair_presentations": {
      "valid_pairs": 102380,
      "quiet_pairs": 82775,
      "moving_pairs": 19605
    },
    "quiet_fraction_of_valid_pair_presentations": 0.8085075210001954,
    "kl_min": 9.96293056232389e-06,
    "kl_max": 0.020301474258303642,
    "kl_mean": 0.01617045102711927,
    "kl_above_adaptation_threshold_0_02": 1,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.2428844800509978,
      "moving_temporal": 0.005993395239929669,
      "spatial": 0.00015517494815867397
    }
  },
  {
    "update": 11,
    "losses": {
      "value": 0.0499084847047925,
      "surrogate": -0.009230245649814606,
      "entropy": -16.00630111694336,
      "caps_temporal": 0.30338945984840393,
      "caps_spatial": 0.0015408387291245163,
      "caps_weighted": 0.2360553041100502,
      "caps_valid_pair_fraction": 0.9784749448299408,
      "caps_quiet_temporal_mean": 0.2985142901734986,
      "caps_moving_temporal_mean": 0.3192690199847853
    },
    "pair_presentations": {
      "valid_pairs": 120235,
      "quiet_pairs": 91995,
      "moving_pairs": 28240
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7651266270220818,
    "kl_min": 1.0571219718258362e-05,
    "kl_max": 0.02076728269457817,
    "kl_mean": 0.012229724682902088,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.22840252694699706,
      "moving_temporal": 0.007498693290140687,
      "spatial": 0.00015408387291245164
    }
  },
  {
    "update": 12,
    "losses": {
      "value": 0.06977077219635248,
      "surrogate": -0.007361770342686214,
      "entropy": -16.006091403961182,
      "caps_temporal": 0.30242019295692446,
      "caps_spatial": 0.0015424425771925598,
      "caps_weighted": 0.2357798807322979,
      "caps_valid_pair_fraction": 0.9763590693473816,
      "caps_quiet_temporal_mean": 0.2973234377693404,
      "caps_moving_temporal_mean": 0.31925073674008086
    },
    "pair_presentations": {
      "valid_pairs": 119975,
      "quiet_pairs": 92085,
      "moving_pairs": 27890
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7675349031048135,
    "kl_min": 1.0776168892334681e-05,
    "kl_max": 0.022508997470140457,
    "kl_mean": 0.013209434427471934,
    "kl_above_adaptation_threshold_0_02": 1,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.2282040190876513,
      "moving_temporal": 0.007421617386927315,
      "spatial": 0.00015424425771925598
    }
  },
  {
    "update": 13,
    "losses": {
      "value": 0.05421048142015934,
      "surrogate": -0.007812914659734816,
      "entropy": -16.01216325759888,
      "caps_temporal": 0.30031781941652297,
      "caps_spatial": 0.0015392229484859853,
      "caps_weighted": 0.23355660066008568,
      "caps_valid_pair_fraction": 0.981038436293602,
      "caps_quiet_temporal_mean": 0.2952426395529054,
      "caps_moving_temporal_mean": 0.31687143949921187
    },
    "pair_presentations": {
      "valid_pairs": 120550,
      "quiet_pairs": 92265,
      "moving_pairs": 28285
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7653670676068022,
    "kl_min": 1.0915430721070152e-05,
    "kl_max": 0.020185597240924835,
    "kl_mean": 0.012847438424569192,
    "kl_above_adaptation_threshold_0_02": 1,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.225967662692872,
      "moving_temporal": 0.007435015672365097,
      "spatial": 0.00015392229484859855
    }
  },
  {
    "update": 14,
    "losses": {
      "value": 0.034962053131312135,
      "surrogate": -0.006373213074402884,
      "entropy": -16.02339153289795,
      "caps_temporal": 0.30057342499494555,
      "caps_spatial": 0.0015308529778849333,
      "caps_weighted": 0.23396380320191384,
      "caps_valid_pair_fraction": 0.9868977963924408,
      "caps_quiet_temporal_mean": 0.29632642546764143,
      "caps_moving_temporal_mean": 0.31431459529282413
    },
    "pair_presentations": {
      "valid_pairs": 121270,
      "quiet_pairs": 92650,
      "moving_pairs": 28620
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7639976911024986,
    "kl_min": 1.0915586244664155e-05,
    "kl_max": 0.03106437250971794,
    "kl_mean": 0.012170466967927496,
    "kl_above_adaptation_threshold_0_02": 4,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 15,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.22639263933847867,
      "moving_temporal": 0.007418078565646688,
      "spatial": 0.00015308529778849334
    }
  },
  {
    "update": 15,
    "losses": {
      "value": 0.06530113313347101,
      "surrogate": -0.007588161685271188,
      "entropy": -16.025729274749757,
      "caps_temporal": 0.2996531590819359,
      "caps_spatial": 0.0015308626636397094,
      "caps_weighted": 0.23312984332442283,
      "caps_valid_pair_fraction": 0.9845377802848816,
      "caps_quiet_temporal_mean": 0.2949561267823844,
      "caps_moving_temporal_mean": 0.3149250377213042
    },
    "pair_presentations": {
      "valid_pairs": 120980,
      "quiet_pairs": 92520,
      "moving_pairs": 28460
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7647545048768392,
    "kl_min": 1.0499352356418967e-05,
    "kl_max": 0.021930545568466187,
    "kl_mean": 0.011273495900968555,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.22556826794429471,
      "moving_temporal": 0.0074084891137641195,
      "spatial": 0.00015308626636397095
    }
  },
  {
    "update": 16,
    "losses": {
      "value": 0.04267026074230671,
      "surrogate": -0.007991883729118854,
      "entropy": -16.035923862457274,
      "caps_temporal": 0.2978382021188736,
      "caps_spatial": 0.0015362097125034778,
      "caps_weighted": 0.2322410136461258,
      "caps_valid_pair_fraction": 0.9801432490348816,
      "caps_quiet_temporal_mean": 0.29345712754521086,
      "caps_moving_temporal_mean": 0.3121737448304643
    },
    "pair_presentations": {
      "valid_pairs": 120440,
      "quiet_pairs": 92255,
      "moving_pairs": 28185
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7659830621056127,
    "kl_min": 9.995138498197775e-06,
    "kl_max": 0.021806985139846802,
    "kl_mean": 0.012723994788166238,
    "kl_above_adaptation_threshold_0_02": 1,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.22478174718109786,
      "moving_temporal": 0.007305645493777571,
      "spatial": 0.0001536209712503478
    }
  },
  {
    "update": 17,
    "losses": {
      "value": 0.8095290064811707,
      "surrogate": -0.004484684590715915,
      "entropy": -16.0418981552124,
      "caps_temporal": 0.29637355357408524,
      "caps_spatial": 0.0015213992795906962,
      "caps_weighted": 0.22529637143015863,
      "caps_valid_pair_fraction": 0.9312744438648224,
      "caps_quiet_temporal_mean": 0.2910026130241576,
      "caps_moving_temporal_mean": 0.3121889962025323
    },
    "pair_presentations": {
      "valid_pairs": 114435,
      "quiet_pairs": 85425,
      "moving_pairs": 29010
    },
    "quiet_fraction_of_valid_pair_presentations": 0.7464936426792502,
    "kl_min": 1.0020982699643355e-05,
    "kl_max": 0.02236345410346985,
    "kl_mean": 0.012206233756069195,
    "kl_above_adaptation_threshold_0_02": 1,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.21722986238310116,
      "moving_temporal": 0.007914369119098407,
      "spatial": 0.00015213992795906964
    }
  },
  {
    "update": 18,
    "losses": {
      "value": 0.09894059821963311,
      "surrogate": -0.006698287790641189,
      "entropy": -16.04946012496948,
      "caps_temporal": 0.2963944286108017,
      "caps_spatial": 0.0014974582300055772,
      "caps_weighted": 0.20519720390439034,
      "caps_valid_pair_fraction": 0.8770752251148224,
      "caps_quiet_temporal_mean": 0.2911365761702373,
      "caps_moving_temporal_mean": 0.3070364872748392
    },
    "pair_presentations": {
      "valid_pairs": 107775,
      "quiet_pairs": 72150,
      "moving_pairs": 35625
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6694502435629784,
    "kl_min": 9.858107659965754e-06,
    "kl_max": 0.03171570599079132,
    "kl_mean": 0.013875719861243852,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.19489779468923288,
      "moving_temporal": 0.01014966339215688,
      "spatial": 0.00014974582300055774
    }
  },
  {
    "update": 19,
    "losses": {
      "value": 0.05677484478801489,
      "surrogate": -0.008971164072863757,
      "entropy": -16.06757926940918,
      "caps_temporal": 0.2977936014533043,
      "caps_spatial": 0.0014833194494713099,
      "caps_weighted": 0.18798418045043946,
      "caps_valid_pair_fraction": 0.9715576171875,
      "caps_quiet_temporal_mean": 0.28992948278486996,
      "caps_moving_temporal_mean": 0.30987238245441423
    },
    "pair_presentations": {
      "valid_pairs": 119385,
      "quiet_pairs": 72315,
      "moving_pairs": 47070
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6057293629852997,
    "kl_min": 9.841965038503986e-06,
    "kl_max": 0.01802857778966427,
    "kl_mean": 0.014074654818023192,
    "kl_above_adaptation_threshold_0_02": 0,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17561832040017988,
      "moving_temporal": 0.012217528105312443,
      "spatial": 0.000148331944947131
    }
  },
  {
    "update": 20,
    "losses": {
      "value": 0.07941435687243939,
      "surrogate": -0.0051121627177053595,
      "entropy": -16.08531322479248,
      "caps_temporal": 0.29668949842453,
      "caps_spatial": 0.001481483137467876,
      "caps_weighted": 0.1877080552279949,
      "caps_valid_pair_fraction": 0.9777832180261612,
      "caps_quiet_temporal_mean": 0.2893016686879309,
      "caps_moving_temporal_mean": 0.3080740505600611
    },
    "pair_presentations": {
      "valid_pairs": 120150,
      "quiet_pairs": 72860,
      "moving_pairs": 47290
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6064086558468581,
    "kl_min": 9.766314178705215e-06,
    "kl_max": 0.03533576801419258,
    "kl_mean": 0.014065812132321298,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17543439674643901,
      "moving_temporal": 0.0121255101678091,
      "spatial": 0.00014814831374678763
    }
  },
  {
    "update": 21,
    "losses": {
      "value": 0.07511111088097096,
      "surrogate": -0.006574493221705779,
      "entropy": -16.092598819732665,
      "caps_temporal": 0.2964966967701912,
      "caps_spatial": 0.0014759036537725478,
      "caps_weighted": 0.1884262040257454,
      "caps_valid_pair_fraction": 0.9736735075712204,
      "caps_quiet_temporal_mean": 0.28834335687971085,
      "caps_moving_temporal_mean": 0.3093132475394635
    },
    "pair_presentations": {
      "valid_pairs": 119645,
      "quiet_pairs": 73135,
      "moving_pairs": 46510
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6112666638806469,
    "kl_min": 9.476741070102435e-06,
    "kl_max": 0.022078249603509903,
    "kl_mean": 0.011611953090095994,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 15,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.1762543822037211,
      "moving_temporal": 0.012024231456647008,
      "spatial": 0.00014759036537725478
    }
  },
  {
    "update": 22,
    "losses": {
      "value": 0.09298398159444332,
      "surrogate": -0.00679889734601602,
      "entropy": -16.098248100280763,
      "caps_temporal": 0.2955773338675499,
      "caps_spatial": 0.0014701794076245277,
      "caps_weighted": 0.18833934217691423,
      "caps_valid_pair_fraction": 0.9771728664636612,
      "caps_quiet_temporal_mean": 0.28764902766096295,
      "caps_moving_temporal_mean": 0.30812307471279843
    },
    "pair_presentations": {
      "valid_pairs": 120075,
      "quiet_pairs": 73580,
      "moving_pairs": 46495
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6127836768686238,
    "kl_min": 9.757757652550936e-06,
    "kl_max": 0.022968441247940063,
    "kl_mean": 0.011724896961823106,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17626065649932976,
      "moving_temporal": 0.011931667736822017,
      "spatial": 0.0001470179407624528
    }
  },
  {
    "update": 23,
    "losses": {
      "value": 0.10833143405616283,
      "surrogate": -0.006703263754025102,
      "entropy": -16.097524452209473,
      "caps_temporal": 0.29354301542043687,
      "caps_spatial": 0.001470066385809332,
      "caps_weighted": 0.1873355083167553,
      "caps_valid_pair_fraction": 0.9759928733110428,
      "caps_quiet_temporal_mean": 0.28579922430880417,
      "caps_moving_temporal_mean": 0.3058422160935803
    },
    "pair_presentations": {
      "valid_pairs": 119930,
      "quiet_pairs": 73590,
      "moving_pairs": 46340
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6136079379638122,
    "kl_min": 9.794573998078704e-06,
    "kl_max": 0.02106737159192562,
    "kl_mean": 0.013275378492835443,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17537133348458964,
      "moving_temporal": 0.011817168193584724,
      "spatial": 0.00014700663858093323
    }
  },
  {
    "update": 24,
    "losses": {
      "value": 0.07263490278273821,
      "surrogate": -0.005439464050868992,
      "entropy": -16.09598054885864,
      "caps_temporal": 0.2926799699664116,
      "caps_spatial": 0.001473475934471935,
      "caps_weighted": 0.1871438205242157,
      "caps_valid_pair_fraction": 0.9764404594898224,
      "caps_quiet_temporal_mean": 0.2846124943726987,
      "caps_moving_temporal_mean": 0.30560672583425913
    },
    "pair_presentations": {
      "valid_pairs": 119985,
      "quiet_pairs": 73880,
      "moving_pairs": 46105
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6157436346209942,
    "kl_min": 9.708455763757229e-06,
    "kl_max": 0.03194205462932587,
    "kl_mean": 0.014398121781414374,
    "kl_above_adaptation_threshold_0_02": 5,
    "lr_after_exact_literal_floor": 15,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 16,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17525386214903035,
      "moving_temporal": 0.011742610781738126,
      "spatial": 0.00014734759344719352
    }
  },
  {
    "update": 25,
    "losses": {
      "value": 0.07374986074864864,
      "surrogate": -0.008761475794017315,
      "entropy": -16.099480152130127,
      "caps_temporal": 0.29125000536441803,
      "caps_spatial": 0.0014623526483774185,
      "caps_weighted": 0.18743984326720237,
      "caps_valid_pair_fraction": 0.9786784052848816,
      "caps_quiet_temporal_mean": 0.2839341568251778,
      "caps_moving_temporal_mean": 0.3031344859189046
    },
    "pair_presentations": {
      "valid_pairs": 120260,
      "quiet_pairs": 74435,
      "moving_pairs": 45825
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6189506070181274,
    "kl_min": 9.93228422885295e-06,
    "kl_max": 0.017180541530251503,
    "kl_mean": 0.011927080827717873,
    "kl_above_adaptation_threshold_0_02": 0,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.1757428971843587,
      "moving_temporal": 0.011550710818005933,
      "spatial": 0.00014623526483774186
    }
  },
  {
    "update": 26,
    "losses": {
      "value": 0.45726414769887924,
      "surrogate": -0.008307231997605413,
      "entropy": -16.11338129043579,
      "caps_temporal": 0.28972248584032056,
      "caps_spatial": 0.0014598605339415371,
      "caps_weighted": 0.1915689527988434,
      "caps_valid_pair_fraction": 0.7982991635799408,
      "caps_quiet_temporal_mean": 0.28235946419649494,
      "caps_moving_temporal_mean": 0.30275616723118814
    },
    "pair_presentations": {
      "valid_pairs": 98095,
      "quiet_pairs": 62705,
      "moving_pairs": 35390
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6392272796778633,
    "kl_min": 1.0086068868986331e-05,
    "kl_max": 0.02703407034277916,
    "kl_mean": 0.01536543680513205,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.1805007979571302,
      "moving_temporal": 0.010922168788319038,
      "spatial": 0.00014598605339415373
    }
  },
  {
    "update": 27,
    "losses": {
      "value": 0.19897944256663322,
      "surrogate": -0.008210018638055771,
      "entropy": -16.127361679077147,
      "caps_temporal": 0.2893700495362282,
      "caps_spatial": 0.0014572235231753439,
      "caps_weighted": 0.1855703189969063,
      "caps_valid_pair_fraction": 0.9315592646598816,
      "caps_quiet_temporal_mean": 0.28162365271887424,
      "caps_moving_temporal_mean": 0.3018664969954538
    },
    "pair_presentations": {
      "valid_pairs": 114470,
      "quiet_pairs": 70675,
      "moving_pairs": 43795
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6174106752861012,
    "kl_min": 1.0096157893713098e-05,
    "kl_max": 0.022359980270266533,
    "kl_mean": 0.011584896439080694,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 15,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17387510187885102,
      "moving_temporal": 0.011549494765737717,
      "spatial": 0.0001457223523175344
    }
  },
  {
    "update": 28,
    "losses": {
      "value": 0.1514060951769352,
      "surrogate": -0.00789535609073937,
      "entropy": -16.138551235198975,
      "caps_temporal": 0.2882180348038673,
      "caps_spatial": 0.0014551098982337862,
      "caps_weighted": 0.18327949196100235,
      "caps_valid_pair_fraction": 0.9517822563648224,
      "caps_quiet_temporal_mean": 0.28069304063632533,
      "caps_moving_temporal_mean": 0.3000299393927725
    },
    "pair_presentations": {
      "valid_pairs": 116955,
      "quiet_pairs": 71440,
      "moving_pairs": 45515
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6108332264546193,
    "kl_min": 1.0232441127300262e-05,
    "kl_max": 0.02217104844748974,
    "kl_mean": 0.014450366789242252,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17145797498976914,
      "moving_temporal": 0.011676005981409818,
      "spatial": 0.00014551098982337862
    }
  },
  {
    "update": 29,
    "losses": {
      "value": 0.14147876277565957,
      "surrogate": -0.008343863506161142,
      "entropy": -16.14748764038086,
      "caps_temporal": 0.28738644570112226,
      "caps_spatial": 0.0014563375385478138,
      "caps_weighted": 0.18311068415641785,
      "caps_valid_pair_fraction": 0.9566650539636612,
      "caps_quiet_temporal_mean": 0.2798523198033055,
      "caps_moving_temporal_mean": 0.2992933274649569
    },
    "pair_presentations": {
      "valid_pairs": 117555,
      "quiet_pairs": 71985,
      "moving_pairs": 45570
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6123516651780018,
    "kl_min": 1.028281076287385e-05,
    "kl_max": 0.023794321343302727,
    "kl_mean": 0.013303756368532049,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.1713626731471676,
      "moving_temporal": 0.011602377255395466,
      "spatial": 0.00014563375385478138
    }
  },
  {
    "update": 30,
    "losses": {
      "value": 0.12885707505047322,
      "surrogate": -0.006894929945701733,
      "entropy": -16.149920654296874,
      "caps_temporal": 0.28623071759939195,
      "caps_spatial": 0.001457399281207472,
      "caps_weighted": 0.18267495706677436,
      "caps_valid_pair_fraction": 0.9639892727136612,
      "caps_quiet_temporal_mean": 0.2783454894343835,
      "caps_moving_temporal_mean": 0.29879062934444267
    },
    "pair_presentations": {
      "valid_pairs": 118455,
      "quiet_pairs": 72775,
      "moving_pairs": 45680
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6143683255244607,
    "kl_min": 1.0184381608269177e-05,
    "kl_max": 0.0266548004001379,
    "kl_mean": 0.012521180319436099,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 15,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17100682819857158,
      "moving_temporal": 0.011522388940082038,
      "spatial": 0.0001457399281207472
    }
  },
  {
    "update": 31,
    "losses": {
      "value": 0.1455055482685566,
      "surrogate": -0.006673665057678591,
      "entropy": -16.16005458831787,
      "caps_temporal": 0.28450811058282854,
      "caps_spatial": 0.0014501961297355593,
      "caps_weighted": 0.18159045949578284,
      "caps_valid_pair_fraction": 0.9714762568473816,
      "caps_quiet_temporal_mean": 0.2768098849677695,
      "caps_moving_temporal_mean": 0.29676039410209365
    },
    "pair_presentations": {
      "valid_pairs": 119375,
      "quiet_pairs": 73310,
      "moving_pairs": 46065
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6141151832460733,
    "kl_min": 1.0318521162844263e-05,
    "kl_max": 0.019550248980522156,
    "kl_mean": 0.014258002812402993,
    "kl_above_adaptation_threshold_0_02": 0,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.1699940320272516,
      "moving_temporal": 0.011451407855557694,
      "spatial": 0.00014501961297355592
    }
  },
  {
    "update": 32,
    "losses": {
      "value": 0.11816748082637787,
      "surrogate": -0.0053602702857460825,
      "entropy": -16.17690029144287,
      "caps_temporal": 0.2838531360030174,
      "caps_spatial": 0.0014428485825192182,
      "caps_weighted": 0.1815421424806118,
      "caps_valid_pair_fraction": 0.9609782099723816,
      "caps_quiet_temporal_mean": 0.27567453365001815,
      "caps_moving_temporal_mean": 0.29701082080971386
    },
    "pair_presentations": {
      "valid_pairs": 118085,
      "quiet_pairs": 72825,
      "moving_pairs": 45260
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6167167718169115,
    "kl_min": 1.0061145985673647e-05,
    "kl_max": 0.029239822179079056,
    "kl_mean": 0.013593358014395563,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17001393780228682,
      "moving_temporal": 0.01138391982007306,
      "spatial": 0.00014428485825192182
    }
  },
  {
    "update": 33,
    "losses": {
      "value": 0.17720842435956002,
      "surrogate": -0.005195011885371059,
      "entropy": -16.180961227416994,
      "caps_temporal": 0.28248096257448196,
      "caps_spatial": 0.0014528811792843045,
      "caps_weighted": 0.18176143541932105,
      "caps_valid_pair_fraction": 0.9647216945886612,
      "caps_quiet_temporal_mean": 0.27437815533024346,
      "caps_moving_temporal_mean": 0.29576048806362776
    },
    "pair_presentations": {
      "valid_pairs": 118545,
      "quiet_pairs": 73625,
      "moving_pairs": 44920
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6210721666877557,
    "kl_min": 1.0033914804807864e-05,
    "kl_max": 0.03141358494758606,
    "kl_mean": 0.014278357328566926,
    "kl_above_adaptation_threshold_0_02": 4,
    "lr_after_exact_literal_floor": 14,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 15,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17040894560438266,
      "moving_temporal": 0.01120720169700993,
      "spatial": 0.00014528811792843046
    }
  },
  {
    "update": 34,
    "losses": {
      "value": 0.18095924630761145,
      "surrogate": -0.00923253346409183,
      "entropy": -16.178674125671385,
      "caps_temporal": 0.28178748935461045,
      "caps_spatial": 0.0014451414928771556,
      "caps_weighted": 0.18583284690976143,
      "caps_valid_pair_fraction": 0.940389022231102,
      "caps_quiet_temporal_mean": 0.27323918641438993,
      "caps_moving_temporal_mean": 0.2970184716378546
    },
    "pair_presentations": {
      "valid_pairs": 115555,
      "quiet_pairs": 74015,
      "moving_pairs": 41540
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6405175024879928,
    "kl_min": 9.961737305275165e-06,
    "kl_max": 0.016524076461791992,
    "kl_mean": 0.012996198200107755,
    "kl_above_adaptation_threshold_0_02": 0,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17501064869445854,
      "moving_temporal": 0.010677684066015193,
      "spatial": 0.00014451414928771556
    }
  },
  {
    "update": 35,
    "losses": {
      "value": 0.17954982444643974,
      "surrogate": -0.006900647410657257,
      "entropy": -16.184263706207275,
      "caps_temporal": 0.2809903100132942,
      "caps_spatial": 0.001442243281053379,
      "caps_weighted": 0.18988329097628592,
      "caps_valid_pair_fraction": 0.9506022185087204,
      "caps_quiet_temporal_mean": 0.2727745007389839,
      "caps_moving_temporal_mean": 0.29682664636382483
    },
    "pair_presentations": {
      "valid_pairs": 116810,
      "quiet_pairs": 76910,
      "moving_pairs": 39900
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6584196558513826,
    "kl_min": 1.003376928565558e-05,
    "kl_max": 0.02762998268008232,
    "kl_mean": 0.014023073840780853,
    "kl_above_adaptation_threshold_0_02": 4,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 14,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17960003960761237,
      "moving_temporal": 0.010139027040568186,
      "spatial": 0.0001442243281053379
    }
  },
  {
    "update": 36,
    "losses": {
      "value": 0.12259690277278423,
      "surrogate": -0.007302083959802985,
      "entropy": -16.192551040649413,
      "caps_temporal": 0.27957254350185395,
      "caps_spatial": 0.0014344081573653966,
      "caps_weighted": 0.18975295796990393,
      "caps_valid_pair_fraction": 0.9644775539636612,
      "caps_quiet_temporal_mean": 0.272245413100785,
      "caps_moving_temporal_mean": 0.29378109772236394
    },
    "pair_presentations": {
      "valid_pairs": 118515,
      "quiet_pairs": 78190,
      "moving_pairs": 40325
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6597477112601781,
    "kl_min": 1.0035767445515376e-05,
    "kl_max": 0.025476954877376556,
    "kl_mean": 0.012653349899255772,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 14,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.1796136253377578,
      "moving_temporal": 0.009995891816409617,
      "spatial": 0.00014344081573653967
    }
  },
  {
    "update": 37,
    "losses": {
      "value": 0.1500987023115158,
      "surrogate": -0.009206904750317334,
      "entropy": -16.204428100585936,
      "caps_temporal": 0.2785904586315155,
      "caps_spatial": 0.0014421777857933193,
      "caps_weighted": 0.18928403109312059,
      "caps_valid_pair_fraction": 0.9526774287223816,
      "caps_quiet_temporal_mean": 0.2708271751666512,
      "caps_moving_temporal_mean": 0.29377939513936335
    },
    "pair_presentations": {
      "valid_pairs": 117065,
      "quiet_pairs": 77460,
      "moving_pairs": 39605
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6616836800068339,
    "kl_min": 9.85948554443894e-06,
    "kl_max": 0.015618762001395226,
    "kl_mean": 0.012148685717465923,
    "kl_above_adaptation_threshold_0_02": 0,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17920085272376632,
      "moving_temporal": 0.009938960590774919,
      "spatial": 0.00014421777857933192
    }
  },
  {
    "update": 38,
    "losses": {
      "value": 0.13797516375780106,
      "surrogate": -0.00736378128058277,
      "entropy": -16.220652389526368,
      "caps_temporal": 0.2774858012795448,
      "caps_spatial": 0.0014311135048046709,
      "caps_weighted": 0.1879492238163948,
      "caps_valid_pair_fraction": 0.9496663510799408,
      "caps_quiet_temporal_mean": 0.2691019851436135,
      "caps_moving_temporal_mean": 0.2938254306622118
    },
    "pair_presentations": {
      "valid_pairs": 116695,
      "quiet_pairs": 77120,
      "moving_pairs": 39575
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6608680748960967,
    "kl_min": 9.798513019632082e-06,
    "kl_max": 0.03161487728357315,
    "kl_mean": 0.013937841475626555,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17784170259773316,
      "moving_temporal": 0.009964409868181166,
      "spatial": 0.0001431113504804671
    }
  },
  {
    "update": 39,
    "losses": {
      "value": 0.14274909496307372,
      "surrogate": -0.006701223005075007,
      "entropy": -16.230222702026367,
      "caps_temporal": 0.2773420140147209,
      "caps_spatial": 0.0014273695240262895,
      "caps_weighted": 0.18707523122429848,
      "caps_valid_pair_fraction": 0.9492187798023224,
      "caps_quiet_temporal_mean": 0.269001234898045,
      "caps_moving_temporal_mean": 0.2933606928819767
    },
    "pair_presentations": {
      "valid_pairs": 116640,
      "quiet_pairs": 76700,
      "moving_pairs": 39940
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6575788751714677,
    "kl_min": 9.634552043280564e-06,
    "kl_max": 0.025136906653642654,
    "kl_mean": 0.012268396351555567,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 15,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17688699207824862,
      "moving_temporal": 0.01004550219364723,
      "spatial": 0.00014273695240262895
    }
  },
  {
    "update": 40,
    "losses": {
      "value": 0.15869834274053574,
      "surrogate": -0.007147464051377029,
      "entropy": -16.241016483306886,
      "caps_temporal": 0.27734526097774503,
      "caps_spatial": 0.0014257665839977562,
      "caps_weighted": 0.18661025390028954,
      "caps_valid_pair_fraction": 0.954915389418602,
      "caps_quiet_temporal_mean": 0.2689609859102658,
      "caps_moving_temporal_mean": 0.29331722865158977
    },
    "pair_presentations": {
      "valid_pairs": 117340,
      "quiet_pairs": 76945,
      "moving_pairs": 40395
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6557439918186466,
    "kl_min": 9.563082130625844e-06,
    "kl_max": 0.020081449300050735,
    "kl_mean": 0.011615854462434072,
    "kl_above_adaptation_threshold_0_02": 1,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.1763701679379058,
      "moving_temporal": 0.010097509303983924,
      "spatial": 0.00014257665839977562
    }
  },
  {
    "update": 41,
    "losses": {
      "value": 0.11105084642767907,
      "surrogate": -0.009389739006292076,
      "entropy": -16.252570247650148,
      "caps_temporal": 0.27504742443561553,
      "caps_spatial": 0.0014201066398527474,
      "caps_weighted": 0.1847260743379593,
      "caps_valid_pair_fraction": 0.9586181789636612,
      "caps_quiet_temporal_mean": 0.2670710868643736,
      "caps_moving_temporal_mean": 0.2900917102451434
    },
    "pair_presentations": {
      "valid_pairs": 117795,
      "quiet_pairs": 76980,
      "moving_pairs": 40815
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6535082134216224,
    "kl_min": 9.228058843291365e-06,
    "kl_max": 0.022159308195114136,
    "kl_mean": 0.012744742843915446,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17453257914490272,
      "moving_temporal": 0.010051484529071281,
      "spatial": 0.00014201066398527475
    }
  },
  {
    "update": 42,
    "losses": {
      "value": 0.17272159233689308,
      "surrogate": -0.00774708081735298,
      "entropy": -16.260015392303465,
      "caps_temporal": 0.2758292376995087,
      "caps_spatial": 0.0014167194080073387,
      "caps_weighted": 0.18330263420939447,
      "caps_valid_pair_fraction": 0.9579264521598816,
      "caps_quiet_temporal_mean": 0.2664562249986813,
      "caps_moving_temporal_mean": 0.29314157938867835
    },
    "pair_presentations": {
      "valid_pairs": 117710,
      "quiet_pairs": 76365,
      "moving_pairs": 41345
    },
    "quiet_fraction_of_valid_pair_presentations": 0.648755415852519,
    "kl_min": 9.192571269522887e-06,
    "kl_max": 0.020629651844501495,
    "kl_mean": 0.015692652736242964,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17286448722071432,
      "moving_temporal": 0.010296475047879436,
      "spatial": 0.00014167194080073387
    }
  },
  {
    "update": 43,
    "losses": {
      "value": 0.1305916965007782,
      "surrogate": -0.008108964888378978,
      "entropy": -16.270827198028563,
      "caps_temporal": 0.274689082801342,
      "caps_spatial": 0.0014149345166515558,
      "caps_weighted": 0.18084635436534882,
      "caps_valid_pair_fraction": 0.9569092094898224,
      "caps_quiet_temporal_mean": 0.2651285622079962,
      "caps_moving_temporal_mean": 0.2918510127577552
    },
    "pair_presentations": {
      "valid_pairs": 117585,
      "quiet_pairs": 75510,
      "moving_pairs": 42075
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6421737466513586,
    "kl_min": 9.194802260026336e-06,
    "kl_max": 0.023521577939391136,
    "kl_mean": 0.013034813191916328,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17026216959283275,
      "moving_temporal": 0.010442691320850928,
      "spatial": 0.00014149345166515558
    }
  },
  {
    "update": 44,
    "losses": {
      "value": 0.15870631784200667,
      "surrogate": -0.009309810516424478,
      "entropy": -16.269802856445313,
      "caps_temporal": 0.27481522411108017,
      "caps_spatial": 0.0014202065649442375,
      "caps_weighted": 0.18114982545375824,
      "caps_valid_pair_fraction": 0.9575602412223816,
      "caps_quiet_temporal_mean": 0.264801892759303,
      "caps_moving_temporal_mean": 0.2929472946044323
    },
    "pair_presentations": {
      "valid_pairs": 117665,
      "quiet_pairs": 75800,
      "moving_pairs": 41865
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6442017592317172,
    "kl_min": 9.01335988601204e-06,
    "kl_max": 0.021599365398287773,
    "kl_mean": 0.015850167804001104,
    "kl_above_adaptation_threshold_0_02": 3,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 5,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17058475820683977,
      "moving_temporal": 0.010423046590424041,
      "spatial": 0.00014202065649442376
    }
  },
  {
    "update": 45,
    "losses": {
      "value": 0.22859062179923056,
      "surrogate": -0.006010673491982743,
      "entropy": -16.272372913360595,
      "caps_temporal": 0.2736548185348511,
      "caps_spatial": 0.0014174083713442086,
      "caps_weighted": 0.18175796344876288,
      "caps_valid_pair_fraction": 0.9483642876148224,
      "caps_quiet_temporal_mean": 0.2636333327126761,
      "caps_moving_temporal_mean": 0.2922742198786803
    },
    "pair_presentations": {
      "valid_pairs": 116535,
      "quiet_pairs": 75760,
      "moving_pairs": 40775
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6501051186338869,
    "kl_min": 8.547775905753952e-06,
    "kl_max": 0.03002053126692772,
    "kl_mean": 0.013719837535381885,
    "kl_above_adaptation_threshold_0_02": 4,
    "lr_after_exact_literal_floor": 15,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 16,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17138971195349262,
      "moving_temporal": 0.010226510658135846,
      "spatial": 0.00014174083713442087
    }
  },
  {
    "update": 46,
    "losses": {
      "value": 0.13026643320918083,
      "surrogate": -0.0079952547326684,
      "entropy": -16.27641553878784,
      "caps_temporal": 0.27204026430845263,
      "caps_spatial": 0.0014168015448376536,
      "caps_weighted": 0.1799384333193302,
      "caps_valid_pair_fraction": 0.9506836086511612,
      "caps_quiet_temporal_mean": 0.26171889193552933,
      "caps_moving_temporal_mean": 0.29103125531695456
    },
    "pair_presentations": {
      "valid_pairs": 116820,
      "quiet_pairs": 75680,
      "moving_pairs": 41140
    },
    "quiet_fraction_of_valid_pair_presentations": 0.647834274952919,
    "kl_min": 8.79032722878037e-06,
    "kl_max": 0.019773565232753754,
    "kl_mean": 0.014234286496275672,
    "kl_above_adaptation_threshold_0_02": 0,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.16954747414889018,
      "moving_temporal": 0.010249279015956245,
      "spatial": 0.00014168015448376536
    }
  },
  {
    "update": 47,
    "losses": {
      "value": 0.13092098273336888,
      "surrogate": -0.0073221348613515145,
      "entropy": -16.283564853668214,
      "caps_temporal": 0.27224249839782716,
      "caps_spatial": 0.0014078636071644724,
      "caps_weighted": 0.1804485723376274,
      "caps_valid_pair_fraction": 0.9609375298023224,
      "caps_quiet_temporal_mean": 0.26279882545794075,
      "caps_moving_temporal_mean": 0.28956642293111573
    },
    "pair_presentations": {
      "valid_pairs": 118080,
      "quiet_pairs": 76425,
      "moving_pairs": 41655
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6472306910569106,
    "kl_min": 9.194569429382682e-06,
    "kl_max": 0.021442359313368797,
    "kl_mean": 0.011979073313705157,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17009281793014247,
      "moving_temporal": 0.010214968046768469,
      "spatial": 0.00014078636071644725
    }
  },
  {
    "update": 48,
    "losses": {
      "value": 0.1485236518085003,
      "surrogate": -0.008606042491737752,
      "entropy": -16.28859453201294,
      "caps_temporal": 0.27064605355262755,
      "caps_spatial": 0.0014118636318016797,
      "caps_weighted": 0.18042752593755723,
      "caps_valid_pair_fraction": 0.9555257558822632,
      "caps_quiet_temporal_mean": 0.260903089655089,
      "caps_moving_temporal_mean": 0.28894142383965205
    },
    "pair_presentations": {
      "valid_pairs": 117415,
      "quiet_pairs": 76615,
      "moving_pairs": 40800
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6525145850189499,
    "kl_min": 8.98135476745665e-06,
    "kl_max": 0.02322784624993801,
    "kl_mean": 0.013330534200940746,
    "kl_above_adaptation_threshold_0_02": 2,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.17024637135457146,
      "moving_temporal": 0.01003996821980561,
      "spatial": 0.00014118636318016797
    }
  },
  {
    "update": 49,
    "losses": {
      "value": 0.13370357342064382,
      "surrogate": -0.006508563025272451,
      "entropy": -16.298941993713377,
      "caps_temporal": 0.26941494941711425,
      "caps_spatial": 0.0014061127556487917,
      "caps_weighted": 0.17927702739834786,
      "caps_valid_pair_fraction": 0.9547526240348816,
      "caps_quiet_temporal_mean": 0.26009933116239864,
      "caps_moving_temporal_mean": 0.2867170038249696
    },
    "pair_presentations": {
      "valid_pairs": 117320,
      "quiet_pairs": 76275,
      "moving_pairs": 41045
    },
    "quiet_fraction_of_valid_pair_presentations": 0.650144902829867,
    "kl_min": 9.00987652130425e-06,
    "kl_max": 0.02679537609219551,
    "kl_mean": 0.012686060184205417,
    "kl_above_adaptation_threshold_0_02": 4,
    "lr_after_exact_literal_floor": 15,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 16,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.1691054679789684,
      "moving_temporal": 0.010030948143814587,
      "spatial": 0.00014061127556487917
    }
  },
  {
    "update": 50,
    "losses": {
      "value": 0.13541649989783763,
      "surrogate": -0.007680328440619633,
      "entropy": -16.31042766571045,
      "caps_temporal": 0.2675733804702759,
      "caps_spatial": 0.0014085702539887279,
      "caps_weighted": 0.1780475854873657,
      "caps_valid_pair_fraction": 0.958455428481102,
      "caps_quiet_temporal_mean": 0.258151676428595,
      "caps_moving_temporal_mean": 0.28511120003681256
    },
    "pair_presentations": {
      "valid_pairs": 117775,
      "quiet_pairs": 76620,
      "moving_pairs": 41155
    },
    "quiet_fraction_of_valid_pair_presentations": 0.6505625132668224,
    "kl_min": 9.282026439905167e-06,
    "kl_max": 0.020044896751642227,
    "kl_mean": 0.012851176963886246,
    "kl_above_adaptation_threshold_0_02": 1,
    "lr_after_exact_literal_floor": 0,
    "lr_after_floor_with_1e_minus12_relative_tolerance": 0,
    "recovered_mean_weighted_loss_components": {
      "quiet_temporal": 0.16794376712771028,
      "moving_temporal": 0.009962961334256563,
      "spatial": 0.0001408570253988728
    }
  }
]