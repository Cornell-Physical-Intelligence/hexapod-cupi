"""Compute the controlled reward v4 comparison: per-seed metrics, the three hypothesis verdicts and per-probe failures.

The script reads the training metric rows and the 13-probe evaluations of the six packet attempts (arms B, J and Y, seeds
20260917 and 20260918, 5000 updates, evaluations at updates 2000, 3500 and 5000) and of the two reference attempts at update
2000. It copies only attempts whose launcher exit code and cleanup receipt exist, so a running attempt stays missing. A
missing or failed attempt leaves each comparison that needs it inconclusive. It starts no native process.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import tarfile

REMOTE = "/srv/cupi/hexapod/runs/james/"
SEEDS = (20260917, 20260918)
ARMS = {"B": {"joint_margin_weight": 1.0, "yaw_tracking_weight": 0.5},
        "J": {"joint_margin_weight": 3.0, "yaw_tracking_weight": 0.5},
        "Y": {"joint_margin_weight": 1.0, "yaw_tracking_weight": 1.0}}
UPDATES = (2000, 3500, 5000)
TRAINING = {(arm, seed): "ppo_v4_omni5k_%s_seed%d_20261007_001" % (arm, seed) for arm in ARMS for seed in SEEDS}
REFERENCE = {seed: "ppo_v4_omni_seed%d_20261007_001" % seed for seed in SEEDS}
TRANSLATIONS = tuple("learning:translate_0.05_%ddeg" % (45 * bearing) for bearing in range(8))
YAWS = {"-1": "learning:yaw_-1", "+1": "learning:yaw_+1"}
REST_TERMS = ("quiet_joint_rate", "quiet_target_motion", "quiet_contact", "quiet_strain")
WINDOWS = {"joint_limit_early": (601, 2000), "joint_limit_late": (4801, 5000), "rest_2000": (1801, 2000),
           "rest_5000": (4801, 5000)}
TORQUE_BOUNDS = {"computed_demand_over_rating_fraction", "max_requested_torque_saturation_fraction",
                 "max_applied_torque_nm", "requested_saturation_400hz", "requested_saturation_mean_400hz",
                 "applied_cap_all_substeps"}
NONFOOT_BOUNDS = {"nonfoot_fraction", "nonfoot_contact_count_400hz", "canonical_full_trial_400hz_nonfoot_fraction"}
APPLIED_CAP_NM = 1.60001
RECORDS = ("PACK.json", "preparation.json", "launcher.exitcode", "run/cleanup.json", "run/standing/state.json")
# PINS holds the sha256 of each file that the script reads, keyed by its path under REMOTE.
PINS = {
    "ppo_v4_omni_seed20260917_20261007_001/PACK.json": "05b49816b722562678bc3228d1a6c49a9fd712ad377c3ae46a1d8e00f528d2d7",
    "ppo_v4_omni_seed20260917_20261007_001/preparation.json": "c9c678579f8ecb517526a5866735f15ef770b818b9a933caa9d9059f004e6be3",
    "ppo_v4_omni_seed20260917_20261007_001/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni_seed20260917_20261007_001/run/cleanup.json": "c70da22b30f12b0736d39ea4b6a91f35156f02e3e3798c2be1149e4f99f28a8d",
    "ppo_v4_omni_seed20260917_20261007_001/run/standing/state.json": "257ea9d6d7c6dc65915b3a822edff825fb0c605470e8a9087cad391647f3a4fe",
    "ppo_v4_omni_seed20260917_20261007_001/run/standing/metrics.jsonl": "12606ba07f5d2989f2bb765f997d015ddd86b545a046adedbec1b27355012ec2",
    "ppo_v4_omni_seed20260917_20261007_001/run/standing/checkpoint_update002000.json": "fd1a692d2332c256a6471f7e1a7338b10c397be0e15faaef401b23d7276330a5",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/PACK.json": "0435f0445ff67ac04ea7c7bd871aa1453556562bbe3a7e46b536ae26def0ffd7",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/preparation.json": "a28d0a754baaa3320ec298fa5e8b2fcadbfd8ef9b627cdc171a10ee4ac28068a",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/cleanup.json": "af6658a5d91b18372a5c0abc12769769bfe92463f0f8a2f59382c67f0a587e62",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/state.json": "b4970c9aa8a5f1eada6043428051d6df021245bff7adb6833fa1a25f49fd97db",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/summary.json": "6be9f20470bdcd3f8b84483283364ecac2b558d3d00fcc6dd393ac19c952d9b1",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/allocation.json": "bcaa1e514410285120bc52aa914049ab4febdef48102a454ecd7d7081f49d07b",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_000/native400hz/capture.json": "9ebad5ce31996d762525c3b62832a166b6e6988f6b2986e749143805611eebf0",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_001/native400hz/capture.json": "8d22afcedb7b22e334bc897c6cea0a3d2f9c2a2eec33130f2e3a4dc6e8c76e0c",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_002/native400hz/capture.json": "a1ea24238c08381883abdcdca523b1b7468a152649c1b65620d194ec42fe1d58",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_003/native400hz/capture.json": "43ff02a12581d6be5ba665bf7d4a19cca2613a19cb070597cf2697312d36a52f",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_004/native400hz/capture.json": "af48cb131dff3f05c9803c3b04eccc80e13f7b4fc1ea612767e0aa6a93bc5a3e",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_005/native400hz/capture.json": "f789c29d9975f52ec8367311dd80f482fa7ff6e573d049f9f4fc844399e9d03e",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_006/native400hz/capture.json": "c8c6f225819c88b9a63440c5fab91385f09fb280e15fbb7026117412beda4e81",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_007/native400hz/capture.json": "fb187a89939398002b8fca7d491a671e9a73ff9951455227fb7195dcafd043d4",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_008/native400hz/capture.json": "aade8ad1a2059ae464287d4dcd7af1cded98db3521b60e69634304bc37ea8bcd",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_009/native400hz/capture.json": "45f29eac69998cb528483665c10cfeed2df6802a2f2a99fc537d11f036e27f16",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/capture.json": "8be43505a48e86ffa55269054995e3350f8ec7152ddc3efe6189c0adef52b371",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/capture.json": "c70699cb15b6af03da31b494e01454d7738be3a9573d9b45143ad002c945898f",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/capture.json": "20ddc19c2083e1babf036edd21529e90f8bb8d248c3c808de179a0c852b4b1c7",
    "ppo_v4_omni_seed20260918_20261007_001/PACK.json": "028c1040a6a5710d1dbd7055f19da28a4a84c9623f4f9df0d2b158b1d98faf59",
    "ppo_v4_omni_seed20260918_20261007_001/preparation.json": "7f6a20d44f3e85c9a332b9466882e982ae09841cbb7497d5f110e435455cde8a",
    "ppo_v4_omni_seed20260918_20261007_001/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni_seed20260918_20261007_001/run/cleanup.json": "8985203c19f42bf46144f648279121350a3bf29d8788e28e8b2112a72e76c69f",
    "ppo_v4_omni_seed20260918_20261007_001/run/standing/state.json": "43593aebd0e555415d24ff47379238e95706ee97de66b29fa1a95fdca7fffb25",
    "ppo_v4_omni_seed20260918_20261007_001/run/standing/metrics.jsonl": "d024aa55a32237d4191ae09930b50bb8e27c5f05903731c8433297cb09225148",
    "ppo_v4_omni_seed20260918_20261007_001/run/standing/checkpoint_update002000.json": "6200381bc9ce36533091be81642fe0f3e30b0e4411df25b7e1df99dc5eada102",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/PACK.json": "9ec4eb695d967ecbc3f8c81bf8b183aad1f2a90e7eb13331bd82f9a53cde088c",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/preparation.json": "258e8a2b55154afab6c4dd664424f4a5903ff35fefd4d222a5b001d97ac80b9f",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/cleanup.json": "b2084578cadd0e463e7763076b21f0f4920d692917a90c75e6108c9e86fa8194",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/state.json": "4a5747c92f2ac62d77e554501abc993c008022217a80774a78af1495c7cc3f62",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/summary.json": "e7696bdf0af0384fa4ab4812d97cc59590df1e8e24a93e198e81ad7b3e1c27ad",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/allocation.json": "be5216ea37a48dc3e7f66c8eabcc9c626a59e29832e8927abb89c4a906643278",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_000/native400hz/capture.json": "7867c833f8911dc0fe877f6662e8a53792a82412eed57d32403467097289e016",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_001/native400hz/capture.json": "8afa557462d32c9391faf4563ce009db720a12f2ba4719cb900f0d374ed200e5",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_002/native400hz/capture.json": "474941f2aacf8a9cac235083684f84914c4b045ff4407f25d959f98cb774f6e3",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_003/native400hz/capture.json": "9c924a61b393891f01a7f478e700c0dddd0d63b96537a70fee4beca07397c4d7",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_004/native400hz/capture.json": "3a0c1cd3e811953a2572d8710923625f2a2df5c8b1a3ab8879fc0d3574cfa646",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_005/native400hz/capture.json": "9a207194bda0b3ea60bfbd87b89172093c96009717ec61be3efaf25e2ce88e7b",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_006/native400hz/capture.json": "dbcba0e793c08c3bf03cdfa512d54cba8618ab0f9c7afd7a638723e2293c2f93",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_007/native400hz/capture.json": "7fe908293236fc5a8f923b2bd91a5d8091366aa2c91d6612f35972b2b5665843",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_008/native400hz/capture.json": "00808dbd9815bcc7aa558ec68d66f3893a59110ae1f01ac625d3360a4d94a1d2",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_009/native400hz/capture.json": "8db80485402abe900ad418d2c6c2af32b999b2ae5d5a1c0c405136469e6acfd1",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/capture.json": "5405ce19ddb1392accac97006287a13f65795f28539bc11b1f809e3eaea2abdc",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/capture.json": "03baea7bc5e09e119a66c471e72a4a07087698e00d0430a7c94114b3c17da278",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/capture.json": "0c93ed389bd0c0b2033232af8bc251693405f8048a12f22d3566135d4352a1b4",
    "ppo_v4_omni5k_B_seed20260917_20261007_001/PACK.json": "dd9d1bfa41098d9cccb8775efdbae0dc055e1a549b5366ca1de12153adeb55e6",
    "ppo_v4_omni5k_B_seed20260917_20261007_001/preparation.json": "adb27d8f91c1bcec6329106160f04eb1ebb29589e57b8456e0c261bf110bbcc4",
    "ppo_v4_omni5k_B_seed20260917_20261007_001/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_B_seed20260917_20261007_001/run/cleanup.json": "d62129fdaad13eba6b32bd74b4c7f636c3fea70a138ea07113f4e3e8d0bbfb24",
    "ppo_v4_omni5k_B_seed20260917_20261007_001/run/standing/state.json": "fde818f4f8724b20c9055800cf0621c6a2523e62e18a5361bfae8d6d0fa7dd4e",
    "ppo_v4_omni5k_B_seed20260917_20261007_001/run/standing/metrics.jsonl": "bbd134eb32094b73363e14926b65a11ca61546adfe6a1a3ec6d3b1889537a5ac",
    "ppo_v4_omni5k_B_seed20260917_20261007_001/run/standing/checkpoint_update002000.json": "de40e21fbbfef6beb158fbbf598965ff49e94ca9ce6f80e005109d69fc01e336",
    "ppo_v4_omni5k_B_seed20260917_20261007_001/run/standing/checkpoint_update003500.json": "3bc77a1021e37643916b041fee3366e473300305571e207353b5f29dabc23686",
    "ppo_v4_omni5k_B_seed20260917_20261007_001/run/standing/checkpoint_update005000.json": "578d30b27c59a05570f91f6f5fc78c28088a3c959f79fc4fa18177eb1f919387",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/PACK.json": "ee5c23aadd7a8a594ef69f70844ab2444470badf93c4ff4b07d1fd359269744d",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/preparation.json": "c62d429d65e73b76b2e101d0ae465c585ac1c30d8a6fac82730f566d323e68c8",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/cleanup.json": "a64bf47370823cf52509e7bcfe2db470aad5dd9fd8cce7c0d63c9d45d7205731",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/state.json": "d089aec8a34e8d599bf88510eda6de4819f002c9e2fdf0e8166cf7b5162d93b2",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/summary.json": "4646ac3e83e6d42371c779606f417dae6ae901d75556787ccacfc1144191e550",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/allocation.json": "c5f439a5bf2d41ca4d02044357c8fbe8963f79c5c778d1f68b9d484fbf28641a",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_000/native400hz/capture.json": "9ebad5ce31996d762525c3b62832a166b6e6988f6b2986e749143805611eebf0",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_001/native400hz/capture.json": "8d22afcedb7b22e334bc897c6cea0a3d2f9c2a2eec33130f2e3a4dc6e8c76e0c",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_002/native400hz/capture.json": "a1ea24238c08381883abdcdca523b1b7468a152649c1b65620d194ec42fe1d58",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_003/native400hz/capture.json": "43ff02a12581d6be5ba665bf7d4a19cca2613a19cb070597cf2697312d36a52f",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_004/native400hz/capture.json": "af48cb131dff3f05c9803c3b04eccc80e13f7b4fc1ea612767e0aa6a93bc5a3e",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_005/native400hz/capture.json": "f789c29d9975f52ec8367311dd80f482fa7ff6e573d049f9f4fc844399e9d03e",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_006/native400hz/capture.json": "c8c6f225819c88b9a63440c5fab91385f09fb280e15fbb7026117412beda4e81",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_007/native400hz/capture.json": "fb187a89939398002b8fca7d491a671e9a73ff9951455227fb7195dcafd043d4",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_008/native400hz/capture.json": "aade8ad1a2059ae464287d4dcd7af1cded98db3521b60e69634304bc37ea8bcd",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_009/native400hz/capture.json": "45f29eac69998cb528483665c10cfeed2df6802a2f2a99fc537d11f036e27f16",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_010/native400hz/capture.json": "8be43505a48e86ffa55269054995e3350f8ec7152ddc3efe6189c0adef52b371",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_011/native400hz/capture.json": "c70699cb15b6af03da31b494e01454d7738be3a9573d9b45143ad002c945898f",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_012/native400hz/capture.json": "20ddc19c2083e1babf036edd21529e90f8bb8d248c3c808de179a0c852b4b1c7",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/PACK.json": "cedb07c3738be28d2a4186fa5c3f016b05dc5e6f13591eed60d55bd065552f96",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/preparation.json": "c89d92ee74ad07a3498aff1cbf73bc960f56a15e86fcfd1760124e8e574dceda",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/cleanup.json": "8e51ddcc1edbf862a0b00649cfb508e447bf647058467268d9da1dfc47ea10a0",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/state.json": "40b2c0f573e649a1086231935e0ed0fb1991c86bcd47e3264d80b4d3f6d823f7",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/summary.json": "1f84818e9bc355eb3a2967d8ae3b268037389227e6cf8e6eb48a14600960c43e",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/allocation.json": "53bafe38389df26c8e1457a713733396b889c6f11e1489ae7faa6bc6716ce1af",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_000/native400hz/capture.json": "ae7569c60c5fc9bd6fe03ba9475fe68dd212ae8bc67e8c23dc0772bff90347a7",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_001/native400hz/capture.json": "52363c73621bc8b446d09568f8878ff9badd962891f5dce760bc3ed8648fb658",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_002/native400hz/capture.json": "a154a9cc7481941caaf47f1290c8cc87683c5afe4b52efaa3158273d16511888",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_003/native400hz/capture.json": "74cf5c1e3d32b156c414a34ef2b7b970e51234a378cd79076f43f4def964c325",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_004/native400hz/capture.json": "766d0cb1e79e4ba55ec26bc9981b11a63bc22586647e7f86000faa2860f6188f",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_005/native400hz/capture.json": "fe188cb48ad26634b1b6d3b0a6245e925ebe4fa22b9816ca07c24925ebd1b84a",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_006/native400hz/capture.json": "6ada1819f382783541b4588df8e1d6a2e3fa376fe25514ef182dc05d4956829b",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_007/native400hz/capture.json": "ebabd822d675a9b8f8fb5be23e152f76c3c581f398886976fbfb13991d4da158",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_008/native400hz/capture.json": "5d0487c64db4e8728af98f06b52b59f8f8a6f7b84c8e0ea93322f4115b390cd0",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_009/native400hz/capture.json": "d90a6a822c60981f005afa38d0b2665be798838100c87a371dddf2098c6a2fa8",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_010/native400hz/capture.json": "b05ce7c889b950912d83e5b7fa49891789296673ef4a96b75d0d89a27d57d837",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_011/native400hz/capture.json": "faf4d49cb427532e0f93609b4c69c19c96c3fbce73de778b12d0dea52b8a37c0",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_012/native400hz/capture.json": "d68c9233df6b14080e4201f2881c114ff29145723bbc3f43d7195088c50f4bad",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/PACK.json": "a5936041d276899a71ebf4dfa5c638b230cb454a8df151d7af4d5ac6f9101637",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/preparation.json": "499636e6ba648397a26d17c5c59de799ab046177882d5ccca2c2c10eb4691cef",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/cleanup.json": "9c4a2afaa7a9963263c98a1639bb6d960a65390e71f6c2f1cabe1e35a02e37a8",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/state.json": "f87767660be082ad62a1da201934bc4058754b34c4d45acc4017a43a4d61c56e",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/summary.json": "6080a5fda8c9e9859935b309d5047b3186ac60b7e97e210ffeafbb82d2b06053",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/allocation.json": "32cb2125e6151f2aa66ca6139af39a85993a2b86eb626dea8c66d91aa291b089",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_000/native400hz/capture.json": "d6590a750bb213907df62c448a5ec628dcf3f6adfb6a93dcb427d77729433289",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_001/native400hz/capture.json": "35f3b00297cfad2e454b0d5531049b583d97f764876a39b1d13633a704fbcbd2",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_002/native400hz/capture.json": "58acd783123207f9e24ab795a40e093d39117632681fb728d09212f31a6fd361",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_003/native400hz/capture.json": "8dc1034db688dac170b3e227264ca87d6c3fcf0680a1ee98b8f17aca1df02426",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_004/native400hz/capture.json": "738fa3a2492079a8e78f7775c25a5de73ee3e4f44df141c63ee360b3b1746e7a",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_005/native400hz/capture.json": "22b0712148950429f99b23c2190bbc45b097a807b5eedf0b0e7ba9d7007d345a",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_006/native400hz/capture.json": "2501bfbd8233fc67afadf98541f7c9245eb4b8cf1472986604a137a00d4d716a",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_007/native400hz/capture.json": "7db8e960455ff006d5d46af958d3f55002c00405f8108550bbca30cff672a7a2",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_008/native400hz/capture.json": "9ac3baa6e1d486679f43272ccfda421a853bc6371f6f25042128b6081454c0bc",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_009/native400hz/capture.json": "049db96a0d44d0bd3749df8a386c8fc4a91f3a355d65087213b4c62dedbec3e3",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_010/native400hz/capture.json": "ddd555afe653f860802c6b5ec9b96f7fef2ea1c0acf2c6080b1284210238ff47",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_011/native400hz/capture.json": "7e952a7080c058dc117682f920f7e11a2ea2bdc49e4f7467a3d3240698c704fe",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_012/native400hz/capture.json": "1688221e47f51210ab8b7c81f865f2d930ce6ae715ae700bb061197d24d84500",
    "ppo_v4_omni5k_B_seed20260918_20261007_001/PACK.json": "78ee8d5e4c5a0768b89bbc5857d9128e94908969a172a69cf2db03431f9041db",
    "ppo_v4_omni5k_B_seed20260918_20261007_001/preparation.json": "be9b389ea2132994b2277e510718b8c849994c11c71f44a2d9ab533060dee008",
    "ppo_v4_omni5k_B_seed20260918_20261007_001/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_B_seed20260918_20261007_001/run/cleanup.json": "949eca1e134dda7edb0071308e15db90f2309c2149208acc2853f479112da277",
    "ppo_v4_omni5k_B_seed20260918_20261007_001/run/standing/state.json": "e7d9d217076384500c97ab7430798f6a060b8130332196e4dbc22ff71cca9091",
    "ppo_v4_omni5k_B_seed20260918_20261007_001/run/standing/metrics.jsonl": "ab739aa6045b0aeb6636fb57e98388827d762f46a6ca83cc21a92768cb998743",
    "ppo_v4_omni5k_B_seed20260918_20261007_001/run/standing/checkpoint_update002000.json": "6d5088768698badb78920efc182a13f3c9e3473cf37053c3ac2a8343178b5ab6",
    "ppo_v4_omni5k_B_seed20260918_20261007_001/run/standing/checkpoint_update003500.json": "3b2926c29bf093762dda2622f08d7606c8e57b1ca5977dfdd2405f8bc032e689",
    "ppo_v4_omni5k_B_seed20260918_20261007_001/run/standing/checkpoint_update005000.json": "7d163101397b9300c06c11098567702e1019c06966c8eeace719c1d22deb9657",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/PACK.json": "ac4fb1e730dfa802e83854d4bce0f626d54ef4b087a608bd3184e4d4cfd20167",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/preparation.json": "fe3f69bf874c560da9e91b90425ebd2ff391b0e297549b923af7037701bbe312",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/cleanup.json": "8008fcbf3acab10b1032c09659fee777f3f7f5f574c7e0e9cabb26bf3987e7b1",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/state.json": "d96e34faac6209377cdb2815af9b05e7d070bed1b8627a92bec0af0228d30671",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/summary.json": "f46e0817b7d59a0eee62cecc32ae38af3137b8a12eccc03bd41834dfb02242e4",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/allocation.json": "12bf7139621f52b363033ef38f6b833e80c05940665301a969e3eaadfe0ee141",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_000/native400hz/capture.json": "7867c833f8911dc0fe877f6662e8a53792a82412eed57d32403467097289e016",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_001/native400hz/capture.json": "8afa557462d32c9391faf4563ce009db720a12f2ba4719cb900f0d374ed200e5",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_002/native400hz/capture.json": "474941f2aacf8a9cac235083684f84914c4b045ff4407f25d959f98cb774f6e3",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_003/native400hz/capture.json": "9c924a61b393891f01a7f478e700c0dddd0d63b96537a70fee4beca07397c4d7",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_004/native400hz/capture.json": "3a0c1cd3e811953a2572d8710923625f2a2df5c8b1a3ab8879fc0d3574cfa646",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_005/native400hz/capture.json": "9a207194bda0b3ea60bfbd87b89172093c96009717ec61be3efaf25e2ce88e7b",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_006/native400hz/capture.json": "dbcba0e793c08c3bf03cdfa512d54cba8618ab0f9c7afd7a638723e2293c2f93",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_007/native400hz/capture.json": "7fe908293236fc5a8f923b2bd91a5d8091366aa2c91d6612f35972b2b5665843",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_008/native400hz/capture.json": "00808dbd9815bcc7aa558ec68d66f3893a59110ae1f01ac625d3360a4d94a1d2",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_009/native400hz/capture.json": "8db80485402abe900ad418d2c6c2af32b999b2ae5d5a1c0c405136469e6acfd1",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_010/native400hz/capture.json": "5405ce19ddb1392accac97006287a13f65795f28539bc11b1f809e3eaea2abdc",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_011/native400hz/capture.json": "03baea7bc5e09e119a66c471e72a4a07087698e00d0430a7c94114b3c17da278",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u002000/run/standing/evaluation/batch_012/native400hz/capture.json": "0c93ed389bd0c0b2033232af8bc251693405f8048a12f22d3566135d4352a1b4",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/PACK.json": "dbb6cd3df990e5889f2460f86544416e76e375f8efe840ddbc0bbde71d3d3466",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/preparation.json": "2b87809c1f4a82ac141ba0e20e033163e880ebd6b0eb80800e05bf0dab0d81a0",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/cleanup.json": "ef9d8dc83986f9d0f6274a43b52dba9b2a96db010805a16b880c54cd02e44f16",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/state.json": "61a2c9719c170945dcc2908a4c26fdaa16627aa01b6c52cb67dd8311e86381bb",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/summary.json": "7cdb9609f726a2fa0b51a9f26a3cb4f0d315a695be1b6c05446386a6dda1f816",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/allocation.json": "b1f847bf94ec364eeb9f7bf3c735fd638e7ce7921ad0254d656dabb455e3df70",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_000/native400hz/capture.json": "fb073b5bf2facfb4a1954c54e63de92f58f2337634604974940f69a46d700dc9",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_001/native400hz/capture.json": "86c840b55998344b51b25a2a9e28fa1a73ec56e951a0eba70cc9d93a24ae5d3e",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_002/native400hz/capture.json": "304a08e9ebe7d9f224e59433957cf1e1ad95504287d38aa07d19a91980c94c58",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_003/native400hz/capture.json": "80be8791a10163d64123f8e5c30e675ffc1f043be5f36155e36d3b6b8e81af22",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_004/native400hz/capture.json": "b80c9556443f4749b11c540cf59ee17fee39dcd684a8509ce6ccbf82e605b9e1",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_005/native400hz/capture.json": "be59955db700d113be7e19b76d319ef5b4f0ac3feed546deab8b0e3b39076a36",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_006/native400hz/capture.json": "0a0f1027f9cae7f631e8ebea121cc2e5f1a7cf8ef55cc0b8fbcb8ac321b2eb03",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_007/native400hz/capture.json": "d01fe3a4961330152f4299f4e551ca5b373832b5050ea4479bc96c577988f9d4",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_008/native400hz/capture.json": "940a2b20188155608969a0cf0b0a3bea9b57217c128c1d232d564cd5c12567f0",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_009/native400hz/capture.json": "1d96016d34273d9fe55e53ae90c780bab1faccf1571ddf98f54b722720cb9304",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_010/native400hz/capture.json": "9ee07339015d3b04d07ff016103d9b9c240b942b8e7b46a842adbfd35bc84aec",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_011/native400hz/capture.json": "bf51b98ce662b81bc8181edd28b5175c4266b8d0c0f1e9c58d14bd9313d2c8b6",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u003500/run/standing/evaluation/batch_012/native400hz/capture.json": "0d6a5066006b41e5bc775935915c9ecfd90739a58a91c618e2f41111ca8d5363",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/PACK.json": "e938a6bca454450a126af1394051fe350e13abf3144bd08fbebf55d853a51548",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/preparation.json": "e21e310c51d2a7cde380fe125e59c42d316afe29228636097bec54f5b5dec259",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/cleanup.json": "45b2552475c8ab84f7ec888fcbb32981aa2b8fa5c642c1f8d5d52cc9f700b5a1",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/state.json": "1b3ef55434ec6e7500348aa72f7cb93c27b8f5233651811dccf3e74f98ead36d",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/summary.json": "dacc4779f18d96c5585089abab94f6502c475b2b82ea8dec754466651b1ecd89",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/allocation.json": "6040110e8a0d53ef21c9a11ddc32b4f4a196119d15fa48a60feec37e0a93043a",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_000/native400hz/capture.json": "e77da890bfb9bb587b82472f1a05e24e1cc1a015fca963cc438ce08dbc9346b7",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_001/native400hz/capture.json": "c29ffeb64edacde2a0a142e7dd4d436ce6ed2ddab2f62f360d71652f00f7861a",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_002/native400hz/capture.json": "3bdb0d65e3a10c2f9a50e6b0c97f12d995f7aa31b7bf18bba539a1be8d360f0d",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_003/native400hz/capture.json": "ce61fdac79a427f44dc3edf7245e3c44da74644328db178c6b64711bbecb90cf",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_004/native400hz/capture.json": "6f29955883d412bb65fac836f8e327d062082c4cedcd4199730a01f9923aa562",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_005/native400hz/capture.json": "7b46d6390faad8c10ff44067cd22e3250a80d8e24a5884e7411dc8837c97ff10",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_006/native400hz/capture.json": "ecda12431e95ddf83edbcb9b435b59b5de1a6f60ce31020514eeaf60072ab267",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_007/native400hz/capture.json": "de8e2ce9140278d163a8a10697cf013855d2523a3237a4785b7df75ff26422b3",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_008/native400hz/capture.json": "54d41a1ad917865206363c18e5c2c4f8dce4382fd98558b694e0ab3c269a7dd9",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_009/native400hz/capture.json": "a63d50ba55744e3ebdd4170192af63e0bc70e1e8f85ccff7272379f1d23f5d05",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_010/native400hz/capture.json": "3c1f8964c7f1d8dfd73e60ffc0a8241c64213e674adf1956ed84408e6e515c82",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_011/native400hz/capture.json": "10c6b09a68647b16731cbbbd2725c3657022e0fc8c02603d2cf0a3a68026432e",
    "ppo_v4_omni5k_B_seed20260918_20261007_001_evaluate_u005000/run/standing/evaluation/batch_012/native400hz/capture.json": "08c939501d86a94fe4c95025a12a6cb17d60efd0c538a70b242a3a14669e5b4c",
    "ppo_v4_omni5k_J_seed20260917_20261007_001/PACK.json": "77a052d1febb45017d8ffc17f34c97421b18fd3447167adbf724adbdf14e5a72",
    "ppo_v4_omni5k_J_seed20260917_20261007_001/preparation.json": "7f25d551560e61d89f8d49c78ae9ffcfb92a865f9b5999bb330c8fd2d0cee8aa",
    "ppo_v4_omni5k_J_seed20260917_20261007_001/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_J_seed20260917_20261007_001/run/cleanup.json": "0fd5a71516f9d65d559aa041e2ff63b419d387300e53657d633ff17d993822c9",
    "ppo_v4_omni5k_J_seed20260917_20261007_001/run/standing/state.json": "c225c78f7acb21e32b2da80ce09e5422fc7b45f71d5d368449345820259a9f9c",
    "ppo_v4_omni5k_J_seed20260917_20261007_001/run/standing/metrics.jsonl": "e1d4f4aa36c69b20fd76269802bba8f7a934dc5293f68cad03079f45dbfe9050",
    "ppo_v4_omni5k_J_seed20260917_20261007_001/run/standing/checkpoint_update002000.json": "4c1c35bbbdfaa109de57945409c876761aba0d0dddfe4ee410054b157f565ceb",
    "ppo_v4_omni5k_J_seed20260917_20261007_001/run/standing/checkpoint_update003500.json": "a8b2dd7ecbb76bab51af397e85fb870a31de7bb5f03ec0417f2e3baef070001e",
    "ppo_v4_omni5k_J_seed20260917_20261007_001/run/standing/checkpoint_update005000.json": "ea7654e2e30ae012ff0e26b97024d0a55eff8d879e44a4821e4ca4541a806b6b",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/PACK.json": "a17ddd491258ac838f17f0858072ed8cb045bc140ed6a9e86d62142b50bfa20c",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/preparation.json": "17eb9491f19ab9098eb590b1151c8330e5ed440b0b02e899bbe5eaede3626573",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/cleanup.json": "72a22ccb791cdc54a976b56b197d3d40fb74d00eac9c29066d17ad94420d99f1",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/state.json": "a08502ed7c2755234bf66a46f330dca9d3e9cc35f4c6e37bcaff20d1bc349672",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/summary.json": "22cd3b01d59b4958707b2ef45960bc04e7d5e849a4f5081a1cbed75ec6d15346",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/allocation.json": "fd69b9e7d08983519801c47bb1697ec52f24c6d269be9047c82d91b348d27036",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_000/native400hz/capture.json": "3dde3e4e782fc65cefda68d9ae702a4653fdd0bc552845e19deea6d9e0010166",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_001/native400hz/capture.json": "655aba0bd27ea9b6b97ecb783458727d9b63c9388ec0bb4e619fe630c73237a4",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_002/native400hz/capture.json": "4f5a123107bf043d224a043f7693d2e7949da54b0880b4fc043bf5a514973a9d",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_003/native400hz/capture.json": "4149f9eaa8f29598fa9431f7727a71849910823a4ee1a64ae8fe76efa0b05b17",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_004/native400hz/capture.json": "442db951f26fd5653b7994b9d0650fc56ef96929095681bd4c06b9936c3eb7dd",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_005/native400hz/capture.json": "fc697ab43b6e7498299ee6bc347676d5baa24a50d1e8ff69474f490587fb8fcf",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_006/native400hz/capture.json": "17a477cb02dacf3f0cf7a53b26fe10a9f834f629fee49a1b49279d27fc250246",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_007/native400hz/capture.json": "6c1a4d7ad79f61e1a213823deea3b7ba9573bde13ca69c23dbdfd93036312f8c",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_008/native400hz/capture.json": "daf43b055bebc1fe830cb66c5a65a3d82b80ed6ae0a9cf42da33875cbf217c35",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_009/native400hz/capture.json": "93352b8ef53f89a6abfa7198c5290104dc8ada353287254b2a77cb4b19eda885",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_010/native400hz/capture.json": "ff0832db39be136836c84de41c880383cf918592371977da563a0133f23d98cd",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_011/native400hz/capture.json": "4ef33f2d2d08af320038ee519853c1cef7af43b5a2650ec78586cc2de50c0d6a",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u002000/run/standing/evaluation/batch_012/native400hz/capture.json": "03d00821a801ea3282025340e8e4fabae6a2a65d068a745ab116bf6ecc9c9440",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/PACK.json": "0600819f489286f7aa4620b13668943aa238813a161f5518a26e4ca52460020a",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/preparation.json": "6358a4c653aaf63e7f3c400f11e5f56da35861365dd4ce3789376dd376c5f20f",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/cleanup.json": "f382c0ae100b088f3ac621a77a8895c3604c0737bcfbf0faef8145305c94268d",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/state.json": "4f98182401f3858ff1425d735aef331bffdb9797a1bc7bfbfa55fcd299ffae3a",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/summary.json": "858efc07ca2a5249c8c0c23d9e610a87fea2fb06b3a666df41036a758775ad12",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/allocation.json": "d0eb755e13da9cf26f58ff86071881a5befb5506877893499bad3b3d10ecdad4",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_000/native400hz/capture.json": "4521c5f0db486f7b9c91f3305d946105416d6fb015d0e1b9229e23440fad4456",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_001/native400hz/capture.json": "c08689ec75d292044313b2e24ebafd9d794cd39db79c567c63a75b709c229d58",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_002/native400hz/capture.json": "00ea221fa8d983b4492fb045570d389f847bfbfb5ab67eab26bd197f3a11db98",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_003/native400hz/capture.json": "66ee9795eb65df4bc6198f1fed15c213249de23b4f412b27b60da9629247d3f9",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_004/native400hz/capture.json": "944d34454d84eabea4511dacc9f8742bfd1729316be07aa3f9d67675ffe6451d",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_005/native400hz/capture.json": "df9eb3694153e9fab2cc8db2d813ce3c94f28efca63ce9f4e547e2a005770d84",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_006/native400hz/capture.json": "927c00295bfe23eaf3433ad50fe054c15f0262981fa97328af06afa8a46aec90",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_007/native400hz/capture.json": "c7178bdff924c9ade3493344e41b213d331b8047804db78c0be2752be6a33545",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_008/native400hz/capture.json": "8e92e1df1d2dfbf7170813f70bc9c846615e66872669c3ec5f673c0a8de36935",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_009/native400hz/capture.json": "6c457e5f8af217e95f28a991dac709bc9dc09d0740eefe140752bcc9e6c8011e",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_010/native400hz/capture.json": "984c4661a83bb5556d4fbd73077e2360e4005b54e0fc795aa78d2ed042adb27c",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_011/native400hz/capture.json": "794e1fdd31e5d09d93a85c7020225d401e1a074b72a4d13018b7ee3dc011fea0",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u003500/run/standing/evaluation/batch_012/native400hz/capture.json": "18e4ad2f19e1ba1e38e8caa1ce02604a2828b27b17dcde296b5d93e90aa263f0",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/PACK.json": "05df5541475e530be9141b754dd407d68be9ce163912b028deed8e158ea4df1e",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/preparation.json": "5c9361cc73e181a8043f85b8dcaaa48ed55e32df4b7f23175398f628bd65d964",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/launcher.exitcode": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/cleanup.json": "c3f597fa0bc98c609438678f17b6462791ec006daeebd287b7f3529ee44bcbfb",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/state.json": "a620faa420c3492f379f356a7043eb70ccf960abaf323be7407198253687a8e0",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/summary.json": "6bda6f032c77894fee0da75ac0d1cc107b862b85af66b54b0346206d9ac0474b",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/allocation.json": "29a5d241b8104afdeddc80d91fd307325b13e7e33b9cbf798ae15836a04b974e",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_000/native400hz/capture.json": "f1d83ccc1e0812f40c853cf44b006ccda6a6cee4f57df87830fb9f88e2191505",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_001/native400hz/capture.json": "1dba50f491e664251a1376f187b03dc11b2ed17f4f4729d050be5b80b9a9ec63",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_002/native400hz/capture.json": "336156f718eace232bcb8abadf2d8a4ab972250ced8c44d8cb9def63c4adf22f",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_003/native400hz/capture.json": "7ff2ca0acb0977f55ed4f2614a2b6a46fd9f31be921de39b0de45b2f7056730c",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_004/native400hz/capture.json": "b92ef658b99551b10c76776d7f5efcc62f6bc890406227e69a189acdf44d40c7",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_005/native400hz/capture.json": "d56df187ad94ae28b41febcd9a467c7c92f896bb640dcbc3c4b836339b08a0c1",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_006/native400hz/capture.json": "5a138cc5c538f9d2a3fa707f69dc29d82d17747cfae21dfb66c69768b1a961ec",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_007/native400hz/capture.json": "c1b8753d315d46d7f704e1a85ab9913866c33181cde698ae1c29ac6a079c961c",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_008/native400hz/capture.json": "16d42b0cc068ad6f1d0ff97d2538ec41a472a97c60334526093f1321a0f883b0",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_009/native400hz/capture.json": "d254993cfbb980ccab57a9441b942b30112c3a7c17f198e13e39f4341898658b",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_010/native400hz/capture.json": "98d676fcde480f5352eede99caf5fbd97e66b510dffeac2cfaa85a5f8dd635be",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_011/native400hz/capture.json": "cec1833d0063ded64ed31e49a3f6c6f785b3fd82d5b0333148ecdcbcc1723114",
    "ppo_v4_omni5k_J_seed20260917_20261007_001_evaluate_u005000/run/standing/evaluation/batch_012/native400hz/capture.json": "867b5363bda12918a8fde7b97cbf3e11ef6fe82e7f8b64db58d30e34b9bc9f6f",
}
METHODS = {
    "translation_passes": "Count of the eight 0.05 m/s translation probes whose evaluator result passes every existing check.",
    "translation_speed": "Mean over the eight translation probes of mean_velocity_mps (navigation frame, scored window) "
                         "projected onto the unit command direction.",
    "yaw_error": "The evaluator's yaw_error_rad_s, the mean absolute yaw-rate error over the scored window, for each sign; "
                 "mean_yaw_rate_rad_s gives the achieved sign.",
    "tracking_E": "(sum of planar_error_mps / 0.05 over the eight translations + sum of yaw_error_rad_s / 0.2 over the two "
                  "yaw probes) / 10.",
    "terminated_probes": "A probe that ends in a native terminal state fails, and the evaluator scores its recorded prefix "
                         "from control 100. Its speed, planar and yaw errors then cover fewer controls than a complete "
                         "probe; terminated_probes lists each such tracking probe so a reader can weigh those values.",
    "joint_limit_rate": "Sum of task.interval_metrics.termination_reasons.rows.joint_limit over the update window divided by "
                        "the sum of interval environment_controls, times 1e6. The companion divides the same rows by the "
                        "completed episodes, terminations plus truncations.",
    "rest_costs": "For each quiet term, minus the control-weighted mean of command_classes.zero.reward_component_means over "
                  "the update window, weighted by command_classes.zero.environment_controls.",
    "regression_guard": "Every probe that passes at the comparison checkpoint must pass at the candidate, and no probe may "
                        "add a failed torque or nonfoot-contact bound. Torque bounds: " + ", ".join(sorted(TORQUE_BOUNDS))
                        + ", plus the native applied-torque cap inside native_capture_or_original_physical_bounds. "
                        "Nonfoot-contact bounds: " + ", ".join(sorted(NONFOOT_BOUNDS)) + ", plus the native 400 Hz contact "
                        "screen inside the same aggregate.",
    "H1_budget": "B at update 5000 against B at update 2000 within each seed: at least one more translation pass, each rest "
                 "cost in updates 4801-5000 at least 10 percent below its value in updates 1801-2000, and the guard "
                 "against the update-2000 evaluation.",
    "H2_margin": "J against B with the matching seed: joint-limit rate over updates 601-2000 at most half of B's, "
                 "joint-limit rate over updates 4801-5000 at most B's, final translation speed at least 5 percent above "
                 "B's, measured as (J - B) / |B|, and the guard against B at update 5000. A zero B rate over 601-2000 "
                 "requires a zero J rate and cannot establish the reduction, so that requirement stays unmet.",
    "H3_yaw": "Y against B at update 5000 with the matching seed: mean yaw rate with the commanded sign in both yaw probes, "
              "each yaw probe's error at most 80 percent of B's, and the guard against B at update 5000.",
    "verdicts": "supported: both seeds meet every requirement. unsupported_at_tested_setting: no predicted outcome, that is "
                "no requirement other than the guard and no per-term or per-sign part of one, is met in either seed. "
                "mixed: any other complete result, including seeds that disagree. inconclusive: a required attempt or "
                "evaluation is missing or failed.",
    "deviation_parity": "mean_action_std of each packet attempt over updates 1-2000 against the reference attempt of the "
                        "same seed, whose decay spanned its 2000 updates, and its largest departure from 0.05 afterwards. "
                        "Each metric row records the deviation that its update trained with.",
    "claim_limits": ["Reward totals do not establish gradient competition.",
                     "Lower terminations alone do not establish a speed benefit.",
                     "Two seeds establish no confidence interval.",
                     "A failed comparison does not disprove other weights or budgets.",
                     "The tested bearings are training commands, not unseen directions.",
                     "Historical forward runs use a different bootstrap implementation; only the matched attempts here "
                     "support causal comparisons."],
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=int)


def remote_complete(names):
    """Names of attempts whose launcher exit code and cleanup receipt exist on Spark."""
    command = "cd " + shlex.quote(REMOTE) + " && for n in " + " ".join(map(shlex.quote, names)) + \
              "; do [ -e \"$n/launcher.exitcode\" ] && [ -e \"$n/run/cleanup.json\" ] && echo \"$n\"; done; true"
    return set(subprocess.run(["ssh", "spark", command], check=True, capture_output=True, text=True).stdout.split())


def fetch(names, destination):
    """Copy the missing named files that exist on Spark through one tar stream."""
    missing = [name for name in names if not (destination / name).exists()]
    if missing:
        destination.mkdir(parents=True, exist_ok=True)
        command = "cd " + shlex.quote(REMOTE) + " && for f in " + " ".join(map(shlex.quote, missing)) + \
                  "; do [ -e \"$f\" ] && echo \"$f\"; done | tar -cf - -T -"
        process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(destination, filter="data")
        assert process.wait() == 0


class Workspace:
    def __init__(self, root, enabled, complete):
        self.inputs, self.enabled, self.complete, self.files = root / "inputs", enabled, complete, {}

    def load(self, attempt, names):
        """Read the named files of a complete attempt; return None for an attempt that is missing or running."""
        if attempt not in self.complete:
            return None
        paths = [attempt + "/" + name for name in names]
        if self.enabled:
            fetch(paths, self.inputs)
        if not all((self.inputs / path).exists() for path in paths):
            return None
        self.files.update({path: sha(self.inputs / path) for path in paths})
        return [self.inputs / path for path in paths]


def launched(workspace, attempt):
    """Launch records of one attempt: status fields, or None when the attempt is missing or still running."""
    paths = workspace.load(attempt, RECORDS)
    if paths is None:
        return None
    pack, preparation, cleanup, state = (read(paths[index]) for index in (0, 1, 3, 4))
    exitcode = int(paths[2].read_text())
    absent = bool(cleanup["inspections"]) and all(row == {"absent": True} for row in cleanup["inspections"])
    return {"attempt": REMOTE + attempt, "launcher_exitcode": exitcode, "state_status": state["status"],
            "errors": state["errors"], "container_absent": absent and cleanup["cleanup_checked"],
            "completed": exitcode == 0 and state["status"] == "completed" and absent and cleanup["cleanup_checked"],
            "source_commit": preparation["source_commit"], "binding_sha256": pack["binding_sha256"],
            "source_freeze_sha256": pack["source_freeze_sha256"], "updates_requested": pack["updates"],
            "updates_completed": state.get("updates"), "reward_options": pack.get("reward_options"),
            "seed": pack["seed"], "pack": pack, "state": state}


def window(rows, first, last):
    """Joint-limit rate, its companion and rest costs over update rows first to last."""
    chosen = [row for row in rows if first <= row["update"] <= last]
    if len(chosen) != last - first + 1:
        return None
    intervals = [row["task"]["interval_metrics"] for row in chosen]
    controls = sum(row["environment_controls"] for row in intervals)
    limit = sum(row["termination_reasons"]["rows"]["joint_limit"] for row in intervals)
    episodes = sum(row["terminations"] + row["truncations"] for row in intervals)
    zero = [row["command_classes"]["zero"] for row in intervals]
    weight = sum(row["environment_controls"] for row in zero)
    rest = {term: -sum(row["reward_component_means"][term] * row["environment_controls"] for row in zero
                       if row["environment_controls"]) / weight for term in REST_TERMS} if weight else None
    return {"first_update": first, "last_update": last, "environment_controls": controls, "joint_limit_rows": limit,
            "joint_limit_per_million_controls": limit / controls * 1e6, "completed_episodes": episodes,
            "joint_limit_fraction_of_completed_episodes": limit / episodes if episodes else None,
            "zero_command_controls": weight, "rest_costs": rest}


def training(workspace, attempt):
    record = launched(workspace, attempt)
    if record is None:
        return None
    paths = workspace.load(attempt, ["run/standing/metrics.jsonl"])
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert [row["update"] for row in rows] == list(range(1, len(rows) + 1))
    record["metrics_rows"] = len(rows)
    record["windows"] = {name: window(rows, *span) for name, span in WINDOWS.items()}
    record["deviation_at"] = {str(update): rows[update - 1]["mean_action_std"] for update in (1, 2000, 2001, 5000)
                              if update <= len(rows)}
    record["_deviation"] = [row["mean_action_std"] for row in rows]
    sidecars = {}
    for update in UPDATES:
        name = "run/standing/checkpoint_update%06d.json" % update
        loaded = workspace.load(attempt, [name]) if update <= len(rows) else None
        if loaded:
            sidecar = read(loaded[0])
            assert sidecar["updates"] == update and sidecar["identity"]["seed"] == record["seed"]
            sidecars[str(update)] = sidecar["checkpoint_sha256"]
    record["checkpoints"] = sidecars
    del record["pack"], record["state"]
    return record


def native_parts(result, capture):
    """The failed parts of native_capture_or_original_physical_bounds, as locomotion/evaluate.py composes it."""
    parts = {"incomplete_capture": not result["native_capture_complete"],
             "joint_bound_violation_steps": capture["joint_bound_violation_steps"][0] != 0,
             "speed_bound_violation_steps": capture["speed_bound_violation_steps"][0] != 0,
             "maximum_applied_nm": capture["maximum_applied_nm"][0] > APPLIED_CAP_NM,
             "minimum_non_toe_floor_m": capture["minimum_non_toe_floor_m"][0] < -.001,
             "minimum_plate_height_m": capture["minimum_plate_height_m"][0] < .055,
             "native_contact_screen": not result["native_contact_screen"]["pass"]}
    return sorted(name for name, failed in parts.items() if failed)


def guard_sets(result, capture):
    """Torque and nonfoot-contact failures of one probe, with the native aggregate split into its parts."""
    failed, native = set(result["failed_bounds"]), native_parts(result, capture)
    torque, nonfoot = failed & TORQUE_BOUNDS, failed & NONFOOT_BOUNDS
    if "native_capture_or_original_physical_bounds" in failed:
        assert native, "the native aggregate failed without a failed part"
        torque |= {"native_" + name for name in native if name == "maximum_applied_nm"}
        nonfoot |= {name for name in native if name == "native_contact_screen"}
    return sorted(torque), sorted(nonfoot), native


def evaluation(workspace, attempt, checkpoint):
    record = launched(workspace, attempt)
    if record is None:
        return None
    paths = workspace.load(attempt, ["run/standing/evaluation/summary.json", "run/standing/evaluation/allocation.json"])
    if paths is None:
        return {**{key: record[key] for key in ("attempt", "launcher_exitcode", "state_status", "errors", "completed")},
                "missing": "summary"}
    summary, allocation = read(paths[0]), read(paths[1])
    assert summary["allocation"] == allocation and (checkpoint is None or allocation["checkpoint_sha256"] == checkpoint)
    captures = workspace.load(attempt, ["run/standing/evaluation/batch_%03d/native400hz/capture.json" % index
                                        for index in range(len(allocation["selected_case_ids"]))])
    results = {row["case_id"]: row for row in summary["results"]}
    probes = {}
    for index, case in enumerate(allocation["selected_case_ids"]):
        result = results.get(case)
        if result is None:
            probes[case] = {"missing": True}
            continue
        torque, nonfoot, native = guard_sets(result, read(captures[index]))
        metrics = result["metrics"]
        probes[case] = {"pass": result["pass"], "failed_bounds": result["failed_bounds"], "torque_failures": torque,
                        "nonfoot_failures": nonfoot, "native_parts_failed": native,
                        "recorded_controls": result["recorded_controls"],
                        "native_terminal_prefix": result["checks"]["terminated"]["status"] == "fail",
                        **{key: metrics.get(key) for key in ("planar_error_mps", "yaw_error_rad_s", "mean_yaw_rate_rad_s",
                                                             "mean_velocity_mps")}}
    complete = all(case in results for case in (*TRANSLATIONS, *YAWS.values()))
    out = {key: record[key] for key in ("attempt", "launcher_exitcode", "state_status", "errors", "completed",
                                        "binding_sha256")}
    out.update(checkpoint_sha256=allocation["checkpoint_sha256"], summary_sha256=workspace.files[
        attempt + "/run/standing/evaluation/summary.json"], probes=probes, passes=sum(
            row.get("pass") is True for row in probes.values()))
    if complete and record["completed"]:
        speeds, planar = [], []
        for case in TRANSLATIONS:
            command = [0.05 * math.cos(math.radians(int(case.split("_")[-1][:-3]))),
                       0.05 * math.sin(math.radians(int(case.split("_")[-1][:-3])))]
            velocity = probes[case]["mean_velocity_mps"]
            speeds.append((velocity[0] * command[0] + velocity[1] * command[1]) / 0.05)
            planar.append(probes[case]["planar_error_mps"])
        yaw = {sign: probes[case]["yaw_error_rad_s"] for sign, case in YAWS.items()}
        out["metrics"] = {
            "terminated_probes": [case for case in (*TRANSLATIONS, *YAWS.values()) if probes[case]["native_terminal_prefix"]],
            "translation_passes": sum(probes[case]["pass"] for case in TRANSLATIONS),
            "translation_speed_mps": sum(speeds) / 8, "translation_speed_by_probe_mps": dict(zip(TRANSLATIONS, speeds)),
            "yaw_error_rad_s": yaw, "mean_yaw_rate_rad_s": {sign: probes[case]["mean_yaw_rate_rad_s"]
                                                            for sign, case in YAWS.items()},
            "tracking_E": (sum(value / 0.05 for value in planar) + sum(value / 0.2 for value in yaw.values())) / 10}
    return out


def guard(candidate, comparison):
    lost = [case for case, row in comparison["probes"].items() if row.get("pass") and not candidate["probes"][case].get("pass")]
    added = {case: {"torque": sorted(set(row["torque_failures"]) - set(comparison["probes"][case]["torque_failures"])),
                    "nonfoot": sorted(set(row["nonfoot_failures"]) - set(comparison["probes"][case]["nonfoot_failures"]))}
             for case, row in candidate["probes"].items() if "torque_failures" in row}
    added = {case: value for case, value in added.items() if value["torque"] or value["nonfoot"]}
    return {"met": not lost and not added, "lost_passing_probes": lost, "new_torque_or_nonfoot_failures": added}


def ratio(candidate, control):
    return None if control == 0 else (candidate - control) / abs(control)


def h1(train, evaluations):
    early, late = (evaluations.get(str(update)) for update in (2000, 5000))
    windows = train["windows"] if train else {}
    if not (train and train["completed"] and early and late and "metrics" in early and "metrics" in late
            and windows.get("rest_2000") and windows.get("rest_5000")):
        return None
    before, after = windows["rest_2000"]["rest_costs"], windows["rest_5000"]["rest_costs"]
    costs = {term: {"update_2000": before[term], "update_5000": after[term], "change": ratio(after[term], before[term]),
                    "met": before[term] > 0 and after[term] <= 0.9 * before[term]} for term in REST_TERMS}
    gain = late["metrics"]["translation_passes"] - early["metrics"]["translation_passes"]
    requirements = {"translation_pass_gain": {"update_2000": early["metrics"]["translation_passes"],
                                              "update_5000": late["metrics"]["translation_passes"], "met": gain >= 1},
                    "rest_costs_each_10_percent_lower": {"terms": costs, "met": all(row["met"] for row in costs.values())},
                    "regression_guard": guard(late, early)}
    parts = [requirements["translation_pass_gain"]["met"], *(row["met"] for row in costs.values())]
    return requirements, parts


def h2(control_train, candidate_train, control, candidate):
    if not (control_train and candidate_train and control_train["completed"] and candidate_train["completed"]
            and control and candidate and "metrics" in control and "metrics" in candidate):
        return None
    rate = lambda train, name: train["windows"][name]["joint_limit_per_million_controls"]
    early_b, early_j = rate(control_train, "joint_limit_early"), rate(candidate_train, "joint_limit_early")
    late_b, late_j = rate(control_train, "joint_limit_late"), rate(candidate_train, "joint_limit_late")
    speed_b, speed_j = control["metrics"]["translation_speed_mps"], candidate["metrics"]["translation_speed_mps"]
    requirements = {
        "joint_limit_rate_601_2000_halved": {"B": early_b, "J": early_j, "change": ratio(early_j, early_b),
                                             "zero_baseline": early_b == 0, "met": early_b > 0 and early_j <= .5 * early_b},
        "joint_limit_rate_4801_5000_not_increased": {"B": late_b, "J": late_j, "met": late_j <= late_b},
        "translation_speed_5_percent_higher": {"B": speed_b, "J": speed_j, "change": ratio(speed_j, speed_b),
                                               "met": speed_b != 0 and (speed_j - speed_b) / abs(speed_b) >= .05},
        "regression_guard": guard(candidate, control)}
    parts = [row["met"] for key, row in requirements.items() if key != "regression_guard"]
    return requirements, parts


def h3(control, candidate):
    if not (control and candidate and "metrics" in control and "metrics" in candidate):
        return None
    signs = {sign: {"command_rad_s": float(sign) * .2, "mean_yaw_rate_rad_s": candidate["metrics"]["mean_yaw_rate_rad_s"][sign],
                    "met": candidate["metrics"]["mean_yaw_rate_rad_s"][sign] * float(sign) > 0} for sign in YAWS}
    errors = {sign: {"B": control["metrics"]["yaw_error_rad_s"][sign], "Y": candidate["metrics"]["yaw_error_rad_s"][sign],
                     "change": ratio(candidate["metrics"]["yaw_error_rad_s"][sign], control["metrics"]["yaw_error_rad_s"][sign]),
                     "met": candidate["metrics"]["yaw_error_rad_s"][sign] <= .8 * control["metrics"]["yaw_error_rad_s"][sign]}
              for sign in YAWS}
    requirements = {"commanded_yaw_sign_both_probes": {"signs": signs, "met": all(row["met"] for row in signs.values())},
                    "yaw_error_each_20_percent_lower": {"signs": errors, "met": all(row["met"] for row in errors.values())},
                    "regression_guard": guard(candidate, control)}
    parts = [row["met"] for row in signs.values()] + [row["met"] for row in errors.values()]
    return requirements, parts


def verdict(per_seed):
    if any(value is None for value in per_seed.values()):
        return "inconclusive"
    met = [all(row["met"] for row in requirements.values()) for requirements, _ in per_seed.values()]
    if all(met):
        return "supported"
    if not any(any(parts) for _, parts in per_seed.values()):
        return "unsupported_at_tested_setting"
    return "mixed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    output = args.output or args.workspace / "comparison.json"
    if output.exists():
        raise FileExistsError("Use a fresh output path")
    names = [*TRAINING.values(), *REFERENCE.values()]
    names += [name + "_evaluate_u%06d" % update for name in TRAINING.values() for update in UPDATES]
    names += [name + "_evaluate" for name in REFERENCE.values()]
    complete_file = args.workspace / "complete_attempts.json"
    if args.fetch:
        complete_file.parent.mkdir(parents=True, exist_ok=True)
        complete_file.write_text(json.dumps(sorted(remote_complete(names)), indent=1) + "\n")
    workspace = Workspace(args.workspace, args.fetch, set(read(complete_file)) if complete_file.exists() else set())
    references = {}
    for seed, name in REFERENCE.items():
        train = training(workspace, name)
        references[str(seed)] = {"training": train,
                                 "evaluation_2000": evaluation(workspace, name + "_evaluate",
                                                               train["checkpoints"].get("2000") if train else None)}
    attempts = {}
    for (arm, seed), name in TRAINING.items():
        train = training(workspace, name)
        evaluations = {str(update): evaluation(workspace, name + "_evaluate_u%06d" % update,
                                               (train or {}).get("checkpoints", {}).get(str(update)))
                       for update in UPDATES}
        attempts.setdefault(arm, {})[str(seed)] = {"reward_options": ARMS[arm], "training": train,
                                                   "evaluations": evaluations}
    for seed in map(str, SEEDS):
        reference = (references[seed]["training"] or {}).pop("_deviation", None)
        for arm in ARMS:
            train = attempts[arm][seed]["training"]
            deviation = None if train is None else train.pop("_deviation")
            if deviation is not None and reference is not None and len(deviation) >= 2000:
                train["deviation_schedule_parity"] = {
                    "max_abs_difference_updates_1_2000": max(abs(a - b) for a, b in zip(deviation[:2000], reference)),
                    "max_abs_departure_from_0_05_after_2000": max((abs(value - .05) for value in deviation[2000:]),
                                                                   default=None)}
    hypotheses = {"H1_budget": {}, "H2_margin": {}, "H3_yaw": {}}
    for seed in map(str, SEEDS):
        b, j, y = (attempts[arm][seed] for arm in "BJY")
        hypotheses["H1_budget"][seed] = h1(b["training"], b["evaluations"])
        hypotheses["H2_margin"][seed] = h2(b["training"], j["training"], b["evaluations"]["5000"], j["evaluations"]["5000"])
        hypotheses["H3_yaw"][seed] = h3(b["evaluations"]["5000"], y["evaluations"]["5000"])
    verdicts = {name: {"verdict": verdict(rows), "seeds": {seed: None if row is None else
                                                           {"requirements": row[0], "all_met": all(
                                                               value["met"] for value in row[0].values()),
                                                            "predicted_outcomes_met": sum(row[1]),
                                                            "predicted_outcomes": len(row[1])}
                                                           for seed, row in rows.items()}}
                for name, rows in hypotheses.items()}
    if workspace.files != {name: digest for name, digest in PINS.items() if name in workspace.files} or not set(
            workspace.files) <= set(PINS):
        raise SystemExit("PINS differ from the retained files. Measured pins: " + json.dumps(workspace.files, indent=4))
    report = {"schema": "hexapod_reward_v4_controlled_comparison_v1", "analysis_sha256": sha(__file__),
              "remote_directory": REMOTE, "work_packet": "reward-v4-omnidirectional-controlled-comparison",
              "arms": ARMS, "seeds": list(SEEDS), "evaluation_updates": list(UPDATES), "methods": METHODS,
              "sha256": workspace.files, "complete_attempts": sorted(workspace.complete),
              "references": references, "attempts": attempts, "hypotheses": verdicts,
              "native_started_by_analysis": False, "stage2_complete": False}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verdicts": {name: value["verdict"] for name, value in verdicts.items()},
                      "references": {seed: (value["evaluation_2000"] or {}).get("metrics", {}).get("tracking_E")
                                     for seed, value in references.items()}}))


if __name__ == "__main__":
    main()
