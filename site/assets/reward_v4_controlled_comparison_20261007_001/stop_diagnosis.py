"""Diagnose joint drift at rest in the retained stop and quiet traces of the two full-command-bank reward v4 policies.

The script reads the 13-probe evaluation of each reference attempt at update 2000 and its three zero-command probes:
quiet 20 s, quiet 32 s and the stop after forward 0.05 m/s. Over each probe's scored window it compares target drift
with joint drift on the same joint, pairs each drifting leg with its contact-normal load and contact-loss events, and
reports the action spectrum and the requested and applied torque of each drifting joint. It starts no native process.
Run it from the repository root with PYTHONPATH set to that root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

import numpy as np

from locomotion.env_config import JOINT_NAMES, LEGS
from locomotion.evaluation import QUIET_GATES

REMOTE = "/srv/cupi/hexapod/runs/james/"
ATTEMPTS = {20260917: "ppo_v4_omni_seed20260917_20261007_001", 20260918: "ppo_v4_omni_seed20260918_20261007_001"}
UPDATE, CONTROL_HZ, PHYSICS_HZ, SUBSTEPS = 2000, 50, 400, 8
CASES = ("learning:quiet_20s", "learning:quiet_32s", "learning:forward_0.05_to_stop")
DRIFT_RANGE_RAD = QUIET_GATES["max_joint_position_range_rad"]
CONTACT_N = 1.
BANDS_HZ = ((0., 1.), (1., 5.), (5., 15.), (15., 25.01))
LOSS_MARGIN_SAMPLES = 10  # 25 ms on each side of a 400 Hz sample without distal contact
# PINS holds the sha256 of each file that the script reads, keyed by its path under REMOTE.
PINS = {
    "ppo_v4_omni_seed20260917_20261007_001/run/standing/checkpoint_update002000.json": "fd1a692d2332c256a6471f7e1a7338b10c397be0e15faaef401b23d7276330a5",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/summary.json": "6be9f20470bdcd3f8b84483283364ecac2b558d3d00fcc6dd393ac19c952d9b1",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/allocation.json": "bcaa1e514410285120bc52aa914049ab4febdef48102a454ecd7d7081f49d07b",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/report.json": "f2c1f3144358c9efe6e5e858773f4f2492164d00c41f10ef251badab5cab580d",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/capture.json": "8be43505a48e86ffa55269054995e3350f8ec7152ddc3efe6189c0adef52b371",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/report.json": "1a78952f41b6c5d974d699c4e49d372f15455fb33bc34d317d3facdebf1df598",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/capture.json": "c70699cb15b6af03da31b494e01454d7738be3a9573d9b45143ad002c945898f",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/report.json": "dda1cd9e02d210e4f55cec84e0b8dab7d36c7714434e559391e7b585187c56b8",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/capture.json": "20ddc19c2083e1babf036edd21529e90f8bb8d248c3c808de179a0c852b4b1c7",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/control_trace.npz": "a8faa541f8b61f8717d95260555ae2be05ed899cc87e2a386ba6c532ee5ced19",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_000.npz": "6abeae03d52f05afec2b02641fcf2e57c2c52fbe0044f246c012adfd233b413d",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_001.npz": "6c55cd843aa1d23943a3e287022e3a618411023725f567569b50588dc30e9d0c",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_002.npz": "b903a603495e9b08590cc226e8c38cf1741b370d0e7f30512b5f1dae0e70f7af",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_003.npz": "f54f1f4882b81dfbcfddc5192d5ae867edbd7afe8646cd81023cc28a0debee13",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_004.npz": "cc3e4cac31da8c1382485b5ad6373fc5512b352be1262db335f0d2d472ba3ee2",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_005.npz": "b7f8ff693e83f386b6c63bbbe0f14f7e3252570debe972216fb6e3989611081f",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_006.npz": "644bccc43507cb1f7e99df37ed5324f0afe7ed2ea70523c4dab7cf2ee89242eb",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_007.npz": "c4e205ff2b9b17acd58b02f793601933d6fe335a5caf36e563cb00b59eb7ad95",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_008.npz": "11d8eb3d3fde45d69142017248b8dd0afcc143d1363d4f8709fcee0b95641d7d",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_009.npz": "4946a3b367ae212eba59885880efd81b7883c9f127ba3c3b124560dbfe0224c6",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/control_trace.npz": "1ba46136a6936af232f285511ac1b7aea68a05c40d721f55167e02de758ddcb4",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_000.npz": "5c5d911361dde22807614d848ff87e1e356b5394df75efe7da38a210a72a90b1",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_001.npz": "e435f7f2b9899c19c1124e76d3163eeeab320939b45ec19b246321e221fb510b",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_002.npz": "2b276e3acf8bc5d622da1db5b4b247f858c1ec9c68ca387365b2f814f76f6d07",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_003.npz": "4913f7fad00530febab993921b094e923f202aee5a87887b24c7d4a537e66184",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_004.npz": "dad57cd35f9503326cfebf0ca9e1b46c27845aa131daf43efca18530a678f0f9",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_005.npz": "c43b76bef4a40e8e7fa8efc5654a38227093e0a767a677c259a58ff08e35bc62",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_006.npz": "77a8de31b62ce93f2fed598e0a07fc1a2308f2d80f3d7c529dc756d1dae543b5",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_007.npz": "d3e479c91b932d36eb2154545eb3438e60a09307a9ec6ea1d2d78873c27ad514",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_008.npz": "53636bfd1367d519ccbd60f28c3543e0bc15e14cf80f99e042f28f768f7e1d1e",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_009.npz": "1f7f8e360d2dc7dc70a3f3a2a6b4a4cb437bc1e3042a3b66504ebe1ad9e6b6c7",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_010.npz": "5d4ab381f8538f847db199fc553bed21198e26be22f3defff81a7d9c82d23831",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_011.npz": "1b4fac270452731b5c5a675ebc226e316312487c8efe36cd909e73e4bdd50342",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_012.npz": "6c4544ba789e9f4aeac8d78c3542dc033666eeb3ed77573624f996561569ab5b",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_013.npz": "148ced7d2818eecda8dc19fd8593f28215e421a9c2f44734f35ae445218f2a05",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_014.npz": "2f0db56c5809ceedc9f206bb4aa2e886237202d290d6b656aab04ddfdebb83df",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_015.npz": "65fdde916003a9229280d36fdce29113da01578686332c7ff5fed7a6b3488bc9",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/control_trace.npz": "71ade219777563a25eec3de25b4dccca1213f1a866e05aee4ac3555e4010189a",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_000.npz": "78ee1803d99447b0b5e7a270a5c4911383b37846c429280bd6fb8fe57a352739",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_001.npz": "bbae4d7ce31aaa9a0056dc585ab6837dbf9689bffb328cb76f81e99a7f3b72ea",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_002.npz": "4c259cf5698e2d07b8cc718190340ad296d089875d05cb393e37bb72cbb4d2e0",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_003.npz": "fcec1c8bebc5da5c045ac82e0f3485318c42e9e1e27df686dcae6a371c274fc5",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_004.npz": "2222d58d9372af28b63c15587f1bac5cb43dbf2043510d482f3c29244a3d4126",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_005.npz": "116e9e91ce85748590a47ed65ca52e45d6727c26308000aee9bf84a784c0e47b",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_006.npz": "60093e20832d5806690ca28796601d2fff435fb5f587e77f0d77e11eb2809af8",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_007.npz": "5c74d9f350bce95d0137f20eb3dd10bd616bbd47f57279664739ac152544b9a3",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_008.npz": "255f200bfa4d46b8bf43529f16e9bf0c4a38a7c6e37d668a389bbf32c081378b",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_009.npz": "9a27994e189622fa29f28d9099b56668b6ffc58b323457ee33122368be7d4376",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_010.npz": "0e00dad3a8d3a7af0a834200e15a1ed2e4d83950193b9f00b027de123ed4c292",
    "ppo_v4_omni_seed20260918_20261007_001/run/standing/checkpoint_update002000.json": "6200381bc9ce36533091be81642fe0f3e30b0e4411df25b7e1df99dc5eada102",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/summary.json": "e7696bdf0af0384fa4ab4812d97cc59590df1e8e24a93e198e81ad7b3e1c27ad",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/allocation.json": "be5216ea37a48dc3e7f66c8eabcc9c626a59e29832e8927abb89c4a906643278",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/report.json": "2de98f828e4c165786bb445b16a86fc31d9821c1ffc52018d5881a7f67a9c053",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/capture.json": "5405ce19ddb1392accac97006287a13f65795f28539bc11b1f809e3eaea2abdc",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/report.json": "908467d930828f9ff301bbbb953192f10f0a23a5cd8980ff8c59f03995ada01f",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/capture.json": "03baea7bc5e09e119a66c471e72a4a07087698e00d0430a7c94114b3c17da278",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/report.json": "9a07ad58fd13746e0acf598830c571df3f99b59929f38e4d050e3659e6f77fc8",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/capture.json": "0c93ed389bd0c0b2033232af8bc251693405f8048a12f22d3566135d4352a1b4",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/control_trace.npz": "295ade1201fb9532951ea949bddd6fcbc50f1612eef75bbc963218774285165f",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_000.npz": "248ac67c9689b4d9884d92a176036a894a154ad7f97b548850fc391508fcd373",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_001.npz": "fc83482276d3d6b291c2c827c10ea4f7500d45b27178d1cbadc5a34932ab673b",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_002.npz": "9faf96d921439aa1c244984f20dd2d35c8509a044bbc53b6a2b51a16c42758e7",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_003.npz": "232fb1d4d01ee474fb3f723ecdcadf767f4f6fa114c9edbe44888a200f9da7da",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_004.npz": "5f7c73cce51e6ef4bd5a9642b707fc043a0c76b7f91beea80833bca173cb8c9e",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_005.npz": "445097aa6d42af6e200206391c5594d436b490152a365f41fde12653544c0494",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_006.npz": "be2e1d66da5b256249e587ef477c5a2bcf8eea540d237d868969dda534841424",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_007.npz": "090d8ccca37cb84d61237b8aa63fb97c61d6dbab5d9b5ede16794f7775da6a57",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_008.npz": "278a29f3c75e64369773311deb408d5da68a759af81a4cc8bf8ca6a6b5524324",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_010/native400hz/substeps_009.npz": "1a11b7ef4c5da9476b23a978e135ae0975562230284ec7efa414c27e479e1399",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/control_trace.npz": "68c7cc3ca9afa1b837547c77a44aa8c21eaea8cad08faaaa364d7fd4f9a0b314",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_000.npz": "41077cef4dfd13fbba17ba3d354f5bea9bcad20728c92829b4de82db5721aa89",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_001.npz": "20baa329badc8ee2fbb1ce971928fd659c9a0b749a574e5cf24770146e4ac8d4",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_002.npz": "3e82658e575f8ef7829cf9a63e19293a81675a1ad223b52fe34a1c8734fcfbe2",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_003.npz": "ce85024c146e3e6b8376f14fb0e5d85e6b49704e2eb7938b82f0f16fa47baff9",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_004.npz": "727224dde288b59f40759a9d689b2fa9a085382034ba12fd022fe9043e0c7bec",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_005.npz": "7ef016f1a62ae7528e65b6680cd2f1df9bba8b8ffe414e21f4bd920b3732e3f0",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_006.npz": "53073840094d911028a932c03af20183f38fa86f5149d2f7d534ea421128d62e",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_007.npz": "60e667cc00ebfd5f5b2dcc80e3af551db556b8d6e431d9389a0d5f10e434cd92",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_008.npz": "4b03a076fe1ae79183164520ec7b680d6dd1b4f888b78e7fecc5cf0017146057",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_009.npz": "d58512859b3ad19dc502326044130f7e6ca3529b478bed4a200e1e28598b75bd",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_010.npz": "8273d07cf0eb7f4e7eb339479d2e55a7649dc6e13f818aae91b210ab373ff3c8",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_011.npz": "e830d9c1313544d4499d970a8f0767410f7ad214b61b7085027dfd9f9b6d0d34",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_012.npz": "c2bd8cf04dbb3c589443a0a7c39318ccb7452b31a89327e881ca55331369e633",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_013.npz": "31fdf0d97d9f1b12a75476250f0be2cd04c7b7a45ce54527d50771d16c781b91",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_014.npz": "711712d5486efea03e28809d89d2a642f174772a891a1a605a342d23af9bc246",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_011/native400hz/substeps_015.npz": "05dd93ccdd5afd0c858c885062d847344c73d1e001f3b682a60620452198cafc",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/control_trace.npz": "dd4528129fa0ce88b1e710f1f932fbaf0bcc598a6c66403f97d49c4821a02924",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_000.npz": "d6fbe89455489532e1ec183a328c21515db4089405d81ec96c99a8f58727405a",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_001.npz": "5fa843fe0089efcb4a98f5a3f34c475392ab8515176079de7d57b55c2fc5952f",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_002.npz": "e0a00c0d5d227e2e661b39632f1e2b5768d8e1deb1ecf925259f4c6e1a8cb889",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_003.npz": "eadb11b15beb83f9e76ff9ee297202ebc6833073a5d03ccecfb9f92d8d9994fa",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_004.npz": "b7fdfe51ba379563f8a79c2552568ed0414cfb8ffc1ada79a6c5b2564daea507",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_005.npz": "461e4297c0416f137c8f8e5856516616b32842fcd7663295f5a051adfc99da14",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_006.npz": "ba6d295ea1abd1cb167cc6b5040457d117663260c1aa74a36c7850cc09f8c871",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_007.npz": "772eeebaaeaf3f7d177196aa41634b24722e078f9ea44f192cff829117d69d39",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_008.npz": "aa45ef3bfc609b70bc34c62304518431e55580437e41f33ee17afdcde2b19a55",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_009.npz": "9be90c3daabaacb312e4dae3aabe9550caecdbc6b7f582e98f7f16be2322d04c",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate/run/standing/evaluation/batch_012/native400hz/substeps_010.npz": "09fcab76210a74e69fb639883a684c6d472d2baedf475ea1425fb48a447d63ca",
}
METHODS = {
    "scope": "Two retained policies, one seed each, update 2000, one deterministic trial per probe (actor mean without "
             "sampling noise). The record measures three zero-command probes; it adds no acceptance limit and retunes "
             "no quiet reward term.",
    "window": "Each probe uses the evaluator's scored window: window_start_control from its result to the last recorded "
              "control. The stop probe holds a zero command from control 400 and scores from control 500.",
    "drift": "For each joint, target_drift and joint_drift are the last minus the first value of joint_target_rad and "
             "joint_position_rad in the window; range is the peak-to-peak value; slope is the least-squares rate in rad/s. "
             "offset is the mean of target minus position over the first and the last 50 controls. A drifting joint has a "
             "position range above the quiet gate's max_joint_position_range_rad bound of 0.02 rad.",
    "contact": "Forces are contact-normal forces: the magnitude of each foot's recorded distal normal-force resultant at "
               "400 Hz. They exclude tangential friction, so the record makes no friction or slip-force claim. A "
               "contact-loss sample is a 400 Hz sample with distal_contact false; an event is a run of such samples. "
               "support_share is the foot's mean world-Z normal force divided by the mean sum over six feet and the "
               "nonfoot categories. path_near_loss is the share of the leg's 400 Hz joint path length (sum of |dq| per "
               "sample over its three joints) inside 25 ms of a contact-loss sample.",
    "spectrum": "policy_action of each drifting joint minus its window mean, rFFT at 50 Hz. peak_hz is the largest non-DC "
                "bin; band shares divide each band's non-DC power by the total non-DC power.",
    "torque": "Requested torque is computed_torque_nm, the motor command before the effort ceiling; applied torque is "
              "applied_torque_nm after it. Both come from the 400 Hz capture over the window and stay separate fields: "
              "signed mean, RMS and peak magnitude per joint.",
    "continuity": "Each substep file must hold sequence, control_index * 8 + substep_index and explicit_counter in one "
                  "contiguous run from the capture's initial counter; its rows for the window must cover every window "
                  "control eight times. Every array read must be finite.",
}
PROPOSED_MECHANISMS = [
    {"status": "proposed, not tested by this record",
     "mechanism": "The policy commands the drift. Joints follow a slowly moving target with a near-constant target-minus-"
                  "position offset, so the servo tracks the command and the slow change starts in the action."},
    {"status": "proposed, not tested by this record",
     "mechanism": "Reward v4 charges a slow ramp almost nothing. quiet_target_motion and quiet_joint_rate are quadratic in "
                  "the per-control change, so the measured drift rates cost orders of magnitude less per control than "
                  "the gait and tracking terms pay; quiet_rate_cost_at_median_drift gives the computed values. A gradient "
                  "that small would leave a slow closed-loop creep of the target in place."},
    {"status": "proposed, not tested by this record",
     "mechanism": "Brief contact losses of a lightly loaded foot and joint motion of that leg occur together in some "
                  "windows. The record does not order them in time beyond the 25 ms pairing, so it does not establish "
                  "whether a loss starts the motion or the motion unloads the foot."}]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(enabled, names, destination):
    """Copy the missing named files from Spark through one tar stream when the caller enables it."""
    missing = [name for name in names if enabled and not (destination / name).exists()]
    if missing:
        destination.mkdir(parents=True, exist_ok=True)
        command = "tar -C " + shlex.quote(REMOTE) + " -cf - " + " ".join(map(shlex.quote, missing))
        process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(destination, filter="data")
        assert process.wait() == 0


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "rms": float(np.sqrt((values ** 2).mean())), "peak_abs": float(np.abs(values).max())}


def slope(values):
    time = np.arange(len(values)) / CONTROL_HZ
    return float(np.polyfit(time, values.astype(np.float64), 1)[0])


def spectrum(values):
    x = values.astype(np.float64) - values.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    frequency = np.fft.rfftfreq(len(x), 1 / CONTROL_HZ)
    total = power[1:].sum()
    peak = int(power[1:].argmax()) + 1
    return {"std": float(x.std()), "peak_hz": float(frequency[peak]),
            "peak_share": float(power[peak] / total) if total else None,
            "band_shares": {f"{low:g}-{min(high, 25):g}_hz": float(power[1:][(frequency[1:] >= low) & (frequency[1:] < high)].sum() / total)
                            if total else None for low, high in BANDS_HZ}}


def runs(mask):
    """Start index and length of each run of true values."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), (stops - starts).tolist()))


def substeps(folder, capture, first, stop):
    """Concatenate the 400 Hz rows of controls first to stop - 1 after continuity checks."""
    keys = ("control_index", "distal_contact", "distal_force_world_n", "nonfoot_force_world_n", "joint_position_rad",
            "computed_torque_nm", "applied_torque_nm")
    count, parts = 0, {key: [] for key in keys}
    for name in capture["substep_files"]:
        with np.load(folder / "native400hz" / name, allow_pickle=False) as data:
            step = count + np.arange(len(data["sequence"]))
            assert (data["sequence"] == step).all() and (data["explicit_counter"] == capture["initial_counter"] + step + 1).all()
            assert (data["control_index"] * SUBSTEPS + data["substep_index"] == step).all()
            chosen = (data["control_index"] >= first) & (data["control_index"] < stop)
            for key in keys:
                value = data[key][chosen]
                assert value.dtype.kind == "b" or np.isfinite(value).all(), key
                parts[key].append(value)
            count += len(step)
    assert count == capture["steps"] == capture["final_counter"] - capture["initial_counter"]
    rows = {key: np.concatenate(value) for key, value in parts.items()}
    assert (rows["control_index"] == np.repeat(np.arange(first, stop), SUBSTEPS)).all(), "incomplete window"
    return {key: (value[:, 0] if key != "control_index" else value) for key, value in rows.items()}


def probe(folder, result, capture):
    """Measure drift, contact, spectrum and torque over one probe's scored window."""
    with np.load(folder / "control_trace.npz", allow_pickle=False) as data:
        trace = {key: data[key][:, 0] for key in ("joint_target_rad", "joint_position_rad", "policy_action", "command")}
        assert not data["reset"].any() and all(np.isfinite(value).all() for value in trace.values())
    first, stop = result["window_start_control"], result["recorded_controls"]
    assert stop == len(trace["command"]) and not trace["command"][first:stop].any()
    target, position, action = (trace[key][first:stop].astype(np.float64)
                                for key in ("joint_target_rad", "joint_position_rad", "policy_action"))
    joints = {}
    for index, name in enumerate(JOINT_NAMES):
        joints[name] = {
            "target_drift_rad": float(target[-1, index] - target[0, index]),
            "joint_drift_rad": float(position[-1, index] - position[0, index]),
            "target_range_rad": float(np.ptp(target[:, index])), "joint_range_rad": float(np.ptp(position[:, index])),
            "target_slope_rad_s": slope(target[:, index]), "joint_slope_rad_s": slope(position[:, index]),
            "offset_first_50_rad": float((target[:50, index] - position[:50, index]).mean()),
            "offset_last_50_rad": float((target[-50:, index] - position[-50:, index]).mean())}
    drifting = [name for name, row in joints.items() if row["joint_range_rad"] > DRIFT_RANGE_RAD]
    raw = substeps(folder, capture, first, stop)
    force = np.linalg.norm(raw["distal_force_world_n"], axis=-1)
    support = raw["distal_force_world_n"][..., 2].sum(-1) + raw["nonfoot_force_world_n"][..., 2].sum(-1)
    path = np.abs(np.diff(raw["joint_position_rad"].astype(np.float64), axis=0, prepend=raw["joint_position_rad"][:1]))
    legs = {}
    for leg_index, leg in enumerate(LEGS):
        lost = ~raw["distal_contact"][:, leg_index]
        events = runs(lost)
        near = np.convolve(lost.astype(float), np.ones(2 * LOSS_MARGIN_SAMPLES + 1), "same") > 0
        leg_path = path[:, 3 * leg_index:3 * leg_index + 3].sum(-1)
        members = [JOINT_NAMES[3 * leg_index + offset] for offset in range(3)]
        legs[leg] = {
            "drifting_joints": [name for name in members if name in drifting],
            "contact_normal_force_n": {**stats(force[:, leg_index]), "minimum": float(force[:, leg_index].min()),
                                       "p05": float(np.quantile(force[:, leg_index], .05)),
                                       "below_1n_fraction": float((force[:, leg_index] <= CONTACT_N).mean())},
            "support_share": float(raw["distal_force_world_n"][:, leg_index, 2].mean() / support.mean()),
            "contact_loss_samples": int(lost.sum()), "contact_loss_events": len(events),
            "longest_contact_loss_ms": max((length for _, length in events), default=0) * 1000 / PHYSICS_HZ,
            "first_contact_loss_control": int(raw["control_index"][events[0][0]]) if events else None,
            "joint_path_rad": float(leg_path.sum()),
            "path_near_loss": float(leg_path[near].sum() / leg_path.sum()) if leg_path.sum() else None}
    torque = {name: {"requested_nm": stats(raw["computed_torque_nm"][:, index]),
                     "applied_nm": stats(raw["applied_torque_nm"][:, index])}
              for index, name in enumerate(JOINT_NAMES) if name in drifting or name.endswith("coxa_yaw")}
    return {
        "first_control": first, "last_control": stop - 1, "pass": result["pass"], "failed_bounds": result["failed_bounds"],
        "evaluator_metrics": {key: result["metrics"].get(key) for key in (
            "max_joint_position_range_rad", "max_joint_velocity_rms_rad_s", "max_target_step_abs_p95_rad_per_20ms",
            "max_planar_excursion_m", "max_heading_excursion_deg")},
        "missing_six_toe_samples": int(result["checks"]["missing_six_toe_count_400hz"]["value"]),
        "drifting_joints": drifting, "joints": joints, "legs": legs,
        "action_spectra": {name: spectrum(action[:, JOINT_NAMES.index(name)]) for name in drifting},
        "torque": torque,
        "total_vertical_support_n": stats(support)}


def observations(records):
    """Aggregate the per-probe measurements into the statements the record reports as observed."""
    from locomotion.task_v4 import REWARD_V4_CONFIG as reward
    rows = []
    for seed, record in records.items():
        for case, row in record["probes"].items():
            drifting = row["drifting_joints"]
            joints = [row["joints"][name] for name in drifting]
            spectra = [row["action_spectra"][name] for name in drifting]
            legs = {leg: value for leg, value in row["legs"].items() if value["contact_loss_events"]}
            slopes = sorted(abs(joint["target_slope_rad_s"]) for joint in joints)
            median = slopes[len(slopes) // 2] if slopes else None
            requested = [value["requested_nm"]["peak_abs"] for value in row["torque"].values()]
            applied = [value["applied_nm"]["peak_abs"] for value in row["torque"].values()]
            rows.append({
                "seed": seed, "case_id": case, "pass": row["pass"], "drifting_joint_count": len(drifting),
                "same_sign_target_and_joint_drift": sum((joint["target_drift_rad"] > 0) == (joint["joint_drift_rad"] > 0)
                                                        for joint in joints),
                "largest_offset_change_rad": max((abs(joint["offset_last_50_rad"] - joint["offset_first_50_rad"])
                                                  for joint in joints), default=None),
                "median_abs_target_slope_rad_s": median,
                "quiet_rate_cost_at_median_drift": None if median is None else {
                    "quiet_target_motion_per_control": reward.quiet_target_weight
                        * (median * .02 / reward.quiet_target_scale_rad) ** 2,
                    "quiet_joint_rate_per_control": reward.quiet_joint_rate_weight
                        * (median / reward.quiet_joint_rate_scale_rad_s) ** 2},
                "smallest_below_1hz_action_share": min((value["band_shares"]["0-1_hz"] for value in spectra), default=None),
                "largest_15_25hz_action_share": max((value["band_shares"]["15-25_hz"] for value in spectra), default=None),
                "legs_with_contact_loss": {leg: {key: value[key] for key in (
                    "contact_loss_events", "longest_contact_loss_ms", "support_share", "path_near_loss")}
                    | {"mean_contact_normal_force_n": value["contact_normal_force_n"]["mean"]} for leg, value in legs.items()},
                "support_share_range": [min(value["support_share"] for value in row["legs"].values()),
                                        max(value["support_share"] for value in row["legs"].values())],
                "requested_peak_above_1_6nm": sum(value > 1.6 for value in requested),
                "largest_applied_peak_nm": max(applied)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    output, inputs = args.output or args.workspace / "stop_diagnosis.json", args.workspace / "inputs"
    if output.exists():
        raise FileExistsError("Use a fresh output path")
    measured, records = {}, {}
    for seed, attempt in ATTEMPTS.items():
        evaluation = attempt + "_evaluate/run/standing/evaluation/"
        record = attempt + "/run/standing/checkpoint_update%06d.json" % UPDATE
        names = [record, evaluation + "summary.json", evaluation + "allocation.json"]
        fetch(args.fetch, names, inputs)
        sidecar, summary, allocation = (json.loads((inputs / name).read_text()) for name in names)
        batches = {case: evaluation + "batch_%03d/" % allocation["selected_case_ids"].index(case) for case in CASES}
        heads = [batch + name for batch in batches.values() for name in ("report.json", "native400hz/capture.json")]
        fetch(args.fetch, heads, inputs)
        chunks = [batch + name for batch in batches.values() for name in (
            "control_trace.npz", *("native400hz/" + item for item in
                                   json.loads((inputs / batch / "native400hz/capture.json").read_text())["substep_files"]))]
        fetch(args.fetch, chunks, inputs)
        names += heads + chunks
        measured.update({name: sha(inputs / name) for name in names})
        assert sidecar["identity"]["seed"] == seed == allocation["seed"] and summary["allocation"] == allocation
        assert sidecar["checkpoint_sha256"] == allocation["checkpoint_sha256"] and sidecar["updates"] == UPDATE
        results, probes = {row["case_id"]: row for row in summary["results"]}, {}
        for case, batch in batches.items():
            report = json.loads((inputs / batch / "report.json").read_text())
            capture = json.loads((inputs / batch / "native400hz/capture.json").read_text())
            assert report["assigned_case_ids"] == [case] and report["results"] == [results[case]]
            assert report["checkpoint_sha256"] == allocation["checkpoint_sha256"] and capture["failure"] is None
            assert report["files"]["control_trace.npz"] == measured[batch + "control_trace.npz"]
            assert all(capture["files"][item] == measured[batch + "native400hz/" + item] for item in capture["substep_files"])
            probes[case] = {"batch": batch, **probe(inputs / batch, results[case], capture)}
        records[str(seed)] = {"attempt": REMOTE + attempt, "checkpoint_sha256": sidecar["checkpoint_sha256"],
                              "update": UPDATE, "probes": probes}
    if measured != PINS:
        raise SystemExit("PINS differ from the retained files. Measured pins: " + json.dumps(measured, indent=4))
    receipt = {"schema": "hexapod_reward_v4_stop_diagnosis_v1", "analysis_sha256": sha(__file__), "remote_directory": REMOTE,
               "cases": list(CASES), "drift_range_bound_rad": DRIFT_RANGE_RAD, "sha256": PINS, "methods": METHODS,
               "seeds": records, "observations": observations(records), "proposed_mechanisms": PROPOSED_MECHANISMS,
               "native_started_by_analysis": False}
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({seed: {case: {"drifting": row["drifting_joints"], "pass": row["pass"]}
                             for case, row in record["probes"].items()} for seed, record in records.items()}))


if __name__ == "__main__":
    main()
