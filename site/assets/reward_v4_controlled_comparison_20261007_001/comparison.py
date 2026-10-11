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
}
METHODS = {
    "translation_passes": "Count of the eight 0.05 m/s translation probes whose evaluator result passes every existing check.",
    "translation_speed": "Mean over the eight translation probes of mean_velocity_mps (navigation frame, scored window) "
                         "projected onto the unit command direction.",
    "yaw_error": "The evaluator's yaw_error_rad_s, the mean absolute yaw-rate error over the scored window, for each sign; "
                 "mean_yaw_rate_rad_s gives the achieved sign.",
    "tracking_E": "(sum of planar_error_mps / 0.05 over the eight translations + sum of yaw_error_rad_s / 0.2 over the two "
                  "yaw probes) / 10.",
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
