"""CPU-only tests for the fail-closed Stage-2C sharded report merger."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ISAACLAB_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ISAACLAB_DIR))
MODULE_PATH = ISAACLAB_DIR / "merge_stage2c_probe_shards.py"
SPEC = importlib.util.spec_from_file_location(
    "hexapod_merge_stage2c_probe_shards", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
merger = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merger
SPEC.loader.exec_module(merger)


PARENT = "/workspace/logs/control/model_2.pt"
CANDIDATES = [f"/workspace/logs/probe/model_{index}.pt" for index in range(4)]


def _row(index: int, command: tuple[float, float, float]) -> dict[str, object]:
    vx, vy, yaw = command
    return {
        "index": index,
        "command": {
            "frame": "navigation",
            "body_vx_mps": vx,
            "body_vy_mps": vy,
            "yaw_rate_radps": yaw,
        },
        "samples": merger.admission.EXPECTED_SAMPLES,
        "measured_seconds": merger.admission.EXPECTED_MEASURED_SECONDS,
    }


def _report(checkpoint: str) -> dict[str, object]:
    return {
        "checkpoint": checkpoint,
        "task": merger.admission.TASK_ID,
        "command_frame": "navigation",
        "deterministic_policy": True,
        "seed": merger.admission.EXPECTED_SEED,
        "policy_step_seconds": merger.admission.EXPECTED_POLICY_STEP_SECONDS,
        "requested_steps": merger.admission.EXPECTED_REQUESTED_STEPS,
        "warmup_steps": merger.admission.EXPECTED_WARMUP_STEPS,
        "commands_evaluated_in_parallel": len(merger.admission.COMMAND_CONTRACT),
        "reset_stance_override": None,
        "startup_randomization_enabled": False,
        "action_processing": copy.deepcopy(merger.EXPECTED_ACTION_PROCESSING),
        "results": [
            _row(index, command)
            for index, (_, command) in enumerate(merger.admission.COMMAND_CONTRACT)
        ],
    }


def _payload(checkpoints: list[str]) -> dict[str, object]:
    return {
        "action_processing": copy.deepcopy(merger.EXPECTED_ACTION_PROCESSING),
        "evaluations": [_report(checkpoint) for checkpoint in checkpoints],
    }


class MergeStage2CProbeShardsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shard_a = _payload([PARENT, CANDIDATES[0], CANDIDATES[2]])
        self.shard_b = _payload([PARENT, CANDIDATES[1], CANDIDATES[3]])

    def merge(self) -> dict[str, object]:
        return merger.merge_payloads(
            self.shard_a,
            self.shard_b,
            parent_checkpoint=PARENT,
            candidate_checkpoints=CANDIDATES,
        )

    def test_valid_merge_keeps_one_exact_parent_and_caller_candidate_order(self):
        result = self.merge()
        self.assertEqual(result["action_processing"], merger.EXPECTED_ACTION_PROCESSING)
        self.assertEqual(
            [report["checkpoint"] for report in result["evaluations"]],
            [PARENT, *CANDIDATES],
        )
        self.assertEqual(result["evaluations"][0], self.shard_a["evaluations"][0])

    def test_parent_reports_must_be_exactly_equal_after_parsing(self):
        self.shard_b["evaluations"][0]["otherwise_valid_extra_field"] = "drift"
        with self.assertRaisesRegex(merger.MergeError, "parent reports differ"):
            self.merge()

    def test_expected_shard_membership_and_order_are_fail_closed(self):
        self.shard_a["evaluations"][1], self.shard_a["evaluations"][2] = (
            self.shard_a["evaluations"][2],
            self.shard_a["evaluations"][1],
        )
        with self.assertRaisesRegex(merger.MergeError, "checkpoint must be"):
            self.merge()

    def test_duplicate_or_parent_candidate_is_rejected_before_merge(self):
        for candidates, message in (
            ([CANDIDATES[0], CANDIDATES[0]], "contains duplicates"),
            ([PARENT, CANDIDATES[0]], "must not also appear"),
        ):
            with self.subTest(candidates=candidates), self.assertRaisesRegex(
                merger.MergeError, message
            ):
                merger.merge_payloads(
                    self.shard_a,
                    self.shard_b,
                    parent_checkpoint=PARENT,
                    candidate_checkpoints=candidates,
                )

    def test_every_playback_metadata_field_is_exact_and_type_strict(self):
        mutations = {
            "task": "wrong-task",
            "command_frame": "body",
            "deterministic_policy": 1,
            "seed": True,
            "policy_step_seconds": 0.021,
            "requested_steps": 499,
            "warmup_steps": 24,
            "commands_evaluated_in_parallel": 3,
            "reset_stance_override": {},
            "startup_randomization_enabled": True,
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                shard_a = copy.deepcopy(self.shard_a)
                shard_a["evaluations"][1][key] = value
                with self.assertRaisesRegex(merger.MergeError, key):
                    merger.merge_payloads(
                        shard_a,
                        self.shard_b,
                        parent_checkpoint=PARENT,
                        candidate_checkpoints=CANDIDATES,
                    )

    def test_wrapper_and_report_action_processing_are_exact(self):
        for location in ("wrapper", "report"):
            with self.subTest(location=location):
                shard_a = copy.deepcopy(self.shard_a)
                target = (
                    shard_a["action_processing"]
                    if location == "wrapper"
                    else shard_a["evaluations"][1]["action_processing"]
                )
                target["processed_joint_target_slew_limit_rad_per_20ms"] = 0.05
                with self.assertRaisesRegex(
                    merger.MergeError,
                    "processed_joint_target_slew_limit_rad_per_20ms",
                ):
                    merger.merge_payloads(
                        shard_a,
                        self.shard_b,
                        parent_checkpoint=PARENT,
                        candidate_checkpoints=CANDIDATES,
                    )

    def test_command_sample_and_duration_contracts_are_enforced(self):
        cases = (
            ("samples", 474, "samples"),
            ("measured_seconds", 9.49, "measured_seconds"),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                shard_a = copy.deepcopy(self.shard_a)
                shard_a["evaluations"][1]["results"][0][key] = value
                with self.assertRaisesRegex(merger.MergeError, message):
                    merger.merge_payloads(
                        shard_a,
                        self.shard_b,
                        parent_checkpoint=PARENT,
                        candidate_checkpoints=CANDIDATES,
                    )

        shard_a = copy.deepcopy(self.shard_a)
        shard_a["evaluations"][1]["results"][0]["command"]["body_vx_mps"] = 0.01
        with self.assertRaisesRegex(merger.MergeError, "command contract"):
            merger.merge_payloads(
                shard_a,
                self.shard_b,
                parent_checkpoint=PARENT,
                candidate_checkpoints=CANDIDATES,
            )

    def test_strict_loader_rejects_duplicate_keys_nonfinite_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"evaluations": [], "evaluations": []}\n')
            with self.assertRaisesRegex(merger.MergeError, "duplicate JSON"):
                merger.load_payload(duplicate)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"value": NaN}\n')
            with self.assertRaisesRegex(merger.MergeError, "non-finite JSON"):
                merger.load_payload(nonfinite)

            link = root / "link.json"
            link.symlink_to(duplicate)
            with self.assertRaisesRegex(merger.MergeError, "symlink"):
                merger.load_payload(link)

    def test_atomic_writer_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "merged.json"
            merger.write_json_exclusive_atomic(output, {"first": True})
            first_bytes = output.read_bytes()
            with self.assertRaises(FileExistsError):
                merger.write_json_exclusive_atomic(output, {"second": True})
            self.assertEqual(output.read_bytes(), first_bytes)
            self.assertEqual(json.loads(first_bytes), {"first": True})
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])


class ShardedLauncherStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            ISAACLAB_DIR / "deploy" / "screen-stage2c-probe-sharded"
        ).read_text(encoding="utf-8")

    def test_lock_hash_and_read_only_mount_contracts_are_present(self):
        self.assertIn("gpu_lock=/tmp/hexapod-isaac-gpu.lock", self.source)
        self.assertIn("flock -n 9", self.source)
        self.assertEqual(
            self.source.count('sha256sum -c "${evaluation_dir}/inputs.sha256"'),
            2,
        )
        self.assertIn('seen_checkpoint_numbers[${checkpoint_key}]', self.source)
        self.assertIn('"${workspace_host}:${workspace_container}:ro"', self.source)
        self.assertIn('"${output_host}:${output_container}:rw"', self.source)

    def test_merge_and_analysis_precede_the_only_success_marker(self):
        merge_index = self.source.index('python3 "${merger_host}"')
        analysis_index = self.source.index('python3 "${analyzer_host}"')
        success_index = self.source.index(
            'echo "formal sharded Stage2C probe screen complete:'
        )
        self.assertLess(merge_index, analysis_index)
        self.assertLess(analysis_index, success_index)
        self.assertIn("trap cleanup_on_exit EXIT", self.source)


if __name__ == "__main__":
    unittest.main()
