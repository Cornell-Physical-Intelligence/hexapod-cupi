"""CPU-only tests for Stage-2C admission-plus-probe analysis."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

ISAACLAB_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ISAACLAB_DIR))
MODULE_PATH = ISAACLAB_DIR / "analyze_stage2c_probe_sweep.py"
SPEC = importlib.util.spec_from_file_location(
    "hexapod_analyze_stage2c_probe_sweep", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


JOINT_NAMES = tuple(
    f"joint_{index:02d}" for index in range(analyzer.admission.EXPECTED_JOINT_COUNT)
)


def _row(
    command: tuple[float, float, float],
    index: int,
    *,
    torque_duty: float = 0.08,
) -> dict[str, object]:
    vx, vy, yaw = command
    stand = command == (0.0, 0.0, 0.0)
    rms_targets = (
        analyzer.admission.STAND_ABSOLUTE_STABILITY_TARGETS
        if stand
        else analyzer.admission.ABSOLUTE_STABILITY_TARGETS
    )
    return {
        "index": index,
        "command": {
            "frame": "navigation",
            "body_vx_mps": vx,
            "body_vy_mps": vy,
            "yaw_rate_radps": yaw,
        },
        "samples": analyzer.admission.EXPECTED_SAMPLES,
        "measured_seconds": analyzer.admission.EXPECTED_MEASURED_SECONDS,
        "mean_command_frame_linear_velocity_mps": [vx, vy, 0.0],
        "mean_command_frame_angular_velocity_radps": [0.0, 0.0, yaw],
        "planar_velocity_rmse_mps": 0.04,
        "falls": 0,
        "timeouts": 0,
        **{key: target * 0.5 for key, target in rms_targets.items()},
        **{
            key: target * 0.5
            for key, target in analyzer.admission.TAIL_STABILITY_TARGETS.items()
        },
        "torque": {
            "rms_applied_nm": 0.8,
            "max_per_joint_rms_applied_nm": 1.0,
            "computed_demand_over_rating_fraction": torque_duty,
            "maximum_computed_over_rating_burst_s": 0.08,
            "peak_abs_computed_nm": 3.6,
            "per_joint": [
                {
                    "name": name,
                    "rms_applied_nm": 0.8,
                    "peak_abs_applied_nm": 1.6,
                    "peak_abs_computed_nm": 3.6,
                    "applied_at_rating_fraction": 0.08,
                    "computed_demand_over_rating_fraction": 0.10,
                    "maximum_computed_over_rating_burst_s": 0.08,
                }
                for name in JOINT_NAMES
            ],
        },
    }


def _report(checkpoint: str, *, duty_at_0p24: float = 0.14) -> dict[str, object]:
    commands = [
        *(command for _, command in analyzer.admission.COMMAND_CONTRACT),
        *(command for _, command in analyzer.DIAGNOSTIC_COMMANDS),
    ]
    rows = [
        _row(
            command,
            100 + index,
            torque_duty=(duty_at_0p24 if abs(command[0] - 0.24) < 1.0e-9 else 0.05),
        )
        for index, command in enumerate(commands)
    ]
    rows.reverse()
    return {
        "checkpoint": checkpoint,
        # The canonical grader rejects this probe metadata.  The analyzer must
        # still select and grade the embedded four-command contract.
        "commands_evaluated_in_parallel": 7,
        "results": rows,
    }


def _short_report(
    checkpoint: str, *, expected_samples: int = 275
) -> dict[str, object]:
    report = _report(checkpoint)
    measured_seconds = (
        expected_samples * analyzer.admission.EXPECTED_POLICY_STEP_SECONDS
    )
    for row in report["results"]:
        row["samples"] = expected_samples
        row["measured_seconds"] = measured_seconds
    return report


class AnalyzeStage2CProbeSweepTest(unittest.TestCase):
    def test_extra_shuffled_rows_are_selected_by_command(self):
        payload = {
            "evaluations": [
                _report("/logs/control/model_119.pt"),
                _report("/logs/probe/model_39.pt"),
            ]
        }
        result = analyzer.analyze_payload(payload)

        self.assertTrue(result["formal_admission_eligible"])
        self.assertEqual(result["sample_contract"]["mode"], "formal_admission")
        self.assertEqual(
            result["sample_contract"]["expected_samples"],
            analyzer.admission.EXPECTED_SAMPLES,
        )
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["four_command_contract_safe_count"], 2)
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["input_result_row_count"], 7)
        self.assertEqual(len(evaluation["admission_results"]), 4)
        self.assertEqual(len(evaluation["diagnostic_results"]), 3)
        self.assertEqual(
            [item["label"] for item in evaluation["admission_results"]],
            [label for label, _ in analyzer.admission.COMMAND_CONTRACT],
        )
        self.assertTrue(evaluation["threshold_diagnostics"]["complete"])

    def test_duplicate_admission_command_is_rejected(self):
        report = _report("/logs/probe/model_39.pt")
        report["results"].append(copy.deepcopy(report["results"][-2]))
        # The penultimate source row after reversal is forward 0.16.
        duplicate_command = report["results"][-1]["command"]
        report["results"][-1] = next(
            copy.deepcopy(row)
            for row in report["results"][:-1]
            if row["command"] == duplicate_command
        )
        with self.assertRaisesRegex(analyzer.admission.ReportError, "more than once"):
            analyzer.analyze_payload({"evaluations": [report]})

    def test_worst_normalized_violation_names_the_gate(self):
        report = _report("/logs/probe/model_39.pt")
        command = dict(analyzer.admission.COMMAND_CONTRACT)["forward_0p20"]
        row = next(
            item
            for item in report["results"]
            if analyzer.admission._commands_match(
                analyzer.admission._command_from_row(item, "row"), command
            )
        )
        row["torque"]["computed_demand_over_rating_fraction"] = 0.30

        evaluation = analyzer.analyze_payload({"evaluations": [report]})["evaluations"][
            0
        ]
        worst = evaluation["worst_normalized_gate"]
        self.assertEqual(worst["command"], "forward_0p20")
        self.assertEqual(worst["metric"], "torque_duty")
        self.assertAlmostEqual(worst["ratio_to_limit"], 2.0)
        self.assertAlmostEqual(worst["normalized_violation"], 1.0)

    def test_threshold_boundary_reports_normalized_jump(self):
        report = _report("/logs/probe/model_39.pt", duty_at_0p24=0.14)
        evaluation = analyzer.analyze_payload({"evaluations": [report]})["evaluations"][
            0
        ]
        boundary = evaluation["threshold_diagnostics"][
            "threshold_boundary_0p23_to_0p24"
        ]
        self.assertIsNotNone(boundary)
        self.assertAlmostEqual(boundary["metric_deltas"]["torque_duty_fraction"], 0.09)
        self.assertEqual(
            boundary["largest_normalized_jump_metric"], "torque_duty_fraction"
        )
        self.assertAlmostEqual(boundary["largest_normalized_jump"], 0.60)

    def test_missing_probe_rows_are_reported_but_not_required_for_admission(self):
        report = _report("/logs/probe/model_39.pt")
        report["results"] = report["results"][3:]
        evaluation = analyzer.analyze_payload({"evaluations": [report]})["evaluations"][
            0
        ]
        self.assertTrue(evaluation["four_command_contract_safe"])
        self.assertFalse(evaluation["threshold_diagnostics"]["complete"])
        self.assertEqual(len(evaluation["diagnostic_results"]), 0)

    def test_short_screen_requires_explicit_diagnostic_mode(self):
        report = _short_report("/logs/probe/model_7.pt")
        with self.assertRaisesRegex(
            analyzer.admission.ReportError, "samples must be 475"
        ):
            analyzer.analyze_payload({"evaluations": [report]})

    def test_explicit_short_screen_ranks_but_is_never_formal_admission(self):
        payload = {
            "evaluations": [
                _short_report("/logs/control/model_2.pt"),
                _short_report("/logs/probe/model_7.pt"),
            ]
        }
        result = analyzer.analyze_payload(
            payload, diagnostic_expected_samples=275
        )

        self.assertFalse(result["formal_admission_eligible"])
        self.assertEqual(
            result["analysis_kind"],
            "stage2c_diagnostic_short_duration_probe_screen",
        )
        self.assertEqual(
            result["sample_contract"],
            {
                "mode": "diagnostic_short_duration",
                "expected_samples": 275,
                "expected_measured_seconds": 5.5,
                "warmup_steps": 25,
                "requested_steps": 300,
                "policy_step_seconds": 0.02,
                "formal_expected_samples": 475,
                "formal_expected_measured_seconds": 9.5,
                "formal_admission_eligible": False,
            },
        )
        self.assertEqual(len(result["ranked_checkpoints"]), 2)
        for evaluation in result["evaluations"]:
            self.assertFalse(evaluation["formal_admission_eligible"])
            self.assertEqual(
                evaluation["sample_contract_mode"],
                "diagnostic_short_duration",
            )
            for row in (
                *evaluation["admission_results"],
                *evaluation["diagnostic_results"],
            ):
                self.assertEqual(row["metrics"]["samples"], 275)
                self.assertEqual(row["metrics"]["measured_seconds"], 5.5)

    def test_diagnostic_mode_rejects_wrong_duration_and_non_short_contracts(self):
        report = _short_report("/logs/probe/model_7.pt")
        report["results"][0]["measured_seconds"] = 6.0
        with self.assertRaisesRegex(
            analyzer.admission.ReportError, "measured_seconds must be 5.5"
        ):
            analyzer.analyze_payload(
                {"evaluations": [report]}, diagnostic_expected_samples=275
            )

        formal_payload = {"evaluations": [_report("/logs/probe/model_7.pt")]}
        for invalid in (0, 475, 500, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                analyzer.admission.ReportError,
                "positive integer below",
            ):
                analyzer.analyze_payload(
                    formal_payload,
                    diagnostic_expected_samples=invalid,
                )


if __name__ == "__main__":
    unittest.main()
