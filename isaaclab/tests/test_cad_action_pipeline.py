"""CAD identity, named action mapping, failure boundaries and Torch parity."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for package in ("hexapod_core", "hexapod_runtime", "hexapod_env"):
    sys.path.insert(0, str(ROOT / "packages" / package))

from hexapod_core import cad_manifest_v2 as contract
from hexapod_runtime.cad_action_pipeline import create_cad_action_pipeline


URDF = ROOT / contract.URDF_PATH
NAMES = contract.RUNTIME_JOINT_NAMES
DEFAULTS = [contract.DEFAULT_JOINT_POSITIONS_BY_NAME[name] for name in NAMES]


def pipeline(names=NAMES, **overrides):
    kwargs = dict(manifest=contract.simulation_manifest(), urdf_path=URDF,
                  asset_name=contract.ASSET_NAME, articulation_joint_names=names)
    kwargs.update(overrides)
    return create_cad_action_pipeline(**kwargs)


def step(instance, action, command=(0.0, 0.0, 0.0), **overrides):
    kwargs = dict(action_joint_names=NAMES, command_frame=contract.COMMAND_FRAME)
    kwargs.update(overrides)
    return instance.step(action, command, **kwargs)


class CadManifestTests(unittest.TestCase):
    def test_source_identity_and_named_joint_values(self):
        self.assertEqual(hashlib.sha256(URDF.read_bytes()).hexdigest(), contract.URDF_SHA256)
        actual = {joint.get("name"): joint for joint in ET.parse(URDF).getroot().findall("joint")}
        self.assertEqual(set(actual), set(NAMES))
        limits_json = json.loads((URDF.parents[1] / "joint_limits.json").read_text())
        stance = json.loads((URDF.parents[1] / "stance.json").read_text())
        for name, joint in actual.items():
            group = name.split("_", 1)[1]
            lower, upper = contract.HARD_LIMITS_BY_NAME[name]
            self.assertEqual((float(joint.find("limit").get("lower")),
                              float(joint.find("limit").get("upper"))), (lower, upper))
            self.assertEqual((limits_json[group]["lower"], limits_json[group]["upper"]),
                             (lower, upper))
            self.assertEqual(contract.DEFAULT_JOINT_POSITIONS_BY_NAME[name], stance[f"{group}_rad"])
        self.assertEqual({contract.DEFAULT_JOINT_POSITIONS_BY_NAME[n] for n in NAMES if "femur" in n}, {-0.25})
        self.assertEqual({contract.DEFAULT_JOINT_POSITIONS_BY_NAME[n] for n in NAMES if "tibia" in n}, {-0.55})

    def test_asymmetric_soft_limits_shrink_about_midpoint(self):
        self.assertAlmostEqual(contract.SOFT_LIMITS_BY_NAME["lf_femur_pitch"][1], 0.492616775)
        self.assertEqual(contract.SOFT_LIMITS_BY_NAME["lf_tibia_pitch"], (-0.8825, 1.6825))

    def test_json_round_trip_and_checked_in_manifest(self):
        expected = contract.simulation_manifest()
        contract.validate_simulation_manifest(json.loads(json.dumps(expected)))
        self.assertEqual(contract.load_simulation_manifest(ROOT / "configs/mkii_v2_simulation_manifest.json"), expected)

    def test_wrong_or_stale_manifest_is_rejected(self):
        changes = {
            "schema_version": 1, "asset_name": "mock", "urdf_sha256": "0" * 64,
            "command_frame": "body", "action_scale_rad": 0.2,
            "stand_action_scale": 0.0, "policy_step_dt_s": 0.01,
            "soft_joint_pos_limit_factor": 1.0,
            "policy_joint_names": list(reversed(NAMES)),
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                manifest = contract.simulation_manifest()
                manifest[field] = value
                with self.assertRaises(ValueError):
                    pipeline(manifest=manifest)
        for bad in (math.nan, math.inf, -math.inf, True):
            manifest = contract.simulation_manifest()
            manifest["action_scale_rad"] = bad
            with self.assertRaises(ValueError):
                pipeline(manifest=manifest)

    def test_missing_extra_and_mutated_named_defaults_are_rejected(self):
        for mutation in ("missing", "extra", "pose", "limits"):
            manifest = contract.simulation_manifest()
            if mutation == "missing":
                del manifest["policy_joint_names"]
            elif mutation == "extra":
                manifest["allow_wrong_asset"] = True
            elif mutation == "pose":
                manifest["default_joint_positions_rad"]["lf_tibia_pitch"] = 2.17
            else:
                manifest["soft_joint_limits_rad"]["lf_femur_pitch"] = [-10.0, 10.0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                pipeline(manifest=manifest)

    def test_actual_asset_and_source_hash_are_mandatory(self):
        with self.assertRaises(ValueError):
            pipeline(asset_name="mkii_v1")
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.urdf"
            changed.write_bytes(URDF.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                pipeline(urdf_path=changed)

    def test_wrong_duplicate_and_missing_articulation_joints_are_rejected(self):
        for names in (NAMES[:-1], (NAMES[0],) * 18, ("revolute_1",) + NAMES[1:]):
            with self.assertRaises(ValueError):
                pipeline(names)

    def test_runtime_imports_without_site_packages(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            str(ROOT / "packages" / name) for name in ("hexapod_core", "hexapod_runtime")
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", "from hexapod_runtime.cad_action_pipeline import create_cad_action_pipeline"],
            env=environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class CadActionPipelineTests(unittest.TestCase):
    def test_explicit_shuffled_order_preserves_each_physical_joint(self):
        shuffled = list(NAMES)
        random.Random(19).shuffle(shuffled)
        reference = pipeline()
        reordered = pipeline(shuffled)
        reference.reset(DEFAULTS, joint_names=NAMES)
        reordered.reset([contract.DEFAULT_JOINT_POSITIONS_BY_NAME[n] for n in shuffled], joint_names=shuffled)
        action = [(index - 8) / 9 for index in range(18)]
        for _ in range(12):
            expected = dict(zip(NAMES, step(reference, action)))
            actual = dict(zip(shuffled, step(reordered, action)))
            self.assertEqual(actual, expected)

    def test_measured_reset_is_required_and_first_step_is_limited(self):
        instance = pipeline()
        with self.assertRaises(RuntimeError):
            step(instance, [1.0] * 18)
        instance.reset(DEFAULTS, joint_names=NAMES)
        targets = step(instance, [1.0] * 18)
        for target, initial in zip(targets, DEFAULTS):
            self.assertAlmostEqual(target - initial, 0.040)

    def test_zero_action_requests_actual_cad_pose(self):
        instance = pipeline()
        instance.reset(DEFAULTS, joint_names=NAMES)
        self.assertEqual(step(instance, [0.0] * 18), DEFAULTS)
        self.assertNotIn(2.17, instance.default_joint_positions)

    def test_order_and_frame_mismatches_do_not_advance_history(self):
        instance = pipeline()
        instance.reset(DEFAULTS, joint_names=NAMES)
        with self.assertRaises(ValueError):
            step(instance, [1.0] * 18, action_joint_names=tuple(reversed(NAMES)))
        with self.assertRaises(ValueError):
            step(instance, [1.0] * 18, command_frame="body")
        self.assertAlmostEqual(step(instance, [1.0] * 18)[0], 0.040)

    def test_bad_feedback_invalidates_session_until_valid_reset(self):
        for invalid in (math.nan, math.inf, -math.inf, 99.0):
            instance = pipeline()
            instance.reset(DEFAULTS, joint_names=NAMES)
            with self.assertRaises(ValueError):
                instance.reset([invalid] + DEFAULTS[1:], joint_names=NAMES)
            with self.assertRaises(RuntimeError):
                step(instance, [0.0] * 18)
        instance = pipeline()
        with self.assertRaises(ValueError):
            instance.reset(DEFAULTS, joint_names=tuple(reversed(NAMES)))

    def test_nonfinite_actions_and_commands_rejected_even_with_stand_passthrough(self):
        instance = pipeline()
        instance.reset(DEFAULTS, joint_names=NAMES)
        for invalid in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                step(instance, [invalid] + [0.0] * 17)
            with self.assertRaises(ValueError):
                step(instance, [0.0] * 18, command=(0.0, invalid, 0.0))
        self.assertEqual(step(instance, [0.0] * 18), DEFAULTS)

    def test_bad_shapes_rejected(self):
        instance = pipeline()
        instance.reset(DEFAULTS, joint_names=NAMES)
        for action, command in (([0.0] * 17, [0.0] * 3), ([0.0] * 18, [0.0] * 2)):
            with self.assertRaises(ValueError):
                step(instance, action, command)

    def test_nonfinite_output_cannot_escape_if_inner_pipeline_regresses(self):
        instance = pipeline()
        instance.reset(DEFAULTS, joint_names=NAMES)
        instance._pipeline.step = lambda *_: [math.inf] * 18
        with self.assertRaises(ValueError):
            step(instance, [0.0] * 18)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch is unavailable")
class CadActionTorchParityTests(unittest.TestCase):
    def test_multistep_named_pipeline_matches_training_helpers_float32_and_float64(self):
        import torch
        from hexapod_env.rewards.actions import (
            apply_command_conditioned_stand_action_scale,
            limit_processed_joint_target_slew,
        )
        generator = random.Random(20260904)
        # Offset initial poses near opposing limits, forcing both soft clamps
        # and slew transitions. Subsequent sign flips expose stale history.
        soft = [contract.SOFT_LIMITS_BY_NAME[name] for name in NAMES]
        initial = [bounds[index % 2] for index, bounds in enumerate(soft)]
        for dtype in (torch.float32, torch.float64):
            instance = pipeline()
            instance.reset(initial, joint_names=NAMES)
            previous = torch.tensor([initial], dtype=dtype)
            limits = torch.tensor([soft], dtype=dtype)
            defaults = torch.tensor([DEFAULTS], dtype=dtype)
            for index in range(30):
                action = [generator.uniform(-2.0, 2.0) for _ in NAMES]
                command = ([0.0, 0.0, 0.0], [0.1, -0.1, 0.3], [0.01, 0.0, 0.0])[index % 3]
                scaled = apply_command_conditioned_stand_action_scale(
                    torch.tensor([action], dtype=dtype).clamp(-1.0, 1.0),
                    torch.tensor([command], dtype=dtype),
                    stand_action_scale=contract.STAND_ACTION_SCALE,
                    command_active_threshold=contract.COMMAND_ACTIVE_THRESHOLD,
                )
                target = (contract.ACTION_SCALE_RAD * scaled + defaults).clamp(
                    min=limits[:, :, 0], max=limits[:, :, 1]
                )
                previous, fraction = limit_processed_joint_target_slew(
                    target, previous, torch.tensor([True]),
                    max_delta_rad_per_20ms=contract.SLEW_LIMIT_RAD_PER_20MS,
                    step_dt=contract.POLICY_STEP_DT_S,
                )
                actual = step(instance, action, command)
                for joint_index, value in enumerate(actual):
                    self.assertAlmostEqual(value, previous[0, joint_index].item(), delta=1e-6)
                self.assertAlmostEqual(instance.last_limited_fraction, fraction[0].item(), delta=1e-6)


if __name__ == "__main__":
    unittest.main()
