"""CAD reset/axis integrity regressions, runnable without Isaac Sim or GPU."""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "packages/hexapod_env", ROOT / "packages/hexapod_core"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools.assets.audit_mkii_stance import (  # noqa: E402
    audit_stance, forward_kinematics, measure_stance, measure_reset_jitter, primitive_bottom_z,
)
from hexapod_env.assets.mkii_v2 import MKII_V2_ASSET as ASSET  # noqa: E402
from hexapod_env.assets.spec import MKII_V1_ASSET  # noqa: E402
from hexapod_core import cad_manifest_v2 as contract  # noqa: E402
from hexapod_core.frames import body_to_navigation, navigation_to_body  # noqa: E402

ASSEMBLY = ROOT / "robot/hexapod_mkii_assy"
URDF = ROOT / ASSET.urdf_path
STANCE = json.loads((ASSEMBLY / "stance_v2.json").read_text())


class InstanceConfigFieldTests(unittest.TestCase):
    def test_v2_config_supports_instance_only_inherited_sim_fields(self):
        """Reproduce the installed Lab 3 configclass shape without importing Kit."""
        @dataclass
        class SimConfig:
            dt: float = .01
            render_interval: int = 1

            def replace(self, **changes):
                return replace(self, **changes)

        class ParentConfig:
            def __init__(self):
                self.sim = SimConfig()

        class Articulation:
            def replace(self, **changes):
                return self

        path = ROOT / "packages/hexapod_env/hexapod_env/tasks/mkii_v2/config.py"
        source = ast.parse(path.read_text())
        source.body = [node for node in source.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        namespace = {name: getattr(contract, name) for name in dir(contract) if name.isupper()}
        namespace.update(configclass=lambda cls: cls, MKII_V2_ASSET=ASSET,
                         articulation_cfg_from_spec=lambda spec: Articulation(),
                         HexapodMkiiV1FlatEnvCfg=ParentConfig, HexapodPPORunnerCfg=object)
        self.assertFalse(hasattr(ParentConfig, "sim"))
        exec(compile(source, str(path), "exec"), namespace)
        actual = namespace["HexapodMkiiV2FlatEnvCfg"].sim
        self.assertEqual(actual.dt, contract.PHYSICS_DT_S)
        self.assertEqual(actual.render_interval, contract.DECIMATION)
        self.assertEqual(ParentConfig().sim.dt, .01)


class StanceGeometryTests(unittest.TestCase):
    def test_six_feet_have_explicit_clearance_and_body_is_clear(self):
        result = audit_stance(URDF, STANCE)
        self.assertEqual(result["links"], 19)
        self.assertEqual(result["joints"], 18)
        self.assertEqual(result["collision_primitives"], 171)
        self.assertEqual(set(result["foot_clearance_at_reset_m"]), set(ASSET.leg_names))
        for clearance in result["foot_clearance_at_reset_m"].values():
            self.assertGreaterEqual(clearance, 0.005)
            self.assertLess(clearance, 0.00505)
        self.assertGreater(result["nonfoot_min_clearance_at_reset_m"], .030)
        self.assertLess(result["foot_vertical_spread_m"], .00005)
        self.assertAlmostEqual(result["geometric_ground_contact_root_height_m"], .137963888698354, places=12)
        self.assertEqual(result["physical_validation"], "not_performed")

    def test_old_8_mm_penetration_is_rejected(self):
        old_stance = dict(STANCE, reset_root_height_m=.130)
        result = measure_stance(URDF, old_stance)
        self.assertAlmostEqual(min(result["foot_clearance_at_reset_m"].values()), -.007963888698354, places=12)
        with self.assertRaisesRegex(ValueError, "reset clearance"):
            audit_stance(URDF, old_stance)

    def test_stale_geometry_hash_and_nominal_height_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Source URDF hash"):
            audit_stance(URDF, dict(STANCE, source_urdf_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Nominal height"):
            audit_stance(URDF, dict(STANCE, root_height_m=.124))
        with self.assertRaisesRegex(ValueError, "reset clearance"):
            audit_stance(URDF, dict(STANCE, reset_clearance_m=.006))
        with self.assertRaisesRegex(ValueError, "finite"):
            audit_stance(URDF, dict(STANCE, reset_root_height_m=float("nan")))

    def test_stance_and_asset_match_each_other_and_runtime_contract(self):
        self.assertEqual(ASSET.nominal_height_m, STANCE["root_height_m"])
        self.assertEqual(ASSET.reset_root_height_m, STANCE["reset_root_height_m"])
        self.assertEqual(ASSET.runtime_joint_names, contract.RUNTIME_JOINT_NAMES)
        self.assertEqual(ASSET.default_joint_positions(), dict(contract.DEFAULT_JOINT_POSITIONS_BY_NAME))
        self.assertEqual(STANCE["source_urdf_sha256"], contract.URDF_SHA256)
        self.assertEqual(ASSET.urdf_path, contract.URDF_PATH)
        self.assertEqual(ASSET.name, contract.ASSET_NAME)
        self.assertNotEqual(ASSET.usd_path_container, MKII_V1_ASSET.usd_path_container)
        self.assertNotEqual(ASSET.usd_env_var, MKII_V1_ASSET.usd_env_var)
        self.assertEqual(MKII_V1_ASSET.reset_root_height_m, .130)

    def test_actual_joint_names_limits_and_soft_reset_range(self):
        joints = ET.parse(URDF).getroot().findall("joint")
        self.assertEqual({j.get("name") for j in joints}, set(ASSET.runtime_joint_names))
        limits_json = json.loads((ASSEMBLY / "joint_limits.json").read_text())
        for joint in joints:
            name = joint.get("name")
            group = name.split("_", 1)[1]
            limit = joint.find("limit")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            self.assertEqual((lower, upper), contract.HARD_LIMITS_BY_NAME[name])
            self.assertEqual((lower, upper), (limits_json[group]["lower"], limits_json[group]["upper"]))
            soft_low, soft_high = contract.SOFT_LIMITS_BY_NAME[name]
            self.assertGreaterEqual(ASSET.default_joint_positions()[name], soft_low)
            self.assertLessEqual(ASSET.default_joint_positions()[name], soft_high)
            self.assertEqual(STANCE[f"{group}_rad"], ASSET.default_joint_positions()[name])

    def test_missing_joints_invalid_limits_and_cycles_fail_closed(self):
        root = ET.parse(URDF).getroot()
        positions = ASSET.default_joint_positions()
        missing = dict(positions)
        del missing[ASSET.coxa_joints[0]]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            forward_kinematics(root, missing)
        with self.assertRaisesRegex(ValueError, "outside URDF limits"):
            forward_kinematics(root, dict(positions, lf_femur_pitch=2.0))
        root.find("joint/parent").set("link", "lf_tibia")
        with self.assertRaisesRegex(ValueError, "cyclic"):
            forward_kinematics(root, positions)

    def test_analytic_collision_bounds_include_orientation(self):
        transform = np.eye(4)
        transform[2, 3] = 2.
        sphere = ET.fromstring('<sphere radius=".3"/>')
        self.assertAlmostEqual(primitive_bottom_z(transform, sphere), 1.7)
        cylinder = ET.fromstring('<cylinder radius=".2" length="1.2"/>')
        self.assertAlmostEqual(primitive_bottom_z(transform, cylinder), 1.4)
        transform[:3, :3] = ((1., 0., 0.), (0., 0., -1.), (0., 1., 0.))
        self.assertAlmostEqual(primitive_bottom_z(transform, cylinder), 1.8)
        box = ET.fromstring('<box size=".4 .8 1.2"/>')
        self.assertAlmostEqual(primitive_bottom_z(transform, box), 1.6)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            primitive_bottom_z(transform, ET.fromstring('<mesh filename="unchecked.stl"/>'))


class ResetJitterGeometryTests(unittest.TestCase):
    def test_jitter_interval_matches_actual_inherited_reset_and_soft_factor(self):
        tree = ast.parse((ROOT / "packages/hexapod_env/hexapod_env/env.py").read_text())
        reset = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx")
        uniform_calls = [node for node in ast.walk(reset) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "uniform_"]
        self.assertEqual(len(uniform_calls), 1)
        self.assertEqual(tuple(ast.literal_eval(arg) for arg in uniform_calls[0].args), (-.03, .03))
        self.assertEqual(contract.SOFT_JOINT_POS_LIMIT_FACTOR, .95)
        jitter = measure_reset_jitter(URDF, STANCE)
        self.assertEqual(jitter["joint_jitter_half_width_rad"], .03)

    def test_jitter_sampling_distinguishes_nominal_from_actual_reset_samples(self):
        report = audit_stance(URDF, STANCE)
        jitter = report["reset_jitter_sampling"]
        self.assertEqual(jitter["samples_per_leg"], 125)
        self.assertEqual(jitter["leg_pose_samples"], 750)
        self.assertIn(0., jitter["per_joint_offset_grid_rad"])
        self.assertGreaterEqual(jitter["nominal_before_jitter_min_foot_clearance_m"], .005)
        self.assertAlmostEqual(jitter["sampled_minimum_foot"]["clearance_m"], .00126469062611856, places=12)
        self.assertAlmostEqual(jitter["sampled_minimum_nonfoot"]["clearance_m"], .0264986104410838, places=12)
        self.assertEqual(jitter["sampled_minimum_foot"]["link"], "lr_tibia")
        self.assertEqual(jitter["sampled_minimum_foot"]["joint_offsets_rad"], {"coxa_yaw": .03, "femur_pitch": -.03, "tibia_pitch": .03})
        self.assertTrue(jitter["all_sampled_ground_clearances_positive"])
        self.assertTrue(jitter["continuous_joint_intervals_within_soft_limits"])
        self.assertFalse(jitter["nominal_5mm_clearance_preserved_at_all_samples"])
        self.assertFalse(jitter["continuous_geometry_clearance_proven"])
        self.assertEqual(set(jitter["sampled_minimum_foot_clearance_by_leg_m"]), set(ASSET.leg_names))

    def test_sampling_finds_penetration_even_when_nominal_pose_is_clear(self):
        lowered = dict(STANCE, reset_root_height_m=STANCE["reset_root_height_m"] - .002)
        nominal = measure_stance(URDF, lowered)
        self.assertGreater(min(nominal["foot_clearance_at_reset_m"].values()), .003)
        sampled = measure_reset_jitter(URDF, lowered)
        self.assertFalse(sampled["all_sampled_ground_clearances_positive"])
        self.assertLess(sampled["sampled_minimum_foot"]["clearance_m"], 0.)

    def test_full_jitter_interval_can_violate_soft_limits_with_legal_nominal(self):
        near_soft_stop = dict(STANCE, tibia_pitch_rad=-.87)
        self.assertGreater(near_soft_stop["tibia_pitch_rad"], contract.SOFT_LIMITS_BY_NAME["lf_tibia_pitch"][0])
        jitter = measure_reset_jitter(URDF, near_soft_stop)
        self.assertTrue(jitter["continuous_joint_intervals_within_hard_limits"])
        self.assertFalse(jitter["continuous_joint_intervals_within_soft_limits"])
        self.assertFalse(jitter["joint_intervals"]["lf_tibia_pitch"]["within_soft_limits"])


class AnatomicalFrameTests(unittest.TestCase):
    def test_positive_and_negative_forward_left_yaw_keep_sign(self):
        cases = (
            ((0., -1., 0.), (1., 0., 0.)),
            ((0., 1., 0.), (-1., 0., 0.)),
            ((1., 0., 0.), (0., 1., 0.)),
            ((-1., 0., 0.), (0., -1., 0.)),
            ((0., 0., 1.), (0., 0., 1.)),
            ((0., 0., -1.), (0., 0., -1.)),
        )
        rotation = np.asarray(contract.BODY_TO_NAVIGATION_MATRIX)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3))
        for body, navigation in cases:
            self.assertEqual(body_to_navigation(body), navigation)
            self.assertEqual(navigation_to_body(navigation), body)
            np.testing.assert_allclose(rotation @ body, navigation)
        vector = (.23, -.47, .11)
        self.assertEqual(navigation_to_body(body_to_navigation(vector)), vector)

    def test_tensor_environment_transform_agrees_with_runtime_on_all_axes(self):
        import torch
        from hexapod_env.command_sampling import body_to_navigation_frame

        vectors = torch.cat((torch.eye(3), -torch.eye(3)), dim=0)
        expected = torch.tensor([body_to_navigation(tuple(row.tolist())) for row in vectors])
        torch.testing.assert_close(body_to_navigation_frame(vectors), expected)

    def test_urdf_leg_mounts_and_positive_coxa_axis_match_anatomy(self):
        root = ET.parse(URDF).getroot()
        zero = {name: 0. for name in ASSET.runtime_joint_names}
        original = forward_kinematics(root, zero)
        moved = forward_kinematics(root, {name: .05 if name.endswith("_coxa_yaw") else 0. for name in zero})
        for leg in ASSET.leg_names:
            hip = original[f"{leg}_coxa"][:3, 3]
            self.assertEqual(math.copysign(1., hip[0]), 1. if leg[0] == "l" else -1.)
            if leg[1] == "f":
                self.assertLess(hip[1], 0.)
            elif leg[1] == "r":
                self.assertGreater(hip[1], 0.)
            initial_ray = original[f"{leg}_femur"][:3, 3] - hip
            moved_ray = moved[f"{leg}_femur"][:3, 3] - hip
            self.assertGreater(np.cross(initial_ray, moved_ray)[2], 0., leg)

    def test_v2_config_explicitly_consumes_shared_navigation_action_contract(self):
        path = ROOT / "packages/hexapod_env/hexapod_env/tasks/mkii_v2/config.py"
        tree = ast.parse(path.read_text())
        cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HexapodMkiiV2FlatEnvCfg")
        assignments = {target.id: node.value for node in cfg.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
        for field, constant in {
            "command_frame": "COMMAND_FRAME", "action_scale": "ACTION_SCALE_RAD",
            "stand_action_scale": "STAND_ACTION_SCALE", "axis_command_active_threshold": "COMMAND_ACTIVE_THRESHOLD",
            "processed_joint_target_slew_limit_rad_per_20ms": "SLEW_LIMIT_RAD_PER_20MS", "decimation": "DECIMATION",
        }.items():
            self.assertEqual(ast.unparse(assignments[field]), constant)
        self.assertEqual(contract.COMMAND_FRAME, "navigation")
        self.assertAlmostEqual(contract.POLICY_STEP_DT_S, .020)
        self.assertEqual(contract.SLEW_LIMIT_RAD_PER_20MS, .040)

    def test_new_task_registration_is_opt_in_and_idempotent(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from hexapod_env.tasks.mkii_v2.register import MKII_V2_FLAT_TASK_ID, register_mkii_v2

        registry = {}
        calls = []
        def register(**kwargs):
            calls.append(kwargs)
            registry[kwargs["id"]] = kwargs
        with patch.dict(sys.modules, {"gymnasium": SimpleNamespace(registry=registry, register=register)}):
            self.assertEqual(register_mkii_v2(), [MKII_V2_FLAT_TASK_ID])
            register_mkii_v2()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["kwargs"]["env_cfg_entry_point"], "hexapod_rl.tasks.mkii_v2:HexapodMkiiV2FlatEnvCfg")
        self.assertEqual(len(registry), 1)
        self.assertNotIn(MKII_V2_FLAT_TASK_ID, (ROOT / "packages/hexapod_env/hexapod_env/register.py").read_text())


if __name__ == "__main__":
    unittest.main()
