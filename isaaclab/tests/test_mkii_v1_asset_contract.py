"""Contracts for asset v1 (the CAD assembly) and the asset-spec mechanism.

Three things are bound here, all without Isaac Lab, gymnasium, torch or a GPU:

1. ``hexapod_env.assets.spec.MKII_V1_ASSET`` matches the URDF it describes
   (``robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf``) and the JSON
   files the import wrote (``joint_limits.json``, ``stance.json``).
2. ``hexapod_core.joints_v2`` (the v2 runtime joint order) is the spec's order,
   stands beside v1 without overlapping it, and is honestly labelled
   provisional until the imported articulation confirms it.
3. The mock is untouched: ``MOCK_ASSET`` equals the frozen literals in
   ``asset_cfg.py`` / ``env_cfg.py`` and the v1 joint contract, the historical
   task ID still loads ``HexapodFlatEnvCfg`` on ``HEXAPOD_CFG``, and asset v1
   is registered under its own, new task ID.

The spec module is loaded by file path so that importing it does not import
``hexapod_env/__init__``; the module is stdlib-only by design.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = REPO_ROOT / "packages"
ENV_PACKAGE = PACKAGES / "hexapod_env" / "hexapod_env"
CORE_PACKAGE_DIR = PACKAGES / "hexapod_core"
SHIM_DIR = REPO_ROOT / "isaaclab" / "hexapod_rl"
ASSEMBLY_DIR = REPO_ROOT / "robot" / "hexapod_mkii_assy"

if str(CORE_PACKAGE_DIR) not in sys.path:  # mirrors the isaaclab/hexapod_rl bootstrap
    sys.path.insert(0, str(CORE_PACKAGE_DIR))

from hexapod_core import joints as joints_v1  # noqa: E402
from hexapod_core import joints_v2  # noqa: E402


MKII_V1_TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0"
MOCK_TASK_ID = "Isaac-Velocity-Flat-Hexapod-RobStride-Direct-v0"


def _load_spec_module():
    path = ENV_PACKAGE / "assets" / "spec.py"
    module_spec = importlib.util.spec_from_file_location("_hexapod_asset_spec", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    # ``dataclasses`` resolves string annotations through sys.modules, so the
    # module has to be registered before its body executes.
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


SPEC = _load_spec_module()
MKII = SPEC.MKII_V1_ASSET
MOCK = SPEC.MOCK_ASSET


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _module_assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    raise AssertionError(f"no module-level assignment to {name}")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"no class {name}")


def _class_assignment(class_node: ast.ClassDef, name: str) -> ast.expr:
    for node in class_node.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return node.value
    raise AssertionError(f"no assignment to {class_node.name}.{name}")


def _string_constants(tree: ast.Module) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _urdf_joints(path: Path) -> dict[str, dict[str, object]]:
    root = ET.parse(path).getroot()
    joints: dict[str, dict[str, object]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        joints[joint.get("name")] = {
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "lower": float(limit.get("lower")) if limit is not None else None,
            "upper": float(limit.get("upper")) if limit is not None else None,
            "effort": float(limit.get("effort")) if limit is not None else None,
        }
    return joints


def _urdf_links(path: Path) -> list[str]:
    return [link.get("name") for link in ET.parse(path).getroot().findall("link")]


class MkiiV1SpecMatchesTheUrdfTests(unittest.TestCase):
    URDF = REPO_ROOT / MKII.urdf_path

    def test_urdf_exists_where_the_spec_says(self) -> None:
        self.assertTrue(self.URDF.is_file(), self.URDF)

    def test_joint_names_root_and_links(self) -> None:
        joints = _urdf_joints(self.URDF)
        self.assertEqual(set(joints), set(MKII.all_joints))
        self.assertTrue(all(j["type"] == "revolute" for j in joints.values()))
        links = _urdf_links(self.URDF)
        children = {j["child"] for j in joints.values()}
        roots = [link for link in links if link not in children]
        self.assertEqual(roots, [MKII.root_link])
        self.assertEqual(len(links), 19)
        for coxa, femur, tibia in MKII.leg_link_names:
            for link in (coxa, femur, tibia):
                self.assertIn(link, links)

    def test_kinematic_chain_per_leg(self) -> None:
        joints = _urdf_joints(self.URDF)
        for leg_index, (coxa, femur, tibia) in enumerate(MKII.leg_link_names):
            coxa_joint = MKII.coxa_joints[leg_index]
            femur_joint = MKII.femur_joints[leg_index]
            tibia_joint = MKII.tibia_joints[leg_index]
            self.assertEqual((joints[coxa_joint]["parent"], joints[coxa_joint]["child"]), (MKII.root_link, coxa))
            self.assertEqual((joints[femur_joint]["parent"], joints[femur_joint]["child"]), (coxa, femur))
            self.assertEqual((joints[tibia_joint]["parent"], joints[tibia_joint]["child"]), (femur, tibia))

    def test_limits_match_spec_and_joint_limits_json(self) -> None:
        joints = _urdf_joints(self.URDF)
        limits_json = json.loads((ASSEMBLY_DIR / "joint_limits.json").read_text())
        for group, spec_limits, key in (
            (MKII.coxa_joints, MKII.coxa_limits, "coxa_yaw"),
            (MKII.femur_joints, MKII.femur_limits, "femur_pitch"),
            (MKII.tibia_joints, MKII.tibia_limits, "tibia_pitch"),
        ):
            self.assertEqual(
                (limits_json[key]["lower"], limits_json[key]["upper"]), spec_limits, key
            )
            for name in group:
                self.assertAlmostEqual(joints[name]["lower"], spec_limits[0], places=6, msg=name)
                self.assertAlmostEqual(joints[name]["upper"], spec_limits[1], places=6, msg=name)
                # RS05 published peak; the actuator model caps continuous effort.
                self.assertEqual(joints[name]["effort"], 5.5, name)

    def test_stance_matches_stance_json(self) -> None:
        stance = json.loads((ASSEMBLY_DIR / "stance.json").read_text())
        self.assertEqual(stance["coxa_yaw_rad"], MKII.stance_coxa_rad)
        self.assertEqual(stance["femur_pitch_rad"], MKII.stance_femur_rad)
        self.assertEqual(stance["tibia_pitch_rad"], MKII.stance_tibia_rad)
        self.assertEqual(stance["root_height_m"], MKII.nominal_height_m)
        self.assertEqual(stance["reset_root_height_m"], MKII.reset_root_height_m)

    def test_spec_invariants(self) -> None:
        self.assertEqual(MKII.name, "mkii_v1")
        self.assertEqual(MKII.usd_env_var, "HEXAPOD_MKII_V1_USD_PATH")
        self.assertNotEqual(MKII.usd_env_var, MOCK.usd_env_var)
        self.assertEqual(
            MKII.geometry_root_prim("/World/envs/env_.*/Robot"),
            "/World/envs/env_.*/Robot/Geometry/body",
        )
        self.assertEqual(len(MKII.default_joint_positions()), 18)
        self.assertLess(MKII.distal_foot_min_y_m, MKII.swing_clearance_pad_offset_y_m)
        self.assertGreater(MKII.reset_root_height_m, MKII.nominal_height_m)

    def test_spec_rejects_a_stance_outside_its_limits(self) -> None:
        import dataclasses

        with self.assertRaises(ValueError):
            dataclasses.replace(MKII, stance_femur_rad=MKII.femur_limits[1] + 0.1)
        with self.assertRaises(ValueError):
            dataclasses.replace(MKII, runtime_joint_names=MKII.runtime_joint_names[::-1])


class JointsV2ContractTests(unittest.TestCase):
    def test_v2_is_the_spec_order(self) -> None:
        self.assertEqual(joints_v2.SCHEMA_VERSION, 2)
        self.assertEqual(joints_v2.ASSET_NAME, MKII.name)
        self.assertEqual(joints_v2.RUNTIME_JOINT_NAMES, MKII.runtime_joint_names)
        self.assertEqual(joints_v2.RUNTIME_ORDER_STATUS, MKII.runtime_order_status)
        self.assertEqual(joints_v2.LEG_NAMES, MKII.leg_names)
        self.assertEqual(joints_v2.DEFAULT_RESET_ROOT_HEIGHT_M, MKII.reset_root_height_m)
        expected_defaults = (
            (MKII.stance_coxa_rad,) * 6
            + (MKII.stance_femur_rad,) * 6
            + (MKII.stance_tibia_rad,) * 6
        )
        self.assertEqual(joints_v2.DEFAULT_JOINT_POSITIONS_RAD, expected_defaults)

    def test_groups_and_slices(self) -> None:
        names = joints_v2.RUNTIME_JOINT_NAMES
        self.assertEqual(len(names), 18)
        self.assertEqual(len(set(names)), 18)
        self.assertEqual(set(names[joints_v2.COXA_SLICE]), set(MKII.coxa_joints))
        self.assertEqual(set(names[joints_v2.FEMUR_SLICE]), set(MKII.femur_joints))
        self.assertEqual(set(names[joints_v2.TIBIA_SLICE]), set(MKII.tibia_joints))
        for index, name in enumerate(names):
            self.assertTrue(name.startswith(joints_v2.leg_of(index) + "_"), name)
            self.assertIn(joints_v2.group_of(index), name)
            self.assertEqual(joints_v2.index_of(name), index)
        with self.assertRaises(IndexError):
            joints_v2.group_of(18)
        with self.assertRaises(IndexError):
            joints_v2.leg_of(-1)
        with self.assertRaises(KeyError):
            joints_v2.index_of("revolute_1_1")

    def test_v1_and_v2_never_share_a_name(self) -> None:
        # A v1 checkpoint indexes revolute_*; a v2 articulation has no such
        # joint, so a mix-up fails loudly instead of permuting the robot.
        self.assertEqual(
            set(joints_v1.RUNTIME_JOINT_NAMES) & set(joints_v2.RUNTIME_JOINT_NAMES), set()
        )
        self.assertEqual(joints_v1.SCHEMA_VERSION, 1)

    def test_status_is_a_known_value(self) -> None:
        self.assertIn(
            joints_v2.RUNTIME_ORDER_STATUS,
            (SPEC.RUNTIME_ORDER_CONFIRMED, SPEC.RUNTIME_ORDER_PROVISIONAL),
        )
        if joints_v2.RUNTIME_ORDER_STATUS == SPEC.RUNTIME_ORDER_PROVISIONAL:
            # The environment must then enforce the order at construction.
            env_source = (ENV_PACKAGE / "env.py").read_text(encoding="utf-8")
            self.assertIn("expected_runtime_joint_names", env_source)
            self.assertIn("self._robot.joint_names", env_source)


class MockAssetIsUnchangedTests(unittest.TestCase):
    ASSET_CFG = _tree(ENV_PACKAGE / "asset_cfg.py")
    ENV_CFG = _tree(ENV_PACKAGE / "env_cfg.py")

    def test_mock_spec_equals_the_asset_cfg_literals(self) -> None:
        for name, expected in (
            ("COXA_JOINTS", MOCK.coxa_joints),
            ("FEMUR_JOINTS", MOCK.femur_joints),
            ("TIBIA_JOINTS", MOCK.tibia_joints),
            ("ROOT_LINK_NAME", MOCK.root_link),
        ):
            self.assertEqual(ast.literal_eval(_module_assignment(self.ASSET_CFG, name)), expected, name)
        usd_default = _module_assignment(self.ASSET_CFG, "USD_PATH")
        self.assertIsInstance(usd_default, ast.Call)
        env_var, default = (ast.literal_eval(arg) for arg in usd_default.args)
        self.assertEqual(env_var, MOCK.usd_env_var)
        self.assertEqual(default, MOCK.usd_path_container)
        self.assertEqual(MOCK.root_link, "root")

    def test_mock_stance_equals_hexapod_cfg(self) -> None:
        hexapod_cfg = _module_assignment(self.ASSET_CFG, "HEXAPOD_CFG")
        init_state = next(kw.value for kw in hexapod_cfg.keywords if kw.arg == "init_state")
        pos = next(kw.value for kw in init_state.keywords if kw.arg == "pos")
        self.assertEqual(ast.literal_eval(pos), (0.0, 0.0, MOCK.reset_root_height_m))
        joint_pos = next(kw.value for kw in init_state.keywords if kw.arg == "joint_pos")
        values = [
            ast.literal_eval(comp.value)
            for comp in joint_pos.values
            if isinstance(comp, ast.DictComp)
        ]
        self.assertEqual(values, [MOCK.stance_coxa_rad, MOCK.stance_femur_rad, MOCK.stance_tibia_rad])

    def test_mock_spec_equals_the_env_cfg_literals(self) -> None:
        self.assertEqual(
            ast.literal_eval(_module_assignment(self.ENV_CFG, "LEG_LINK_NAMES")),
            MOCK.leg_link_names,
        )
        flat = _class(self.ENV_CFG, "HexapodFlatEnvCfg")
        self.assertEqual(ast.literal_eval(_class_assignment(flat, "nominal_height_m")), MOCK.nominal_height_m)
        self.assertEqual(ast.literal_eval(_class_assignment(flat, "distal_foot_min_y_m")), MOCK.distal_foot_min_y_m)
        self.assertEqual(
            ast.literal_eval(_class_assignment(flat, "swing_clearance_pad_offset_y_m")),
            MOCK.swing_clearance_pad_offset_y_m,
        )
        # The historical config still spawns the mock articulation ...
        robot = _class_assignment(flat, "robot")
        self.assertEqual(ast.unparse(robot.func.value), "HEXAPOD_CFG")
        # ... resolves feet from the mock table, and does not enforce an order.
        self.assertEqual(ast.unparse(_class_assignment(flat, "leg_link_names")), "LEG_LINK_NAMES")
        self.assertIsNone(ast.literal_eval(_class_assignment(flat, "expected_runtime_joint_names")))
        base_mass = _class_assignment(_class(self.ENV_CFG, "EventCfg"), "base_mass")
        self.assertIn("body_names='root'", ast.unparse(base_mass))

    def test_mock_runtime_order_is_the_v1_contract(self) -> None:
        self.assertEqual(MOCK.runtime_joint_names, joints_v1.RUNTIME_JOINT_NAMES)
        self.assertEqual(MOCK.runtime_order_status, SPEC.RUNTIME_ORDER_CONFIRMED)
        expected_defaults = (
            (MOCK.stance_coxa_rad,) * 6
            + (MOCK.stance_femur_rad,) * 6
            + (MOCK.stance_tibia_rad,) * 6
        )
        self.assertEqual(joints_v1.BASE_DEFAULT_JOINT_POSITIONS_RAD, expected_defaults)


class MkiiV1TaskRegistrationTests(unittest.TestCase):
    REGISTER = _tree(ENV_PACKAGE / "register.py")
    ENV_CFG = _tree(ENV_PACKAGE / "env_cfg.py")
    PPO_CFG = _tree(ENV_PACKAGE / "ppo_cfg.py")

    def test_new_task_id_is_new_and_the_old_one_is_unchanged(self) -> None:
        self.assertEqual(
            ast.literal_eval(_module_assignment(self.REGISTER, "MKII_V1_FLAT_TASK_ID")),
            MKII_V1_TASK_ID,
        )
        self.assertEqual(ast.literal_eval(_module_assignment(self.REGISTER, "TASK_ID")), MOCK_TASK_ID)
        task_ids = [value for value in _string_constants(self.REGISTER) if value.endswith("-v0")]
        self.assertEqual(task_ids.count(MKII_V1_TASK_ID), 1)
        self.assertEqual(len(task_ids), len(set(task_ids)), "duplicate task ID literal")

    def test_entry_points_name_the_new_classes_through_the_shims(self) -> None:
        constants = _string_constants(self.REGISTER)
        self.assertIn("hexapod_rl.env_cfg:HexapodMkiiV1FlatEnvCfg", constants)
        self.assertIn("hexapod_rl.ppo_cfg:HexapodMkiiV1PPORunnerCfg", constants)
        # The historical task keeps its historical entry points.
        self.assertIn("hexapod_rl.env_cfg:HexapodFlatEnvCfg", constants)
        self.assertIn("hexapod_rl.ppo_cfg:HexapodPPORunnerCfg", constants)

    def test_env_cfg_defines_the_asset_v1_config_from_the_spec(self) -> None:
        cfg = _class(self.ENV_CFG, "HexapodMkiiV1FlatEnvCfg")
        self.assertEqual([ast.unparse(base) for base in cfg.bases], ["HexapodFlatEnvCfg"])
        self.assertEqual(ast.unparse(_class_assignment(cfg, "robot").func.value), "HEXAPOD_MKII_V1_CFG")
        for attribute in (
            "leg_link_names",
            "expected_runtime_joint_names",
            "nominal_height_m",
            "distal_foot_min_y_m",
            "swing_clearance_pad_offset_y_m",
        ):
            self.assertTrue(
                ast.unparse(_class_assignment(cfg, attribute)).startswith("MKII_V1_ASSET."),
                attribute,
            )
        events = _class_assignment(cfg, "events")
        self.assertEqual(ast.unparse(events), "MkiiV1EventCfg()")
        base_mass = _class_assignment(_class(self.ENV_CFG, "MkiiV1EventCfg"), "base_mass")
        self.assertIn("MKII_V1_ASSET.root_link", ast.unparse(base_mass))
        _class(self.PPO_CFG, "HexapodMkiiV1PPORunnerCfg")

    def test_shims_and_package_init_export_the_new_names(self) -> None:
        for shim, names in (
            ("env_cfg.py", ("HexapodMkiiV1FlatEnvCfg", "MkiiV1EventCfg")),
            ("ppo_cfg.py", ("HexapodMkiiV1PPORunnerCfg",)),
            ("register.py", ("MKII_V1_FLAT_TASK_ID",)),
            ("asset_cfg.py", ("ROOT_LINK_NAME",)),
            ("__init__.py", ("MKII_V1_FLAT_TASK_ID",)),
        ):
            source = (SHIM_DIR / shim).read_text(encoding="utf-8")
            for name in names:
                self.assertIn(name, source, f"{shim} does not re-export {name}")
        package_all = ast.literal_eval(_module_assignment(_tree(ENV_PACKAGE / "__init__.py"), "__all__"))
        self.assertIn("MKII_V1_FLAT_TASK_ID", package_all)

    def test_validate_defaults_to_asset_v1_and_keeps_the_mock(self) -> None:
        source = (REPO_ROOT / "isaaclab" / "validate.py").read_text(encoding="utf-8")
        self.assertIn('"--asset"', source)
        self.assertIn("MKII_V1_FLAT_TASK_ID", source)
        self.assertIn("MOCK_ASSET", source)
        self.assertIn("joint_names=", source)


if __name__ == "__main__":
    unittest.main()
