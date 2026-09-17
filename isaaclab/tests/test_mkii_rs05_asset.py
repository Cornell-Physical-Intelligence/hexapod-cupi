"""Contracts for the corrected RS05 asset spec, runnable without Isaac Lab.

The spec must bind the exact model selected by ``robot/active_model.json``: its
URDF, model JSON, generated USD and the pinned kinematic prior metadata. The
per-leg coxa limits of this model differ, so the spec carries an explicit
per-joint limit table beside the shared per-group fields. Nothing here imports
Isaac Sim, gymnasium or a GPU; ``articulation.py`` is executed from its source
with recording stubs so the v1/v2 articulation defaults stay observable.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PACKAGE = REPO_ROOT / "packages" / "hexapod_env" / "hexapod_env"
for directory in (REPO_ROOT, REPO_ROOT / "packages" / "hexapod_env"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiments.paper_walk.env_config import (  # noqa: E402  (read-only reference)
    BODY_NAMES,
    JOINT_NAMES,
    LEGS,
    MASS_KG,
    MODEL_SHA256,
    URDF_SHA256,
    USD_SHA256,
)

MODEL_PATH = REPO_ROOT / "robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json"
METADATA_PATH = REPO_ROOT / "artifacts/restart_2026-09-14/paper_walk_execution_001/prior_001/prior_metadata.json"
NATIVE_READBACK_PATH = (
    REPO_ROOT / "artifacts/restart_2026-09-14/paper_walk_execution_001/"
    "results_startup_cold_001/standing/native/native_readback.json"
)


def _load_module(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


spec_module = _load_module("_hexapod_rs05_spec", ENV_PACKAGE / "assets" / "spec.py")
HexapodAssetSpec = spec_module.HexapodAssetSpec
MKII_V1_ASSET = spec_module.MKII_V1_ASSET
MOCK_ASSET = spec_module.MOCK_ASSET

from hexapod_env.assets.mkii_rs05 import (  # noqa: E402
    MKII_RS05_ASSET,
    MKII_RS05_METADATA_PATH,
    MKII_RS05_METADATA_SHA256,
    MKII_RS05_MODEL_SHA256,
    MKII_RS05_URDF_SHA256,
    MKII_RS05_USD_ARTIFACT_PATH,
    MKII_RS05_USD_SHA256,
)
from hexapod_env.assets.mkii_v2 import MKII_V2_ASSET  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_json():
    return json.loads(MODEL_PATH.read_text())


def metadata_json():
    return json.loads(METADATA_PATH.read_text())


class RecordedCfg:
    """Collect the keyword arguments ``articulation.py`` passes on."""

    def __init__(self, **values):
        self.values = values

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError as error:
            raise AttributeError(name) from error


def articulation_namespace():
    """Execute ``articulation.py`` with recording stubs in place of Isaac Lab."""

    path = ENV_PACKAGE / "assets" / "articulation.py"
    source = ast.parse(path.read_text())
    source.body = [node for node in source.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    sim_utils = types.SimpleNamespace(
        UsdFileCfg=lambda **kwargs: RecordedCfg(**kwargs),
        RigidBodyPropertiesCfg=lambda **kwargs: RecordedCfg(**kwargs),
        ArticulationRootPropertiesCfg=lambda **kwargs: RecordedCfg(**kwargs),
    )

    class ArticulationCfg(RecordedCfg):
        InitialStateCfg = staticmethod(lambda **kwargs: RecordedCfg(**kwargs))

    namespace = {
        "os": __import__("os"),
        "sim_utils": sim_utils,
        "ArticulationCfg": ArticulationCfg,
        "DCMotorCfg": object,
        "ROBSTRIDE_RS05_CFG": "robstride-rs05-sentinel",
        "MKII_V1_ASSET": MKII_V1_ASSET,
        "HexapodAssetSpec": HexapodAssetSpec,
    }
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


class PinnedIdentityTests(unittest.TestCase):
    def test_pinned_hashes_match_the_selected_files(self):
        selected = json.loads((REPO_ROOT / "robot/active_model.json").read_text())
        self.assertEqual(MKII_RS05_ASSET.urdf_path, selected["urdf"]["path"])
        self.assertEqual(digest(REPO_ROOT / MKII_RS05_ASSET.urdf_path), MKII_RS05_URDF_SHA256)
        self.assertEqual(digest(MODEL_PATH), MKII_RS05_MODEL_SHA256)
        self.assertEqual(digest(REPO_ROOT / MKII_RS05_USD_ARTIFACT_PATH), MKII_RS05_USD_SHA256)
        self.assertEqual(digest(REPO_ROOT / MKII_RS05_METADATA_PATH), MKII_RS05_METADATA_SHA256)

    def test_pinned_hashes_match_the_prototype_reference(self):
        self.assertEqual(MKII_RS05_URDF_SHA256, URDF_SHA256)
        self.assertEqual(MKII_RS05_MODEL_SHA256, MODEL_SHA256)
        self.assertEqual(MKII_RS05_USD_SHA256, USD_SHA256)
        self.assertEqual(MKII_RS05_ASSET.mass_kg, MASS_KG)

    def test_container_usd_path_is_the_bind_mounted_artifact(self):
        self.assertEqual(
            MKII_RS05_ASSET.usd_path_container,
            "/workspace/hexapod/" + MKII_RS05_USD_ARTIFACT_PATH,
        )
        self.assertEqual(MKII_RS05_ASSET.usd_env_var, "HEXAPOD_MKII_RS05_USD_PATH")

    def test_the_metadata_belongs_to_the_selected_model(self):
        metadata = metadata_json()
        self.assertEqual(metadata["urdf_sha256"], MKII_RS05_URDF_SHA256)
        self.assertEqual(metadata["model_sha256"], MKII_RS05_MODEL_SHA256)
        self.assertEqual(metadata["mass_kg"], MKII_RS05_ASSET.mass_kg)


class NamesAndLimitsTests(unittest.TestCase):
    def test_canonical_names_match_the_prototype(self):
        self.assertEqual(MKII_RS05_ASSET.leg_names, LEGS)
        self.assertEqual(set(MKII_RS05_ASSET.all_joints), set(JOINT_NAMES))
        self.assertEqual(MKII_RS05_ASSET.root_link, "body")
        links = {MKII_RS05_ASSET.root_link}
        for group in MKII_RS05_ASSET.leg_link_names:
            links.update(group)
        self.assertEqual(links, set(BODY_NAMES))

    def test_runtime_order_matches_the_recorded_native_readback(self):
        recorded = json.loads(NATIVE_READBACK_PATH.read_text())
        self.assertEqual(list(MKII_RS05_ASSET.runtime_joint_names), recorded["native_joint_names"])
        self.assertEqual(MKII_RS05_ASSET.runtime_order_status, spec_module.RUNTIME_ORDER_PROVISIONAL)

    def test_per_joint_limits_equal_the_model_json(self):
        expected = {joint["name"]: (joint["lower"], joint["upper"]) for joint in model_json()["joints"]}
        self.assertEqual(MKII_RS05_ASSET.joint_limits(), expected)
        # The per-leg coxa travel differs, which is why the table exists.
        coxa = {expected[name] for name in MKII_RS05_ASSET.coxa_joints}
        self.assertGreater(len(coxa), 1)

    def test_stance_and_heights_come_from_the_pinned_metadata(self):
        metadata = metadata_json()
        expected = dict(zip(metadata["joint_names"], metadata["nominal_joint_position_rad"]))
        self.assertEqual(MKII_RS05_ASSET.default_joint_positions(), expected)
        self.assertEqual(MKII_RS05_ASSET.reset_root_height_m, metadata["reset_root_height_m"])
        self.assertEqual(MKII_RS05_ASSET.nominal_height_m, metadata["root_height_m"])

    def test_declared_mass_equals_the_model_ledger(self):
        self.assertAlmostEqual(
            sum(link["mass"] for link in model_json()["links"]), MKII_RS05_ASSET.mass_kg, places=9
        )

    def test_the_stance_lies_inside_every_named_limit(self):
        limits = MKII_RS05_ASSET.joint_limits()
        for name, value in MKII_RS05_ASSET.default_joint_positions().items():
            lower, upper = limits[name]
            self.assertGreaterEqual(value, lower, name)
            self.assertLessEqual(value, upper, name)


class SpecOverrideTests(unittest.TestCase):
    def test_assets_without_a_table_keep_the_shared_group_limits(self):
        self.assertIsNone(MKII_V1_ASSET.joint_limits_rad)
        self.assertIsNone(MOCK_ASSET.joint_limits_rad)
        limits = MKII_V1_ASSET.joint_limits()
        self.assertEqual(len(limits), 18)
        for name in MKII_V1_ASSET.coxa_joints:
            self.assertEqual(limits[name], MKII_V1_ASSET.coxa_limits)
        for name in MKII_V1_ASSET.femur_joints:
            self.assertEqual(limits[name], MKII_V1_ASSET.femur_limits)
        for name in MKII_V1_ASSET.tibia_joints:
            self.assertEqual(limits[name], MKII_V1_ASSET.tibia_limits)

    def test_a_table_that_misses_a_joint_is_rejected(self):
        import dataclasses

        table = tuple(
            (name, lower, upper)
            for name, (lower, upper) in list(MKII_RS05_ASSET.joint_limits().items())[:-1]
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(MKII_RS05_ASSET, joint_limits_rad=table)

    def test_a_stance_outside_one_named_limit_is_rejected(self):
        import dataclasses

        # Narrow one femur joint above the declared -0.30 rad stance while the
        # shared group envelope still accepts it. Only the table rejects this.
        table = tuple(
            (name, lower, -0.10) if name == "rm_femur_pitch" else (name, lower, upper)
            for name, lower, upper in MKII_RS05_ASSET.joint_limits_rad
        )
        dataclasses.replace(MKII_RS05_ASSET, joint_limits_rad=table)
        table = tuple(
            (name, -0.10, upper) if name == "rm_femur_pitch" else (name, lower, upper)
            for name, lower, upper in MKII_RS05_ASSET.joint_limits_rad
        )
        with self.assertRaisesRegex(ValueError, "rm_femur_pitch"):
            dataclasses.replace(MKII_RS05_ASSET, joint_limits_rad=table)
        with self.assertRaises(ValueError):
            dataclasses.replace(MKII_RS05_ASSET, stance_coxa_rad=1.30)


class IdentitySeparationTests(unittest.TestCase):
    def test_the_new_asset_never_redirects_v1_or_v2(self):
        for other in (MKII_V1_ASSET, MKII_V2_ASSET, MOCK_ASSET):
            self.assertNotEqual(MKII_RS05_ASSET.name, other.name)
            self.assertNotEqual(MKII_RS05_ASSET.usd_env_var, other.usd_env_var)
            self.assertNotEqual(MKII_RS05_ASSET.usd_path_container, other.usd_path_container)
            self.assertNotEqual(MKII_RS05_ASSET.urdf_path, other.urdf_path)
        self.assertEqual(MKII_V1_ASSET.reset_root_height_m, 0.130)
        self.assertEqual(MKII_V2_ASSET.reset_root_height_m, 0.142964)
        self.assertEqual(MKII_V2_ASSET.mass_kg, 8.26081134)


class ArticulationDefaultTests(unittest.TestCase):
    def setUp(self):
        self.namespace = articulation_namespace()

    def articulation(self, spec, **kwargs):
        return self.namespace["articulation_cfg_from_spec"](spec, **kwargs)

    def test_defaults_reproduce_the_existing_v1_and_v2_articulations(self):
        for spec in (MKII_V1_ASSET, MKII_V2_ASSET):
            cfg = self.articulation(spec)
            props = cfg.spawn.articulation_props
            self.assertEqual(props.solver_position_iteration_count, 8)
            self.assertEqual(props.solver_velocity_iteration_count, 2)
            self.assertEqual(props.enabled_self_collisions, False)
            self.assertEqual(cfg.soft_joint_pos_limit_factor, 0.95)
            self.assertEqual(cfg.actuators, {"legs": "robstride-rs05-sentinel"})
            self.assertEqual(cfg.spawn.activate_contact_sensors, True)
            self.assertEqual(cfg.init_state.pos, (0.0, 0.0, spec.reset_root_height_m))
            self.assertEqual(cfg.init_state.joint_pos, spec.default_joint_positions())

    def test_the_module_level_v1_configuration_is_unchanged(self):
        cfg = self.namespace["HEXAPOD_MKII_V1_CFG"]
        self.assertEqual(cfg.spawn.articulation_props.solver_position_iteration_count, 8)
        self.assertEqual(cfg.spawn.articulation_props.solver_velocity_iteration_count, 2)
        self.assertEqual(cfg.soft_joint_pos_limit_factor, 0.95)
        self.assertEqual(cfg.spawn.usd_path, self.namespace["MKII_V1_USD_PATH"])

    def test_solver_iterations_and_actuator_are_selectable(self):
        cfg = self.articulation(
            MKII_RS05_ASSET,
            solver_position_iterations=32,
            solver_velocity_iterations=0,
            actuator_cfg="paper-walk-sentinel",
            usd_path="/workspace/hexapod/robot.usda",
        )
        props = cfg.spawn.articulation_props
        self.assertEqual(props.solver_position_iteration_count, 32)
        self.assertEqual(props.solver_velocity_iteration_count, 0)
        self.assertEqual(cfg.actuators, {"legs": "paper-walk-sentinel"})
        self.assertEqual(cfg.spawn.usd_path, "/workspace/hexapod/robot.usda")
        self.assertEqual(cfg.init_state.pos, (0.0, 0.0, MKII_RS05_ASSET.reset_root_height_m))


if __name__ == "__main__":
    unittest.main()
