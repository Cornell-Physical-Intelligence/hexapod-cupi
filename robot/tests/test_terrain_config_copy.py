"""Preserve the admitted omni controller fields across terrain cfg adaptation."""
import ast
from dataclasses import dataclass, field, replace
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab")]
from hexapod_terrain.fixture_adapter import MildTerrainSpec, adapt_flat_cfg_for_fixture_smoke


@dataclass
class Config:
    scene: object = field(default_factory=lambda: SimpleNamespace(num_envs=32))
    robot: object = field(default_factory=lambda: SimpleNamespace(init_state=SimpleNamespace(pos=(0., 0., .13))))
    actuators: object = field(default_factory=lambda: {"effort_limit": 1.6})

    def copy(self):
        # Exact installed configclass._copy_class behavior, inspected on Spark.
        return replace(self)


class TerrainConfigCopyTests(unittest.TestCase):
    def test_actual_omni_dynamic_fields_and_nested_state_survive_without_aliasing(self):
        tree = ast.parse((ROOT / "tools/omni_flat_env.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure_omni")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "configure_omni.py", "exec"), namespace)
        cfg = Config()
        namespace["configure_omni"](cfg)
        cfg.processed_joint_target_slew_limit_rad_per_20ms = .03
        expected_dynamic = {k: v for k, v in vars(cfg).items() if k not in ("scene", "robot", "actuators")}
        self.assertFalse(hasattr(cfg.copy(), "omni_observation_noise_scale"))
        cfg.base_contact_sensor = SimpleNamespace(track_contact_points=False)
        cfg.coxa_contact_sensor = SimpleNamespace(track_contact_points=False)
        cfg.feet_contact_sensors = tuple(SimpleNamespace(track_contact_points=True,
            filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"]) for _ in range(6))
        cfg.femur_contact_sensors = tuple(SimpleNamespace(track_contact_points=False) for _ in range(6))
        terrains = ModuleType("isaaclab.terrains")
        terrains.TerrainImporter = type("TerrainImporter", (), {})
        terrains.TerrainImporterCfg = SimpleNamespace
        identity = dict(variant="f050_t060", urdf_sha256="a" * 64, plan_sha256="b" * 64, stance_index=0)
        with patch.dict(sys.modules, {"isaaclab.terrains": terrains}), \
             patch.object(MildTerrainSpec, "load", return_value=({"start_xy_m": [-1., 0.]}, Path("fixture.usda"))):
            actual = adapt_flat_cfg_for_fixture_smoke(cfg, MildTerrainSpec(Path("catalog"), "ramp"),
                admission=dict(identity, gate={"passed": True}), asset_identity=identity)
        for key, value in expected_dynamic.items():
            self.assertEqual(getattr(actual, key), value, key)
        self.assertEqual(actual.actuators, {"effort_limit": 1.6})
        self.assertEqual(actual.scene.num_envs, 1)
        self.assertEqual(cfg.scene.num_envs, 32)
        actual.omni_reward_weights["linear_tracking"] = 999.
        self.assertEqual(cfg.omni_reward_weights["linear_tracking"], 4.)
        self.assertEqual(actual.feet_contact_sensors[0].filter_prim_paths_expr, ["/World/ground/terrain"])
        self.assertEqual(cfg.feet_contact_sensors[0].filter_prim_paths_expr, ["/World/ground/GroundPlane/CollisionPlane"])
        self.assertEqual(cfg.robot.init_state.pos, (0., 0., .13))
        # Independent geometric check of installed AssetBaseCfg's XYZW order.
        # The former WXYZ tuple rotates gravity sideways instead of yawing.
        rotation = Rotation.from_quat(actual.robot.init_state.rot)
        np.testing.assert_allclose(rotation.apply([0., 0., -1.], inverse=True),
                                   [0., 0., -1.], atol=1e-12)
        np.testing.assert_allclose(rotation.apply([0., -1., 0.]), [1., 0., 0.], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
