"""CPU contracts for the direct RS05 task: action path, observations, rewards.

Every equality here runs the frozen prototype source. ``emitted_target`` and
``executed_action_feature`` are imported from ``experiments/paper_walk/env.py``
and compared value for value. The reward and termination expressions live
inside that file's ``step`` method, so the statements are read from the source
with ``ast`` and executed against the same inputs. The task config is executed
from its own source with recording stubs, so no Isaac Sim import happens.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "packages/hexapod_env/hexapod_env/tasks/mkii_rs05"
for directory in (ROOT, ROOT / "packages/hexapod_env", ROOT / "packages/hexapod_core"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiments.paper_walk import env as prototype  # noqa: E402
from experiments.paper_walk.env_config import JOINT_NAMES, KD  # noqa: E402
from hexapod_env.actuators.rs05_paper_walk_model import damping_vector, motor_effort  # noqa: E402
from hexapod_env.assets.mkii_rs05 import (  # noqa: E402
    MKII_RS05_ASSET,
    MKII_RS05_CANONICAL_JOINT_NAMES,
    MKII_RS05_JOINT_LIMITS_RAD,
    MKII_RS05_TOE_LOCAL_POINTS_M,
    MKII_RS05_USD_SHA256,
)
from hexapod_env.tasks.mkii_rs05 import math as task_math  # noqa: E402
from hexapod_env.tasks.mkii_rs05.register import (  # noqa: E402
    MKII_RS05_FLAT_TASK_ID,
    register_mkii_rs05,
)

PROTOTYPE_SOURCE = ast.parse((ROOT / "experiments/paper_walk/env.py").read_text())


def prototype_statements(names):
    """Return the ``step`` assignments that produce ``names``, in source order."""

    classes = [node for node in PROTOTYPE_SOURCE.body if isinstance(node, ast.ClassDef)]
    step = next(
        node
        for cls in classes
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    wanted, selected = list(names), []
    for node in step.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            selected.append(node)
    if not selected:
        raise AssertionError("The prototype step no longer assigns these names")
    module = ast.Module(body=selected, type_ignores=[])
    return compile(ast.fix_missing_locations(module), "paper_walk_step", "exec")


def limits_in_canonical_order():
    table = MKII_RS05_ASSET.joint_limits()
    lower = torch.tensor([table[name][0] for name in JOINT_NAMES], dtype=torch.float32)
    upper = torch.tensor([table[name][1] for name in JOINT_NAMES], dtype=torch.float32)
    return lower, upper


def neutral_in_canonical_order():
    stance = MKII_RS05_ASSET.default_joint_positions()
    return torch.tensor([stance[name] for name in JOINT_NAMES], dtype=torch.float32)


class ActionPathTests(unittest.TestCase):
    def setUp(self):
        self.lower, self.upper = limits_in_canonical_order()
        self.neutral = neutral_in_canonical_order()

    def test_emitted_target_matches_the_prototype_over_a_held_sequence(self):
        generator = torch.Generator().manual_seed(20260916)
        held = self.neutral.expand(8, -1).clone()
        mine = held.clone()
        for _ in range(24):
            action = torch.empty(8, 18).uniform_(-3.0, 3.0, generator=generator)
            expected = prototype.emitted_target(
                action, held, self.neutral, self.lower, self.upper,
                task_math.ACTION_SCALE_RAD, task_math.TARGET_SLEW_RAD,
            )
            actual = task_math.emitted_target(action, mine, self.neutral, self.lower, self.upper)
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            held, mine = expected, actual
        self.assertTrue(bool(((mine >= self.lower) & (mine <= self.upper)).all()))

    def test_the_target_never_leaves_the_hard_limit_or_the_slew_bound(self):
        held = self.lower.expand(4, -1).clone()
        for direction in (1.0, -1.0, 1.0):
            target = task_math.emitted_target(
                torch.full_like(held, 9.0 * direction), held, self.neutral, self.lower, self.upper
            )
            self.assertTrue(bool(((target.double() - held.double()).abs() <= 0.040).all()))
            self.assertTrue(bool(((target >= self.lower) & (target <= self.upper)).all()))
            held = target

    def test_executed_action_feature_matches_the_prototype(self):
        target = self.neutral + torch.linspace(-0.3, 0.3, 18)
        expected = prototype.executed_action_feature(target, self.neutral, task_math.ACTION_SCALE_RAD)
        actual = task_math.executed_action_feature(target, self.neutral)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_the_declared_scale_and_slew_are_the_accepted_values(self):
        self.assertEqual(task_math.ACTION_SCALE_RAD, 0.35)
        self.assertEqual(task_math.TARGET_SLEW_RAD, 0.040)
        self.assertEqual(task_math.PHYSICS_DT_S, 0.0025)
        self.assertEqual(task_math.DECIMATION, 8)


class ObservationLayoutTests(unittest.TestCase):
    def setUp(self):
        self.neutral = neutral_in_canonical_order()
        generator = torch.Generator().manual_seed(4)
        self.angular = torch.empty(3, 3).uniform_(-1, 1, generator=generator)
        self.gravity = torch.empty(3, 3).uniform_(-1, 1, generator=generator)
        self.linear = torch.empty(3, 3).uniform_(-1, 1, generator=generator)
        self.joint_pos = torch.empty(3, 18).uniform_(-1, 1, generator=generator)
        self.joint_vel = torch.empty(3, 18).uniform_(-1, 1, generator=generator)

    def test_widths_are_the_accepted_231_and_234(self):
        self.assertEqual(task_math.PROPRIO_WIDTH, 42)
        self.assertEqual(task_math.HISTORY_LENGTH, 5)
        self.assertEqual(task_math.POLICY_OBSERVATION_WIDTH, 231)
        self.assertEqual(task_math.CRITIC_OBSERVATION_WIDTH, 234)
        self.assertEqual(
            task_math.POLICY_OBSERVATION_WIDTH,
            task_math.HISTORY_LENGTH * task_math.PROPRIO_WIDTH + 3 + 18,
        )

    def test_the_proprio_row_matches_the_prototype_composition(self):
        row = task_math.proprio_row(self.angular, self.gravity, self.joint_pos, self.joint_vel, self.neutral)
        expected = torch.cat(
            (
                prototype.navigation(self.angular),
                prototype.navigation(self.gravity),
                self.joint_pos - self.neutral,
                self.joint_vel,
            ),
            dim=-1,
        )
        self.assertEqual(row.shape, (3, 42))
        np.testing.assert_array_equal(row.numpy(), expected.numpy())

    def test_the_policy_and_critic_observations_carry_the_declared_blocks(self):
        history = torch.arange(3 * 5 * 42, dtype=torch.float32).reshape(3, 5, 42)
        commands = torch.tensor([[0.1, 0.0, 0.0]]).expand(3, -1).contiguous()
        previous = torch.full((3, 18), 0.25)
        policy = task_math.policy_observation(history, commands, previous)
        critic = task_math.critic_observation(policy, self.linear)
        self.assertEqual(policy.shape, (3, 231))
        self.assertEqual(critic.shape, (3, 234))
        np.testing.assert_array_equal(policy[:, :210].numpy(), history.flatten(1).numpy())
        np.testing.assert_array_equal(policy[:, 210:213].numpy(), commands.numpy())
        np.testing.assert_array_equal(policy[:, 213:].numpy(), previous.numpy())
        np.testing.assert_array_equal(critic[:, :231].numpy(), policy.numpy())
        np.testing.assert_array_equal(critic[:, 231:].numpy(), prototype.navigation(self.linear).numpy())

    def test_history_advances_oldest_first(self):
        history = torch.zeros(2, 5, 42)
        for index in range(1, 8):
            history = task_math.advance_history(history, torch.full((2, 42), float(index)))
        np.testing.assert_array_equal(
            history[:, :, 0].numpy(), np.tile(np.arange(3, 8, dtype=np.float32), (2, 1))
        )

    def test_a_wrong_block_width_is_rejected(self):
        with self.assertRaises(ValueError):
            task_math.policy_observation(torch.zeros(2, 5, 41), torch.zeros(2, 3), torch.zeros(2, 18))
        with self.assertRaises(ValueError):
            task_math.policy_observation(torch.zeros(2, 5, 42), torch.zeros(2, 2), torch.zeros(2, 18))
        with self.assertRaises(ValueError):
            task_math.critic_observation(torch.zeros(2, 231), torch.zeros(2, 2))


class RewardAndTerminationTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(99)
        self.rows = 6
        self.lower, self.upper = limits_in_canonical_order()
        self.neutral = neutral_in_canonical_order()
        self.state = {
            "linear": torch.empty(self.rows, 3).uniform_(-0.5, 0.5, generator=generator),
            "angular": torch.empty(self.rows, 3).uniform_(-1.0, 1.0, generator=generator),
            "gravity": torch.empty(self.rows, 3).uniform_(-1.0, 1.0, generator=generator),
            "q": self.neutral + torch.empty(self.rows, 18).uniform_(-0.2, 0.2, generator=generator),
            "root": torch.empty(self.rows, 7).uniform_(0.0, 0.3, generator=generator),
        }
        # Two rows fall below the height gate and one violates a joint limit.
        self.state["root"][0, 2] = 0.02
        self.state["gravity"][1, 2] = 0.9
        self.state["q"][2, 5] = self.upper[5] + 0.01
        self.commands = torch.tensor([[0.1, -0.05, 0.2]]).expand(self.rows, -1).contiguous()
        self.target = self.neutral + torch.empty(self.rows, 18).uniform_(-0.1, 0.1, generator=generator)
        self.old_target = self.neutral + torch.empty(self.rows, 18).uniform_(-0.1, 0.1, generator=generator)
        self.torque_square = torch.empty(self.rows, 18).uniform_(0.0, 4.0, generator=generator)

    def prototype_values(self):
        namespace = {
            "torch": torch,
            "math": math,
            "navigation": prototype.navigation,
            "state": self.state,
            "self": types.SimpleNamespace(
                commands=self.commands, lower=self.lower, upper=self.upper
            ),
            "target": self.target,
            "old_target": self.old_target,
            "torque_square": self.torque_square,
        }
        exec(
            prototype_statements(
                (
                    "velocity_nav",
                    "angular_nav",
                    "tracking",
                    "yaw_tracking",
                    "tilt_cost",
                    "torque_cost",
                    "slew_cost",
                    "reward",
                    "joint_violation",
                    "terminated",
                )
            ),
            namespace,
        )
        return namespace

    def test_reward_matches_the_prototype_expression_exactly(self):
        expected = self.prototype_values()
        components, total = task_math.reward_terms(
            commands=self.commands,
            linear_body=self.state["linear"],
            angular_body=self.state["angular"],
            gravity_body=self.state["gravity"],
            torque_square_sum=self.torque_square,
            target=self.target,
            previous_target=self.old_target,
            terminated=expected["terminated"],
        )
        np.testing.assert_array_equal(total.numpy(), expected["reward"].numpy())
        np.testing.assert_array_equal(components["tracking"].numpy(), expected["tracking"].numpy())
        np.testing.assert_array_equal(components["yaw_tracking"].numpy(), expected["yaw_tracking"].numpy())
        np.testing.assert_array_equal(components["tilt_cost"].numpy(), expected["tilt_cost"].numpy())
        np.testing.assert_array_equal(components["torque_cost"].numpy(), expected["torque_cost"].numpy())
        np.testing.assert_array_equal(components["slew_cost"].numpy(), expected["slew_cost"].numpy())

    def test_termination_matches_the_prototype_expression_exactly(self):
        expected = self.prototype_values()
        terminated, reasons = task_math.termination_flags(
            root_height=self.state["root"][:, 2],
            gravity_body=self.state["gravity"],
            joint_pos=self.state["q"],
            lower=self.lower,
            upper=self.upper,
        )
        np.testing.assert_array_equal(terminated.numpy(), expected["terminated"].numpy())
        np.testing.assert_array_equal(
            reasons["joint_violation"].numpy(), expected["joint_violation"].numpy()
        )
        self.assertTrue(bool(reasons["low_plate"][0]))
        self.assertTrue(bool(reasons["tilted"][1]))
        self.assertTrue(bool(reasons["joint_violation"][2]))

    def test_the_torque_cost_reads_the_eight_substep_sum(self):
        _, total_free = task_math.reward_terms(
            commands=self.commands,
            linear_body=self.state["linear"],
            angular_body=self.state["angular"],
            gravity_body=self.state["gravity"],
            torque_square_sum=torch.zeros_like(self.torque_square),
            target=self.target,
            previous_target=self.old_target,
            terminated=torch.zeros(self.rows, dtype=torch.bool),
        )
        components, total_loaded = task_math.reward_terms(
            commands=self.commands,
            linear_body=self.state["linear"],
            angular_body=self.state["angular"],
            gravity_body=self.state["gravity"],
            torque_square_sum=torch.full_like(self.torque_square, 8.0),
            target=self.target,
            previous_target=self.old_target,
            terminated=torch.zeros(self.rows, dtype=torch.bool),
        )
        self.assertTrue(bool((total_free > total_loaded).all()))
        np.testing.assert_allclose(components["torque_cost"].numpy(), np.ones(self.rows), rtol=0, atol=1e-7)


@dataclass
class StubSim:
    dt: float = 0.01
    render_interval: int = 1
    gravity: tuple = (0.0, 0.0, -9.81)
    physics: object = None
    physics_material: object = None

    def replace(self, **changes):
        return replace(self, **changes)


class Recorded:
    def __init__(self, **values):
        self.values = dict(values)

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def replace(self, **changes):
        return Recorded(**{**self.values, **changes})


def config_namespace():
    """Execute the task config with recording stubs in place of Isaac Lab."""

    path = TASK_DIR / "config.py"
    source = ast.parse(path.read_text())
    source.body = [node for node in source.body if not isinstance(node, (ast.Import, ast.ImportFrom))]

    def recorded(**kwargs):
        return Recorded(**kwargs)

    stub_modules = {
        "sim_utils": types.SimpleNamespace(
            UsdFileCfg=recorded,
            RigidBodyPropertiesCfg=recorded,
            ArticulationRootPropertiesCfg=recorded,
            RigidBodyMaterialCfg=recorded,
            DomeLightCfg=recorded,
        ),
    }
    articulation_module = importlib.util.spec_from_file_location(
        "_rs05_articulation", ROOT / "packages/hexapod_env/hexapod_env/assets/articulation.py"
    )
    namespace = {
        "os": __import__("os"),
        "Path": Path,
        "configclass": lambda cls: cls,
        "DirectRLEnvCfg": object,
        "SimulationCfg": lambda **kwargs: StubSim(**kwargs),
        "PhysxCfg": recorded,
        "InteractiveSceneCfg": recorded,
        "ContactSensorCfg": recorded,
        "TerrainImporterCfg": recorded,
        "ArticulationCfg": type(
            "ArticulationCfg", (Recorded,), {"InitialStateCfg": staticmethod(recorded)}
        ),
        "HexapodPPORunnerCfg": object,
        "MKII_RS05_ASSET": MKII_RS05_ASSET,
        "MKII_RS05_CANONICAL_JOINT_NAMES": MKII_RS05_CANONICAL_JOINT_NAMES,
        "MKII_RS05_JOINT_LIMITS_RAD": MKII_RS05_JOINT_LIMITS_RAD,
        "MKII_RS05_TOE_LOCAL_POINTS_M": MKII_RS05_TOE_LOCAL_POINTS_M,
        "MKII_RS05_USD_SHA256": MKII_RS05_USD_SHA256,
        "make_rs05_paper_walk_cfg": lambda names: Recorded(joint_names_expr=list(names)),
        "articulation_cfg_from_spec": None,
        "sim_utils": stub_modules["sim_utils"],
    }
    namespace.update(
        {
            name: getattr(task_math, name)
            for name in dir(task_math)
            if name.isupper()
        }
    )
    namespace["articulation_cfg_from_spec"] = _articulation_builder(articulation_module, namespace)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def _articulation_builder(module_spec, config_namespace):
    """Run the real ``articulation_cfg_from_spec`` body against the stubs."""

    path = ROOT / "packages/hexapod_env/hexapod_env/assets/articulation.py"
    source = ast.parse(path.read_text())
    source.body = [node for node in source.body if not isinstance(node, (ast.Import, ast.ImportFrom))]

    def recorded(**kwargs):
        return Recorded(**kwargs)

    namespace = {
        "os": __import__("os"),
        "sim_utils": types.SimpleNamespace(
            UsdFileCfg=recorded,
            RigidBodyPropertiesCfg=recorded,
            ArticulationRootPropertiesCfg=recorded,
        ),
        "ArticulationCfg": type(
            "ArticulationCfg", (Recorded,), {"InitialStateCfg": staticmethod(recorded)}
        ),
        "DCMotorCfg": object,
        "ROBSTRIDE_RS05_CFG": "robstride-rs05-sentinel",
        "MKII_V1_ASSET": None,
        "HexapodAssetSpec": object,
    }
    from hexapod_env.assets.spec import MKII_V1_ASSET

    namespace["MKII_V1_ASSET"] = MKII_V1_ASSET
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["articulation_cfg_from_spec"]


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = config_namespace()
        cls.cfg = cls.namespace["HexapodMkiiRs05FlatEnvCfg"]

    def test_timing_and_spaces_match_the_accepted_runtime(self):
        self.assertEqual(self.cfg.sim.dt, 0.0025)
        self.assertEqual(self.cfg.decimation, 8)
        self.assertEqual(self.cfg.sim.render_interval, 8)
        self.assertEqual(self.cfg.episode_length_s, 20.0)
        self.assertEqual(self.cfg.action_space, 18)
        self.assertEqual(self.cfg.observation_space, 231)
        self.assertEqual(self.cfg.state_space, 234)
        self.assertEqual(self.cfg.action_scale, 0.35)
        self.assertEqual(self.cfg.target_slew_rad_per_control, 0.040)

    def test_the_physics_recipe_matches_the_accepted_native_readback(self):
        self.assertIs(self.cfg.sim.physics.enable_external_forces_every_iteration, True)
        props = self.cfg.robot.spawn.articulation_props
        self.assertEqual(props.solver_position_iteration_count, 32)
        self.assertEqual(props.solver_velocity_iteration_count, 0)
        self.assertIs(props.enabled_self_collisions, True)
        self.assertEqual(self.cfg.robot.soft_joint_pos_limit_factor, 1.0)
        material = self.cfg.sim.physics_material
        self.assertEqual((material.static_friction, material.dynamic_friction, material.restitution), (1.0, 1.0, 0.0))
        terrain = self.cfg.terrain.physics_material
        self.assertEqual((terrain.static_friction, terrain.dynamic_friction, terrain.restitution), (1.0, 1.0, 0.0))
        self.assertEqual(self.cfg.terrain.terrain_type, "plane")
        self.assertEqual(self.cfg.scene.env_spacing, 2.0)

    def test_the_robot_uses_the_corrected_asset_and_its_named_motors(self):
        self.assertEqual(self.cfg.robot.spawn.usd_path, MKII_RS05_ASSET.usd_path_container)
        self.assertEqual(self.cfg.robot.init_state.pos, (0.0, 0.0, MKII_RS05_ASSET.reset_root_height_m))
        self.assertEqual(self.cfg.robot.init_state.joint_pos, MKII_RS05_ASSET.default_joint_positions())
        motors = self.cfg.robot.actuators["legs"]
        self.assertEqual(set(motors.joint_names_expr), set(MKII_RS05_ASSET.all_joints))
        self.assertEqual(self.cfg.expected_runtime_joint_names, MKII_RS05_ASSET.runtime_joint_names)
        self.assertEqual(self.cfg.canonical_joint_names, JOINT_NAMES)

    def test_contact_sensors_cover_every_body_and_filter_the_ground(self):
        sensors = self.cfg.body_contact_sensors
        self.assertEqual(len(sensors), 19)
        for name, sensor in sensors.items():
            self.assertTrue(sensor.prim_path.endswith("/" + name), name)
            self.assertEqual(sensor.update_period, 0.0025)
            if name.endswith("_tibia"):
                self.assertEqual(len(sensor.filter_prim_paths_expr), 1)
            else:
                self.assertEqual(sensor.filter_prim_paths_expr, [])

    def test_the_runner_config_has_its_own_experiment_name(self):
        runner = self.namespace["HexapodMkiiRs05PPORunnerCfg"]
        self.assertEqual(runner.experiment_name, "hexapod_mkii_rs05_flat_direct")


class RegistrationTests(unittest.TestCase):
    def test_the_task_id_is_new_and_the_entry_points_name_hexapod_rl(self):
        self.assertEqual(MKII_RS05_FLAT_TASK_ID, "Isaac-Velocity-Flat-Hexapod-MKII-RS05-Direct-v0")
        registry = {}

        def register(**kwargs):
            registry[kwargs["id"]] = kwargs

        module = types.ModuleType("gymnasium")
        module.registry = registry
        module.register = register
        original = sys.modules.get("gymnasium")
        sys.modules["gymnasium"] = module
        try:
            self.assertEqual(register_mkii_rs05(), [MKII_RS05_FLAT_TASK_ID])
            register_mkii_rs05()
        finally:
            if original is None:
                del sys.modules["gymnasium"]
            else:
                sys.modules["gymnasium"] = original
        self.assertEqual(list(registry), [MKII_RS05_FLAT_TASK_ID])
        entry = registry[MKII_RS05_FLAT_TASK_ID]
        self.assertEqual(entry["entry_point"], "hexapod_rl.tasks.mkii_rs05:HexapodMkiiRs05Env")
        self.assertEqual(
            entry["kwargs"]["env_cfg_entry_point"], "hexapod_rl.tasks.mkii_rs05:HexapodMkiiRs05FlatEnvCfg"
        )
        self.assertEqual(
            entry["kwargs"]["rsl_rl_cfg_entry_point"],
            "hexapod_rl.tasks.mkii_rs05:HexapodMkiiRs05PPORunnerCfg",
        )

    def test_the_deployment_shim_exposes_every_entry_point_name(self):
        shim = ROOT / "isaaclab/hexapod_rl/tasks/mkii_rs05.py"
        tree = ast.parse(shim.read_text())
        imported = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        for name in ("HexapodMkiiRs05FlatEnvCfg", "HexapodMkiiRs05PPORunnerCfg"):
            self.assertIn(name, imported)
        source = shim.read_text()
        self.assertIn("HexapodMkiiRs05Env", source)

    def test_the_historical_task_ids_stay_bound_to_their_own_models(self):
        from hexapod_env.tasks.mkii_v2.register import MKII_V2_FLAT_TASK_ID

        self.assertNotEqual(MKII_RS05_FLAT_TASK_ID, MKII_V2_FLAT_TASK_ID)
        self.assertEqual(MKII_V2_FLAT_TASK_ID, "Isaac-Velocity-Flat-Hexapod-MKII-V2-Direct-v0")


class MotorBindingTests(unittest.TestCase):
    def test_the_damping_order_follows_the_canonical_joint_names(self):
        np.testing.assert_array_equal(
            damping_vector(JOINT_NAMES).numpy(), np.asarray(KD, dtype=np.float32)
        )

    def test_the_task_effort_equals_the_prototype_for_a_held_target(self):
        neutral = neutral_in_canonical_order()
        q = neutral + 0.01
        dq = torch.full((2, 18), 0.5)
        target = neutral.expand(2, -1).contiguous()
        expected = prototype.motor_force(q.expand(2, -1), dq, target, torch.tensor(KD))
        actual = motor_effort(q.expand(2, -1), dq, target, damping_vector(JOINT_NAMES))
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left.numpy(), right.numpy())


if __name__ == "__main__":
    unittest.main()
