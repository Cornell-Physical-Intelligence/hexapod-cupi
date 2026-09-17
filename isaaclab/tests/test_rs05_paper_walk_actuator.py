"""CPU contracts for the accepted paper-walk RS05 actuator model.

The torch model must reproduce ``experiments/paper_walk/env.py:motor_force``
exactly, including the float64 speed curve, the zero ceiling at and above
480 rpm and the float32 result. The frozen scorer recomputes the same values
with ``_diagnostic_servo``, so both references are compared here. Nothing in
this file imports Isaac Sim; the configuration and runtime modules are executed
against a small faithful ``IdealPDActuator`` stub.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "packages/hexapod_env"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiments.paper_walk.env import _diagnostic_servo, motor_force  # noqa: E402
from experiments.paper_walk.env_config import JOINT_NAMES, KD  # noqa: E402
from hexapod_env.actuators.rs05_paper_walk_model import (  # noqa: E402
    CEILING_NM_KNOTS,
    DAMPING_NM_S_PER_RAD_BY_NAME,
    MAX_APPLIED_NM,
    SPEED_RPM_KNOTS,
    STIFFNESS_NM_PER_RAD,
    ZERO_AUTHORITY_RPM,
    damping_vector,
    effort_ceiling,
    motor_effort,
)

BLOCK_ORDER = tuple(
    f"{leg}_{joint}"
    for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch")
    for leg in ("lf", "lm", "lr", "rf", "rm", "rr")
)


def sample_states(seed=20260916, rows=96):
    generator = np.random.default_rng(seed)
    q = generator.uniform(-2.0, 3.0, (rows, 18)).astype(np.float32)
    dq = generator.uniform(-60.0, 60.0, (rows, 18)).astype(np.float32)
    target = generator.uniform(-2.0, 3.0, (rows, 18)).astype(np.float32)
    # Every published knot, both signs, and speeds above the zero-authority rpm.
    knots = np.array(SPEED_RPM_KNOTS, dtype=np.float64) * (2 * math.pi / 60)
    dq[0, : len(knots)] = knots
    dq[1, : len(knots)] = -knots
    dq[2, :] = 481.0 * (2 * math.pi / 60)
    dq[3, :] = -700.0 * (2 * math.pi / 60)
    dq[4, :] = 0.0
    return q, dq, target


class PureModelTests(unittest.TestCase):
    def test_model_reproduces_the_prototype_motor_force_exactly(self):
        q, dq, target = sample_states()
        kd = torch.tensor(KD)
        expected = motor_force(torch.from_numpy(q), torch.from_numpy(dq), torch.from_numpy(target), kd)
        actual = motor_effort(torch.from_numpy(q), torch.from_numpy(dq), torch.from_numpy(target), kd)
        for name, left, right in zip(("requested", "applied", "ceiling"), actual, expected):
            self.assertEqual(left.dtype, torch.float32, name)
            np.testing.assert_array_equal(left.numpy(), right.numpy(), err_msg=name)

    def test_model_reproduces_the_frozen_scorer_recurrence_exactly(self):
        q, dq, target = sample_states(seed=7)
        expected = _diagnostic_servo(q, dq, target, np.full(18, 12.0, np.float32), np.asarray(KD, np.float32))
        actual = motor_effort(torch.from_numpy(q), torch.from_numpy(dq), torch.from_numpy(target), torch.tensor(KD))
        for name, left, right in zip(("requested", "applied", "ceiling"), actual, expected):
            np.testing.assert_array_equal(left.numpy(), right, err_msg=name)

    def test_the_ceiling_follows_the_published_knots_and_stops_at_480_rpm(self):
        rpm = torch.tensor([0.0, 70.0, 275.0, 340.0, 450.0, 477.0, 480.0, 481.0, 5000.0], dtype=torch.float32)
        ceiling = effort_ceiling(rpm * (2 * math.pi / 60))
        expected = [min(value, MAX_APPLIED_NM) for value in CEILING_NM_KNOTS] + [0.0, 0.0]
        # The knot speeds arrive as float32 radians per second, so each knot is
        # reproduced to that representation, and the zero knots are exact.
        np.testing.assert_allclose(ceiling.numpy(), expected, rtol=0, atol=3e-6)
        np.testing.assert_array_equal(ceiling.numpy()[-3:], np.zeros(3, dtype=np.float32))
        self.assertEqual(float(effort_ceiling(torch.tensor([-ZERO_AUTHORITY_RPM * 2 * math.pi / 60]))), 0.0)
        self.assertLessEqual(float(ceiling.max()), 1.60001)
        self.assertGreaterEqual(float(ceiling.min()), 0.0)

    def test_the_applied_effort_stays_inside_the_ceiling_for_both_signs(self):
        q, dq, target = sample_states(seed=11)
        requested, applied, ceiling = motor_effort(
            torch.from_numpy(q), torch.from_numpy(dq), torch.from_numpy(target), torch.tensor(KD)
        )
        self.assertTrue(bool((applied.abs() <= ceiling + 1e-7).all()))
        self.assertLessEqual(float(applied.abs().max()), 1.60001)
        self.assertTrue(bool((applied.sign() * requested.sign() >= 0).all()))

    def test_the_stiffness_is_the_accepted_twelve(self):
        self.assertEqual(STIFFNESS_NM_PER_RAD, 12.0)
        q = torch.zeros(1, 18)
        target = torch.full((1, 18), 0.01)
        requested, _, _ = motor_effort(q, torch.zeros_like(q), target, torch.zeros(18))
        np.testing.assert_allclose(requested.numpy(), np.full((1, 18), 0.12), rtol=1e-6, atol=0)


class DampingBindingTests(unittest.TestCase):
    def test_damping_binds_the_prototype_values_by_joint_name(self):
        self.assertEqual(dict(DAMPING_NM_S_PER_RAD_BY_NAME), dict(zip(JOINT_NAMES, KD)))
        self.assertEqual(tuple(name for name, _ in DAMPING_NM_S_PER_RAD_BY_NAME), JOINT_NAMES)

    def test_damping_vector_follows_the_requested_order(self):
        table = dict(DAMPING_NM_S_PER_RAD_BY_NAME)
        vector = damping_vector(BLOCK_ORDER)
        self.assertEqual(vector.shape, (18,))
        np.testing.assert_array_equal(
            vector.numpy(), np.asarray([table[name] for name in BLOCK_ORDER], dtype=np.float32)
        )
        np.testing.assert_array_equal(damping_vector(JOINT_NAMES).numpy(), np.asarray(KD, dtype=np.float32))

    def test_an_unknown_or_incomplete_order_is_rejected(self):
        for order in (BLOCK_ORDER[:-1], BLOCK_ORDER + ("extra",), ("wrong",) + BLOCK_ORDER[1:]):
            with self.subTest(order=order[-1]), self.assertRaises(ValueError):
                damping_vector(order)


def isaac_stub_modules():
    class StubCfg:
        def __init__(self, **values):
            self.__dict__.update(values)

    class StubIdealPD:
        def __init__(self, cfg, joint_names, joint_ids=None, num_envs=2, device="cpu"):
            self.cfg, self.joint_names, self.joint_ids = cfg, list(joint_names), joint_ids
            self.num_envs, self._device = num_envs, device
            self.computed_effort = torch.zeros(num_envs, len(joint_names))
            self.applied_effort = torch.zeros_like(self.computed_effort)

        def compute(self, action, joint_pos, joint_vel):
            raise AssertionError("The paper-walk actuator computes its own effort")

    actuators = types.ModuleType("isaaclab.actuators")
    actuators.IdealPDActuator, actuators.IdealPDActuatorCfg = StubIdealPD, StubCfg
    configclass = types.ModuleType("isaaclab.utils.configclass")
    configclass.configclass = lambda cls: cls
    return {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.actuators": actuators,
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.configclass": configclass,
    }


def load_binding():
    modules = isaac_stub_modules()
    with patch.dict(sys.modules, modules):
        loaded = []
        for name, relative in (
            ("hexapod_env.actuators.rs05_paper_walk", "packages/hexapod_env/hexapod_env/actuators/rs05_paper_walk.py"),
            ("hexapod_env.actuators.rs05_paper_walk_runtime",
             "packages/hexapod_env/hexapod_env/actuators/rs05_paper_walk_runtime.py"),
        ):
            spec = importlib.util.spec_from_file_location(name, ROOT / relative)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded.append(module)
    return loaded


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding, cls.runtime = load_binding()

    def test_the_configuration_binds_the_accepted_gains(self):
        cfg = self.binding.make_rs05_paper_walk_cfg(BLOCK_ORDER)
        self.assertEqual(
            cfg.class_type, "hexapod_env.actuators.rs05_paper_walk_runtime:RS05PaperWalkActuator"
        )
        self.assertEqual(cfg.stiffness, 12.0)
        self.assertEqual(cfg.armature, 0.0)
        self.assertEqual(cfg.friction, 0.0)
        self.assertEqual(list(cfg.joint_names_expr), list(BLOCK_ORDER))
        self.assertEqual(list(cfg.active_joint_names), list(BLOCK_ORDER))
        self.assertEqual(cfg.damping, dict(DAMPING_NM_S_PER_RAD_BY_NAME))

    def test_an_incomplete_motor_group_is_rejected(self):
        for names in (BLOCK_ORDER[:-1], BLOCK_ORDER + ("passive",), ("wrong",) + BLOCK_ORDER[1:]):
            with self.subTest(names=names[-1]), self.assertRaises(ValueError):
                self.binding.make_rs05_paper_walk_cfg(names)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding, cls.runtime = load_binding()

    def actuator(self, order=BLOCK_ORDER):
        cfg = self.binding.make_rs05_paper_walk_cfg(BLOCK_ORDER)
        return self.runtime.RS05PaperWalkActuator(cfg, list(order))

    def test_physx_receives_only_the_applied_effort(self):
        actuator = self.actuator()
        q = torch.full((2, 18), 0.1)
        dq = torch.full((2, 18), 0.2)
        target = torch.full((2, 18), 0.3)
        action = types.SimpleNamespace(
            joint_positions=target.clone(), joint_velocities=torch.zeros_like(q), joint_efforts=torch.zeros_like(q)
        )
        result = actuator.compute(action, q, dq)
        requested, applied, ceiling = motor_effort(q, dq, target, damping_vector(BLOCK_ORDER))
        np.testing.assert_array_equal(result.joint_efforts.numpy(), applied.numpy())
        np.testing.assert_array_equal(actuator.computed_effort.numpy(), requested.numpy())
        np.testing.assert_array_equal(actuator.applied_effort.numpy(), applied.numpy())
        np.testing.assert_array_equal(actuator.effort_ceiling.numpy(), ceiling.numpy())
        self.assertIsNone(result.joint_positions)
        self.assertIsNone(result.joint_velocities)

    def test_the_resolved_order_selects_the_matching_damping(self):
        reversed_order = tuple(reversed(BLOCK_ORDER))
        actuator = self.actuator(reversed_order)
        np.testing.assert_array_equal(
            actuator.damping_nm_s_per_rad.numpy(), damping_vector(reversed_order).numpy()
        )
        q = torch.zeros(2, 18)
        dq = torch.full((2, 18), 1.0)
        action = types.SimpleNamespace(
            joint_positions=torch.zeros_like(q), joint_velocities=torch.zeros_like(q), joint_efforts=torch.zeros_like(q)
        )
        actuator.compute(action, q, dq)
        expected = motor_effort(q, dq, torch.zeros_like(q), damping_vector(reversed_order))[0]
        np.testing.assert_array_equal(actuator.computed_effort.numpy(), expected.numpy())

    def test_the_clip_hook_uses_the_last_motor_velocity(self):
        actuator = self.actuator()
        q = torch.zeros(2, 18)
        dq = torch.full((2, 18), 450.0 * 2 * math.pi / 60)
        action = types.SimpleNamespace(
            joint_positions=torch.zeros_like(q), joint_velocities=torch.zeros_like(q), joint_efforts=torch.zeros_like(q)
        )
        actuator.compute(action, q, dq)
        clipped = actuator._clip_effort(torch.full_like(q, 9.0))
        np.testing.assert_allclose(clipped.numpy(), np.full((2, 18), 1.6), rtol=0, atol=1e-7)

    def test_a_group_that_is_not_the_eighteen_named_motors_is_rejected(self):
        cfg = self.binding.make_rs05_paper_walk_cfg(BLOCK_ORDER)
        for order in (list(BLOCK_ORDER)[:-1], list(BLOCK_ORDER) + ["passive"]):
            with self.subTest(count=len(order)), self.assertRaises(ValueError):
                self.runtime.RS05PaperWalkActuator(cfg, order)

    def test_a_changed_gain_or_armature_is_rejected(self):
        for attribute, value in (("stiffness", 30.0), ("armature", 0.001), ("friction", 0.1)):
            cfg = self.binding.make_rs05_paper_walk_cfg(BLOCK_ORDER)
            setattr(cfg, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                self.runtime.RS05PaperWalkActuator(cfg, list(BLOCK_ORDER))


class PreKitImportTests(unittest.TestCase):
    def test_the_configuration_imports_before_kit_without_the_runtime(self):
        # The probe runs from a file because torch reads its own source while
        # importing, which a ``-c`` program cannot provide.
        code = '''
import importlib.abc, sys, types
class ForbidUSD(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pxr" or fullname.startswith("pxr."):
            raise AssertionError("USD imported before Kit")
sys.meta_path.insert(0, ForbidUSD())
class FakeCfg:
    def __init__(self, **values): self.__dict__.update(values)
actuators = types.ModuleType("isaaclab.actuators")
actuators.IdealPDActuatorCfg = FakeCfg
def forbidden(name):
    if name.startswith("__"): raise AttributeError(name)
    raise AssertionError("Runtime SDK symbol requested before Kit: " + name)
actuators.__getattr__ = forbidden
configclass = types.ModuleType("isaaclab.utils.configclass")
configclass.configclass = lambda cls: cls
sys.modules.update({"isaaclab": types.ModuleType("isaaclab"),
    "isaaclab.actuators": actuators, "isaaclab.utils": types.ModuleType("isaaclab.utils"),
    "isaaclab.utils.configclass": configclass})
sys.path[:0] = PACKAGE_PATHS
from hexapod_env.actuators.rs05_paper_walk import make_rs05_paper_walk_cfg
cfg = make_rs05_paper_walk_cfg(NAMES)
assert cfg.class_type == "hexapod_env.actuators.rs05_paper_walk_runtime:RS05PaperWalkActuator"
assert not any(n == "pxr" or n.startswith("pxr.") for n in sys.modules)
assert "hexapod_env.actuators.rs05_paper_walk_runtime" not in sys.modules
print("PRE_KIT_CONFIG_PASS")
'''.replace("PACKAGE_PATHS", repr([str(ROOT / "packages" / p) for p in ("hexapod_core", "hexapod_env")]))
        code = code.replace("NAMES", repr(list(BLOCK_ORDER)))
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory) / "pre_kit_probe.py"
            probe.write_text(code)
            result = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PRE_KIT_CONFIG_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
