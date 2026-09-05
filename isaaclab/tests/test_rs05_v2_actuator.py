"""CPU regressions for bounded RS05 bursts; no Isaac/Kit or hardware required."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
for package in ("hexapod_core", "hexapod_env"):
    sys.path.insert(0, str(ROOT / "packages" / package))

from hexapod_core import rs05_v2 as contract
from hexapod_env.actuators.rs05_v2_model import RS05V2BudgetModel, TELEMETRY_FIELDS


def model(headroom=1.0, shape=(1, 1), voltage=48.0):
    return RS05V2BudgetModel(shape, dtype=torch.float64, reset_burst_headroom=headroom,
                            assumed_bus_voltage_v=voltage)


def step(m, torque, rpm=0):
    demand = torch.full(m.shape, float(torque), dtype=torch.float64)
    speed = torch.full(m.shape, float(rpm) * 2 * math.pi / 60, dtype=torch.float64)
    return m.step(demand, speed)


class VendorEnvelopeTests(unittest.TestCase):
    def test_published_torque_speed_knots_not_archived_dc_line(self):
        for rpm, torque in contract.CONFIG["vendor"]["torque_speed_rpm_nm"]:
            m = model()
            actual = step(m, 100, rpm).item()
            self.assertAlmostEqual(actual, torque, places=10)
            self.assertAlmostEqual(m.torque_speed_limit_nm.item(), torque, places=10)
        m = model()
        self.assertAlmostEqual(step(m, 100, 400).item(), 3 + (1.6 - 3) * 60 / 110)

    def test_braking_uses_explicit_symmetric_absolute_speed_assumption(self):
        for rpm in (-450, -100, 0, 100, 450):
            positive, negative = model(), model()
            self.assertAlmostEqual(step(positive, 100, rpm).item(), -step(negative, -100, rpm).item())
        self.assertEqual(step(model(), -5.5, 600).item(), 0)

    def test_stall_rotating_and_high_speed_continuous_envelopes(self):
        for rpm, expected in [(0, 1.2), (50, 1.4), (100, 1.6), (200, .8), (400, .4), (480, 0)]:
            m = model(0)
            actual = step(m, 100, rpm).item()
            self.assertAlmostEqual(actual, expected, places=10)
            self.assertAlmostEqual(m.continuous_limit_nm.item(), expected, places=10)
            self.assertEqual(m.budget_consumed.item(), 0)

    def test_stall_vendor_point_rates(self):
        for torque, duration in contract.CONFIG["vendor"]["stall_overload_nm_s"]:
            m = model()
            self.assertAlmostEqual(step(m, torque).item(), torque)
            expected = 0 if duration is None else .005 / duration
            self.assertAlmostEqual(m.budget_consumed.item(), expected, places=12)

    def test_rotating_70mm_vendor_point_rates(self):
        for torque, duration in contract.CONFIG["vendor"]["rotating_overload_nm_s"]:
            # At100rpm the speed curve permits5.2805Nm, so5.5Nm is
            # intentionally unattainable there. Test rate interpolation
            # separately from the physical speed cap for the top point.
            if torque == 5.5:
                continue
            m = model()
            self.assertAlmostEqual(step(m, torque, 100).item(), torque)
            self.assertAlmostEqual(m.budget_consumed.item(), 0 if duration is None else .005 / duration, places=12)

    def test_voltage_is_assumption_and_cannot_increase_authority(self):
        self.assertAlmostEqual(step(model(voltage=24), 100).item(), 2.75)
        self.assertAlmostEqual(step(model(voltage=24), 100, 240).item(), 0)
        self.assertAlmostEqual(step(model(voltage=60), 100).item(), 5.5)
        for bad in (0, 14.9, 60.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): model(voltage=bad)


class BudgetTests(unittest.TestCase):
    def test_one_second_peak_at_stall_then_continuous_holding(self):
        m = model()
        for _ in range(200):
            self.assertAlmostEqual(step(m, 5.5).item(), 5.5, places=9)
        self.assertAlmostEqual(m.burst_headroom.item(), 0, places=10)
        for _ in range(5):
            self.assertAlmostEqual(step(m, 5.5).item(), 1.2, places=9)
        self.assertAlmostEqual(m.applied_peak_exposure_s.item(), 1, places=9)
        self.assertAlmostEqual(m.applied_overload_exposure_s.item(), 1, places=9)

    def test_partial_final_step_cannot_overdraw_budget(self):
        m = model(.001)
        actual = step(m, 5.5).item()
        self.assertGreater(actual, 1.2)
        self.assertLess(actual, 5.5)
        self.assertAlmostEqual(m.budget_consumed.item(), .001, places=12)
        self.assertAlmostEqual(m.burst_headroom.item(), 0, places=12)
        self.assertEqual(m.envelope_violation_nm.item(), 0)
        self.assertAlmostEqual(step(m, 100).item(), 1.2)

    def test_mixed_load_and_reversal_preserve_one_budget(self):
        m = model(.7)
        initial = m.burst_headroom.item()
        step(m, 4, 0)
        first_cost = m.budget_consumed.item()
        step(m, -3, -100)
        self.assertAlmostEqual(m.burst_headroom.item(), initial - first_cost - m.budget_consumed.item())

    def test_no_recovery_at_half_continuous_or_above(self):
        for torque in (.6, 1.0, 1.2, 2.0):
            m = model(.25)
            step(m, torque)
            self.assertEqual(m.budget_recovered.item(), 0)
            self.assertLessEqual(m.burst_headroom.item(), .25)

    def test_low_load_recovery_is_explicit_bounded_60_second_assumption(self):
        m = model(0)
        step(m, 0)
        self.assertAlmostEqual(m.burst_headroom.item(), .005 / 60)
        m = model(0)
        step(m, .3)
        self.assertAlmostEqual(m.burst_headroom.item(), .5 * .005 / 60)
        m = model(1)
        step(m, 0)
        self.assertEqual(m.burst_headroom.item(), 1)
        self.assertEqual(m.budget_recovered.item(), 0)

    def test_selected_reset_does_not_recharge_other_environments(self):
        m = model(.5, shape=(3, 2))
        step(m, 5.5)
        before = m.burst_headroom.clone()
        m.reset(torch.tensor([1]))
        self.assertTrue(torch.equal(m.burst_headroom[[0, 2]], before[[0, 2]]))
        self.assertTrue(torch.equal(m.burst_headroom[1], torch.full((2,), .5, dtype=torch.float64)))
        self.assertTrue(torch.all(m.applied_peak_exposure_s[1] == 0))
        self.assertTrue(torch.all(m.applied_peak_exposure_s[0] > 0))

    def test_raw_demand_clipping_continuous_excess_peak_and_current_distinct(self):
        m = model()
        applied = step(m, 10).item()
        self.assertEqual(applied, 5.5)
        self.assertEqual(m.raw_demand_nm.item(), 10)
        self.assertEqual(m.clipping_nm.item(), 4.5)
        self.assertAlmostEqual(m.continuous_overload_nm.item(), 8.8)
        self.assertAlmostEqual(m.applied_overload_nm.item(), 4.3)
        self.assertAlmostEqual(m.estimated_phase_current_arms.item(), 5.5 / .94)
        self.assertAlmostEqual(m.estimated_phase_current_apk.item(), 5.5 / .94 * math.sqrt(2))
        self.assertIn("not battery current", contract.CONFIG["units"])

    def test_heterogeneous_batch_and_float32_remain_bounded(self):
        torch.manual_seed(17)
        m = RS05V2BudgetModel((32, 18))
        for _ in range(40):
            demand, speed = torch.randn(32, 18) * 8, torch.randn(32, 18) * 40
            actual = m.step(demand, speed)
            self.assertTrue(torch.all(torch.isfinite(actual)))
            self.assertTrue(torch.all(actual.abs() <= m.instantaneous_limit_nm + 1e-6))
            self.assertTrue(torch.all((m.burst_headroom >= 0) & (m.burst_headroom <= 1)))
            self.assertTrue(torch.all(m.envelope_violation_nm == 0))

    def test_nonfinite_inputs_fail_closed_and_latch_until_reset(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            m = model()
            self.assertEqual(step(m, bad).item(), 0)
            self.assertTrue(m.invalid_input.item())
            self.assertEqual(step(m, 1).item(), 0)
            m.reset()
            self.assertEqual(step(m, 1).item(), 1)
            m = model()
            self.assertEqual(step(m, 1, bad).item(), 0)
            self.assertTrue(m.invalid_input.item())

    def test_invalid_budget_and_shape_rejected(self):
        for bad in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError): model(bad)
        m = model()
        m.burst_headroom.fill_(1.01)
        self.assertEqual(step(m, 1).item(), 0)
        self.assertTrue(m.invalid_input.item())
        with self.assertRaises(ValueError): m.step(torch.zeros(2, 2), torch.zeros(2, 2))
        with self.assertRaises(ValueError): RS05V2BudgetModel((1, 1), physics_dt_s=.02)

    def test_finite_extreme_speed_cannot_overflow_budget_arithmetic(self):
        m = RS05V2BudgetModel((1, 1))
        maximum = torch.finfo(torch.float32).max
        actual = m.step(torch.tensor([[maximum]]), torch.tensor([[maximum]]))
        self.assertEqual(actual.item(), 0)
        for name in TELEMETRY_FIELDS:
            self.assertTrue(torch.isfinite(getattr(m, name)).all(), name)


class ContractTests(unittest.TestCase):
    def test_hash_covers_actual_model_and_configuration_bytes(self):
        actual = contract.contract_manifest()
        self.assertEqual(set(actual["implementation_sha256"]), set(contract.IMPLEMENTATION_PATHS))
        self.assertEqual(actual["configuration"]["vendor"]["peak_output_torque_nm"], 5.5)
        self.assertEqual(actual["configuration"]["vendor"]["stall_continuous_nm"], 1.2)
        self.assertNotEqual(actual["contract_sha256"], contract.contract_manifest(assumed_bus_voltage_v=24)["contract_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rel in contract.IMPLEMENTATION_PATHS:
                destination = root / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / rel).read_bytes())
            self.assertEqual(actual, contract.contract_manifest(repo_root=root))
            (root / contract.IMPLEMENTATION_PATHS[-1]).write_text("# changed model")
            self.assertNotEqual(actual["contract_sha256"], contract.contract_manifest(repo_root=root)["contract_sha256"])
            (root / contract.IMPLEMENTATION_PATHS[-1]).unlink()
            with self.assertRaises(FileNotFoundError): contract.contract_manifest(repo_root=root)

    def test_runtime_factory_values_are_exact_and_reject_legacy_overrides(self):
        names = [f"motor_{i}" for i in range(18)]
        values = contract.configuration_values(names)
        cfg = types.SimpleNamespace(**values, class_type=contract.ACTUATOR_CLASS)
        self.assertEqual(contract.verify_runtime_cfg(cfg, names)["effort_limit"], 5.5)
        for attribute, bad in [("effort_limit", 1.6), ("stiffness", 100), ("reset_burst_headroom", 1), ("class_type", "DCMotor")]:
            damaged = copy.deepcopy(cfg)
            setattr(damaged, attribute, bad)
            with self.assertRaises(ValueError): contract.verify_runtime_cfg(damaged, names)
        for bad_names in (names[:-1], names + ["passive"], names[:-1] + [names[0]]):
            with self.assertRaises(ValueError): contract.configuration_values(bad_names)

    def test_sdk_binding_calls_budget_once_per_compute_and_reset(self):
        """Exercise actual binding against a minimal faithful IdealPD interface."""
        class FakeIdealPD:
            def __init__(self, cfg, joint_names, joint_ids=None, num_envs=2, device="cpu"):
                self.cfg, self.joint_names = cfg, joint_names
                self.computed_effort = torch.zeros(num_envs, len(joint_names), dtype=torch.float64)
                self.stiffness, self.damping = cfg.stiffness, cfg.damping

            def compute(self, action, position, velocity):
                self.computed_effort = self.stiffness * (action.joint_positions - position) + self.damping * (action.joint_velocities - velocity) + action.joint_efforts
                self.applied_effort = self._clip_effort(self.computed_effort)
                action.joint_efforts = self.applied_effort
                action.joint_positions = action.joint_velocities = None
                return action

        class FakeCfg:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        actuator_module = types.ModuleType("isaaclab.actuators")
        actuator_module.IdealPDActuator, actuator_module.IdealPDActuatorCfg = FakeIdealPD, FakeCfg
        utils_module = types.ModuleType("isaaclab.utils")
        configclass_module = types.ModuleType("isaaclab.utils.configclass")
        configclass_module.configclass = lambda cls: cls
        modules = {"isaaclab": types.ModuleType("isaaclab"), "isaaclab.actuators": actuator_module,
                   "isaaclab.utils": utils_module, "isaaclab.utils.configclass": configclass_module}
        name = "hexapod_env.actuators.rs05_v2"
        spec = importlib.util.spec_from_file_location(name, ROOT / contract.IMPLEMENTATION_PATHS[2])
        binding = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(binding)
            runtime_name = "hexapod_env.actuators.rs05_v2_runtime"
            runtime_spec = importlib.util.spec_from_file_location(runtime_name, ROOT / contract.IMPLEMENTATION_PATHS[4])
            runtime = importlib.util.module_from_spec(runtime_spec)
            runtime_spec.loader.exec_module(runtime)
        names = [f"motor_{i}" for i in range(18)]
        cfg = binding.make_rs05_v2_cfg(names)
        actuator = runtime.RS05V2Actuator(cfg, list(reversed(names)))
        zeros = torch.zeros(2, 18, dtype=torch.float64)
        action = types.SimpleNamespace(joint_positions=torch.ones_like(zeros), joint_velocities=zeros.clone(), joint_efforts=zeros.clone())
        result = actuator.compute(action, zeros, zeros)
        self.assertTrue(torch.all(actuator.computed_effort == 30))
        self.assertTrue(torch.all(result.joint_efforts == 5.5))
        self.assertTrue(torch.allclose(actuator.burst_headroom, torch.full_like(zeros, .495)))
        self.assertIsNone(result.joint_positions)
        for field in TELEMETRY_FIELDS:
            self.assertEqual(getattr(actuator, field).shape, (2, 18))
        actuator.reset([0])
        self.assertTrue(torch.all(actuator.burst_headroom[0] == .5))
        self.assertTrue(torch.all(actuator.burst_headroom[1] == .495))
        with self.assertRaises(ValueError): runtime.RS05V2Actuator(cfg, names[:-1] + ["passive"])

    def test_configuration_import_cannot_resolve_runtime_or_usd(self):
        """Separate interpreter: runtime SDK symbol access fails before Kit."""
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
    if name == "__path__": raise AttributeError(name)
    raise AssertionError("Runtime SDK symbol requested before Kit: " + name)
actuators.__getattr__ = forbidden
configclass = types.ModuleType("isaaclab.utils.configclass")
configclass.configclass = lambda cls: cls
sys.modules.update({"isaaclab": types.ModuleType("isaaclab"),
    "isaaclab.actuators": actuators, "isaaclab.utils": types.ModuleType("isaaclab.utils"),
    "isaaclab.utils.configclass": configclass})
sys.path[:0] = PACKAGE_PATHS
from hexapod_env.actuators.rs05_v2 import make_rs05_v2_cfg
cfg = make_rs05_v2_cfg(["motor_"+str(i) for i in range(18)])
assert cfg.class_type == "hexapod_env.actuators.rs05_v2_runtime:RS05V2Actuator"
assert not any(n == "pxr" or n.startswith("pxr.") for n in sys.modules)
assert "hexapod_env.actuators.rs05_v2_runtime" not in sys.modules
assert "hexapod_env.actuators.rs05_v2_model" not in sys.modules
print("PRE_KIT_CONFIG_PASS")
'''.replace("PACKAGE_PATHS", repr([str(ROOT / "packages" / p) for p in ("hexapod_core", "hexapod_env")]))
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PRE_KIT_CONFIG_PASS", result.stdout)

    def test_task_entry_point_defers_runtime_environment_until_requested(self):
        """Guard the shim independently of SDK availability and Hydra internals."""
        shim = ROOT / "isaaclab/hexapod_rl/tasks/mkii_fourbar_v1.py"
        code = '''
import importlib.abc, sys, types
runtime_name = "hexapod_env.tasks.mkii_fourbar_v1.env"
class ForbidRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == runtime_name or fullname == "pxr" or fullname.startswith("pxr."):
            raise AssertionError("RUNTIME_IMPORT_BEFORE_KIT:" + fullname)
sys.meta_path.insert(0, ForbidRuntime())
for name in ("hexapod_env", "hexapod_env.tasks", "hexapod_env.tasks.mkii_fourbar_v1"):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
config_name = "hexapod_env.tasks.mkii_fourbar_v1.config"
config = types.ModuleType(config_name)
config.HexapodMkiiFourbarV1EnvCfg = type("EnvCfg", (), {})
config.HexapodMkiiFourbarV1PPORunnerCfg = type("RunnerCfg", (), {})
sys.modules[config_name] = config
namespace = {"__name__": "entry_point_probe", "__file__": SHIM_PATH}
source = SHIM_SOURCE
if NEGATIVE_CONTROL:
    source += "\\nfrom hexapod_env.tasks.mkii_fourbar_v1.env import HexapodMkiiFourbarEnv\\n"
exec(compile(source, SHIM_PATH, "exec"), namespace)
assert namespace["HexapodMkiiFourbarV1EnvCfg"] is config.HexapodMkiiFourbarV1EnvCfg
assert namespace["HexapodMkiiFourbarV1PPORunnerCfg"] is config.HexapodMkiiFourbarV1PPORunnerCfg
assert runtime_name not in sys.modules
assert not any(n == "pxr" or n.startswith("pxr.") for n in sys.modules)
assert "HexapodMkiiFourbarEnv" not in namespace
# Once the runtime is available, the historical environment export resolves.
runtime = types.ModuleType(runtime_name)
runtime.HexapodMkiiFourbarEnv = type("RuntimeEnv", (), {})
sys.modules[runtime_name] = runtime
assert namespace["__getattr__"]("HexapodMkiiFourbarEnv") is runtime.HexapodMkiiFourbarEnv
try:
    namespace["__getattr__"]("unknown")
except AttributeError:
    pass
else:
    raise AssertionError("Unknown export did not fail")
print("PRE_KIT_TASK_SHIM_PASS")
'''.replace("SHIM_PATH", repr(str(shim))).replace("SHIM_SOURCE", repr(shim.read_text()))
        for negative in (False, True):
            result = subprocess.run([sys.executable, "-I", "-c", code.replace("NEGATIVE_CONTROL", repr(negative))],
                                    capture_output=True, text=True)
            if negative:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("RUNTIME_IMPORT_BEFORE_KIT", result.stderr)
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("PRE_KIT_TASK_SHIM_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
