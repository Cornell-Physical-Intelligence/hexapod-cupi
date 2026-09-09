"""Named controller selection and actual actuator binding, with no Isaac launch."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/hexapod_core"))
from hexapod_core import rs05_v2 as contract
from hexapod_core import fourbar_v1 as fourbar_contract


class ControllerProfileTests(unittest.TestCase):
    names = [f"motor_{i}" for i in range(18)]
    candidate = "mkii_pd_damping_030_v1"

    def test_explicit_candidate_changes_only_damping_and_profile_identity(self):
        baseline = contract.configuration_values(self.names, physics_dt_s=.00125)
        candidate = contract.configuration_values(self.names, physics_dt_s=.00125,
                                                  controller_profile=self.candidate)
        self.assertEqual(baseline["controller_profile"], "rs05_pd_default_v2")
        self.assertEqual((baseline["stiffness"], baseline["damping"]), (30., .6))
        self.assertEqual((candidate["stiffness"], candidate["damping"]), (30., .3))
        self.assertEqual({key for key in baseline if baseline[key] != candidate[key]},
                         {"controller_profile", "damping"})
        self.assertEqual(contract.CONFIG["provisional"]["damping_nm_s_per_rad"], .6)
        self.assertEqual(contract.CONFIG["provisional"]["stiffness_nm_per_rad"], 30.)
        self.assertEqual(contract.CONFIG, json.loads(contract.CONFIG_PATH.read_text()))

    def test_profiles_resolve_immutable_known_parameters_without_aliases(self):
        self.assertEqual(contract.LOWER_DAMPING_CONTROLLER_PROFILE, self.candidate)
        resolved = contract.resolve_controller_profile(self.candidate)
        resolved["damping_nm_s_per_rad"] = 1.
        self.assertEqual(contract.resolve_controller_profile(self.candidate)["damping_nm_s_per_rad"], .3)
        with self.assertRaises(TypeError):
            contract.CONTROLLER_PROFILES[self.candidate]["damping_nm_s_per_rad"] = 1.
        for invalid in (None, "", "default", "mkii_pd_damping_030", .3, {"damping": .3}):
            with self.subTest(profile=invalid):
                with self.assertRaises(ValueError):
                    contract.configuration_values(self.names, controller_profile=invalid)
                with self.assertRaises(ValueError):
                    contract.contract_manifest(controller_profile=invalid)

    def test_runtime_verification_rejects_hidden_gain_or_label_override(self):
        values = contract.configuration_values(self.names, controller_profile=self.candidate)
        cfg = types.SimpleNamespace(**values, class_type=contract.ACTUATOR_CLASS)
        self.assertEqual(contract.verify_runtime_cfg(cfg, self.names),
                         dict(values, class_type=contract.ACTUATOR_CLASS))
        for field, value in (("damping", .6), ("stiffness", 29.), ("effort_limit", 1.6),
                             ("armature", .001), ("controller_profile", contract.DEFAULT_CONTROLLER_PROFILE),
                             ("controller_profile", "unknown")):
            damaged = copy.deepcopy(cfg)
            setattr(damaged, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                contract.verify_runtime_cfg(damaged, self.names)
        del cfg.controller_profile
        with self.assertRaises(ValueError):
            contract.verify_runtime_cfg(cfg, self.names)
        baseline = types.SimpleNamespace(**contract.configuration_values(self.names),
                                         class_type=contract.ACTUATOR_CLASS)
        baseline.damping = .3
        with self.assertRaises(ValueError):
            contract.verify_runtime_cfg(baseline, self.names)

    def test_manifest_records_actual_profile_and_hashes_without_changing_vendor_data(self):
        baseline = contract.contract_manifest(physics_dt_s=.00125)
        selected = contract.contract_manifest(physics_dt_s=.00125, controller_profile=self.candidate)
        self.assertEqual(selected["controller_profile"], self.candidate)
        self.assertEqual(selected["controller_parameters"],
                         {"stiffness_nm_per_rad": 30., "damping_nm_s_per_rad": .3})
        self.assertEqual(baseline["controller_parameters"]["damping_nm_s_per_rad"], .6)
        self.assertEqual(selected["configuration"], baseline["configuration"])
        self.assertEqual(selected["implementation_sha256"], baseline["implementation_sha256"])
        self.assertEqual(selected["model_id"], baseline["model_id"])
        self.assertNotEqual(selected["contract_sha256"], baseline["contract_sha256"])
        self.assertEqual({key for key in selected if selected[key] != baseline[key]},
                         {"controller_profile", "controller_parameters", "contract_sha256"})
        digest = selected.pop("contract_sha256")
        self.assertEqual(digest, hashlib.sha256(json.dumps(selected, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest())

    def test_actual_factory_runtime_binding_changes_velocity_feedback_only(self):
        # A separate interpreter keeps fake SDK modules from disturbing native
        # libraries imported by the surrounding full CPU test suite.
        code = r'''
import sys, types
import torch
sys.path[:0] = PACKAGE_PATHS
class FakeCfg:
    def __init__(self, **values): self.__dict__.update(values)
class FakePD:
    def __init__(self, cfg, joint_names):
        self.joint_names = joint_names
        self.computed_effort = torch.zeros(1, len(joint_names), dtype=torch.float64)
        self.stiffness, self.damping = cfg.stiffness, cfg.damping
    def compute(self, action, position, velocity):
        self.computed_effort = self.stiffness*(action.joint_positions-position) + self.damping*(action.joint_velocities-velocity) + action.joint_efforts
        self.applied_effort = self._clip_effort(self.computed_effort)
        action.joint_efforts = self.applied_effort
        action.joint_positions = action.joint_velocities = None
        return action
actuators = types.ModuleType("isaaclab.actuators")
actuators.IdealPDActuatorCfg, actuators.IdealPDActuator = FakeCfg, FakePD
configclass = types.ModuleType("isaaclab.utils.configclass")
configclass.configclass = lambda cls: cls
sys.modules.update({"isaaclab": types.ModuleType("isaaclab"), "isaaclab.actuators": actuators,
    "isaaclab.utils": types.ModuleType("isaaclab.utils"), "isaaclab.utils.configclass": configclass})
from hexapod_core import rs05_v2 as contract
from hexapod_env.actuators.rs05_v2 import make_rs05_v2_cfg
from hexapod_env.actuators.rs05_v2_runtime import RS05V2Actuator
names = ["motor_"+str(i) for i in range(18)]
zeros = torch.zeros(1,18,dtype=torch.float64)
results = []
for profile in (contract.DEFAULT_CONTROLLER_PROFILE, "mkii_pd_damping_030_v1"):
    cfg = make_rs05_v2_cfg(names, physics_dt_s=.00125, controller_profile=profile)
    verified = contract.verify_runtime_cfg(cfg, names)
    manifest = contract.contract_manifest(physics_dt_s=.00125, controller_profile=profile)
    assert verified["controller_profile"] == manifest["controller_profile"]
    assert verified["damping"] == manifest["controller_parameters"]["damping_nm_s_per_rad"]
    motor = RS05V2Actuator(cfg, list(reversed(names)))
    outputs = []
    for position_target, velocity in ((0.,2.), (.04,0.), (1.,0.)):
        motor.reset()
        action = types.SimpleNamespace(joint_positions=torch.full_like(zeros,position_target),
            joint_velocities=zeros.clone(), joint_efforts=zeros.clone())
        out = motor.compute(action, zeros, torch.full_like(zeros,velocity))
        outputs.append((motor.computed_effort.clone(), out.joint_efforts.clone()))
        assert out.joint_positions is None and out.joint_velocities is None
        assert motor._budget.dt == .00125
    results.append(outputs)
torch.testing.assert_close(results[0][0][0], torch.full_like(zeros,-1.2))
torch.testing.assert_close(results[1][0][0], torch.full_like(zeros,-.6))
for outputs in results:
    torch.testing.assert_close(outputs[1][0], torch.full_like(zeros,1.2))
    torch.testing.assert_close(outputs[1][1], torch.full_like(zeros,1.2))
    torch.testing.assert_close(outputs[2][0], torch.full_like(zeros,30.))
    torch.testing.assert_close(outputs[2][1], torch.full_like(zeros,5.5))
print("PROFILE_FACTORY_AND_ACTUATOR_PASS")
'''.replace("PACKAGE_PATHS", repr([str(ROOT / "packages" / package)
                                    for package in ("hexapod_core", "hexapod_env")]))
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PROFILE_FACTORY_AND_ACTUATOR_PASS", result.stdout)


class PhysicalTaskControllerWiringTests(unittest.TestCase):
    """Execute the exact task statements at the SDK-independent boundaries."""

    @classmethod
    def setUpClass(cls):
        directory = ROOT / "packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1"
        cfg_path, env_path = directory / "config.py", directory / "env.py"
        cfg_tree, env_tree = ast.parse(cfg_path.read_text()), ast.parse(env_path.read_text())
        factories = [node for node in ast.walk(cfg_tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "make_rs05_v2_cfg"]
        if len(factories) != 1:
            raise AssertionError("Expected one explicit physical-task motor factory call")
        cls.factory_call = compile(ast.Expression(factories[0]), str(cfg_path), "eval")
        env_class = next(node for node in env_tree.body
                         if isinstance(node, ast.ClassDef) and node.name == "HexapodMkiiFourbarEnv")
        initializer = next(node for node in env_class.body
                           if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        assignments = [node for node in initializer.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "motor_cfg"
                               for target in node.targets)]
        def tests_motor_key(node, key):
            return isinstance(node, ast.If) and any(
                isinstance(item, ast.Subscript) and isinstance(item.value, ast.Name)
                and item.value.id == "motor_cfg" and isinstance(item.slice, ast.Constant)
                and item.slice.value == key for item in ast.walk(node.test))
        guards = [node for node in initializer.body if tests_motor_key(node, "controller_profile")]
        timing = [node for node in initializer.body if tests_motor_key(node, "physics_dt_s")]
        if len(assignments) != 1 or len(guards) != 1 or len(timing) != 1:
            raise AssertionError("Expected actual motor verification, controller and timing guards")
        if not assignments[0].lineno < guards[0].lineno < timing[0].lineno:
            raise AssertionError("The actual controller profile must be checked before timing")
        cls.verify_and_guard = compile(ast.Module(body=[assignments[0], guards[0]], type_ignores=[]),
                                       str(env_path), "exec")
        manifests = [node for node in ast.walk(initializer) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "motor_contract_manifest"]
        if len(manifests) != 1:
            raise AssertionError("Expected one explicit actual-motor manifest call")
        cls.manifest_call = compile(ast.Expression(manifests[0]), str(env_path), "eval")

    def test_actual_task_factory_selects_named_candidate_with_matching_timing(self):
        # The actual SDK factory has its own integration test above; substitute
        # its core value producer here to execute the task's exact arguments.
        values = eval(self.factory_call, {"contract": fourbar_contract,
                                         "make_rs05_v2_cfg": contract.configuration_values})
        self.assertEqual(fourbar_contract.MOTOR_CONTROLLER_PROFILE, "mkii_pd_damping_030_v1")
        self.assertEqual(values["controller_profile"], fourbar_contract.MOTOR_CONTROLLER_PROFILE)
        self.assertEqual(values["active_joint_names"], list(fourbar_contract.ACTIVE_JOINT_NAMES))
        self.assertEqual(values["physics_dt_s"], fourbar_contract.PHYSICS_DT_S)
        self.assertEqual((values["stiffness"], values["damping"]), (30., .3))

    def test_actual_env_guard_rejects_valid_default_profile_and_accepts_candidate(self):
        for profile in (contract.DEFAULT_CONTROLLER_PROFILE, fourbar_contract.MOTOR_CONTROLLER_PROFILE):
            cfg = types.SimpleNamespace(**contract.configuration_values(fourbar_contract.ACTIVE_JOINT_NAMES,
                physics_dt_s=fourbar_contract.PHYSICS_DT_S, controller_profile=profile),
                class_type=contract.ACTUATOR_CLASS)
            # The default profile is valid for the motor in isolation, but is
            # explicitly rejected by the physical task's real constructor guard.
            self.assertEqual(contract.verify_runtime_cfg(cfg, fourbar_contract.ACTIVE_JOINT_NAMES)
                             ["controller_profile"], profile)
            raw = types.SimpleNamespace(_motor_model=types.SimpleNamespace(cfg=cfg),
                                        active_joint_names=list(fourbar_contract.ACTIVE_JOINT_NAMES))
            scope = {"self": raw, "contract": fourbar_contract, "verify_runtime_cfg": contract.verify_runtime_cfg}
            if profile == contract.DEFAULT_CONTROLLER_PROFILE:
                with self.assertRaisesRegex(ValueError, "Motor controller profile differs"):
                    exec(self.verify_and_guard, scope)
            else:
                exec(self.verify_and_guard, scope)
                self.assertEqual(scope["motor_cfg"]["controller_profile"], profile)

    def test_actual_env_manifest_call_forwards_resolved_profile_and_operating_values(self):
        for profile in (contract.DEFAULT_CONTROLLER_PROFILE, fourbar_contract.MOTOR_CONTROLLER_PROFILE):
            values = contract.configuration_values(fourbar_contract.ACTIVE_JOINT_NAMES,
                physics_dt_s=.00125, assumed_bus_voltage_v=24., controller_profile=profile)
            manifest = eval(self.manifest_call, {"motor_cfg": values,
                                                 "motor_contract_manifest": contract.contract_manifest})
            self.assertEqual(manifest["controller_profile"], values["controller_profile"])
            self.assertEqual(manifest["controller_parameters"]["damping_nm_s_per_rad"], values["damping"])
            self.assertEqual(manifest["physics_dt_s"], values["physics_dt_s"])
            self.assertEqual(manifest["assumed_bus_voltage_v"], values["assumed_bus_voltage_v"])


if __name__ == "__main__":
    unittest.main()
