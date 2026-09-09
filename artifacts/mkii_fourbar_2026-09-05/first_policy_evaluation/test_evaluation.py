"""Local CPU tests. These do not claim live Isaac/RSL integration coverage."""
import copy
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import evaluation_math as model
import evaluate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sdk_definition(path_suffix, name):
    evidence = json.loads((HERE/"sdk_contract_evidence.json").read_text())["files"]
    entry = next(value for path, value in evidence.items() if path.endswith(path_suffix))
    return entry["definitions"][name]["source"]


def compile_definition(source, name, namespace=None):
    values = {} if namespace is None else namespace
    exec(compile("from __future__ import annotations\n"+source,"<reviewed-source>","exec"),values)
    return values[name]


def perfect_rows(scenario, count=model.STEPS):
    initial = {"position_world_m": [0.,0.,.14], "quaternion_wxyz": [1.,0.,0.,0.]}
    pose = [0.,0.,-math.pi/2]
    rows = []
    for step in range(count):
        command, window = model.scheduled_command(scenario, step)
        pose = model.reference_step(pose, command)
        yaw = pose[2]+math.pi/2
        rows.append({"step": step, "command": command, "window": window,
            "position_world_m": [*pose[:2],.14], "quaternion_wxyz": [math.cos(yaw/2),0.,0.,math.sin(yaw/2)],
            "velocity_navigation_m_s": [*command[:2],0.], "yaw_rate_rad_s": command[2],
            "velocity_world_m_s": [math.cos(pose[2])*command[0]-math.sin(pose[2])*command[1],
                                   math.sin(pose[2])*command[0]+math.cos(pose[2])*command[1],0.],
            "height_m": .14, "support_count": 3, "nonfoot_contacts": 0,
            "loaded_foot_slip_rms_m_s": 0., "max_raw_demand_nm": .7, "max_applied_nm": .7,
            "max_clipping_nm": 0., "max_continuous_overload_nm": 0., "min_headroom": .5,
            "max_motor_speed_rad_s": 0., "max_estimated_phase_current_arms": 0.,
            "max_motor_overload_exposure_s": 0., "max_motor_peak_exposure_s": 0.,
            "reward": 1., "reward_components": {"track": 50.},
            "terminated": False, "truncated": False, "termination_reasons": []})
    return initial, rows


class ScoringTests(unittest.TestCase):
    def test_counts_axes_and_diagonal_speed(self):
        cases = model.scenarios()
        self.assertEqual(len(cases), 14)
        self.assertEqual(len({c["name"] for c in cases}), 14)
        self.assertEqual(len(model.SEEDS)*len(cases), 42)
        self.assertEqual(len(model.SEEDS)*model.STEPS, 1500)
        for case in cases:
            if case["name"].startswith("diagonal"):
                self.assertAlmostEqual(math.hypot(*case["command"][:2]), .1)

    def test_schedule_boundaries_and_no_timeout(self):
        forward = model.scenarios()[1]
        self.assertEqual(model.scheduled_command(forward,99)[1], "startup")
        self.assertEqual(model.scheduled_command(forward,100), ([.1,0.,0.], "motion"))
        self.assertEqual(model.scheduled_command(forward,299)[1], "motion")
        self.assertEqual(model.scheduled_command(forward,300)[1], "stop")
        reverse = model.scenarios()[-3]
        self.assertEqual(model.scheduled_command(reverse,249)[0][0], .1)
        self.assertEqual(model.scheduled_command(reverse,250)[0][0], -.1)
        self.assertEqual(model.scheduled_command(reverse,400)[1], "stop")
        for invalid in (-1,500,True,1.2):
            with self.assertRaises(ValueError): model.scheduled_command(reverse,invalid)

    def test_anatomical_heading_and_body_twist(self):
        self.assertAlmostEqual(model.quaternion_attitude([1.,0.,0.,0.])[2], -math.pi/2)
        rotated = model.quaternion_attitude([math.sqrt(.5),0.,0.,math.sqrt(.5)])[2]
        self.assertAlmostEqual(rotated, 0.)
        x,y,h = model.reference_step([0.,0.,0.],[1.,0.,1.],math.pi/2)
        self.assertAlmostEqual(x,1.)
        self.assertAlmostEqual(y,1.)
        self.assertAlmostEqual(h,math.pi/2)

    def test_native_sdk_xyzw_alias_conversion_and_identity_reset(self):
        alias = compile_definition(sdk_definition("base_articulation_data.py","root_quat_w"),"root_quat_w")
        source = sdk_definition("isaaclab_physx/assets/articulation/articulation_data.py","root_link_quat_w")
        self.assertIn("(x, y, z, w)",ast.get_docstring(ast.parse(source).body[0]))
        identity = alias(types.SimpleNamespace(root_link_quat_w=[0.,0.,0.,1.]))
        self.assertEqual(model.native_xyzw_to_wxyz(identity),[1.,0.,0.,0.])
        self.assertTrue(model.verify_reset_identity([identity,[0.,0.,0.,-1.]])["pass"])
        with self.assertRaises(ValueError):
            model.verify_reset_identity([[1.,0.,0.,0.]])
        native_yaw90 = alias(types.SimpleNamespace(root_link_quat_w=[0.,0.,math.sqrt(.5),math.sqrt(.5)]))
        roll,pitch,heading = model.quaternion_attitude(model.native_xyzw_to_wxyz(native_yaw90))
        self.assertAlmostEqual(roll,0.)
        self.assertAlmostEqual(pitch,0.)
        self.assertAlmostEqual(heading,0.)

    def test_all_perfect_scenarios_have_zero_tracking_and_path_error(self):
        for scenario in model.scenarios():
            initial, rows = perfect_rows(scenario)
            with self.subTest(scenario=scenario["name"]):
                result = model.score_trial(scenario,101,initial,rows)
                self.assertTrue(result["completed"])
                self.assertEqual(result["path_position_error_m"]["max"],0.)
                self.assertEqual(result["stop_travel_m"],0.)
                for window in result["windows"].values():
                    self.assertEqual(window["rmse_forward_left_yaw"], [0.,0.,0.])

    def test_zero_motion_does_not_earn_tracking_success_from_reward(self):
        scenario = model.scenarios()[1]
        initial, rows = perfect_rows(scenario)
        for row in rows:
            row["position_world_m"] = [0.,0.,.14]
            row["velocity_navigation_m_s"] = [0.,0.,0.]
        result = model.score_trial(scenario,101,initial,rows)
        self.assertAlmostEqual(result["windows"]["motion"]["rmse_forward_left_yaw"][0], .1)
        self.assertAlmostEqual(result["path_position_error_m"]["max"], .4)
        self.assertEqual(result["reward_total"],500.)

    def test_first_terminal_state_kept_but_respawn_jump_excluded(self):
        scenario = model.scenarios()[1]
        initial, rows = perfect_rows(scenario)
        rows[119].update(terminated=True,termination_reasons=["low_height"])
        rows[120]["position_world_m"] = [999.,999.,.14]
        result = model.score_trial(scenario,101,initial,rows)
        self.assertFalse(result["completed"])
        self.assertEqual(result["steps_scored"],120)
        self.assertEqual(result["termination_reasons"],["low_height"])
        self.assertTrue(result["fall"])
        self.assertEqual(result["path_position_error_m"]["max"],0.)

    def test_timeout_is_separate_and_unreached_reversal_is_censored(self):
        scenario = model.scenarios()[-3]
        initial, rows = perfect_rows(scenario)
        for row in rows:
            row["velocity_navigation_m_s"] = [.1,0.,0.]
        rows[-1]["truncated"] = True
        result = model.score_trial(scenario,101,initial,rows)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["terminated"])
        self.assertFalse(result["reversal_reached"])
        self.assertIsNone(result["reversal_delay_s"])

    def test_actual_task_termination_names_are_classified_without_invented_limit_falls(self):
        source = ROOT/"packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py"
        tree = ast.parse(source.read_text())
        fn = next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name == "_get_dones")
        declaration = next(node.value for node in ast.walk(fn) if isinstance(node,ast.Assign)
                           and any(isinstance(target,ast.Name) and target.id == "reasons" for target in node.targets))
        names = {node.value for node in declaration.keys}
        self.assertEqual(names, model.FALL_REASONS | model.MODEL_INTEGRITY_REASONS)
        scenario = model.scenarios()[1]
        initial,rows = perfect_rows(scenario,2)
        rows[-1].update(terminated=True,termination_reasons=["closure_coordinate_error"])
        result = model.score_trial(scenario,101,initial,rows)
        self.assertFalse(result["fall"])
        self.assertEqual(result["model_integrity_termination_reasons"],["closure_coordinate_error"])
        self.assertNotIn("joint_limit",names)

    def test_invalid_commands_sequence_or_nonfinite_data_rejected(self):
        scenario = model.scenarios()[1]
        initial, rows = perfect_rows(scenario,2)
        for key, value in (("command",[.1,0.,0.]),("step",7),("height_m",float("nan"))):
            modified = copy.deepcopy(rows)
            modified[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                model.score_trial(scenario,101,initial,modified)


class DriverPreflightTests(unittest.TestCase):
    def test_execute_cannot_reach_sdk_with_invalid_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"source").mkdir()
            report = root/"output/report.json"
            result = subprocess.run([sys.executable,str(HERE/"evaluate.py"),"--execute","--controller","zero",
                "--source",str(root/"source"),"--admission",str(root/"missing.json"),"--report",str(report)],
                capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,1)
            data = json.loads(report.read_text())
            self.assertIn("Source is not a physical MKII training snapshot",data["errors"][0])
            self.assertFalse(data["runtime_execution_exercised"])
            self.assertFalse(data["evaluation_complete"])

    def test_help_does_not_import_native_dependencies(self):
        result = subprocess.run([sys.executable,str(HERE/"evaluate.py"),"--help"],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("Default is CPU preflight",result.stdout)

    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)/"report.json"
            report.write_text("KEEP")
            result = subprocess.run([sys.executable,str(HERE/"evaluate.py"),"--source",tmp,"--controller","zero",
                "--admission",tmp,"--report",str(report)],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(report.read_text(),"KEEP")

    def test_prior_partial_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"output").mkdir()
            prior = root/"output/trials_seed_101.json"
            prior.write_text("KEEP")
            result = subprocess.run([sys.executable,str(HERE/"evaluate.py"),"--source",str(root/"source"),
                "--controller","zero","--admission",tmp,"--report",str(root/"output/report.json")],
                capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(prior.read_text(),"KEEP")


class RuntimeBoundaryTests(unittest.TestCase):
    def test_installed_reset_has_no_scene_update_and_observer_rejects_added_samples(self):
        for name in ("reset","_reset_idx"):
            tree = ast.parse(sdk_definition("direct_rl_env.py",name))
            calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)}
            self.assertNotIn("self.scene.update",calls)
        guard = types.SimpleNamespace(total=16,policy_samples=16)
        env = types.SimpleNamespace(reset=lambda seed: (seed,{}))
        self.assertEqual(evaluate.reset_without_physics(env,202,guard),(202,{}))
        def bad_reset(seed): guard.total += 1
        env.reset = bad_reset
        with self.assertRaisesRegex(ValueError,"unexpectedly generated physics"):
            evaluate.reset_without_physics(env,303,guard)

    def test_frozen_training_guard_rejects_extra_same_counter_substep(self):
        tree = ast.parse((ROOT/"isaaclab/train_mkii_fourbar.py").read_text())
        definition = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name == "PhysicalTrainingGuard")
        namespace = {"PHYSICS_DT_S":.00125,"DECIMATION":16,"check_training_physics":lambda windows: None}
        exec(compile(ast.Module(body=[definition],type_ignores=[]),"<frozen-guard>","exec"),namespace)
        pending = []
        metrics = types.SimpleNamespace(pending=pending,windows={},capture=lambda: pending.append(1),drain=pending.clear)
        raw = types.SimpleNamespace(_physics_handles_decimation=False,common_step_counter=0,
            cfg=types.SimpleNamespace(sim=types.SimpleNamespace(dt=.00125),decimation=16),
            _body_contact_sensors={},scene=types.SimpleNamespace(update=lambda dt: None))
        with namespace["PhysicalTrainingGuard"](raw,metrics) as guard:
            for _ in range(16): raw.scene.update(dt=.00125)
            with self.assertRaisesRegex(ValueError,"Extra training physics substep"):
                raw.scene.update(dt=.00125)
            self.assertEqual(guard.total,16)
            raw.common_step_counter += 1
            for _ in range(16): raw.scene.update(dt=.00125)
            guard.require_coverage(2)

    def test_reviewed_rsl_eval_mode_keeps_serialized_digest_and_freezes_normalization(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch-only source-bound checkpoint test; use project uv environment")
        torch.manual_seed(7)
        actor = torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2))
        critic = torch.nn.Linear(2,1)
        optimizer = torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=.001)
        algorithm = types.SimpleNamespace(actor=actor,critic=critic,optimizer=optimizer,rnd=None,learning_rate=.001)
        save = compile_definition(sdk_definition("algorithms/ppo.py","save"),"save")
        switch = compile_definition(sdk_definition("algorithms/ppo.py","eval_mode"),"eval_mode")
        infer = compile_definition(sdk_definition("runners/on_policy_runner.py","get_inference_policy"),"get_inference_policy")
        algorithm.save = lambda: save(algorithm)
        algorithm.eval_mode = lambda: switch(algorithm)
        algorithm.get_policy = lambda: actor
        runner = types.SimpleNamespace(alg=algorithm)
        tree = ast.parse((ROOT/"isaaclab/train_mkii_fourbar.py").read_text())
        functions = [node for node in tree.body if isinstance(node,ast.FunctionDef)
                     and node.name in ("finite_state_digest","algorithm_state_digest")]
        namespace = {"hashlib":hashlib,"json":json,"math":math}
        exec(compile(ast.Module(body=functions,type_ignores=[]),"<frozen-digest>","exec"),namespace)
        digest = namespace["algorithm_state_digest"]
        before = digest(runner)
        policy = infer(runner,"cpu")
        self.assertTrue(all(not module.training for module in policy.modules()))
        self.assertEqual(before,digest(runner))
        with torch.inference_mode(): policy(torch.randn(4,2))
        self.assertEqual(before,digest(runner))
        with torch.no_grad(): actor[0].weight.add_(1.)
        self.assertNotEqual(before,digest(runner))


if __name__ == "__main__":
    unittest.main()
