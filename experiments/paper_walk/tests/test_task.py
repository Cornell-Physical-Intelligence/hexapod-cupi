"""Command/reward timing and selected-row isolation without a simulator."""
from types import SimpleNamespace
from pathlib import Path
import json
import hashlib
import importlib.util
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from experiments.paper_walk.env_config import JOINT_NAMES
from experiments.paper_walk.task import TaskConfig, TrainingTask, command_bank, measured_reward, articulation_reach, quiet_cost_tail, REWARD_VERSION

ROOT = Path(__file__).resolve().parents[3]
GEOMETRY = ROOT/"artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry.json"
MODEL = json.loads((ROOT/"robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json").read_text())


class NativeDouble:
    def __init__(self, n=3):
        self.num_envs, self.device = n, "cpu"
        self.cfg = SimpleNamespace(control_dt=.02, decimation=8, target_slew_rad=.04)
        self.joint_names = list(JOINT_NAMES)
        self.lower, self.upper = torch.full((18,), -1.), torch.full((18,), 1.)
        self.position = torch.zeros(n, 18)
        self.gravity_z = torch.full((n,), -1.)
        self.reference_metadata = {"root_height_m": .1}
        self.reset_height = .1
        self.model, self.geometry_path = MODEL, GEOMETRY
        self.origins = torch.zeros(n, 3)
        self.origins[:, 0] = torch.arange(n)*2.
        roots = torch.zeros(n, 7)
        roots[:, :3] = self.origins
        roots[:, 2], roots[:, 6] = .1, 1.
        self.current = {"root": roots, "linear": torch.zeros(n, 3)}
        self.held = torch.zeros(n, 18)
        self.commands = torch.zeros(n, 3)
        self.history = torch.zeros(n, 210)
        self.amp = torch.zeros(n, 61)
        self.steps = 0
        self.reset_calls = []
        self.telemetry = {}
        self.rate = torch.zeros(n, 18)
        self.last_result = None

    def state(self):
        obs = torch.cat((self.history, self.commands, self.held), -1)
        obs[:, 173] = self.gravity_z
        return {"obs": obs.clone(), "critic": torch.cat((obs, torch.zeros(self.num_envs, 3)), -1), "amp": self.amp.clone()}

    def reset(self, indices):
        self.reset_calls.append(indices.clone())
        self.history[indices] = 0
        self.amp[indices] = 0
        self.held[indices] = 0
        self.current["root"][indices, :3] = self.origins[indices]
        self.current["root"][indices, 2] = self.reset_height
        self.current["linear"][indices] = 0
        return self.state()

    def step(self, action):
        self.steps += 1
        self.history += 1
        self.amp += .01
        pose = self.current["root"].clone()
        velocity = self.commands.clone()
        velocity[:, 2] = 0
        angular = torch.zeros(self.num_envs, 3)
        angular[:, 2] = self.commands[:, 2]
        self.telemetry = {"linear_velocity_nav": velocity, "angular_velocity_body": angular,
            "root_pose_xyzw": pose, "joint_velocity_rad_s": self.rate.clone(),
            "joint_position_rad": self.position.clone(),
            "joint_target_rad": self.held.clone(), "torque_square_sum_400hz": torch.zeros(self.num_envs, 18),
            "saturation_count_400hz": torch.zeros(self.num_envs, 18, dtype=torch.long),
            "other_body_force_max_400hz": torch.zeros(self.num_envs), "command": self.commands.clone()}
        output = self.state()
        output.update(reward=torch.full((self.num_envs,), -999.), terminated=torch.zeros(self.num_envs, dtype=torch.bool),
                      truncated=torch.zeros(self.num_envs, dtype=torch.bool))
        self.last_result = {k: v.clone() for k, v in output.items()}
        return output


class TaskTests(unittest.TestCase):
    def test_exact_geometry_all_articulation_reach(self):
        bound = articulation_reach(MODEL, GEOMETRY)
        self.assertAlmostEqual(bound["mesh_radius_m"], .46988232997647794, places=12)
        self.assertGreater(bound["radius_m"], bound["mesh_radius_m"])
        self.assertEqual(set(bound["body_radius_bounds_m"]), {link["name"] for link in MODEL["links"]})

    def test_initial_and_before_control_proximity_stop_without_physics_or_reset(self):
        native = NativeDouble(2)
        native.current["root"][1, 0] = 1.
        with self.assertRaisesRegex(RuntimeError, "proximity guard"):
            TrainingTask(native)
        self.assertEqual((native.steps, len(native.reset_calls)), (0, 0))
        native = NativeDouble(2)
        task = TrainingTask(native)
        task.reset()
        native.current["root"][1, 0] = 1.1
        with self.assertRaisesRegex(RuntimeError, "before_control"):
            task.step(torch.zeros(2, 18))
        self.assertEqual((native.steps, len(native.reset_calls)), (0, 1))

    def test_swept_crossing_is_preserved_and_no_result_is_returned(self):
        native = NativeDouble(2)
        with tempfile.TemporaryDirectory() as temporary:
            task = TrainingTask(native, output_dir=temporary)
            task.reset()
            original = native.step
            def crossing(action):
                result = original(action)
                native.current["root"][:, 0] = torch.tensor([2., 0.])
                return result
            native.step = crossing
            with self.assertRaisesRegex(RuntimeError, "after_control"):
                task.step(torch.zeros(2, 18))
            receipt = json.loads((Path(temporary)/"proximity_failure.json").read_text())
            self.assertLess(receipt["sphere_gap_m"], 0.)
            self.assertFalse(receipt["contaminated_control_returned_to_learner"])
            self.assertEqual((native.steps, len(native.reset_calls), task.controls_completed), (1, 1, 0))

    def test_prospective_reset_cannot_teleport_into_another_robot(self):
        native = NativeDouble(2)
        task = TrainingTask(native)
        task.reset()
        native.current["root"][:, 0] = torch.tensor([4., 0.])
        with self.assertRaisesRegex(RuntimeError, "prospective_reset"):
            task.reset(torch.tensor([0]))
        self.assertEqual(len(native.reset_calls), 1)

    def test_measured_tracking_and_interval_metrics(self):
        native = NativeDouble(2)
        task = TrainingTask(native)
        task.reset()
        native.commands[:] = torch.tensor([[.05, 0., 0.], [0., 0., 0.]])
        task.step(torch.zeros(2, 18))
        status = task.status(reset_interval=True)
        metrics = status["cumulative_metrics"]
        self.assertEqual(metrics["environment_controls"], 2)
        self.assertEqual(metrics["signed_navigation_velocity_error_mps"], [0., 0.])
        self.assertAlmostEqual(metrics["requested_planar_speed_mps"], .025)
        self.assertAlmostEqual(metrics["achieved_planar_speed_mps"], .025)
        self.assertEqual(metrics["zero_command_rows"], 1)
        # Preserve the historical diluted aggregate and expose its corrected
        # moving-only counterpart with an explicit denominator.
        self.assertAlmostEqual(metrics["signed_command_direction_speed_mps"], .025)
        self.assertAlmostEqual(metrics["moving_signed_command_direction_speed_mps"], .05)
        self.assertEqual(metrics["command_classes"]["linear_0.05"]["environment_controls"], 1)
        self.assertIsNone(metrics["command_classes"]["yaw"]["task_reward_mean"])
        self.assertEqual(task.status()["interval_metrics"]["environment_controls"], 0)

    def test_reward_decomposition_command_denominators_and_quiet_rms(self):
        native = NativeDouble(7)
        task = TrainingTask(native)
        task.reset()
        native.commands[:] = torch.tensor([[0., 0., 0.], [0., 0., 0.], [.025, 0., 0.],
            [-.05, 0., 0.], [0., 0., .2], [.04, 0., .15], [.04, 0., -.15]])
        native.rate[0, 0] = .03
        native.rate[1] = .03
        native.rate[1, 0] = .09
        native.held[0] = .002
        native.held[1] = .006
        original = native.step
        def measured(action):
            output = original(action)
            native.telemetry["linear_velocity_nav"][3, 0] = .05  # Wrong direction.
            native.telemetry["other_body_force_max_400hz"][[1, 3]] = torch.tensor([2., 3.])
            return output
        native.step = measured
        output = task.step(torch.zeros(7, 18))
        components = task.last_components
        status = task.status(reset_interval=True)
        metrics = status["interval_metrics"]
        classes = metrics["command_classes"]
        self.assertEqual({name: entry["environment_controls"] for name, entry in classes.items()},
            {"zero": 2, "linear_0.025": 1, "linear_0.05": 1, "yaw": 1, "arc": 2, "other": 0})
        self.assertAlmostEqual(sum(c["time_fraction"] for c in classes.values()), 1.)
        self.assertAlmostEqual(metrics["task_reward_sum"], float(output["reward"].double().sum()))
        component_sum = 0.
        for key, value in components.items():
            entry = metrics["reward_components"][key]
            expected = float(value.double().sum())
            component_sum += expected
            self.assertAlmostEqual(entry["sum"], expected)
            self.assertAlmostEqual(entry["mean"], expected/7)
            self.assertAlmostEqual(entry["absolute_sum"], float(value.double().abs().sum()))
            self.assertAlmostEqual(classes["zero"]["reward_component_means"][key], float(value[:2].double().mean()))
            self.assertAlmostEqual(classes["arc"]["reward_component_mean_absolute"][key], float(value[5:].double().abs().mean()))
            self.assertAlmostEqual(sum(c["reward_component_means"][key]*c["environment_controls"]
                for c in classes.values() if c["environment_controls"]), expected)
        self.assertAlmostEqual(component_sum, metrics["task_reward_sum"], places=6)
        self.assertEqual(metrics["moving_command_rows"], 4)
        self.assertAlmostEqual(metrics["moving_signed_command_direction_speed_mps"], (.025-.05+.04+.04)/4)
        self.assertAlmostEqual(classes["linear_0.05"]["moving_signed_command_direction_speed_mps"], -.05)
        self.assertIsNone(classes["zero"]["moving_signed_command_direction_speed_mps"])
        self.assertEqual(metrics["nonfoot_event_rows"], 2)
        self.assertAlmostEqual(metrics["nonfoot_event_fraction"], 2/7)
        self.assertAlmostEqual(classes["zero"]["nonfoot_event_fraction"], .5)
        self.assertEqual(classes["linear_0.05"]["nonfoot_event_fraction"], 1.)
        expected_rms = math.sqrt((.03**2+.09**2)/2)
        self.assertAlmostEqual(metrics["zero_hold_joint_rate_rms_rad_s"][0], expected_rms)
        self.assertAlmostEqual(metrics["zero_hold_worst_joint_rate_rms_rad_s"], expected_rms)
        self.assertAlmostEqual(metrics["zero_hold_joint_rate_rms_rad_s"][1], .03/math.sqrt(2))
        # The next interval has no stops; absence is null, never an invented zero.
        native.commands[:] = torch.tensor([.05, 0., 0.])
        task.step(torch.zeros(7, 18))
        later = task.status()
        self.assertEqual(later["interval_metrics"]["environment_controls"], 7)
        self.assertIsNone(later["interval_metrics"]["zero_hold_joint_rate_rms_rad_s"])
        self.assertEqual(later["cumulative_metrics"]["command_classes"]["zero"]["environment_controls"], 2)
        self.assertEqual(later["cumulative_metrics"]["environment_controls"], 14)
        self.assertAlmostEqual(later["cumulative_metrics"]["zero_hold_worst_joint_rate_rms_rad_s"], expected_rms)
        json.dumps(later, allow_nan=False)

    def test_reporting_absolute_values_do_not_cancel_and_reject_nonfinite(self):
        native = NativeDouble(2)
        task = TrainingTask(native)
        task.reset()
        native.commands.zero_()
        output = native.step(torch.zeros(2, 18))
        values = {"diagnostic_fixture": torch.tensor([3., -2.])}
        task._metrics(native.commands, output, values["diagnostic_fixture"], values)
        metrics = task.status()["interval_metrics"]
        self.assertEqual(metrics["reward_components"]["diagnostic_fixture"]["sum"], 1.)
        self.assertEqual(metrics["reward_components"]["diagnostic_fixture"]["absolute_sum"], 5.)
        self.assertEqual(metrics["command_classes"]["zero"]["reward_component_mean_absolute"]["diagnostic_fixture"], 2.5)
        before = json.dumps(task.status(), sort_keys=True, allow_nan=False)
        for bad in (float("nan"), float("inf"), -float("inf")):
            broken = {"diagnostic_fixture": torch.tensor([bad, 1.])}
            with self.assertRaisesRegex(FloatingPointError, "reporting metrics"):
                task._metrics(native.commands, output, torch.ones(2), broken)
            self.assertEqual(json.dumps(task.status(), sort_keys=True, allow_nan=False), before)

    def test_old_command_reward_and_next_command_observation(self):
        native = NativeDouble(2)
        task = TrainingTask(native)
        task.reset()
        native.commands[:] = torch.tensor([[.05, 0., 0.], [0., .05, 0.]])
        task.remaining_controls[:] = torch.tensor([1, 7])
        def next_commands(indices):
            native.commands[indices] = torch.tensor([0., 0., .2])
            task.remaining_controls[indices] = 100
        task._resample = next_commands
        output = task.step(torch.zeros(2, 18))
        self.assertEqual(native.steps, 1)
        self.assertEqual(len(native.reset_calls), 1)
        torch.testing.assert_close(output["reward"], torch.full((2,), 1.3))
        torch.testing.assert_close(task.last_held_command, torch.tensor([[.05, 0., 0.], [0., .05, 0.]]))
        torch.testing.assert_close(output["obs"][:, 210:213], torch.tensor([[0., 0., .2], [0., .05, 0.]]))
        torch.testing.assert_close(output["critic"][:, :231], output["obs"], rtol=0, atol=0)
        torch.testing.assert_close(output["obs"][:, :210], native.last_result["obs"][:, :210], rtol=0, atol=0)
        torch.testing.assert_close(output["amp"], native.last_result["amp"], rtol=0, atol=0)

    def test_selected_reset_does_not_touch_another_robot(self):
        native = NativeDouble(3)
        task = TrainingTask(native)
        initial = task.reset()
        terminal = task.step(torch.zeros(3, 18))
        commands = native.commands.clone()
        clocks = task.remaining_controls.clone()
        next_state = task.reset(torch.tensor([1]))
        for key in ("obs", "critic", "amp"):
            torch.testing.assert_close(next_state[key][[0, 2]], terminal[key][[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(native.commands[[0, 2]], commands[[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(task.remaining_controls[[0, 2]], clocks[[0, 2]], rtol=0, atol=0)
        self.assertEqual(native.steps, 1)
        torch.testing.assert_close(next_state["obs"][1, :210], initial["obs"][1, :210], rtol=0, atol=0)

    def test_termination_reason_boundaries_overlap_and_timeout_are_distinct(self):
        native = NativeDouble(8)
        task = TrainingTask(native)
        task.reset()
        height = torch.tensor(.045)
        tilt = torch.tensor(-math.cos(.85))
        upper, lower = native.upper[0]+2e-6, native.lower[0]-2e-6
        native.current["root"][0, 2] = height
        native.current["root"][[1, 7], 2] = torch.nextafter(height, torch.tensor(-torch.inf))
        native.gravity_z[2] = tilt
        native.gravity_z[[3, 7]] = torch.nextafter(tilt, torch.tensor(torch.inf))
        native.position[4, 0] = upper
        native.position[5, 0] = torch.nextafter(upper, torch.tensor(torch.inf))
        native.position[6, 0] = lower
        native.position[7, 0] = torch.nextafter(lower, torch.tensor(-torch.inf))
        original = native.step
        def terminal_rows(action):
            result = original(action)
            result["terminated"][[1, 3, 5, 7]] = True
            result["truncated"][[0, 7]] = True
            return result
        native.step = terminal_rows
        task.step(torch.zeros(8, 18))
        metrics = task.status()["interval_metrics"]
        reasons = metrics["termination_reasons"]
        self.assertEqual(metrics["terminations"], 4)
        self.assertEqual(metrics["truncations"], 2)
        self.assertEqual(reasons["rows"], {"height":2,"tilt":2,"joint_limit":2,"timeout":2})
        self.assertEqual(reasons["multiple_physical_reason_rows"], 1)
        self.assertEqual(reasons["termination_flag_mismatch_rows"], 0)
        self.assertEqual(reasons["unattributed_termination_rows"], 0)

    def test_actual_target_slew_excludes_first_and_selected_reset_rows(self):
        native = NativeDouble(2)
        task = TrainingTask(native)
        task.reset()
        native.held[:] = .04
        task.step(torch.zeros(2, 18))
        first = task.status()["interval_metrics"]["actual_target_slew"]
        self.assertEqual((first["environment_controls"], first["excluded_first_or_reset_rows"]), (0, 2))
        self.assertTrue(all(row["at_limit_fraction"] is None for row in first["by_joint"].values()))
        native.held[0, 0] = .08
        native.held[1, 1] = 0.
        task.step(torch.zeros(2, 18))
        second = task.status(reset_interval=True)["interval_metrics"]["actual_target_slew"]
        self.assertEqual(second["environment_controls"], 2)
        for joint in JOINT_NAMES[:2]:
            self.assertEqual(second["by_joint"][joint]["at_limit_rows"], 1)
            self.assertEqual(second["by_joint"][joint]["at_limit_fraction"], .5)
        task.reset(torch.tensor([1]))
        native.held[0, 0] = .12
        native.held[1, 1] = .04
        task.step(torch.zeros(2, 18))
        third = task.status(reset_interval=True)
        slew = third["interval_metrics"]["actual_target_slew"]
        self.assertEqual((slew["environment_controls"], slew["excluded_first_or_reset_rows"]), (1, 1))
        self.assertEqual(slew["by_joint"][JOINT_NAMES[0]]["at_limit_rows"], 1)
        self.assertEqual(slew["by_joint"][JOINT_NAMES[1]]["at_limit_rows"], 0)
        self.assertEqual(third["cumulative_metrics"]["actual_target_slew"]["environment_controls"], 3)
        native.held[0, 0] += .04-2e-6-1e-7
        native.held[1, 1] += .04-2e-6+1e-7
        native.held[1, 2] += .04+2e-6+1e-7
        task.step(torch.zeros(2, 18))
        last = task.status()["interval_metrics"]["actual_target_slew"]["by_joint"]
        self.assertEqual(last[JOINT_NAMES[0]]["at_limit_rows"], 0)
        self.assertEqual(last[JOINT_NAMES[1]]["at_limit_rows"], 1)
        self.assertEqual(last[JOINT_NAMES[2]]["at_limit_rows"], 0)
        self.assertEqual(last[JOINT_NAMES[2]]["beyond_limit_rows"], 1)

    def test_reporting_preserves_frozen014_outputs_commands_and_rng_across_resets(self):
        path = ROOT/"artifacts/restart_2026-09-14/paper_walk_execution_001/source_014/task.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            "55a9517b5f7a8013bc54ba36fcdf0e28e9f306e89693683866a8425c9ceaf4e4")
        name = "paper_task_reporting_equivalence_fixture"
        spec = importlib.util.spec_from_file_location(name, path)
        frozen = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name:frozen}):
            spec.loader.exec_module(frozen)
            old, new = frozen.TrainingTask(NativeDouble(2)), TrainingTask(NativeDouble(2))
            for task in (old, new):task.reset()
            for step in range(5):
                for task in (old, new):
                    task.remaining_controls[:] = 1
                    task.env.held += .01
                    task.env.rate[:] = .02
                a, b = old.step(torch.zeros(2, 18)), new.step(torch.zeros(2, 18))
                for key in a:
                    self.assertEqual(a[key].numpy().tobytes(), b[key].numpy().tobytes(), key)
                self.assertTrue(torch.equal(old.rng.get_state(), new.rng.get_state()))
                self.assertTrue(torch.equal(old.previous_target, new.previous_target))
                self.assertEqual((old.command_draws,old.zero_command_draws), (new.command_draws,new.zero_command_draws))
                if step in (1,3):
                    a, b = old.reset(torch.tensor([1])), new.reset(torch.tensor([1]))
                    for key in a:self.assertEqual(a[key].numpy().tobytes(), b[key].numpy().tobytes(), key)

    def test_quiet_joint_rate_penalty_uses_measured_velocity(self):
        native = NativeDouble(2)
        native.commands[:] = torch.tensor([[0., 0., 0.], [.05, 0., 0.]])
        native.step(torch.zeros(2, 18))
        config = TaskConfig()
        terminated = torch.zeros(2, dtype=torch.bool)
        base, _ = measured_reward(native.telemetry, native.commands, native.held, terminated, config, .1)
        native.telemetry["joint_velocity_rad_s"][:] = .03
        changed, components = measured_reward(native.telemetry, native.commands, native.held, terminated, config, .1)
        torch.testing.assert_close(changed-base, torch.tensor([-.05, 0.]), atol=1e-7, rtol=0)
        self.assertEqual(float(components["quiet_joint_rate"][1]), 0.)

    def test_quiet_tail_exact_threshold_and_finite_monotone_gradient(self):
        u=torch.tensor([0.,.0625,.25,.999,1.,2.,10.,1e10,1e30],dtype=torch.float64,requires_grad=True)
        value=quiet_cost_tail(u)
        torch.testing.assert_close(value[:5],u[:5],rtol=0,atol=0)
        self.assertTrue(bool((torch.diff(value)>0).all()))
        value.sum().backward()
        self.assertTrue(bool(torch.isfinite(value).all() and torch.isfinite(u.grad).all()))
        torch.testing.assert_close(u.grad[:5],torch.ones(5,dtype=torch.float64),rtol=0,atol=0)
        self.assertTrue(bool((u.grad[5:]>0).all()))
        # Taking the tail after the mean preserves the original ranking and
        # equality of concentrated versus distributed joint mean-square errors.
        distributed=torch.tensor([[1.,1.],[0.,2.]],dtype=torch.float64).mean(-1)
        torch.testing.assert_close(quiet_cost_tail(distributed),torch.ones(2,dtype=torch.float64),rtol=0,atol=0)

    def test_quiet_costs_preserve_original_below_scale_and_change_only_tails(self):
        native=NativeDouble(4)
        native.commands[-1]=torch.tensor([.05,0.,0.])
        native.step(torch.zeros(4,18))
        telemetry=native.telemetry
        telemetry["joint_velocity_rad_s"][:]=torch.tensor([.015,.03,.06,1e8])[:,None]
        telemetry["joint_target_rad"][:]=torch.tensor([.001,.002,.004,1e8])[:,None]
        telemetry["linear_velocity_nav"][:,2]=.2
        telemetry["angular_velocity_body"][:]=torch.tensor([.1,.2,.3])
        telemetry["root_pose_xyzw"][:,2]=.12
        telemetry["torque_square_sum_400hz"][:]=1.28
        telemetry["other_body_force_max_400hz"][:]=2.
        terminated=torch.ones(4,dtype=torch.bool)
        args=(telemetry,native.commands,torch.zeros(4,18),terminated,TaskConfig(),.1)
        new_reward,new=measured_reward(*args)
        with patch("experiments.paper_walk.task.quiet_cost_tail",side_effect=lambda u:u):
            old_reward,old=measured_reward(*args)
        for key in new:
            if key in ("quiet_joint_rate","quiet_target_motion"):
                torch.testing.assert_close(new[key][:2],old[key][:2],rtol=0,atol=0)
                self.assertGreater(float(new[key][2]),float(old[key][2]))
                self.assertEqual(float(new[key][3]),0.)
            else:
                torch.testing.assert_close(new[key],old[key],rtol=0,atol=0)
        self.assertTrue(bool(torch.isfinite(new_reward).all()))
        self.assertAlmostEqual(float(new["quiet_joint_rate"][2]),-.05*(1+math.log(4)),places=7)
        self.assertAlmostEqual(float(new["quiet_target_motion"][2]),-.02*(1+math.log(4)),places=7)
        torch.testing.assert_close(new_reward[:2],old_reward[:2],rtol=0,atol=0)
        declaration=TrainingTask(NativeDouble(1)).declaration()
        self.assertEqual(declaration["reward_version"],REWARD_VERSION)
        self.assertEqual(declaration["quiet_cost_tail"]["original_quadratic_preserved_u_interval"],[0.,1.])
        self.assertFalse(declaration["quiet_cost_tail"]["other_reward_terms_changed"])

    def test_tracking_does_not_reward_stationary_robot_for_commanded_walking(self):
        native = NativeDouble(1)
        native.commands[:] = torch.tensor([.05, 0., 0.])
        native.step(torch.zeros(1, 18))
        native.telemetry["linear_velocity_nav"].zero_()
        _, components = measured_reward(native.telemetry, native.commands, native.held, torch.zeros(1, dtype=torch.bool), TaskConfig(), .1)
        self.assertAlmostEqual(float(components["linear_tracking"][0]), math.exp(-.05**2/.0009), places=7)
        self.assertLess(float(components["linear_tracking"][0]), .07)

    def test_command_bank_quota_bounds_and_repeatability(self):
        a, b = TrainingTask(NativeDouble(7)), TrainingTask(NativeDouble(7))
        for _ in range(37):
            for task in (a, b):
                task._resample(torch.arange(7))
                self.assertTrue(bool(((task.remaining_controls >= 100) & (task.remaining_controls <= 250)).all()))
                self.assertGreaterEqual(task.zero_command_draws/task.command_draws, .15)
            torch.testing.assert_close(a.commands, b.commands, rtol=0, atol=0)
            torch.testing.assert_close(a.remaining_controls, b.remaining_controls, rtol=0, atol=0)
        bank = torch.tensor(command_bank(TaskConfig()))
        self.assertEqual(len(bank), 20)
        for bearing in range(8):
            wanted = torch.tensor([.05*math.cos(bearing*math.pi/4), .05*math.sin(bearing*math.pi/4), 0.])
            self.assertTrue(any(torch.allclose(row, wanted) for row in bank))
        low = torch.tensor(command_bank(TaskConfig(maximum_speed_mps=.025)))
        self.assertTrue(bool((torch.linalg.vector_norm(low[:, :2], dim=-1) <= .02500001).all()))
        self.assertFalse(a.declaration()["stage2_gates_changed"])

    def test_measured_command_mismatch_is_rejected(self):
        native = NativeDouble(1)
        native.step(torch.zeros(1, 18))
        with self.assertRaisesRegex(ValueError, "completed hold"):
            measured_reward(native.telemetry, torch.tensor([[.05, 0., 0.]]), native.held,
                            torch.zeros(1, dtype=torch.bool), TaskConfig(), .1)


if __name__ == "__main__":
    unittest.main()
