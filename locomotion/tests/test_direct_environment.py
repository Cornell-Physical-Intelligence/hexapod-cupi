"""Check CPU adapter contracts, without claiming Isaac Lab native parity."""
import ast
from pathlib import Path
import unittest
import torch
from locomotion.omni.env import reset_rows
from locomotion.omni.env import LocomotionEnv
from locomotion.omni.motor import emitted_target, executed_action_feature

BACKEND = Path(__file__).resolve().parents[1] / "omni"


class DirectEnvironmentContractTests(unittest.TestCase):
    def test_selected_rows_preserve_order_and_empty_selection(self):
        self.assertEqual(reset_rows([3, 1], 4, "cpu").tolist(), [3, 1])
        self.assertEqual(reset_rows([], 4, "cpu").tolist(), [])
        self.assertEqual(reset_rows(None, 4, "cpu").tolist(), [0, 1, 2, 3])
        for rows in [[1, 1], [-1], [4], [[1]]]:
            with self.assertRaises(ValueError):
                reset_rows(rows, 4, "cpu")

    def test_candidate_reuses_current_frames_history_and_observation_order(self):
        tree = ast.parse((BACKEND / "env.py").read_text())
        candidate = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LocomotionEnv")
        self.assertEqual(candidate.bases, [])
        self.assertFalse((BACKEND / "direct_env.py").exists())
        # Exercise the shared tensor path with distinct values in each field.
        env = LocomotionEnv.__new__(LocomotionEnv)
        env.neutral = torch.arange(18, dtype=torch.float32) / 100
        env.history = torch.arange(2 * 5 * 42, dtype=torch.float32).reshape(2, 5, 42)
        env.commands = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        env.previous_action = torch.arange(36, dtype=torch.float32).reshape(2, 18)
        from types import SimpleNamespace
        env.cfg = SimpleNamespace(record_motion_features=False)
        state = {"linear": torch.tensor([[2., -1., 3.], [5., -4., 6.]])}
        obs = env._observations(state)
        self.assertEqual(obs["obs"].shape, (2, 231))
        self.assertEqual(obs["critic"].shape, (2, 234))
        self.assertTrue(torch.equal(obs["obs"][:, :210], env.history.flatten(1)))
        self.assertTrue(torch.equal(obs["obs"][:, 210:213], env.commands))
        self.assertTrue(torch.equal(obs["obs"][:, 213:], env.previous_action))
        self.assertTrue(torch.equal(obs["critic"][:, -3:], torch.tensor([[1., 2., 3.], [4., 5., 6.]])))

    def test_stock_loop_owns_steps_and_task_owns_resets(self):
        tree = ast.parse((BACKEND / "native.py").read_text())
        native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeEnv")
        self.assertEqual(ast.unparse(native.bases[0]), "DirectRLEnv")
        self.assertNotIn("step", {n.name for n in native.body if isinstance(n, ast.FunctionDef)})
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertNotIn("self.sim.step", calls)
        self.assertNotIn("self.scene.update", calls)
        adapter = ast.parse((BACKEND / "env.py").read_text())
        adapter_calls = [ast.unparse(n.func) for n in ast.walk(adapter) if isinstance(n, ast.Call)]
        self.assertIn("self.native._reset_idx", adapter_calls)
        self.assertIn("self.robot.write_joint_position_to_sim_index", adapter_calls)

    def test_episode_clock_reuses_the_stock_buffer(self):
        tree = ast.parse((BACKEND / "env.py").read_text())
        bindings = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(ast.unparse(t) == "self.episode_steps" for t in n.targets)]
        self.assertEqual(len(bindings), 1)
        self.assertEqual(ast.unparse(bindings[0].value), "self.native.episode_length_buf")
        writes = [n for n in ast.walk(tree) if isinstance(n, ast.AugAssign)
                  and ast.unparse(n.target) == "self.episode_steps"]
        self.assertEqual(writes, [])

    def test_stock_collision_filter_keeps_the_shared_ground(self):
        tree = ast.parse((BACKEND / "native.py").read_text())
        native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeEnv")
        setup = next(n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == "_setup_scene")
        filters = [n for n in ast.walk(setup) if isinstance(n, ast.Call)
                   and ast.unparse(n.func) == "self.scene.filter_collisions"]
        self.assertEqual(len(filters), 1)
        self.assertEqual(ast.literal_eval(filters[0].keywords[0].value), ["/Ground"])
        ground = next(n for n in ast.walk(setup) if isinstance(n, ast.Call)
                      and ast.unparse(n.func) == "GroundPlaneCfg")
        self.assertEqual(ast.literal_eval(ground.keywords[0].value), (80., 80.))
        self.assertLess(ground.lineno, filters[0].lineno)
        material = next(n for n in ast.walk(setup) if isinstance(n, ast.Call)
                        and ast.unparse(n.func) == "RigidBodyMaterialCfg")
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in material.keywords},
                         {"static_friction": 1., "dynamic_friction": 1., "restitution": 0.,
                          "friction_combine_mode": "multiply", "restitution_combine_mode": "multiply"})
        config = ast.parse((BACKEND / "env_cfg.py").read_text())
        scenes = [n for n in ast.walk(config) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == "InteractiveSceneCfg"]
        self.assertEqual(len(scenes), 2)
        for scene in scenes:
            values = {k.arg: ast.literal_eval(k.value) for k in scene.keywords
                      if k.arg in ("filter_collisions", "replicate_physics")}
            self.assertEqual(values, {"replicate_physics": False, "filter_collisions": True})

    def test_force_readback_brackets_physics_at_articulation_hooks(self):
        # Execute the production hook methods against a call recorder. This
        # verifies hook order, not PhysX contact or torque behavior.
        tree = ast.parse((BACKEND / "native.py").read_text())
        hook = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CaptureArticulation")
        events = []
        class Articulation:
            def write_data_to_sim(self): events.append("submit")
            def update(self, dt): events.append("update")
        namespace = {"Articulation": Articulation}
        exec(compile(ast.Module(body=[hook], type_ignores=[]), str(BACKEND / "native.py"), "exec"), namespace)
        robot = namespace["CaptureArticulation"]()
        from types import SimpleNamespace
        def read_force(name):
            self.assertEqual(name, "get_dof_actuation_forces")
            events.append("read_submitted")
            return torch.tensor([[.5, 1.]])
        def capture():
            events.append("capture")
            self.assertTrue(torch.equal(robot.owner.native_pre, torch.tensor([[1., .5]])))
        robot.owner = SimpleNamespace(_control_active=True, capture=object(), get=read_force,
                                      joint_index=torch.tensor([1, 0]), _collect_substep=capture)
        robot.write_data_to_sim()
        events.append("physics")
        robot.update(.0025)
        self.assertEqual(events, ["submit", "read_submitted", "physics", "update", "capture"])
        events.clear()
        robot.owner._control_active = False
        robot.write_data_to_sim()
        robot.update(.0025)
        self.assertEqual(events, ["submit", "update"])

    def test_slew_and_action_feature_share_current_arithmetic(self):
        q = torch.zeros((2, 18))
        held = q.clone()
        neutral = torch.zeros(18)
        for _ in range(12):
            target = emitted_target(torch.ones_like(q), held, neutral, neutral - .3, neutral + .3)
            self.assertTrue(bool(((target.double() - held.double()).abs() <= .040).all()))
            self.assertTrue(torch.equal(executed_action_feature(target, neutral, .35), target / .35))
            held = target
        self.assertTrue(torch.equal(held, torch.full_like(held, .3)))


if __name__ == "__main__":
    unittest.main()
