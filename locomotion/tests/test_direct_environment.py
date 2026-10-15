"""Check CPU adapter contracts, without claiming Isaac Lab native parity."""
import ast
from pathlib import Path
import unittest
import torch
from locomotion.backends.isaac.direct_control import reset_rows
from locomotion.backends.isaac.env import LocomotionEnv
from locomotion.core.motor import emitted_target, executed_action_feature

BACKEND = Path(__file__).resolve().parents[1] / "backends" / "isaac"


class DirectEnvironmentContractTests(unittest.TestCase):
    def test_selected_rows_preserve_order_and_empty_selection(self):
        self.assertEqual(reset_rows([3, 1], 4, "cpu").tolist(), [3, 1])
        self.assertEqual(reset_rows([], 4, "cpu").tolist(), [])
        self.assertEqual(reset_rows(None, 4, "cpu").tolist(), [0, 1, 2, 3])
        for rows in [[1, 1], [-1], [4], [[1]]]:
            with self.assertRaises(ValueError):
                reset_rows(rows, 4, "cpu")

    def test_candidate_reuses_current_frames_history_and_observation_order(self):
        tree = ast.parse((BACKEND / "direct_env.py").read_text())
        candidate = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DirectLocomotionEnv")
        self.assertEqual(ast.unparse(candidate.bases[0]), "LocomotionEnv")
        methods = {n.name for n in candidate.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"_read", "_proprio", "_observations", "verify_native_recipe"}.isdisjoint(methods))
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
        tree = ast.parse((BACKEND / "direct_env.py").read_text())
        native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_DirectPhysics")
        self.assertEqual(ast.unparse(native.bases[0]), "DirectRLEnv")
        self.assertNotIn("step", {n.name for n in native.body if isinstance(n, ast.FunctionDef)})
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertNotIn("self.sim.step", calls)
        self.assertNotIn("self.scene.update", calls)
        self.assertIn("self.direct._reset_idx", calls)
        self.assertIn("self.robot.write_joint_position_to_sim_index", calls)

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
