"""Check CPU adapter contracts, without claiming Isaac Lab native parity."""
import ast
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import torch
from locomotion.omniwalk.env import reset_rows
from locomotion.omniwalk.features import observations, proprio
from locomotion.omniwalk.motor import emitted_target, executed_action_feature

BACKEND = Path(__file__).resolve().parents[1] / "omniwalk"
ISAAC_MODULES = ("warp", "pxr", "omni", "omni.physx", "isaaclab", "isaaclab.envs", "isaaclab.sim",
                 "isaaclab.sim.utils", "isaaclab_physx", "isaaclab_physx.assets",
                 "isaaclab_physx.assets.articulation", "isaaclab_physx.sim", "isaaclab_physx.sim.spawners",
                 "isaaclab_physx.sim.spawners.materials")


def load_isaac(direct_env=object, articulation=object):
    """Import isaac.py over stand-ins for Isaac Lab, PhysX, USD and Warp, which this CPU host lacks."""
    stubs = {name: mock.MagicMock(name=name) for name in ISAAC_MODULES}
    stubs["isaaclab.envs"].DirectRLEnv = direct_env
    stubs["isaaclab_physx.assets.articulation"].Articulation = articulation
    package = importlib.import_module("locomotion.omniwalk")
    saved = {name: sys.modules.get(name) for name in (*ISAAC_MODULES, "locomotion.omniwalk.isaac")}
    bound = package.__dict__.get("isaac")
    sys.modules.update(stubs)
    sys.modules.pop("locomotion.omniwalk.isaac", None)
    try:
        return importlib.import_module("locomotion.omniwalk.isaac")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if bound is None:
            package.__dict__.pop("isaac", None)
        else:
            package.isaac = bound


def class_node(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)


class DirectEnvironmentContractTests(unittest.TestCase):
    def test_selected_rows_preserve_order_and_empty_selection(self):
        self.assertEqual(reset_rows([3, 1], 4, "cpu").tolist(), [3, 1])
        self.assertEqual(reset_rows([], 4, "cpu").tolist(), [])
        self.assertEqual(reset_rows(None, 4, "cpu").tolist(), [0, 1, 2, 3])
        for rows in [[1, 1], [-1], [4], [[1]]]:
            with self.assertRaises(ValueError):
                reset_rows(rows, 4, "cpu")

    def test_candidate_reuses_current_frames_history_and_observation_order(self):
        # Exercise the shared tensor path with distinct values in each field.
        history = torch.arange(2 * 5 * 42, dtype=torch.float32).reshape(2, 5, 42)
        commands = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        previous_action = torch.arange(36, dtype=torch.float32).reshape(2, 18)
        state = {"linear": torch.tensor([[2., -1., 3.], [5., -4., 6.]])}
        obs = observations(history, commands, previous_action, state, False)
        self.assertEqual(set(obs), {"obs", "critic"})
        self.assertEqual(obs["obs"].shape, (2, 231))
        self.assertEqual(obs["critic"].shape, (2, 234))
        self.assertTrue(torch.equal(obs["obs"][:, :210], history.flatten(1)))
        self.assertTrue(torch.equal(obs["obs"][:, 210:213], commands))
        self.assertTrue(torch.equal(obs["obs"][:, 213:], previous_action))
        self.assertTrue(torch.equal(obs["critic"][:, -3:], torch.tensor([[1., 2., 3.], [4., 5., 6.]])))
        # One frame: navigation body rates, navigation gravity, joint offsets, joint rates.
        frame_state = {"angular": torch.tensor([[1., 2., 3.]]), "gravity": torch.tensor([[4., 5., 6.]]),
                       "q": torch.full((1, 18), .5), "dq": torch.full((1, 18), 2.)}
        neutral = torch.arange(18, dtype=torch.float32) / 100
        frame = proprio(frame_state, neutral)
        self.assertEqual(frame.shape, (1, 42))
        self.assertTrue(torch.equal(frame[0, :6], torch.tensor([-2., 1., 3., -5., 4., 6.])))
        self.assertTrue(torch.equal(frame[0, 6:24], .5 - neutral))
        self.assertTrue(torch.equal(frame[0, 24:], torch.full((18,), 2.)))

    def test_stock_loop_owns_steps_and_task_owns_resets(self):
        isaac = class_node(BACKEND / "isaac.py", "IsaacEnv")
        self.assertEqual(ast.unparse(isaac.bases[0]), "DirectRLEnv")
        self.assertNotIn("step", {n.name for n in isaac.body if isinstance(n, ast.FunctionDef)})
        calls = [ast.unparse(n.func) for n in ast.walk(ast.parse((BACKEND / "isaac.py").read_text())) if isinstance(n, ast.Call)]
        self.assertNotIn("self.sim.step", calls)
        self.assertNotIn("self.scene.update", calls)
        self.assertIn("super()._reset_idx", calls)
        self.assertIn("self._reset_idx", calls)
        self.assertIn("self.robot.write_joint_position_to_sim_index", calls)
        adapter = ast.parse((BACKEND / "env.py").read_text())
        adapter_calls = [ast.unparse(n.func) for n in ast.walk(adapter) if isinstance(n, ast.Call)]
        self.assertIn("self.isaac.step", adapter_calls)
        self.assertIn("self.isaac.reset_selected", adapter_calls)

    def test_episode_clock_reuses_the_stock_buffer(self):
        def writes(name, target):
            return [n for n in ast.walk(ast.parse((BACKEND / name).read_text())) if isinstance(n, (ast.Assign, ast.AugAssign))
                    and target in {ast.unparse(t) for t in (n.targets if isinstance(n, ast.Assign) else [n.target])}]
        bindings = writes("env.py", "self.episode_steps")
        self.assertEqual(len(bindings), 1)
        self.assertEqual(ast.unparse(bindings[0].value), "isaac.episode_length_buf")
        self.assertEqual(writes("isaac.py", "self.episode_length_buf"), [])

    def test_stock_collision_filter_keeps_the_shared_ground(self):
        setup = next(n for n in class_node(BACKEND / "isaac.py", "IsaacEnv").body
                     if isinstance(n, ast.FunctionDef) and n.name == "_setup_scene")
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
        # Run the production hooks against a call recorder. This verifies
        # hook order, not PhysX contact or torque behavior.
        events = []
        class Articulation:
            def write_data_to_sim(self): events.append("submit")
            def update(self, dt): events.append("update")
        isaac = load_isaac(articulation=Articulation)
        env = isaac.IsaacEnv.__new__(isaac.IsaacEnv)
        def read_force(name):
            self.assertEqual(name, "get_dof_actuation_forces")
            events.append("read_submitted")
            return torch.tensor([[.5, 1.]])
        def capture():
            events.append("capture")
            self.assertTrue(torch.equal(env.native_pre, torch.tensor([[1., .5]])))
        env.control_active, env.capture, env.joint_index = True, object(), torch.tensor([1, 0])
        env.get, env.collect_substep = read_force, capture
        robot = isaac.CaptureArticulation()
        robot.env = env
        robot.write_data_to_sim()
        events.append("physics")
        robot.update(.0025)
        self.assertEqual(events, ["submit", "read_submitted", "physics", "update", "capture"])
        events.clear()
        env.control_active = False
        robot.write_data_to_sim()
        robot.update(.0025)
        self.assertEqual(events, ["submit", "update"])

    def test_capture_hooks_receive_the_task_environment(self):
        from locomotion.omniwalk.env import OmniEnv
        env = OmniEnv.__new__(OmniEnv)
        env.device, env.num_envs = "cpu", 1
        received = []
        def step(action):
            env.isaac.capture("state", "q")
            env.isaac.current, env.isaac.telemetry = {"q": 1}, {"terminated": "t", "truncated": "f"}
            return {"policy": "obs", "critic": "critic"}, None, None, None, None
        env.isaac = SimpleNamespace(step=step, total_controls=4)
        env.capture = lambda *args: received.append(args)
        result = env.step(torch.zeros(1, 18))
        self.assertEqual(received, [(env, "state", "q")])
        self.assertEqual(result, {"obs": "obs", "critic": "critic", "terminated": "t", "truncated": "f"})
        self.assertEqual((env.current, env.total_controls), ({"q": 1}, 4))
        env.capture = None
        env.isaac.step = lambda action: ({"policy": "obs"}, None, None, None, None)
        env.step(torch.zeros(1, 18))
        self.assertIsNone(env.isaac.capture)

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
