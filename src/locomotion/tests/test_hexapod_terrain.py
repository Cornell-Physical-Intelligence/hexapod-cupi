"""Check the terrain v2.2 hexapod task: registration, defaults, reset geometry, logging and the CPU curriculum
arithmetic."""

import ast
import contextlib
import importlib.util
import io
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/locomotion/hexapod/terrain"
HEXAPOD = ROOT / "src/locomotion/hexapod"
APPROVED_USD = ROOT / "robot/hexapod_mkii_updated_v1/usd/robot.usda"
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")


def load(name):
    spec = importlib.util.spec_from_file_location(f"hexapod_terrain_{name}", PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_curriculum():
    return load("curriculum")


def assignments(name, class_name=None):
    body = ast.parse((PACKAGE / name).read_text()).body
    if class_name is not None:
        body = next(
            node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    return {
        item.targets[0].id if isinstance(item, ast.Assign) else item.target.id: item.value
        for item in body
        if isinstance(item, (ast.Assign, ast.AnnAssign))
        and isinstance(getattr(item, "target", None) or item.targets[0], ast.Name)
    }


class TerrainTaskSourceTests(unittest.TestCase):
    def test_registration_adds_one_id_beside_the_original_tasks(self):
        calls = []
        tree = ast.parse((PACKAGE / "__init__.py").read_text())
        tree.body = [
            node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        scope = {
            "__name__": "locomotion.hexapod.terrain",
            "gym": SimpleNamespace(register=lambda **kwargs: calls.append(kwargs)),
        }
        exec(compile(tree, str(PACKAGE / "__init__.py"), "exec"), scope)
        self.assertEqual([entry["id"] for entry in calls], ["Hexapod-Rough-v22-Direct-v0"])
        entry = calls[0]
        self.assertEqual(entry["entry_point"], "locomotion.hexapod.terrain.env:HexapodTerrainEnv")
        self.assertEqual(
            entry["kwargs"]["env_cfg_entry_point"],
            "locomotion.hexapod.terrain.env_cfg:HexapodTerrainEnvCfg",
        )
        self.assertEqual(
            entry["kwargs"]["rsl_rl_cfg_entry_point"],
            "locomotion.hexapod.terrain.agents:HexapodTerrainPPORunnerCfg",
        )
        self.assertTrue(entry["disable_env_checker"])
        self.assertIn("register", scope)

    def test_terrain_constants_are_the_v22_values(self):
        values = assignments("terrains_cfg.py")
        for name, value in {
            "STEP_CAP": 0.052,
            "STAIR_TREAD": 0.25,
            "STAIR_PLATFORM": 1.0,
            "BOX_PLATFORM": 0.8,
            "SLOPE_PLATFORM": 0.8,
            "BOX_GRID_WIDTH": 0.12,
            "TILE_SIZE": 2.0,
            "NUM_LEVELS": 10,
            "HF_BORDER": 0.04,
            "COURSE_BORDER": 9.5,
        }.items():
            self.assertEqual(ast.literal_eval(values[name]), value, name)

    def test_task_defaults_are_the_trained_settings(self):
        values = assignments("env_cfg.py", "HexapodTerrainEnvCfg")
        for name, value in {
            "gait_clock_enabled": True,
            "gait_period_s": 0.4,
            "command_forward_only": True,
            "command_lin_vel_min": 0.1,
            "command_lin_vel_max": 0.35,
            "command_yaw_rate_max": 0.1,
            "feet_air_time_threshold": 0.0,
            "flat_orientation_reward_scale": 0.0,
            "swing_clearance_reward_scale": 0.05,
            "swing_clearance_target": 0.06,
            "success_speed_tol_ratio": 0.25,
            "terrain_curriculum": True,
            "curriculum_move_down_speed_ratio": 0.5,
            "command_curriculum": False,
        }.items():
            self.assertEqual(ast.literal_eval(values[name]), value, name)

    def test_task_imports_no_package_outside_hexapod(self):
        for path in PACKAGE.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    with self.subTest(file=path.name, module=node.module):
                        self.assertFalse(node.module.startswith("locomotion."))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith("locomotion."))


class TerrainCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.c = load_curriculum()

    def test_success_needs_a_timeout_without_fall_and_speed_within_tolerance(self):
        steps = torch.tensor([100.0, 100.0, 100.0, 100.0, 0.0])
        cmd = torch.tensor([0.28, 0.28, 0.28, 0.28, 0.28])
        v_sum = torch.tensor([0.28, 0.20, 0.28, 0.10, 0.0]) * steps
        fell = torch.tensor([False, False, True, False, False])
        timed_out = torch.tensor([True, True, True, True, True])
        success, slow, err, mean_v = self.c.episode_outcome(
            v_sum, steps, cmd, fell, timed_out, 0.35
        )
        # 0.20 m/s is 0.08 below 0.28, outside 25% (0.07); 0.10 m/s is below half the command
        torch.testing.assert_close(success, torch.tensor([True, False, False, False, False]))
        torch.testing.assert_close(slow, torch.tensor([False, False, False, True, True]))
        torch.testing.assert_close(mean_v, torch.tensor([0.28, 0.20, 0.28, 0.10, 0.0]))
        torch.testing.assert_close(err, torch.tensor([0.0, 0.08, 0.0, 0.18, 0.28]))

    def test_small_commands_use_the_tolerance_floor(self):
        steps = torch.tensor([10.0])
        success, *_ = self.c.episode_outcome(
            torch.tensor([0.02]) * steps,
            steps,
            torch.tensor([0.0]),
            torch.tensor([False]),
            torch.tensor([True]),
            0.35,
        )
        # floor 0.1 * 0.35 = 0.035, tolerance 25% of it = 0.00875 < 0.02
        self.assertFalse(bool(success))

    def test_terrain_moves_up_on_success_and_down_on_fall_or_slow(self):
        success = torch.tensor([True, False, False, False, False])
        fell = torch.tensor([False, True, False, False, True])
        slow = torch.tensor([False, False, True, False, False])
        ran = torch.tensor([True, True, True, True, False])
        up, down = self.c.terrain_moves(success, fell, slow, ran)
        torch.testing.assert_close(up, torch.tensor([True, False, False, False, False]))
        torch.testing.assert_close(down, torch.tensor([False, True, True, False, False]))

    def test_swing_clearance_counts_lifted_swing_feet_up_to_the_target(self):
        toe_z = torch.tensor(
            [[0.0, 0.0, 0.0, 0.03, 0.10, 0.0], [0.02, 0.05, 0.08, 0.02, 0.05, 0.08]]
        )
        in_contact = torch.tensor([[True, True, True, False, False, True], [False] * 6])
        value = self.c.swing_clearance(toe_z, in_contact, 0.06)
        # env 0: stance mean 0, swing feet at 0.03 (half) and 0.10 (capped at 1); env 1: lowest toe 0.02 is the base
        torch.testing.assert_close(value, torch.tensor([1.5, (0.0 + 0.5 + 1.0) * 2]))

    def test_command_curriculum_rises_after_a_successful_top_bucket_window(self):
        cc = self.c.CommandCurriculum(
            start=0.15, step=0.05, cap=0.25, success=0.8, window_episodes=4
        )
        ran = torch.ones(4, dtype=torch.bool)
        top = torch.full((4,), 0.14)
        low = torch.full((4,), 0.05)
        self.assertEqual(cc.update(low, torch.ones(4, dtype=torch.bool), ran), 0.15)
        self.assertEqual(cc.update(top, torch.tensor([True, True, True, False]), ran), 0.15)
        self.assertAlmostEqual(cc.update(top, torch.ones(4, dtype=torch.bool), ran), 0.20)
        self.assertAlmostEqual(
            cc.update(torch.full((4,), 0.19), torch.ones(4, dtype=torch.bool), ran), 0.25
        )
        self.assertAlmostEqual(
            cc.update(torch.full((4,), 0.24), torch.ones(4, dtype=torch.bool), ran), 0.25
        )


def literal(name, attr, class_name=None):
    return ast.literal_eval(assignments(name, class_name)[attr])


def default_joint_angles():
    tree = ast.parse((HEXAPOD / "hexapod_robot_cfg.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == ".*_femur_pitch" for k in node.keys
        ):
            return {
                k.value.removeprefix(".*_"): ast.literal_eval(v)
                for k, v in zip(node.keys, node.values)
            }
    raise AssertionError("default joint angles not found")


def toe_tips(angles):
    """Toe tips in the root frame from the approved USD's joint frames (revolute about local +Z)."""
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(APPROVED_USD))

    def frame(pos, rot):
        quat = Gf.Quatd(rot.GetReal(), Gf.Vec3d(rot.GetImaginary())).GetNormalized()
        m = np.eye(4)
        m[:3, :3] = np.array(Gf.Matrix3d(quat)).T
        m[:3, 3] = pos
        return m

    def about_z(angle):
        m = np.eye(4)
        m[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        return m

    tips = []
    for leg in LEGS:
        m = np.eye(4)
        for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch"):
            api = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"/Robot/joints/{leg}_{joint}"))
            assert api.GetAxisAttr().Get() == "Z"
            a = frame(np.array(api.GetLocalPos0Attr().Get()), api.GetLocalRot0Attr().Get())
            b = frame(np.array(api.GetLocalPos1Attr().Get()), api.GetLocalRot1Attr().Get())
            m = m @ a @ about_z(angles[joint]) @ np.linalg.inv(b)
        tips.append((m @ np.array([0.130, 0.0, 0.0, 1.0]))[:3])
    return np.array(tips)


def pinned_sloped_terrain(difficulty, inverted, platform, size=2.0, hs=0.02, vs=0.001, border=0.04):
    """Heights [m] (with the tile border) of Isaac Lab ffff603e ``pyramid_sloped_terrain`` under
    ``height_field_to_mesh``, slope range (0, 0.27), and the 2 m-window origin height the wrapper returns."""
    width = int(size / hs) + 1
    border_px = int(border / hs) + 1
    inner = int((width - 2 * border_px) * hs / hs)
    sub_size = (width - 2 * border_px) * hs
    slope = (-1 if inverted else 1) * 0.27 * difficulty
    height_max = int(slope * sub_size / 2 / vs)
    center = int(inner / 2)
    xx = ((center - np.abs(center - np.arange(inner))) / center).reshape(-1, 1)
    hf = height_max * xx * xx.reshape(1, -1)
    half = int(platform / hs / 2)
    z_pf = hf[inner // 2 - half, inner // 2 - half]
    hf = np.rint(np.clip(hf, min(0, z_pf), max(0, z_pf)))
    heights = np.zeros((width, width))
    heights[border_px:-border_px, border_px:-border_px] = hf
    lo, hi = int((size * 0.5 - 1) / hs), int((size * 0.5 + 1) / hs)
    return heights * vs, float(heights[lo:hi, lo:hi].max() * vs)


class TerrainResetGeometryTests(unittest.TestCase):
    def setUp(self):
        self.g = load("geometry")

    def test_toe_footprint_constants_match_the_approved_robot(self):
        angles = default_joint_angles()
        self.assertEqual(angles, {"coxa_yaw": 0.0, "femur_pitch": -0.3, "tibia_pitch": 0.4})
        tips = toe_tips(angles)
        extent = np.ptp(tips[:, :2], axis=0)
        np.testing.assert_allclose(extent, self.g.TOE_FOOTPRINT, atol=1e-3)
        self.assertAlmostEqual(
            np.linalg.norm(tips[:, :2], axis=1).max(), self.g.TOE_HALF_DIAGONAL, delta=1e-3
        )
        np.testing.assert_allclose(-tips[:, 2], self.g.TOE_DROP, atol=1e-3)
        np.testing.assert_allclose(
            np.abs(tips[:, :2]).max(axis=0) * 2, extent, atol=2e-3
        )  # centred on the root

    def test_root_height_is_the_robot_reset_height(self):
        tree = ast.parse((HEXAPOD / "hexapod_robot_cfg.py").read_text())
        pos = next(
            ast.literal_eval(k.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for k in node.keywords
            if k.arg == "pos"
        )
        self.assertEqual(pos[2], self.g.ROOT_HEIGHT)
        self.assertGreater(self.g.ROOT_HEIGHT - self.g.TOE_DROP, 0.0)

    def test_pinned_stair_generator_shrinks_the_requested_platform(self):
        # the v2.1 request: 3 risers, a 0.42 m platform, inside the 0.61 m footprint
        self.assertEqual(self.g.mesh_stairs_steps(2.0, 0.04, 0.8, 0.25), 3)
        self.assertAlmostEqual(self.g.mesh_stairs_platform(2.0, 0.04, 0.8, 0.25), 0.42)
        self.assertAlmostEqual(self.g.mesh_stairs_platform(2.0, 0.04, 1.0, 0.25), 0.92)
        self.assertEqual(self.g.mesh_stairs_steps(2.0, 0.04, 1.0, 0.25), 2)

    def test_every_spawn_platform_holds_the_toe_footprint(self):
        hx, hy = self.g.spawn_half_extent()
        need = 2 * max(hx, hy)
        self.assertAlmostEqual(need, 0.61 + 2 * self.g.SPAWN_MARGIN, delta=1e-3)
        stairs = self.g.mesh_stairs_platform(
            literal("terrains_cfg.py", "TILE_SIZE"),
            literal("terrains_cfg.py", "HF_BORDER"),
            literal("terrains_cfg.py", "STAIR_PLATFORM"),
            literal("terrains_cfg.py", "STAIR_TREAD"),
        )
        for name, width in {
            "stairs and pits": stairs,
            "boxes": literal("terrains_cfg.py", "BOX_PLATFORM"),
            "slopes": literal("terrains_cfg.py", "SLOPE_PLATFORM"),
        }.items():
            with self.subTest(terrain=name):
                self.assertGreaterEqual(width, need)

    def test_slope_origin_comes_from_the_spawn_area_at_every_level(self):
        hs = 0.02
        hx, hy = self.g.spawn_half_extent()
        platform = literal("terrains_cfg.py", "SLOPE_PLATFORM")
        for level in range(10):
            difficulty = (level + 0.95) / 10
            for inverted in (False, True):
                with self.subTest(level=level, inverted=inverted):
                    heights, window_origin = pinned_sloped_terrain(difficulty, inverted, platform)
                    i, j = np.meshgrid(
                        np.arange(heights.shape[0]), np.arange(heights.shape[1]), indexing="ij"
                    )
                    vertices = np.c_[i.ravel() * hs, j.ravel() * hs, heights.ravel()]
                    center = (1.0, 1.0)
                    origin = self.g.patch_max_height(vertices, center, (hx + hs, hy + hs))
                    near = (np.abs(vertices[:, 0] - 1.0) <= hx + hs) & (
                        np.abs(vertices[:, 1] - 1.0) <= hy + hs
                    )
                    # the spawn area is level with the origin: no toe in the ground, none above a drop
                    np.testing.assert_allclose(vertices[near, 2], origin, atol=1e-9)
                    if inverted:
                        self.assertAlmostEqual(origin, heights.min(), places=9)
                        # Isaac Lab's 2 m window reaches the zero-height border
                        self.assertEqual(window_origin, 0.0)
                    else:
                        self.assertAlmostEqual(origin, window_origin, places=9)
        _, v21 = pinned_sloped_terrain(0.95, True, 0.5)
        self.assertEqual(v21, 0.0)
        floor = pinned_sloped_terrain(0.95, True, 0.5)[0].min()
        self.assertAlmostEqual(floor, -0.135, delta=0.002)  # the reviewed 0.135 m spawn drop

    def test_course_border_covers_a_successful_episode_from_an_edge_tile(self):
        max_command = max(
            literal("env_cfg.py", "command_lin_vel_max", "HexapodTerrainEnvCfg"),
            literal("env_cfg.py", "command_curriculum_max", "HexapodTerrainEnvCfg"),
        )
        episode_s = ast.literal_eval(
            next(
                node.value
                for cls in ast.parse((HEXAPOD / "hexapod_env_cfg.py").read_text()).body
                if isinstance(cls, ast.ClassDef) and cls.name == "HexapodFlatEnvCfg"
                for node in cls.body
                if isinstance(node, ast.Assign) and node.targets[0].id == "episode_length_s"
            )
        )
        tol = literal("env_cfg.py", "success_speed_tol_ratio", "HexapodTerrainEnvCfg")
        need = self.g.required_course_border(
            max_command, episode_s, tol, literal("terrains_cfg.py", "TILE_SIZE")
        )
        self.assertAlmostEqual(need, 1.25 * 0.4 * 20.0 + self.g.TOE_HALF_DIAGONAL - 1.0)
        self.assertGreaterEqual(literal("terrains_cfg.py", "COURSE_BORDER"), need)

    def test_course_bounds_keep_the_toes_on_the_mesh(self):
        origins = np.array([[[x, y] for y in (-1.0, 1.0)] for x in (-3.0, -1.0, 1.0, 3.0)])
        lo, hi = self.g.course_bounds(origins, 2.0, 9.5)
        np.testing.assert_allclose(
            lo, [-4.0 - 9.5 + self.g.TOE_HALF_DIAGONAL, -2.0 - 9.5 + self.g.TOE_HALF_DIAGONAL]
        )
        np.testing.assert_allclose(hi, -lo)


class FakeWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, key, value, step):
        self.scalars[key] = float(value)


class TerrainSuccessLoggingTests(unittest.TestCase):
    """Feed the task's log entries through the pinned RSL-RL logger (rsl_rl 5.0.1)."""

    def setUp(self):
        self.c = load_curriculum()
        self.keys = [
            self.c.SUCCESS_KEY,
            self.c.SPEED_ERR_KEY,
            self.c.EPISODES_KEY,
            self.c.SUCCESSES_KEY,
        ]
        self.keys += [f"{self.c.SUCCESS_KEY}/{name}" for name in ("stairs", "boxes")]

    def logged(self, resets, num_envs=10):
        """``resets``: per step, None (no reset) or (success, ran, type_ids). Mirrors HexapodTerrainEnv.step."""
        from rsl_rl.utils.logger import Logger

        logger = Logger(
            None,
            {"algorithm": {"rnd_cfg": None}, "num_steps_per_env": len(resets)},
            {},
            num_envs,
            False,
            1,
            0,
            "cpu",
        )
        logger.writer, logger.logger_type = FakeWriter(), "tensorboard"
        log = None
        for reset in resets:
            if log is not None:
                log = self.c.cleared_episode_log(log, self.keys)
            if reset is not None:
                success, ran, types = (torch.tensor(v) for v in reset)
                log = {
                    "Episode_Reward/termination": 0.0
                }  # HexapodEnv rebinds the dict at every reset
                log.update(
                    self.c.episode_log(
                        success, ran, torch.zeros(len(ran)), types, ["stairs", "boxes"]
                    )
                )
            logger.process_env_step(torch.zeros(num_envs), torch.zeros(num_envs), {"log": log})
        with contextlib.redirect_stdout(io.StringIO()):
            logger.log(0, 0, 1, 1.0, 1.0, {}, 1e-3, torch.ones(1), None)
        return logger.writer.scalars

    def test_one_failure_then_nine_successes_logs_nine_tenths(self):
        startup = ([False] * 10, [False] * 10, [0] * 10)  # forced full reset, no episode ran
        one_failure = ([False] + [False] * 9, [True] + [False] * 9, [0] * 10)
        nine_successes = ([False] + [True] * 9, [False] + [True] * 9, [0] + [1] * 9)
        scalars = self.logged([startup, one_failure, None, nine_successes, None, None])
        self.assertAlmostEqual(scalars[self.c.SUCCESS_KEY], 0.9, places=6)
        self.assertAlmostEqual(
            scalars[self.c.SUCCESSES_KEY] / scalars[self.c.EPISODES_KEY], 0.9, places=6
        )
        self.assertAlmostEqual(scalars[f"{self.c.SUCCESS_KEY}/stairs"], 0.0)
        self.assertAlmostEqual(scalars[f"{self.c.SUCCESS_KEY}/boxes"], 1.0)

    def test_steps_without_a_reset_do_not_repeat_the_last_outcomes(self):
        one_success = ([True] + [False] * 9, [True] + [False] * 9, [0] * 10)
        one_failure = ([False] * 10, [True] + [False] * 9, [0] * 10)
        scalars = self.logged([one_success, None, None, None, one_failure])
        self.assertAlmostEqual(scalars[self.c.SUCCESS_KEY], 0.5)
        self.assertAlmostEqual(scalars[self.c.EPISODES_KEY], 2 / 5)

    def test_start_up_reset_logs_no_outcome(self):
        log = self.c.episode_log(
            torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), torch.zeros(4)
        )
        self.assertEqual(log[self.c.SUCCESS_KEY].numel(), 0)
        self.assertEqual(log[self.c.EPISODES_KEY], 0.0)


class TerrainOffCourseTests(unittest.TestCase):
    def setUp(self):
        self.c = load_curriculum()

    def test_an_off_course_timeout_is_a_failure_and_a_demotion(self):
        # an episode that left the course terminates; on speed alone it would have been a success
        steps = torch.tensor([1000.0, 1000.0])
        cmd = torch.tensor([0.35, 0.35])
        terminated = torch.tensor([True, False])
        success, slow, *_ = self.c.episode_outcome(
            cmd * steps, steps, cmd, terminated, torch.tensor([False, True]), 0.35
        )
        torch.testing.assert_close(success, torch.tensor([False, True]))
        up, down = self.c.terrain_moves(success, terminated, slow, torch.tensor([True, True]))
        torch.testing.assert_close(up, torch.tensor([False, True]))
        torch.testing.assert_close(down, torch.tensor([True, False]))


class TerrainFunctionIdentityTests(unittest.TestCase):
    def test_sub_terrain_functions_are_module_level_names(self):
        # Isaac Lab serializes cfg.function as module:__name__ and Hydra overrides rebuild it from that string, so each
        # assigned function must be a plain def of terrains_cfg; a functools.wraps wrapper resolves to the stock one
        tree = ast.parse((PACKAGE / "terrains_cfg.py").read_text())
        defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assigned = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "function"
        ]
        self.assertEqual(len(assigned), 3)
        for value in assigned:
            with self.subTest(function=ast.unparse(value)):
                self.assertIsInstance(value, ast.Name)
                self.assertIn(value.id, defs)
        self.assertNotIn("functools", ast.unparse(tree))


class TerrainEnvSourceTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse((PACKAGE / "env.py").read_text())

    def calls(self, attr):
        return [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
        ]

    def test_feet_contacts_follow_the_legs_order(self):
        (call,) = [c for c in self.calls("find_sensors")]
        self.assertIn(("preserve_order", True), [(k.arg, k.value.value) for k in call.keywords])
        self.assertIn("LEGS", ast.unparse(call.args[0]))
        self.assertNotIn("_feet_ids", ast.unparse(self.tree))

    def test_curriculum_never_writes_the_reward_scale(self):
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute):
                        self.assertNotEqual(target.attr, "command_lin_vel_max", ast.unparse(node))

    def test_scene_setup_extends_the_parent(self):
        setup = next(
            n
            for n in ast.walk(self.tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_setup_scene"
        )
        self.assertIn("super()._setup_scene()", ast.unparse(setup))
        self.assertNotIn("Articulation(", ast.unparse(setup))


if __name__ == "__main__":
    unittest.main()
