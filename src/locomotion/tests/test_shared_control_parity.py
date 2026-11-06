"""Retain pre-refactor Direct control behavior on CPU; these tests do not simulate PhysX.

The fixture contains unmodified source from 929e035f, before PR 91. Both versions
read the same synthetic native views. No expected value comes from the new helpers.
"""

import hashlib
import json
import math
import sys
import tarfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from locomotion.omniwalk.mdp.terminations import episode_timeout, physical_termination
from locomotion.robots.hexapod import BODY_NAMES, JOINT_NAMES
from locomotion.tests.test_direct_environment import ISAAC_MODULES, load_isaac

FIXTURE = Path(__file__).parent / "fixtures/direct_control_929e035f.tar.gz"
FIXTURE_SHA = "70912d03b0d59084d654a7c4631883bd70f44140fc0cc870afd1f38d2d028242"
BASE_COMMIT = "929e035f8ba6d6b13beacd0dff78b9fb8eb2085c"


def frozen_direct():
    """Load the old environment and its frame/action/observation helpers without Isaac."""
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != FIXTURE_SHA:
        raise ValueError("Frozen Direct control source changed")
    with tarfile.open(FIXTURE) as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        if manifest["source_commit"] != BASE_COMMIT:
            raise ValueError("Unexpected Direct control baseline")
        source = {path: archive.extractfile(path).read() for path in manifest["files"]}
    for path, content in source.items():
        if hashlib.sha256(content).hexdigest() != manifest["files"][path]:
            raise ValueError("Frozen source bytes differ: " + path)
    stubs = {name: mock.MagicMock(name=name) for name in ISAAC_MODULES}
    stubs["isaaclab.envs"].DirectRLEnv = object
    stubs["isaaclab_physx.assets.articulation"].Articulation = object
    with mock.patch.dict(sys.modules, stubs):
        for suffix in ("mdp.frames", "mdp.actions", "mdp.observations", "env"):
            name = "locomotion.omniwalk." + suffix
            path = "src/" + name.replace(".", "/") + ".py"
            module = types.ModuleType(name)
            exec(compile(source[path], str(FIXTURE) + "/" + path, "exec"), module.__dict__)
            # Later frozen imports must resolve to the frozen helpers, too.
            sys.modules[name] = module
    return module


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    return value


def direct_double(module, rows):
    """Stub native I/O only; keep the real read_state and full control methods."""
    module.wp = SimpleNamespace(to_torch=lambda value: value)
    env = object.__new__(module.IsaacEnv)
    env.config = SimpleNamespace(
        physics_dt=0.0025,
        control_dt=0.02,
        episode_seconds=20.0,
        action_scale_rad=0.35,
        target_slew_rad=0.04,
    )
    env.device, env.num_envs = "cpu", rows
    env.torch = lambda value: torch.as_tensor(value, dtype=torch.float32)
    env.neutral = torch.arange(18, dtype=torch.float32) / 100 - 0.09
    env.lower, env.upper = env.neutral - 1, env.neutral + 1
    env.joint_index = torch.roll(torch.arange(18), 5)
    env.inverse_joint_index = torch.argsort(env.joint_index)
    env.native_body_names = list(reversed(BODY_NAMES))
    env.toe_index = torch.tensor(
        [
            env.native_body_names.index(leg + "_tibia")
            for leg in ("lf", "lm", "lr", "rf", "rm", "rr")
        ]
    )
    env.contact_index = torch.roll(torch.arange(rows * 19), 7)
    env.root_com_local = torch.tensor([[0.012, -0.025, 0.009]]).repeat(rows, 1)
    env.toe_local = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 1000
    env.native = {}
    env.view = SimpleNamespace(
        **{
            getter: (lambda key=getter: env.native[key])
            for getter in (
                "get_dof_positions",
                "get_dof_velocities",
                "get_root_transforms",
                "get_link_transforms",
                "get_root_velocities",
                "get_dof_actuation_forces",
            )
        }
    )
    env.sim = SimpleNamespace(
        count=7, physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda: None)
    )
    env.sim.get_physics_step_count = lambda: env.sim.count
    env.written_targets = []
    env.robot = SimpleNamespace(
        set_joint_position_target_index=lambda target: env.written_targets.append(target.clone())
    )
    env.motor = SimpleNamespace(joint_names=list(JOINT_NAMES[9:] + JOINT_NAMES[:9]))
    env.contact = SimpleNamespace(get_contact_force_matrix=lambda dt: env.raw_forces)
    env.native_errors, env.captures = [], []
    env.capture = lambda *args: env.captures.append(snapshot(args))
    env.held = env.neutral.expand(rows, -1).clone()
    env.previous_action = torch.zeros(rows, 18)
    env.history = torch.arange(rows * 5 * 42, dtype=torch.float32).reshape(rows, 5, 42) / 1000
    env.commands = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) / 100
    env.episode_length_buf = torch.zeros(rows, dtype=torch.long)
    env.episode_length_buf[-1] = 999
    env.total_controls, env.control_active = 0, False
    set_readings(env, 0)
    return env


def set_readings(env, phase):
    rows = env.num_envs
    values = torch.arange(rows * 18, dtype=torch.float32).reshape(rows, 18)
    q = env.neutral + (values % 11 - 5) / 1000 + phase / 10000
    dq = (values % 13 - 6) / 20 + phase / 100
    root = torch.tensor([[0.25, -0.5, 0.1, 0.125, -0.25, 0.375, 1.0]]).repeat(rows, 1)
    root[:, :2] += torch.arange(rows)[:, None] / 10
    if rows >= 5:
        root[1, 2] = 0.04  # Fall, tilt, joint violation, timeout and normal rows coexist.
        root[2, 3:] = torch.tensor([0.75, 0.0, 0.0, 1.0])
        q[3, 7] = env.upper[7] + 0.001
    link = torch.arange(rows * 19 * 7, dtype=torch.float32).reshape(rows, 19, 7) / 1000
    link[:, :, 6] += 1
    velocity = torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 6) / 10 - 0.75
    requested = (values % 17 - 8) / 4 + (phase % 4) / 10
    requested[:, :4] = torch.tensor([1.6, -1.6, 1.60001, -1.60001])
    applied = (values % 13 - 6) / 5 + (phase % 3) / 100
    motor_order = [JOINT_NAMES.index(name) for name in env.motor.joint_names]
    for name, value in (
        ("pre_position", q - 0.002),
        ("pre_velocity", dq - 0.03),
        ("computed_effort", requested),
        ("applied_effort", applied),
        ("ceiling", torch.full_like(applied, 1.6)),
    ):
        setattr(env.motor, name, value[:, motor_order])
    env.native.update(
        get_dof_positions=q[:, env.inverse_joint_index],
        get_dof_velocities=dq[:, env.inverse_joint_index],
        get_root_transforms=root,
        get_link_transforms=link,
        get_root_velocities=velocity,
        get_dof_actuation_forces=(applied + 0.125)[:, env.inverse_joint_index],
    )
    forces = torch.arange(rows * 19 * 3, dtype=torch.float32).reshape(rows, 19, 3) % 23
    forces *= (9 - phase % 8) / (16 if phase > 8 else 8)
    # Two toes stay loaded, and the second control changes which toes unload.
    # Zero-initialized minima and stale per-control extrema must change the trace.
    unloaded = phase % 4 if phase <= 8 else 0
    forces[:, env.toe_index[unloaded]] = 0
    env.raw_forces = torch.empty(rows * 19, 1, 3)
    env.raw_forces[env.contact_index, 0] = forces.reshape(rows * 19, 3)


def control_trace(module, rows):
    env = direct_double(module, rows)
    trace = [snapshot(env.read_state())]
    for control in range(2):
        action = torch.arange(rows * 18, dtype=torch.float32).reshape(rows, 18) / 20 - 1.5
        env._pre_physics_step(action * (-1 if control else 1))
        for step in range(8):
            set_readings(env, 8 * control + step + 1)
            env._apply_action()
            env.record_submitted_force()
            env.sim.count += 1
            env.collect_substep()
        env.episode_length_buf += 1
        stock_dones = env._get_dones()
        trace.append(
            snapshot(
                {
                    "state": env.current,
                    "telemetry": env.telemetry,
                    "capture": env.captures,
                    "stock_dones": stock_dones,
                    "observations": env._get_observations(),
                    "held": env.held,
                    "history": env.history,
                    "previous_action": env.previous_action,
                    "written_targets": env.written_targets,
                    "total_controls": env.total_controls,
                }
            )
        )
        env.captures.clear()
        env.written_targets.clear()
    return trace


class SharedControlParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = frozen_direct()
        cls.candidate = load_isaac()

    def exact(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                with self.subTest(field=key):
                    self.exact(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.exact(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_state_capture_loads_and_telemetry_match_pre_refactor_control(self):
        for rows in (1, 5):
            with self.subTest(rows=rows):
                actual = control_trace(self.candidate, rows)
                self.exact(actual, control_trace(self.baseline, rows))
                for control in actual[1:]:
                    self.assertEqual(len(control["capture"]), 8)
                    self.assertEqual([frame[8] for frame in control["capture"]], list(range(8)))
                    self.assertEqual(len(control["telemetry"]), 22)
                    self.assertEqual(control["telemetry"]["truncated"][-1].item(), True)
                if rows == 5:
                    self.assertEqual(
                        actual[1]["telemetry"]["terminated"].tolist(),
                        [False, True, True, True, False],
                    )

    def test_physical_termination_boundaries_keep_strict_inequalities(self):
        # Test the threshold and its adjacent float64 values, without copying helper arithmetic.
        for field, threshold, index, direction in (
            ("root", 0.045, 2, -1),
            ("gravity", -math.cos(0.85), 2, 1),
            ("q", -1.0 - 2e-6, 7, -1),
            ("q", 1.0 + 2e-6, 7, 1),
        ):
            with self.subTest(field=field, threshold=threshold):
                state = {
                    "root": torch.tensor([[0, 0, 0.1, 0, 0, 0, 1]], dtype=torch.float64).repeat(
                        3, 1
                    ),
                    "gravity": torch.tensor([[0, 0, -1]], dtype=torch.float64).repeat(3, 1),
                    "q": torch.zeros(3, 18, dtype=torch.float64),
                }
                value = torch.tensor(threshold, dtype=torch.float64)
                state[field][:, index] = torch.stack(
                    (
                        torch.nextafter(value, value.new_tensor(-torch.inf)),
                        value,
                        torch.nextafter(value, value.new_tensor(torch.inf)),
                    )
                )
                expected = [True, False, False] if direction < 0 else [False, False, True]
                self.assertEqual(
                    physical_termination(
                        state,
                        -torch.ones(18, dtype=torch.float64),
                        torch.ones(18, dtype=torch.float64),
                    ).tolist(),
                    expected,
                )

    def test_timeout_starts_at_the_rounded_control_count(self):
        for seconds, boundary in ((20.0, 1000), (20.03, 1002)):
            with self.subTest(seconds=seconds):
                steps = torch.tensor([boundary - 1, boundary, boundary + 1])
                self.assertEqual(
                    episode_timeout(steps, seconds, 0.02).tolist(), [False, True, True]
                )

    def test_nonfinite_native_views_are_rejected_before_telemetry(self):
        for getter in (
            "get_dof_positions",
            "get_dof_velocities",
            "get_root_transforms",
            "get_link_transforms",
            "get_root_velocities",
        ):
            for value in (torch.nan, torch.inf, -torch.inf):
                with self.subTest(getter=getter, value=value):
                    for module in (self.baseline, self.candidate):
                        env = direct_double(module, 5)
                        env.native[getter].reshape(-1)[0] = value
                        with self.assertRaisesRegex(FloatingPointError, "Nonfinite native"):
                            env.read_state()


if __name__ == "__main__":
    unittest.main()
