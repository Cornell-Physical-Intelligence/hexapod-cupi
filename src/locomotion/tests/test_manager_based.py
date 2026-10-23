"""Check the manager-based prototype's shared arithmetic against the certified Direct task and wrapper on CPU.

These tests claim no native parity; the native replay harness decides that.
"""

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from locomotion.omniwalk.mdp.commands import TaskConfig
from locomotion.omniwalk.mdp.loads import SubstepLoads, control_telemetry
from locomotion.omniwalk.mdp.state import measured_state
from locomotion.omniwalk.mdp.terminations import episode_timeout, physical_termination
from locomotion.omniwalk.task import TrainingTask
from locomotion.rl.wrapper import OmniVecEnv
from locomotion.robots.hexapod import BODY_NAMES, JOINT_NAMES, LEGS
from locomotion.spark.sources import physics_files, source_files
from locomotion.tests.test_direct_environment import load_isaac

PACKAGE = Path(__file__).resolve().parents[1]


def unit(shape, generator):
    value = torch.randn(*shape, 4, generator=generator)
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)


class NativeArithmeticTests(unittest.TestCase):
    def test_prototype_stays_outside_the_certified_physics_closure(self):
        closure = set(physics_files(source_files(PACKAGE)))
        for name in (
            "omniwalk/manager_based/terms.py",
            "omniwalk/manager_based/env_cfg.py",
            "omniwalk/mdp/memory.py",
            "omniwalk/mdp/state.py",
            "omniwalk/mdp/loads.py",
            "omniwalk/mdp/terminations.py",
        ):
            self.assertNotIn(name, closure)

    def test_measured_state_equals_the_direct_state_read(self):
        module = load_isaac()
        generator = torch.Generator().manual_seed(4)
        n = 3
        arrays = {
            "get_dof_positions": torch.randn(n, 18, generator=generator),
            "get_dof_velocities": torch.randn(n, 18, generator=generator),
            "get_root_transforms": torch.cat(
                (torch.randn(n, 3, generator=generator), unit((n,), generator)), -1
            ),
            "get_link_transforms": torch.cat(
                (torch.randn(n, 19, 3, generator=generator), unit((n, 19), generator)), -1
            ),
            "get_root_velocities": torch.randn(n, 6, generator=generator),
        }
        isaac = module.IsaacEnv.__new__(module.IsaacEnv)
        isaac.get = lambda name: arrays[name].clone()
        isaac.sim = SimpleNamespace(
            physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda: None)
        )
        isaac.joint_index = torch.randperm(18, generator=generator)
        isaac.root_com_local = 0.01 * torch.randn(n, 3, generator=generator)
        isaac.torch = lambda x: torch.as_tensor(x, dtype=torch.float32)
        isaac.num_envs = n
        isaac.toe_index = torch.tensor([BODY_NAMES.index(leg + "_tibia") for leg in LEGS])
        isaac.toe_local = 0.1 * torch.randn(6, 3, generator=generator)
        expected = isaac.read_state()
        actual = measured_state(
            arrays["get_dof_positions"][:, isaac.joint_index],
            arrays["get_dof_velocities"][:, isaac.joint_index],
            arrays["get_root_transforms"],
            arrays["get_link_transforms"],
            arrays["get_root_velocities"],
            isaac.root_com_local,
            isaac.toe_index,
            isaac.toe_local,
        )
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_loads_telemetry_and_terminations_equal_one_direct_control(self):
        module = load_isaac()
        module.wp = SimpleNamespace(to_torch=lambda x: x)
        generator = torch.Generator().manual_seed(9)

        def random(*shape):
            return torch.randn(*shape, generator=generator)

        n, neutral = 2, torch.linspace(-0.3, 0.3, 18)
        isaac = module.IsaacEnv.__new__(module.IsaacEnv)
        isaac.native_errors, isaac.capture, isaac.device, isaac.num_envs = [], None, "cpu", n
        isaac.neutral, isaac.lower, isaac.upper = neutral, neutral - 0.5, neutral + 0.5
        isaac.held = neutral.expand(n, -1).clone()
        isaac.config = SimpleNamespace(
            action_scale_rad=0.35,
            target_slew_rad=0.04,
            physics_dt=0.0025,
            episode_seconds=20.0,
            control_dt=0.02,
        )
        steps = [0]
        isaac.sim = SimpleNamespace(get_physics_step_count=lambda: steps[0])
        isaac.robot = SimpleNamespace(set_joint_position_target_index=lambda **kwargs: None)
        isaac.inverse_joint_index = torch.arange(18)
        isaac.native_body_names = list(BODY_NAMES)
        isaac.toe_index = torch.tensor([BODY_NAMES.index(leg + "_tibia") for leg in LEGS])
        isaac.contact_index = torch.randperm(19 * n, generator=generator)
        isaac.history, isaac.previous_action = torch.zeros(n, 5, 42), torch.zeros(n, 18)
        isaac.episode_length_buf = torch.tensor([3, 999])
        isaac.commands, isaac.total_controls = random(n, 3), 0
        reading = {}
        isaac.read_state = lambda: reading["state"]
        isaac.motor = SimpleNamespace(joint_names=list(JOINT_NAMES))
        isaac.contact = SimpleNamespace(get_contact_force_matrix=lambda dt: reading["forces"])
        action = random(n, 18)
        isaac._pre_physics_step(action)
        other = [i for i, name in enumerate(BODY_NAMES) if not name.endswith("_tibia")]
        loads = SubstepLoads(isaac.target, isaac.toe_index, other)
        for _ in range(8):
            isaac._apply_action()
            isaac.record_submitted_force()
            steps[0] += 1
            for name in ("pre_position", "pre_velocity", "ceiling"):
                setattr(isaac.motor, name, random(n, 18))
            isaac.motor.computed_effort, isaac.motor.applied_effort = (
                2 * random(n, 18),
                random(n, 18).clamp(-1.6, 1.6),
            )
            reading["forces"] = 5 * random(19 * n, 1, 3)
            root = torch.cat(
                (random(n, 2), torch.tensor([[0.08], [0.03]]), unit((n,), generator)), -1
            )
            reading["state"] = {
                "q": neutral + 0.4 * torch.tanh(random(n, 18)),
                "dq": random(n, 18),
                "root": root,
                "linear": random(n, 3),
                "angular": random(n, 3),
                "gravity": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
                "toe_world": random(n, 6, 3),
                "toe_body": random(n, 6, 3),
            }
            isaac.collect_substep()
            forces = reading["forces"][isaac.contact_index, 0].reshape(n, 19, 3)
            loads.add(isaac.motor.computed_effort, isaac.motor.applied_effort, forces)
        isaac.episode_length_buf += 1
        isaac._get_dones()
        state = reading["state"]
        terminated = physical_termination(state, isaac.lower, isaac.upper)
        truncated = episode_timeout(isaac.episode_length_buf, 20.0, 0.02)
        self.assertEqual((terminated.tolist(), truncated.tolist()), ([False, True], [False, True]))
        expected = control_telemetry(
            state,
            isaac.target,
            isaac.motor.computed_effort,
            isaac.motor.applied_effort,
            forces,
            loads,
            isaac.commands,
            action,
            terminated,
            truncated,
        )
        self.assertEqual(list(isaac.telemetry), list(expected))
        for key, value in expected.items():
            self.assertTrue(torch.equal(isaac.telemetry[key], value), key)


class CommandTermStub:
    """The part of ``isaaclab.managers.CommandTerm`` that the bank command inherits."""

    def __init__(self, cfg, env):
        self.cfg, self._env, self.num_envs, self.device = cfg, env, env.num_envs, env.device
        self.command_counter = torch.zeros(env.num_envs, dtype=torch.long)


def load_terms():
    """Import the manager terms over stand-ins for Isaac Lab, PhysX, USD and Warp."""
    stubs = {
        name: mock.MagicMock(name=name)
        for name in (
            "omni",
            "omni.physx",
            "warp",
            "pxr",
            "isaaclab",
            "isaaclab.sim.schemas",
            "isaaclab.sim.utils",
        )
    }
    stubs["isaaclab.managers"] = SimpleNamespace(
        ActionTerm=object,
        ActionTermCfg=object,
        CommandTerm=CommandTermStub,
        CommandTermCfg=object,
        RecorderTerm=object,
    )
    stubs["isaaclab.sim"] = SimpleNamespace(CollisionPropertiesCfg=object, SpawnerCfg=object)
    stubs["isaaclab.utils"] = SimpleNamespace(configclass=lambda cls: cls)
    with mock.patch.dict(sys.modules, stubs):
        sys.modules.pop("locomotion.omniwalk.manager_based.terms", None)
        module = importlib.import_module("locomotion.omniwalk.manager_based.terms")
    sys.modules.pop("locomotion.omniwalk.manager_based.terms", None)
    return module


class BankCommandTests(unittest.TestCase):
    def test_bank_command_draws_what_the_training_task_draws_in_the_manager_call_order(self):
        # The training task draws expired holds inside its step and finished rows at the learner's reset;
        # the manager resets finished rows first and then computes commands.
        terms, n = load_terms(), 8
        env = SimpleNamespace(
            num_envs=n,
            device="cpu",
            held=torch.zeros(n, 18),
            commands=torch.zeros(n, 3),
            current={
                "dq": torch.zeros(n, 18),
                "root": torch.zeros(n, 7),
                "toe_world": torch.zeros(n, 6, 3),
            },
            cfg=SimpleNamespace(control_dt=0.02, decimation=8),
            reference_metadata={},
            reset_height=0.08,
        )
        task = TrainingTask(env, TaskConfig(seed=31))
        cfg = terms.BankCommandCfg()
        cfg.seed = 31
        term = terms.BankCommand(cfg, SimpleNamespace(num_envs=n, device="cpu"))
        task._resample(torch.arange(n))
        term.reset(None)
        self.assertTrue(torch.equal(term.command, env.commands))
        generator = torch.Generator().manual_seed(2)
        both = 0
        for control in range(700):
            done = torch.rand(n, generator=generator) < 0.01
            task.remaining_controls -= 1
            expired = ((task.remaining_controls == 0) & ~done).nonzero(as_tuple=False).squeeze(-1)
            both += bool(len(expired) and done.any())
            task._resample(expired)
            task._resample(done.nonzero().flatten())
            if done.any():
                term.reset(done.nonzero().flatten())
            term.compute(0.02)
            self.assertTrue(torch.equal(term.command, env.commands), control)
            self.assertTrue(torch.equal(term.remaining, task.remaining_controls), control)
        self.assertGreater(both, 0)
        self.assertEqual(term.schedule.command_draws, task.command_draws)

    def test_evaluation_keeps_the_written_commands(self):
        terms = load_terms()
        cfg = terms.BankCommandCfg()
        cfg.resample, cfg.initial = False, (0.1, 0.0, 0.0)
        term = terms.BankCommand(cfg, SimpleNamespace(num_envs=2, device="cpu"))
        term.reset(None)
        term.commands[1] = torch.tensor([0.0, 0.0, 0.2])
        for _ in range(300):
            term.compute(0.02)
        term.reset(torch.tensor([0]))
        self.assertEqual(
            term.command.tolist(),
            [[0.10000000149011612, 0.0, 0.0], [0.0, 0.0, 0.20000000298023224]],
        )
        self.assertEqual(term.schedule.command_draws, 0)


class ScriptedWalk:
    """One scripted rollout read by a training-task double and a manager-environment double.

    Row 1's hold ends at control 3, row 2 falls at control 5, row 3 falls as its episode times out at
    control 6 and row 0 times out at control 7. The reward depends on the action, so smoothing shows.
    """

    def __init__(self, n=4, controls=9):
        generator = torch.Generator().manual_seed(21)

        def random(*shape):
            return torch.randn(*shape, generator=generator)

        self.n, self.controls = n, controls
        self.commands = torch.tensor(
            [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]
        )
        self.steps = torch.zeros(n, dtype=torch.long)
        self.frames = [
            (random(n, 231), random(n, 3), random(n, 231), random(n, 3), random(n, 3))
            for _ in range(controls + 1)
        ]
        self.ends = {3: ([], [], [1]), 5: ([2], [], []), 6: ([3], [3], []), 7: ([], [0], [])}
        self.control = 0

    def output(self, obs, linear):
        obs = obs.clone()
        obs[:, 210:213] = self.commands
        return {"obs": obs, "critic": torch.cat((obs, linear), -1)}

    def advance(self, action):
        """Step every row; return the terminal output, the reward and the end flags, before any reset."""
        obs, linear, *_ = self.frames[self.control]
        falls, timeouts, expired = self.ends.get(self.control, ([], [], []))
        self.steps += 1
        terminated, truncated = (
            torch.zeros(self.n, dtype=torch.bool),
            torch.zeros(self.n, dtype=torch.bool),
        )
        terminated[falls], truncated[timeouts] = True, True
        self.commands[expired] = torch.tensor([0.025, 0.025, 0.0])
        self.control += 1
        return self.output(obs, linear), action.sum(-1), terminated, truncated

    def reset(self, rows):
        _, _, obs, linear, command = self.frames[self.control]
        self.steps[rows] = 0
        self.commands[rows] = command[rows]
        return self.output(obs, linear)


class TaskDouble:
    """The training-task surface that ``OmniVecEnv`` drives."""

    def __init__(self, walk):
        self.walk, self.num_envs, self.device = walk, walk.n, "cpu"
        self.cfg = SimpleNamespace(episode_seconds=20.0, control_dt=0.02, declaration=lambda: {})
        self.declaration = lambda: {}
        self.current = None

    @property
    def episode_steps(self):
        return self.walk.steps

    @property
    def commands(self):
        return self.walk.commands

    def reset(self, rows=None):
        rows = torch.arange(self.num_envs) if rows is None else rows
        fresh = self.walk.reset(rows)
        current = (
            fresh
            if self.current is None
            else {key: value.clone() for key, value in self.current.items()}
        )
        for key in current:
            current[key][rows] = fresh[key][rows]
        self.current = current
        return current

    def step(self, action):
        output, reward, terminated, truncated = self.walk.advance(action)
        self.current = output
        return {**output, "reward": reward, "terminated": terminated, "truncated": truncated}


class ManagerDouble:
    """The manager-based environment surface that ``ManagerVecEnv`` drives; it resets finished rows inside its step."""

    def __init__(self, walk):
        self.walk, self.num_envs, self.device, self.max_episode_length = walk, walk.n, "cpu", 1000
        self.command_manager = SimpleNamespace(get_command=lambda name: walk.commands)
        self.observation_manager = SimpleNamespace(compute=lambda: self.obs)
        self.current = None
        # The stock environment returns this one dictionary from every step.
        self.extras = {"log": {}}

    @property
    def episode_length_buf(self):
        return self.walk.steps

    def groups(self, output):
        return {"policy": output["obs"], "critic": output["critic"]}

    def reset(self):
        fresh = self.walk.reset(torch.arange(self.num_envs))
        self.current = fresh
        self.obs = self.groups(fresh)
        return self.obs, {}

    def step(self, action):
        output, reward, terminated, truncated = self.walk.advance(action)
        rows = (terminated | truncated).nonzero().flatten()
        if len(rows):
            fresh = self.walk.reset(rows)
            for key in output:
                output[key][rows] = fresh[key][rows]
        self.obs = self.groups(output)
        return self.obs, reward, terminated, truncated, self.extras


class StockWrapper:
    """The part of ``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper`` that the manager wrapper inherits."""

    def __init__(self, env, clip_actions=None):
        self.env, self.num_envs, self.device = env, env.num_envs, env.device
        self.env.reset()

    @property
    def unwrapped(self):
        return self.env


def load_manager_wrapper():
    stubs = {
        "isaaclab_rl": mock.MagicMock(),
        "isaaclab_rl.rsl_rl": SimpleNamespace(RslRlVecEnvWrapper=StockWrapper),
    }
    with mock.patch.dict(sys.modules, stubs):
        sys.modules.pop("locomotion.rl.manager_wrapper", None)
        module = importlib.import_module("locomotion.rl.manager_wrapper")
    sys.modules.pop("locomotion.rl.manager_wrapper", None)
    return module


class ManagerWrapperTests(unittest.TestCase):
    def test_manager_wrapper_returns_what_the_omni_wrapper_returns_under_the_c48_long_options(self):
        options = {
            "observation_scaling": "fixed",
            "command_segments": "bootstrap",
            "gait_clock": 60,
            "action_smoothing": "mean2",
            "velocity_noise": 0.5,
        }
        omni = OmniVecEnv(TaskDouble(ScriptedWalk()), **options)
        manager = load_manager_wrapper().ManagerVecEnv(ManagerDouble(ScriptedWalk()), **options)
        torch.manual_seed(0)
        first = omni.get_observations()
        torch.manual_seed(0)
        second = manager.get_observations()
        for key in ("policy", "critic"):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        generator = torch.Generator().manual_seed(5)
        seen = set()
        for control in range(9):
            action = torch.randn(4, 18, generator=generator)
            torch.manual_seed(100 + control)
            a = omni.step(action.clone())
            torch.manual_seed(100 + control)
            b = manager.step(action.clone())
            for key in ("policy", "critic"):
                self.assertTrue(torch.equal(a[0][key], b[0][key]), (control, key))
            self.assertTrue(torch.equal(a[1], b[1]), control)
            self.assertTrue(torch.equal(a[2], b[2]), control)
            self.assertTrue(torch.equal(a[3]["time_outs"], b[3]["time_outs"]), control)
            self.assertEqual(set(a[3]), set(b[3]) - {"log"}, control)
            if "command_time_outs" in a[3]:
                seen.add(control)
                self.assertTrue(torch.equal(a[3]["command_time_outs"], b[3]["command_time_outs"]))
                self.assertTrue(
                    torch.equal(
                        a[3]["command_bootstrap_observation"]["critic"],
                        b[3]["command_bootstrap_observation"]["critic"],
                    )
                )
        self.assertEqual(seen, {3})


if __name__ == "__main__":
    unittest.main()
