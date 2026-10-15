"""Check Shaurya source relocation and CPU hook arithmetic against pinned Git."""
import ast
import hashlib
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT/'locomotion/tripod'
SOURCE_REVISION = '969ea5bc942df3122b2662650f29acd11644a41c'
SOURCE_HASHES = {
    '__init__.py': '6b7b28b0491ff1abef576d5fb6c8f662333143d252195286fa80d1b33b3efa61',
    'hexapod_env.py': '6e68fa237f7a358df12bd1cb1afcef02dc9ee7ad0a3c21d51728d12f2cb9a227',
    'hexapod_env_cfg.py': '252fd40b2788ab6b658bd0d3c354d397716ccca773b1479db8a4fc863507e154',
    'hexapod_robot_cfg.py': '1d3ee917b0822bebc79f859d463686649d9ff41fc14765e6d5ebb8aec9401909',
    'agents/__init__.py': '3be19bd191a2d8780967a91b0d6e6fe4c6d74ad8b54efef94c33f6c2ebd6078b',
    'agents/rsl_rl_ppo_cfg.py': '6cc8751ff7cde95b957aa915a23894236a6f9ec14b6f2a021e85028390bd528e',
}
OLD_ASSET_PATH = 'os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot",'
NEW_ASSET_PATH = 'os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "robot",'


def source(name):
    return (PACKAGE/name).read_text()


def definition(name, class_name):
    return next(node for node in ast.parse(source(name)).body
                if isinstance(node, ast.ClassDef) and node.name == class_name)


def assignments(node):
    return {item.targets[0].id if isinstance(item, ast.Assign) else item.target.id: item.value
            for item in node.body if isinstance(item, (ast.Assign, ast.AnnAssign))}


class DirectDouble:
    def _reset_idx(self, env_ids):
        self.base_reset_ids = env_ids.clone()


class FlatCfg:
    pass


class RoughCfg(FlatCfg):
    pass


def hook_class():
    node = definition('hexapod_env.py', 'HexapodEnv')
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node],
                      type_ignores=[])
    scope = {'torch': torch, 'math': math, 'DirectRLEnv': DirectDouble,
             'HexapodFlatEnvCfg': FlatCfg, 'HexapodRoughEnvCfg': RoughCfg,
             'wp': SimpleNamespace(to_torch=lambda value: value)}
    exec(compile(ast.fix_missing_locations(tree), str(PACKAGE/'hexapod_env.py'), 'exec'), scope)
    return scope['HexapodEnv']


def array(value):
    return SimpleNamespace(torch=value)


class ShauryaSourceTests(unittest.TestCase):
    def test_source_matches_pinned_git_with_only_the_asset_path_relocation(self):
        for name, digest in SOURCE_HASHES.items():
            with self.subTest(file=name):
                original = subprocess.check_output(
                    ['git', 'show', f'{SOURCE_REVISION}:hexapod/{name}'], cwd=ROOT).decode()
                self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), digest)
                current = source(name)
                self.assertEqual(current.splitlines()[:4], original.splitlines()[:4])
                if name == 'hexapod_robot_cfg.py':
                    self.assertEqual(original.count(OLD_ASSET_PATH), 1)
                    self.assertEqual(current.count(NEW_ASSET_PATH), 1)
                    original = original.replace(OLD_ASSET_PATH, NEW_ASSET_PATH)
                self.assertEqual(ast.dump(ast.parse(current)), ast.dump(ast.parse(original)))
                self.assertEqual(current, original)

    def test_registration_keeps_both_ids_and_points_to_the_relocated_package(self):
        calls = []
        tree = ast.parse(source('__init__.py'))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        scope = {'__name__': 'locomotion.tripod', 'gym': SimpleNamespace(register=lambda **kwargs: calls.append(kwargs)),
                 'agents': SimpleNamespace(__name__='locomotion.tripod.agents')}
        exec(compile(tree, str(PACKAGE/'__init__.py'), 'exec'), scope)
        self.assertEqual([entry['id'] for entry in calls], ['Hexapod-Flat-Direct-v0', 'Hexapod-Rough-Direct-v0'])
        for entry, terrain in zip(calls, ('Flat', 'Rough')):
            self.assertEqual(entry['entry_point'], 'locomotion.tripod.hexapod_env:HexapodEnv')
            self.assertEqual(entry['kwargs']['env_cfg_entry_point'],
                             f'locomotion.tripod.hexapod_env_cfg:Hexapod{terrain}EnvCfg')
            self.assertEqual(entry['kwargs']['rsl_rl_cfg_entry_point'],
                             f'locomotion.tripod.agents.rsl_rl_ppo_cfg:Hexapod{terrain}PPORunnerCfg')
            self.assertTrue(entry['disable_env_checker'])

    def test_asset_path_defaults_to_repo_robot_and_keeps_the_override(self):
        node = next(node for node in ast.parse(source('hexapod_robot_cfg.py')).body
                    if isinstance(node, ast.Assign) and node.targets[0].id == 'HEXAPOD_USD_PATH')
        expression = compile(ast.Expression(node.value), str(PACKAGE/'hexapod_robot_cfg.py'), 'eval')
        with patch.dict(os.environ, {}, clear=True):
            value = eval(expression, {'os': os, '__file__': str(PACKAGE/'hexapod_robot_cfg.py')})
            self.assertEqual(Path(value).resolve(), ROOT/'robot/hexapod_mkii_updated_v1/usd/robot.usda')
        with patch.dict(os.environ, {'HEXAPOD_USD': '/chosen/robot.usda'}):
            self.assertEqual(eval(expression, {'os': os, '__file__': str(PACKAGE/'hexapod_robot_cfg.py')}),
                             '/chosen/robot.usda')

    def test_config_keeps_stock_spaces_randomization_and_actor_architectures(self):
        flat = assignments(definition('hexapod_env_cfg.py', 'HexapodFlatEnvCfg'))
        rough = assignments(definition('hexapod_env_cfg.py', 'HexapodRoughEnvCfg'))
        for name, value in {'episode_length_s': 20., 'decimation': 8, 'action_scale': .35,
                            'action_space': 18, 'observation_space': 66, 'gait_clock_enabled': False,
                            'gait_period_s': .8, 'command_lin_vel_max': .3, 'command_yaw_rate_max': .3}.items():
            self.assertEqual(ast.literal_eval(flat[name]), value)
        self.assertEqual(ast.literal_eval(rough['observation_space']), 235)
        event = assignments(definition('hexapod_env_cfg.py', 'EventCfg'))
        self.assertEqual(set(event), {'physics_material', 'add_base_mass'})
        for terrain, dims in (('Flat', [128, 128, 128]), ('Rough', [512, 256, 128])):
            cfg = assignments(definition('agents/rsl_rl_ppo_cfg.py', f'Hexapod{terrain}PPORunnerCfg'))
            for name in ('actor', 'critic'):
                fields = {item.arg: item.value for item in cfg[name].keywords}
                self.assertEqual(ast.literal_eval(fields['hidden_dims']), dims)
                self.assertEqual(ast.literal_eval(fields['activation']), 'elu')
                self.assertFalse(ast.literal_eval(fields['obs_normalization']))


class ShauryaHookTests(unittest.TestCase):
    def test_flat_observation_keeps_66_or_68_values_and_clock_action_order(self):
        cls = hook_class()
        for clock in (False, True):
            env = cls.__new__(cls)
            env.cfg = FlatCfg()
            env.cfg.gait_clock_enabled, env.cfg.gait_period_s, env.cfg.action_scale = clock, .8, .35
            env.step_dt = .02
            env._gait_phase = torch.tensor([0., math.pi])
            env._commands = torch.tensor([[.1, .2, .3], [-.1, -.2, -.3]])
            env._robot = SimpleNamespace(data=SimpleNamespace(
                root_lin_vel_b=array(torch.ones(2, 3)), root_ang_vel_b=array(torch.ones(2, 3)*2),
                projected_gravity_b=array(torch.tensor([[0., 0., -1.]]).repeat(2, 1)),
                joint_pos=array(torch.ones(2, 18)*.4), default_joint_pos=array(torch.ones(2, 18)*.1),
                joint_vel=array(torch.ones(2, 18)*.2)))
            actions = torch.arange(36, dtype=torch.float32).reshape(2, 18)/100
            env._pre_physics_step(actions)
            torch.testing.assert_close(env._processed_actions, actions*.35+.1, rtol=0, atol=0)
            result = env._get_observations()['policy']
            self.assertEqual(result.shape, (2, 68 if clock else 66))
            torch.testing.assert_close(result[:, 9:12], env._commands, rtol=0, atol=0)
            torch.testing.assert_close(result[:, 48:66], actions, rtol=0, atol=0)
            torch.testing.assert_close(env._previous_actions, actions, rtol=0, atol=0)
            expected_phase = torch.tensor([0., math.pi])
            if clock:
                expected_phase = torch.remainder(expected_phase+2*math.pi*.02/.8, 2*math.pi)
                torch.testing.assert_close(result[:, -2:], torch.stack(
                    (torch.sin(expected_phase), torch.cos(expected_phase)), dim=1), rtol=0, atol=0)
            torch.testing.assert_close(env._gait_phase, expected_phase, rtol=0, atol=0)

    def test_selected_reset_preserves_other_rows_and_the_original_rng_order(self):
        cls = hook_class()
        for forward_only in (False, True):
            env = cls.__new__(cls)
            env.num_envs, env.device = 3, 'cpu'
            env.cfg = SimpleNamespace(command_lin_vel_max=.3, command_yaw_rate_max=.3,
                command_forward_only=forward_only, command_lin_vel_min=.02,
                vel_xy_success_ratio=.5, vel_yaw_success_ratio=.4)
            env.max_episode_length, env.max_episode_length_s = 1000, 20.
            env.episode_length_buf = torch.tensor([10, 20, 30])
            env.reset_terminated, env.reset_time_outs = torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)
            env._actions, env._previous_actions = torch.ones(3, 18), torch.ones(3, 18)
            env._commands, env._gait_phase = torch.ones(3, 3), torch.ones(3)
            env._episode_sums, env.extras = {}, {}
            for name in ('_error_xy_sum', '_error_yaw_sum', '_step_count', '_root_z_sum',
                         '_speed_xy_sum', '_swing_sum', '_torque_sat_sum'):
                setattr(env, name, torch.ones(3))
            writes = {}
            def record(name):
                def call(**kwargs):
                    writes[name] = {key: value.clone() for key, value in kwargs.items()}
                return call
            pose = torch.zeros(3, 7)
            pose[:, 2], pose[:, 6] = .103, 1.
            env._robot = SimpleNamespace(_ALL_INDICES=torch.arange(3), reset=lambda ids: None,
                data=SimpleNamespace(default_joint_pos=array(torch.zeros(3, 18)),
                    default_joint_vel=array(torch.zeros(3, 18)), default_root_pose=array(pose),
                    default_root_vel=array(torch.zeros(3, 6))),
                write_root_pose_to_sim_index=record('pose'), write_root_velocity_to_sim_index=record('velocity'),
                write_joint_position_to_sim_index=record('position'), write_joint_velocity_to_sim_index=record('rate'))
            env._terrain = SimpleNamespace(env_origins=torch.tensor([[0., 0., 0.], [4., 0., 0.], [8., 0., 0.]]))
            selected = torch.tensor([1])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20260917)
                env._reset_idx(selected)
                observed_rng = torch.get_rng_state()
                torch.manual_seed(20260917)
                phase = torch.rand(1)*2*math.pi
                if forward_only:
                    magnitude = torch.zeros(1).uniform_(.02, .3)
                    sign = torch.where(torch.rand(1) < .5, -1., 1.)
                    xy = torch.stack((torch.zeros(1), magnitude*sign), dim=1)
                else:
                    xy = torch.zeros(1, 2).uniform_(-.3, .3)
                yaw = torch.zeros(1).uniform_(-.3, .3)
                torch.testing.assert_close(observed_rng, torch.get_rng_state(), rtol=0, atol=0)
            torch.testing.assert_close(env._commands[selected], torch.cat((xy, yaw[:, None]), dim=1), rtol=0, atol=0)
            torch.testing.assert_close(env._gait_phase[selected], phase, rtol=0, atol=0)
            torch.testing.assert_close(env._commands[[0, 2]], torch.ones(2, 3), rtol=0, atol=0)
            torch.testing.assert_close(env._actions[selected], torch.zeros(1, 18), rtol=0, atol=0)
            torch.testing.assert_close(env._actions[[0, 2]], torch.ones(2, 18), rtol=0, atol=0)
            torch.testing.assert_close(env._step_count, torch.tensor([1., 0., 1.]), rtol=0, atol=0)
            torch.testing.assert_close(env.base_reset_ids, selected, rtol=0, atol=0)
            torch.testing.assert_close(writes['pose']['root_pose'][0, :3], torch.tensor([4., 0., .103]), rtol=0, atol=0)

    def test_done_uses_base_contact_and_stock_timeout_boundary(self):
        cls = hook_class()
        env = cls.__new__(cls)
        env.episode_length_buf, env.max_episode_length = torch.tensor([998, 999, 0]), 1000
        forces = torch.zeros(3, 3, 2, 3)
        forces[2, 1, 0, 2] = 1.01
        forces[0, 0, 0, 2] = 1.
        env._base_id = [0]
        env._contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=array(forces)))
        died, timeout = env._get_dones()
        torch.testing.assert_close(died, torch.tensor([False, False, True]), rtol=0, atol=0)
        torch.testing.assert_close(timeout, torch.tensor([False, True, False]), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
