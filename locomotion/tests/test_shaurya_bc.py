"""Compare Shaurya's retained tools with their pinned Git source on CPU."""
import ast
from contextlib import redirect_stdout
import copy
import io
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

REVISION = '969ea5bc942df3122b2662650f29acd11644a41c'
ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'locomotion/shaurya/tools'


def original(name):
    return subprocess.check_output(['git', 'show', f'{REVISION}:hexapod/tools/{name}'], cwd=ROOT).decode()


def tree(source):
    parsed = ast.parse(source)
    parsed.body = parsed.body[1:]  # The module examples use the retained package's path.
    for node in ast.walk(parsed):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('locomotion.shaurya.'):
            node.module = node.module.replace('locomotion.shaurya.', 'hexapod.', 1)
        if isinstance(node, ast.Constant) and node.value == 'logs/bc':
            node.value = '/workspace/learning/logs/bc'
    return parsed


def definitions(source, names, namespace):
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<retained BC>', 'exec'), namespace)


def assigned(source, names, namespace):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<retained BC>', 'exec'), namespace)


class ShauryaBCTests(unittest.TestCase):
    def exact(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.exact(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.exact(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_tools_preserve_source_ast_except_namespace_and_output_path(self):
        for name in ('bc_clone.py', 'diag_policy_rollout.py'):
            with self.subTest(tool=name):
                self.assertEqual(ast.dump(tree((TOOLS / name).read_text())), ast.dump(tree(original(name))))

    def test_teacher_preserves_phase_direction_and_joint_order(self):
        names = [f'{leg}_{joint}' for leg in ('lf', 'lm', 'lr', 'rf', 'rm', 'rr')
                 for joint in ('coxa_yaw', 'femur_pitch', 'tibia_pitch')][::-1]
        env = SimpleNamespace(_robot=SimpleNamespace(joint_names=names), device='cpu', num_envs=4,
            cfg=SimpleNamespace(command_lin_vel_max=.15),
            _commands=torch.tensor([[0., -.15, 0.], [0., .15, 0.], [0., 0., 0.], [0., -.1, 0.]]))
        phase = torch.tensor([0., math.pi / 2, math.pi, 5 * math.pi / 3])
        results = []
        for source in (original('bc_clone.py'), (TOOLS / 'bc_clone.py').read_text()):
            scope = {'torch': torch, 'math': math, 'HexapodEnv': object}
            definitions(source, {'ScriptedTripod'}, scope)
            results.append(scope['ScriptedTripod'](env).delta(env, phase))
        self.exact(results[1], results[0])
        self.assertEqual(float(results[0][2].abs().sum()), 0.)

    def test_fit_preserves_68_input_actor_optimizer_and_rng(self):
        results = []
        for source in (original('bc_clone.py'), (TOOLS / 'bc_clone.py').read_text()):
            torch.manual_seed(41)
            scope = {'torch': torch, 'nn': nn, 'obs_dim': 68, 'act_dim': 18, 'dev': 'cpu',
                     'args': SimpleNamespace(epochs=2)}
            assigned(source, {'mlp', 'opt'}, scope)
            definitions(source, {'fit'}, scope)
            observations, labels = torch.randn(40, 68), torch.randn(40, 18)
            with redirect_stdout(io.StringIO()):
                loss = scope['fit'](observations, labels, 'CPU differential')
            actor = scope['mlp']
            self.assertEqual([layer.out_features for layer in actor if isinstance(layer, nn.Linear)],
                             [128, 128, 128, 18])
            results.append((copy.deepcopy(actor.state_dict()), copy.deepcopy(scope['opt'].state_dict()),
                            loss, torch.get_rng_state()))
        self.exact(results[1], results[0])

    def test_collection_and_dagger_keep_labels_noise_and_original_speed_reference(self):
        results = []
        for source in (original('bc_clone.py'), (TOOLS / 'bc_clone.py').read_text()):
            torch.manual_seed(17)
            commands = torch.tensor([[0., -.1, 0.]])
            velocity = torch.tensor([[0., 1., 0.]])
            env = SimpleNamespace(_commands=commands,
                _robot=SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=SimpleNamespace(torch=velocity))))
            observation = torch.zeros(1, 68)
            env.reset = lambda: ({'policy': observation.clone()}, {})
            def step(action):
                commands[:, 1] = .1
                observation[:, :18] = action
                return {'policy': observation.clone()}, None, torch.tensor([False]), None, None
            env.step = step
            scope = {'torch': torch, 'env': env, 'teacher_action': lambda: commands[:, 1:].repeat(1, 9)}
            definitions(source, {'rollout'}, scope)
            teacher = scope['rollout'](None, 102, .05)
            dagger = scope['rollout'](lambda value: value[:, :18] * .5, 102, .05)
            self.assertEqual(teacher[2], -1.)  # The original tool retains its reset-time direction.
            self.assertEqual(dagger[2], 1.)
            self.assertFalse(torch.equal(teacher[0][1, :18], teacher[1][0]))
            results.append((teacher, dagger, torch.get_rng_state()))
        self.exact(results[1], results[0])

    def test_splice_preserves_critic_and_clears_optimizer_moments(self):
        results = []
        for source in (original('bc_clone.py'), (TOOLS / 'bc_clone.py').read_text()):
            with tempfile.TemporaryDirectory() as directory:
                checkpoint_path = str(Path(directory) / 'model_0.pt')
                actor = {'mlp.0.weight': torch.arange(12.).reshape(3, 4),
                         'distribution.std_param': torch.full((18,), .1)}
                checkpoint = {'actor_state_dict': {key: torch.zeros_like(value) for key, value in actor.items()},
                    'critic_state_dict': {'weight': torch.tensor([7.])},
                    'optimizer_state_dict': {'state': {0: {'step': torch.tensor(3.)}},
                                             'param_groups': [{'lr': .001}]}, 'iter': 17}
                torch.save(checkpoint, checkpoint_path)
                scope = {'torch': torch, 'os': __import__('os'), 'actor': actor,
                         'args': SimpleNamespace(splice_into=checkpoint_path)}
                nodes = [node for node in ast.parse(source).body if isinstance(node, ast.If)
                         and isinstance(node.test, ast.Attribute) and node.test.attr == 'splice_into']
                with redirect_stdout(io.StringIO()):
                    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<retained splice>', 'exec'), scope)
                result = torch.load(Path(directory) / 'model_bc.pt', weights_only=False)
                self.exact(result['actor_state_dict'], actor)
                self.exact(result['critic_state_dict'], checkpoint['critic_state_dict'])
                self.assertEqual(result['optimizer_state_dict']['state'], {})
                self.assertEqual(result['iter'], 0)
                results.append(result)
        self.exact(results[1], results[0])


if __name__ == '__main__':
    unittest.main()
