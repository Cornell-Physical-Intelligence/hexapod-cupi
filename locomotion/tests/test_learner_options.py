"""Check the optional learner path: input scales, command segments, rate ceiling and video case."""
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO

from locomotion import ppo
from locomotion.prepare import prepare
from locomotion.rate_schedule import CappedRatePPO
from locomotion.train import require_learner_configuration

REMOTE = '/srv/cupi/hexapod/runs/james/learner_options_fixture'
OPTIONS = dict(observation_scaling='fixed', command_segments='bootstrap', learning_rate_max=3e-4, action_std=.1,
               action_noise_correlation=.9, action_std_final=.03, gait_clock=60)


class CommandTask:
    """Three replicas: the command of row 1 changes, row 2 terminates while its command changes."""
    num_envs = 3
    device = 'cpu'
    cfg = SimpleNamespace(episode_seconds=20., control_dt=.02, declaration=lambda: {})
    episode_steps = torch.zeros(3, dtype=torch.long)

    def __init__(self):
        self.commands = torch.tensor([[.05, 0., 0.]]*3)
        self.resets = []

    def declaration(self):
        return {}

    def output(self):
        obs = torch.ones(3, 231)
        obs[:, 210:213] = self.commands
        return {'obs': obs, 'critic': torch.cat((obs, torch.ones(3, 3)), -1)}

    def reset(self, selected=None):
        self.resets.append((torch.arange(3) if selected is None else selected).tolist())
        return self.output()

    def step(self, action):
        self.commands = torch.tensor([[.05, 0., 0.], [0., 0., .2], [0., 0., 0.]])
        return {**self.output(), 'reward': torch.tensor([1., 2., 3.]),
                'terminated': torch.tensor([False, False, True]), 'truncated': torch.zeros(3, dtype=torch.bool)}


def construct(config):
    obs = TensorDict({'policy': torch.zeros(8, 231), 'critic': torch.zeros(8, 234)}, batch_size=[8])
    config = dict(config, num_steps_per_env=4, multi_gpu=None)
    config.pop('environment_wrapper', None)
    config.pop('exploration', None)
    for role in ('actor', 'critic'):
        config[role] = dict(config[role], hidden_dims=[8])
    with redirect_stdout(io.StringIO()):
        return PPO.construct_algorithm(obs, SimpleNamespace(num_envs=8, num_actions=18), config, 'cpu')


class LearnerOptionTests(unittest.TestCase):
    def test_options_appear_in_the_configuration_only_when_selected(self):
        base = ppo.ppo_config(7)
        self.assertNotIn('environment_wrapper', base)
        self.assertEqual(base['algorithm']['class_name'], 'PPO')
        self.assertNotIn('learning_rate_max', base['algorithm'])
        self.assertEqual(base['actor']['distribution_cfg']['init_std'], .15)
        chosen = ppo.ppo_config(7, action_mean='tanh', **OPTIONS)
        self.assertEqual(chosen['environment_wrapper'],
                         {'observation_scaling': 'fixed', 'command_segments': 'bootstrap', 'gait_clock': 60})
        self.assertEqual(chosen['exploration'], {'action_std_final': .03})
        self.assertEqual(chosen['algorithm']['class_name'], ppo.RATE_CLASS)
        self.assertEqual(chosen['algorithm']['learning_rate'], 3e-4)
        self.assertEqual(chosen['actor']['distribution_cfg']['init_std'], .1)
        record = {'identity': {'upstream_source_files': {}, 'ppo_config': chosen}}
        require_learner_configuration(record, {'upstream_source_files': {}}, chosen)
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            require_learner_configuration(record, {'upstream_source_files': {}}, base)
        for bad in (dict(observation_scaling='unit'), dict(command_segments='cut'),
                    dict(learning_rate_max=1.), dict(learning_rate_max=1), dict(action_std=0.), dict(action_std=1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ppo.ppo_config(7, **bad)

    def test_fixed_scales_bring_each_input_group_to_unit_size(self):
        actor = torch.zeros(1, 231)
        actor[0, 0], actor[0, 3], actor[0, 6], actor[0, 24] = .5, 1., .35, 2.
        actor[0, 210:213] = torch.tensor([.05, -.05, .2])
        actor[0, 213] = .7
        scaled = ppo.scale_observation(actor, 'fixed')
        self.assertEqual([round(float(scaled[0, i]), 6) for i in (0, 3, 6, 24, 213)], [1., 1., 1., 1., .7])
        torch.testing.assert_close(scaled[0, 210:213], torch.tensor([1., -1., 1.]))
        critic = torch.cat((actor, torch.tensor([[.05, 0., -.05]])), -1)
        torch.testing.assert_close(ppo.scale_observation(critic, 'fixed', critic=True)[0, 231:], torch.tensor([1., 0., -1.]))
        self.assertIs(ppo.scale_observation(actor, 'none'), actor)
        # Every proprioception frame uses the same scales.
        self.assertEqual(ppo.ACTOR_SCALES[:42], ppo.ACTOR_SCALES[168:210])
        with self.assertRaises(ValueError):
            ppo.scale_observation(actor, 'fixed', critic=True)

    def test_bootstrap_segments_end_the_return_at_a_command_change_without_a_reset(self):
        plain_task, segment_task = CommandTask(), CommandTask()
        plain = ppo.VanillaVecEnv(plain_task, tensor_dict=lambda value, **kw: value)
        segments = ppo.VanillaVecEnv(segment_task, tensor_dict=lambda value, **kw: value,
                                     observation_scaling='fixed', command_segments='bootstrap')
        _, reward, done, extras = plain.step(torch.zeros(3, 18))
        self.assertEqual(done.tolist(), [0, 0, 1])
        self.assertEqual(extras['time_outs'].tolist(), [False, False, False])
        obs, reward, done, extras = segments.step(torch.zeros(3, 18))
        self.assertEqual(reward.tolist(), [1., 2., 3.])
        # Row 1 changed command: its return bootstraps. Row 2 terminated: no bootstrap, one reset.
        self.assertEqual(done.tolist(), [0, 1, 1])
        self.assertEqual(extras['time_outs'].tolist(), [False, True, False])
        self.assertEqual(segment_task.resets, [[0, 1, 2], [2]])
        torch.testing.assert_close(obs['policy'][1, 210:213], torch.tensor([0., 0., 1.]))
        self.assertEqual(obs['critic'].shape, (3, 234))

    def test_rate_ceiling_holds_through_the_stock_schedule(self):
        capped = construct(ppo.ppo_config(3, action_mean='tanh', observation_normalization='none', learning_rate_max=3e-4))
        self.assertIs(type(capped), CappedRatePPO)
        self.assertEqual(capped.learning_rate, 3e-4)
        self.assertEqual({group['lr'] for group in capped.optimizer.param_groups}, {3e-4})
        stock = construct(ppo.ppo_config(3, action_mean='tanh', observation_normalization='none'))
        self.assertIs(type(stock), PPO)
        rates = {}
        for name, algorithm in (('capped', capped), ('stock', stock)):
            torch.manual_seed(0)
            algorithm.num_learning_epochs, algorithm.num_mini_batches = 1, 2
            # Saturated tanh means: a parameter step barely moves the action distribution.
            for key, value in algorithm.actor.named_parameters():
                if value.shape == (18,) and 'std' not in key:
                    value.data.fill_(20.)
            obs = TensorDict({'policy': torch.zeros(8, 231), 'critic': torch.zeros(8, 234)}, batch_size=[8])
            with torch.inference_mode():
                for _ in range(4):
                    algorithm.act(obs)
                    algorithm.process_env_step(obs, torch.zeros(8), torch.zeros(8, dtype=torch.long), {})
                algorithm.compute_returns(obs)
            algorithm.update()
            rates[name] = algorithm.learning_rate
        # The stock schedule reads a small KL and raises its rate; the ceiling holds.
        self.assertEqual(rates['capped'], 3e-4)
        self.assertGreater(rates['stock'], 1e-3)

    def test_correlated_noise_keeps_the_gaussian_marginal_and_carries_noise_between_controls(self):
        from locomotion.action_distribution import BoundedMeanGaussian, CorrelatedBoundedMeanGaussian
        config = ppo.ppo_config(3, action_mean='tanh', action_noise_correlation=.9)
        self.assertEqual(config['actor']['distribution_cfg']['class_name'],
                         'locomotion.action_distribution:CorrelatedBoundedMeanGaussian')
        self.assertEqual(config['actor']['distribution_cfg']['noise_correlation'], .9)
        self.assertNotIn('noise_correlation', ppo.ppo_config(3, action_mean='tanh')['actor']['distribution_cfg'])
        for bad in (dict(action_noise_correlation=.9), dict(action_mean='tanh', action_noise_correlation=1.),
                    dict(action_mean='tanh', action_noise_correlation=1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ppo.ppo_config(3, **bad)
        self.assertIs(type(construct(config).actor.distribution), CorrelatedBoundedMeanGaussian)
        torch.manual_seed(5)
        distribution = CorrelatedBoundedMeanGaussian(2, init_std=.15, std_type='log', noise_correlation=.9)
        samples = []
        with torch.no_grad():
            for _ in range(3000):
                distribution.update(torch.full((32, 2), .3))
                samples.append(distribution.sample())
        noise = torch.stack(samples) - math.tanh(.3)
        self.assertAlmostEqual(float(noise.std()), .15, delta=.01)
        self.assertAlmostEqual(float((noise[1:] * noise[:-1]).mean() / noise.var()), .9, delta=.02)
        # The likelihood stays the Gaussian that PPO evaluates for the realized action.
        distribution.update(torch.full((32, 2), .3))
        action = distribution.sample()
        plain = BoundedMeanGaussian(2, init_std=.15, std_type='log')
        plain.update(torch.full((32, 2), .3))
        torch.testing.assert_close(distribution.log_prob(action), plain.log_prob(action))
        with self.assertRaises(ValueError):
            CorrelatedBoundedMeanGaussian(2, noise_correlation=1.)

    def test_deviation_schedule_replaces_the_learned_deviation(self):
        config = ppo.ppo_config(3, action_mean='tanh', observation_normalization='none', action_std_final=.03)
        self.assertEqual(config['exploration'], {'action_std_final': .03})
        self.assertNotIn('exploration', ppo.ppo_config(3))
        for bad in (dict(action_std_final=.2), dict(action_std_final=.001), dict(action_std_final=1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ppo.ppo_config(3, **bad)
        config.pop('exploration')
        algorithm = construct(config)
        calls = []
        algorithm.update = lambda: calls.append(float(algorithm.actor.distribution.log_std_param.exp()[0])) or {'value': 0.}
        schedule = ppo.DeviationSchedule(algorithm, .15, .03, 4)
        parameter = algorithm.actor.distribution.log_std_param
        self.assertFalse(parameter.requires_grad)
        for _ in range(6):
            self.assertEqual(schedule.update(), {'value': 0.})
        # Each update trains with the deviation its rollout used; the value holds after the last scheduled update.
        for seen, expected in zip(calls, (.15, .12, .09, .06, .03, .03)):
            self.assertAlmostEqual(seen, expected, places=6)
        self.assertAlmostEqual(float(parameter.exp()[0]), .03, places=6)

    def test_gait_clock_appends_the_episode_phase_to_actor_and_critic_inputs(self):
        self.assertEqual(ppo.ppo_config(3, gait_clock=60)['environment_wrapper'], {'gait_clock': 60})
        for bad in (dict(gait_clock=5), dict(gait_clock=60.), dict(gait_clock=300)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ppo.ppo_config(3, **bad)
        task = CommandTask()
        task.episode_steps = torch.tensor([0, 15, 75])
        wrapped = ppo.VanillaVecEnv(task, tensor_dict=lambda value, **kw: value, gait_clock=60)
        obs = wrapped.get_observations()
        self.assertEqual((obs['policy'].shape, obs['critic'].shape), ((3, 233), (3, 236)))
        expected = torch.tensor([[0., 1.], [1., 0.], [1., 0.]])
        torch.testing.assert_close(obs['policy'][:, 231:], expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(obs['critic'][:, 234:], expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(obs['policy'][:, :231], task.output()['obs'])
        plain = ppo.VanillaVecEnv(CommandTask(), tensor_dict=lambda value, **kw: value)
        self.assertEqual(plain.get_observations()['policy'].shape, (3, 231))
        moving = torch.tensor([[.05, 0., 0.], [0., 0., .2]])
        torch.testing.assert_close(ppo.clock_features(torch.tensor([30, 45]), 60, moving),
                                   torch.tensor([[0., -1.], [-1., 0.]]), atol=1e-6, rtol=0)
        # Under a zero command the clock input rests at zero.
        self.assertEqual(float(ppo.clock_features(torch.tensor([30, 45]), 60, torch.zeros(2, 3)).abs().sum()), 0.)
        task.commands = torch.tensor([[.05, 0., 0.], [0., 0., 0.], [0., 0., .2]])
        gated = wrapped.observations(task.output())['policy'][:, 231:]
        torch.testing.assert_close(gated, torch.tensor([[0., 1.], [0., 0.], [1., 0.]]), atol=1e-6, rtol=0)

    def test_action_smoothing_sends_the_two_control_mean_and_restarts_with_each_episode(self):
        self.assertEqual(ppo.ppo_config(3, action_smoothing='mean2')['environment_wrapper'], {'action_smoothing': 'mean2'})
        with self.assertRaises(ValueError):
            ppo.ppo_config(3, action_smoothing='mean3')
        mean = ppo.smoothed_action(torch.ones(2, 18), torch.full((2, 18), .4), torch.tensor([5, 0]))
        torch.testing.assert_close(mean[:, 0], torch.tensor([.7, .5]))
        task, received = CommandTask(), []
        step = task.step
        task.step = lambda action: (received.append(action.clone()), step(action))[1]
        wrapped = ppo.VanillaVecEnv(task, tensor_dict=lambda value, **kw: value, action_smoothing='mean2')
        # An action that alternates on consecutive controls reaches the task as a constant.
        for control, sign in enumerate((1., -1., 1., -1.)):
            task.episode_steps = torch.full((3,), control)
            wrapped.step(torch.full((3, 18), .2 * sign))
        torch.testing.assert_close(received[0], torch.full((3, 18), .1))
        for later in received[1:]:
            torch.testing.assert_close(later, torch.zeros(3, 18), atol=1e-7, rtol=0)
        # A new episode pairs its first action with the neutral action, as the evaluation policy does.
        task.episode_steps = torch.tensor([4, 0, 4])
        wrapped.step(torch.full((3, 18), .6))
        torch.testing.assert_close(received[-1][:, 0], torch.tensor([.2, .3, .2]))
        plain_task, direct = CommandTask(), []
        plain_step = plain_task.step
        plain_task.step = lambda action: (direct.append(action.clone()), plain_step(action))[1]
        ppo.VanillaVecEnv(plain_task, tensor_dict=lambda value, **kw: value).step(torch.full((3, 18), .6))
        torch.testing.assert_close(direct[0], torch.full((3, 18), .6))
        with tempfile.TemporaryDirectory() as directory:
            args = prepare(Path(directory)/'smooth', REMOTE, mode='train', action_mean='tanh',
                           action_smoothing='mean2')['command_args']
            self.assertEqual(args[args.index('--action-smoothing')+1], 'mean2')
            self.assertEqual(json.loads((Path(directory)/'smooth/PACK.json').read_text())['action_smoothing'], 'mean2')
            with self.assertRaises(ValueError):
                prepare(Path(directory)/'tripod', REMOTE, mode='tripod', action_smoothing='mean2')

    def test_value_metrics_score_the_critic_on_the_collected_rollout(self):
        storage = SimpleNamespace(returns=torch.tensor([[1.], [2.], [3.], [4.]]), values=torch.tensor([[1.], [2.], [3.], [4.]]))
        self.assertAlmostEqual(ppo.value_metrics(storage)['explained_variance'], 1.)
        storage.values = torch.full((4, 1), 2.5)
        self.assertAlmostEqual(ppo.value_metrics(storage)['explained_variance'], 0.)
        storage.returns = torch.ones(4, 1)
        self.assertIsNone(ppo.value_metrics(storage)['explained_variance'])

    def test_packs_declare_selected_options_and_reject_other_learners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = {'checkpoint': REMOTE+'/checkpoint.pt', 'checkpoint_sha': 'a'*64, 'checkpoint_declaration_sha': 'b'*64}
            for mode in ('train', 'evaluate'):
                extra = {} if mode == 'train' else checkpoint
                binding = prepare(root/mode, REMOTE, mode=mode, action_mean='tanh', **OPTIONS, **extra)
                args = binding['command_args']
                for flag, value in (('--observation-scaling', 'fixed'), ('--command-segments', 'bootstrap'),
                                    ('--learning-rate-max', '0.0003'), ('--action-std', '0.1'),
                                    ('--action-noise-correlation', '0.9'), ('--action-std-final', '0.03'),
                                    ('--gait-clock', '60')):
                    self.assertEqual(args[args.index(flag)+1], value)
                pack = json.loads((root/mode/'PACK.json').read_text())
                self.assertEqual({key: pack[key] for key in OPTIONS}, OPTIONS)
                self.assertIn('locomotion/rate_schedule.py', json.loads((root/mode/'source/FREEZE_SHA256.json').read_text()))
            legacy = prepare(root/'legacy', REMOTE)
            self.assertFalse({'--observation-scaling', '--command-segments', '--learning-rate-max', '--action-std',
                              '--action-noise-correlation', '--action-std-final', '--gait-clock', '--video-case'}
                             & set(legacy['command_args']))
            self.assertFalse(set(OPTIONS) & set(json.loads((root/'legacy/PACK.json').read_text())))
            short = prepare(root/'short', REMOTE, mode='train', episode_seconds=10.)
            self.assertEqual(short['command_args'][-2:], ['--episode-seconds', '10.0'])
            self.assertEqual(json.loads((root/'short/PACK.json').read_text())['episode_seconds'], 10.)
            self.assertNotIn('--episode-seconds', legacy['command_args'])
            video = prepare(root/'video', REMOTE, mode='evaluate', video_case='learning:forward_0.05_to_stop', **checkpoint)
            position = video['command_args'].index('--video-case')
            self.assertEqual(video['command_args'][position+1], 'learning:forward_0.05_to_stop')
            for index, bad in enumerate((dict(mode='diagnostic', observation_scaling='fixed'), dict(learner='amp', action_std=.1),
                    dict(mode='tripod', command_segments='bootstrap'), dict(learning_rate_max=1.),
                    dict(action_noise_correlation=.9), dict(gait_clock=5), dict(episode_seconds=2.), dict(episode_seconds=10),
                    dict(mode='evaluate', episode_seconds=10., **checkpoint),
                    dict(video_case='learning:quiet_20s'), dict(mode='evaluate', eval_scope='full', video_case='learning:quiet_20s', **checkpoint),
                    dict(mode='evaluate', video_case='static:stand', **checkpoint))):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    prepare(root/f'bad{index}', REMOTE, **bad)


if __name__ == '__main__':
    unittest.main()
