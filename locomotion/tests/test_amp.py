"""AMP feature contract, discriminator, Eq. (1) loss, Eq. (2) style reward and their use in candidate rewards."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from locomotion import amp, paper_reward, reward_scorer
from locomotion.amp import extract_features, feature_contract
from locomotion.env import LocomotionEnv, inverse_rotate
from locomotion.task import TaskConfig

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT/'locomotion/tests/fixtures/control_trace.npz'
SMALL = amp.TrainConfig(steps=150, batch_size=64, hidden=(16,), learning_rate=1e-2)


class AMPFeatureTests(unittest.TestCase):
    def state(self):
        return {'q': np.arange(36, dtype=np.float32).reshape(2, 18),
                'dq': np.full((2, 18), 2., np.float32),
                'linear': np.full((2, 3), 3., np.float32),
                'angular': np.full((2, 3), 4., np.float32),
                'root': np.tile(np.array([1, 2, 5, 0, 0, 0, 1], np.float32), (2, 1)),
                'toe_body': np.arange(36, dtype=np.float32).reshape(2, 6, 3)}

    def test_native_policy_and_numpy_export_preserve_historical_layout(self):
        state = self.state()
        expected = np.concatenate((state['q'], state['dq'], state['linear'], state['angular'],
                                   state['root'][:, 2:3], state['toe_body'].reshape(2, 18)), axis=1)
        env = LocomotionEnv.__new__(LocomotionEnv)
        env.cfg = type('Config', (), {'record_motion_features': True})()
        env.history = torch.zeros(2, 5, 42)
        env.commands = torch.zeros(2, 3)
        env.previous_action = torch.zeros(2, 18)
        tensors = {name: torch.from_numpy(value) for name, value in state.items()}
        observed = env._observations(tensors)['amp']
        np.testing.assert_array_equal(observed.numpy(), expected)
        np.testing.assert_array_equal(extract_features(state), expected)
        self.assertEqual(observed.dtype, torch.float32)
        self.assertEqual(observed.shape, (2, 61))

    def test_contract_has_absolute_joints_and_all_toe_coordinates(self):
        contract = feature_contract()
        self.assertEqual(contract['fields'][0]['frame'], 'native_joint_absolute')
        self.assertEqual(contract['fields'][-1]['slice'], [43, 61])
        self.assertEqual(contract['leg_order'], ['lf', 'lm', 'lr', 'rf', 'rm', 'rr'])
        contract['fields'].clear()
        self.assertEqual(len(feature_contract()['fields']), 6)

    def test_six_heights_cannot_substitute_for_six_positions(self):
        state = self.state()
        state['toe_body'] = state['toe_body'][:, :, 2]
        with self.assertRaisesRegex(ValueError, 'toe_body'):
            extract_features(state)


def linear_discriminator(weight, mean=0., std=1.):
    discriminator = amp.Discriminator(torch.full((2*amp.AMP_WIDTH,), mean), torch.full((2*amp.AMP_WIDTH,), std), hidden=())
    with torch.no_grad():
        discriminator.net[0].weight.copy_(torch.as_tensor(weight, dtype=torch.float32).reshape(1, -1))
        discriminator.net[0].bias.zero_()
    return discriminator


class StyleRewardTests(unittest.TestCase):
    def test_equation_two(self):
        scores = torch.tensor([1., 0., 2., -1., 3., -5.])
        torch.testing.assert_close(amp.style_reward(scores), torch.tensor([1., .75, .75, 0., 0., 0.]))

    def test_default_architecture_follows_table_three(self):
        discriminator = amp.Discriminator(torch.zeros(122), torch.ones(122))
        sizes = [layer.out_features for layer in discriminator.net if isinstance(layer, torch.nn.Linear)]
        self.assertEqual(sizes, [1024, 512, 1])
        self.assertEqual(discriminator.net[0].in_features, 2 * amp.AMP_WIDTH)


class LossTests(unittest.TestCase):
    def test_least_squares_targets(self):
        discriminator = linear_discriminator(torch.zeros(122))
        with torch.no_grad():
            discriminator.net[0].bias.fill_(.5)
        _, terms = amp.discriminator_loss(discriminator, torch.zeros(4, 122), torch.zeros(4, 122))
        self.assertAlmostEqual(terms['prior'], .25, places=6)
        self.assertAlmostEqual(terms['policy'], 2.25, places=6)
        self.assertAlmostEqual(terms['gradient_penalty'], 0., places=9)

    def test_gradient_penalty_is_taken_with_respect_to_standardized_prior_inputs(self):
        weight = torch.linspace(-1, 1, 122)
        discriminator = linear_discriminator(weight, std=2.)
        _, terms = amp.discriminator_loss(discriminator, torch.randn(8, 122), torch.randn(8, 122), gradient_penalty=10.)
        self.assertAlmostEqual(terms['gradient_penalty'], .5 * 10 * float(weight.square().sum()), places=4)

    def test_gradient_penalty_ignores_feature_scale(self):
        weight = torch.linspace(-1, 1, 122)
        prior, policy = torch.randn(8, 122), torch.randn(8, 122)
        unit = amp.discriminator_loss(linear_discriminator(weight, std=1.), prior, policy)[1]['gradient_penalty']
        tiny = amp.discriminator_loss(linear_discriminator(weight, std=1e-4), prior * 1e-4, policy * 1e-4)[1]['gradient_penalty']
        self.assertAlmostEqual(unit, tiny, places=4)
        self.assertAlmostEqual(unit, .5 * 10 * float(weight.square().sum()), places=4)


class TrainingTests(unittest.TestCase):
    def data(self):
        generator = torch.Generator().manual_seed(1)
        prior = torch.randn(200, 122, generator=generator) + 2
        policy = torch.randn(200, 122, generator=generator) - 2
        return prior, policy

    def test_training_separates_prior_from_policy_transitions(self):
        prior, policy = self.data()
        discriminator, history = amp.train(prior, policy, SMALL)
        with torch.no_grad():
            self.assertGreater(float(amp.style_reward(discriminator(prior)).mean()), .8)
            self.assertLess(float(amp.style_reward(discriminator(policy)).mean()), .2)
        self.assertEqual(history[-1]['step'], SMALL.steps - 1)

    def test_training_is_deterministic(self):
        prior, policy = self.data()
        first, _ = amp.train(prior, policy, SMALL)
        second, _ = amp.train(prior, policy, SMALL)
        for (name, a), (_, b) in zip(first.state_dict().items(), second.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)

    def test_checkpoint_round_trip(self):
        prior, policy = self.data()
        discriminator, _ = amp.train(prior, policy, SMALL)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'disc.pt'
            amp.save(discriminator, path, {'config': {'hidden': list(SMALL.hidden)}, 'note': 'x'})
            loaded, metadata = amp.load(path)
        with torch.no_grad():
            torch.testing.assert_close(loaded(prior), discriminator(prior))
        self.assertEqual(metadata['note'], 'x')


class RecordedTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = reward_scorer.load_trace(TRACE)
        cls.com = reward_scorer.root_com_local()

    def test_amp_velocity_columns_match_the_recorded_state(self):
        after = torch.as_tensor(self.trace.data['amp_state_after'][:, 0], dtype=torch.float64)
        joint, linear, angular = (after[:, s] for s in reward_scorer.AMP_VELOCITY_SLICES)
        torch.testing.assert_close(joint, torch.as_tensor(self.trace.data['joint_velocity_rad_s'][:, 0], dtype=torch.float64))
        torch.testing.assert_close(angular, torch.as_tensor(self.trace.data['gyro_body_rad_s'][:, 0]), rtol=0, atol=1e-6)
        nav = reward_scorer.origin_velocity_nav(self.trace, self.com)[:, 0]
        body = torch.stack((nav[:, 1], -nav[:, 0], nav[:, 2]), dim=-1)
        torch.testing.assert_close(linear, body, rtol=0, atol=1e-6)

    def test_style_is_added_to_the_calibrated_reward(self):
        prior = amp.trace_transitions(self.trace)
        discriminator, _ = amp.train(prior, prior + 1, SMALL)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'disc.pt'
            amp.save(discriminator, path, {'config': {'hidden': list(SMALL.hidden)}})
            combined = reward_scorer.load_reward(f'locomotion.amp:calibrated_with_style:{path}')
            reward, components = reward_scorer.evaluate(self.trace, combined, TaskConfig(), None, self.com)
            base, base_components = reward_scorer.evaluate(self.trace, paper_reward.paper_reward_calibrated,
                                                           TaskConfig(), None, self.com)
            still, still_components = reward_scorer.evaluate(self.trace, combined, TaskConfig(), None, self.com, motionless=True)
        self.assertEqual(set(components), set(base_components) | {'style'})
        torch.testing.assert_close(reward - components['style'], base)
        self.assertTrue(bool(((components['style'] >= 0) & (components['style'] <= 1)).all()))
        self.assertTrue(bool(torch.isfinite(still).all()))

    def test_motionless_transitions_hold_pose_with_zero_velocity(self):
        telemetry, _, _ = reward_scorer.reward_inputs(self.trace, self.com, motionless=True)
        torch.testing.assert_close(telemetry['amp_state'], telemetry['next_amp_state'])
        for columns in reward_scorer.AMP_VELOCITY_SLICES:
            self.assertTrue(bool((telemetry['amp_state'][:, columns] == 0).all()))
        recorded = torch.as_tensor(self.trace.data['amp_state_before'][1:, 0])
        torch.testing.assert_close(telemetry['amp_state'][:, :18], recorded[:, :18])


if __name__ == '__main__':
    unittest.main()
