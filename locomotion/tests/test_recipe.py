"""Check that the c48_long recipe reproduces the learner and reward options of its compatibility receipt."""
from dataclasses import fields, replace
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from locomotion.evaluation.checkpoint import CLASS_NAMES
from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import RECIPES, ppo_config
from locomotion.omniwalk.mdp.commands import TaskConfig
from locomotion.omniwalk.mdp.rewards import RewardConfig
from locomotion.omniwalk.task import TrainingTask
import locomotion.scripts.diagnose as diagnose
import locomotion.scripts.play as play
import locomotion.scripts.train as train
from locomotion.spark.prepare import prepare
from locomotion.spark.validation import OPTION_DEFAULTS
from locomotion.tests.task_double import GaitDouble

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT/'site/assets/package_cutover_20261015_001/compatibility/receipt.json'
REMOTE = '/srv/cupi/hexapod/runs/james/recipe_fixture'
ENTRY = ['--asset', 'a', '--model', 'm', '--geometry', 'g', '--geometry-extrema', 'e', '--stance', 's',
         '--output', 'o', '--source-freeze-sha256', 'f'*64, '--num-envs', '128', '--preflight-only']


def recorded_identity():
    """The checkpoint identity of the receipt, with each recorded class path mapped to its current class."""
    return json.loads(RECEIPT.read_text(), object_hook=lambda row: {
        key: CLASS_NAMES.get(value, value) if key == 'class_name' else value
        for key, value in row.items()})['original_identity']


def parsed_options(script, argv):
    args = script.parse_args(argv)
    return {key: getattr(args, key) for key in OPTION_DEFAULTS}


class RecipeTests(unittest.TestCase):
    def test_recipe_reproduces_the_recorded_learner_configuration(self):
        self.assertEqual(ppo_config(20260917, **RECIPES['c48_long']), recorded_identity()['ppo_config'])

    def test_reward_defaults_equal_the_recorded_reward_options(self):
        recorded = dict(item.split('=') for item in recorded_identity()['reward_options'].split(','))
        # Reward v4 made this share of moving draws a full-speed forward command; 0 kept the bank draw, the only sampler left.
        self.assertEqual(float(recorded.pop('forward_draw_fraction')), 0.)
        self.assertNotIn('forward_draw_fraction', {field.name for field in fields(RewardConfig)})
        for key, value in recorded.items():
            with self.subTest(key=key):
                self.assertEqual(getattr(RewardConfig(), key), json.loads(value))

    def test_forward_draw_fraction_zero_keeps_the_bank_draw(self):
        env = GaitDouble(64)
        sampler = TrainingTask(env, TaskConfig(seed=7), reward_config=replace(RewardConfig(), turn_draw_fraction=0.))
        replay = torch.Generator().set_state(sampler.rng.get_state())
        sampler.reset()
        drawn = sampler.bank[torch.randint(len(sampler.bank), (64,), generator=replay)]
        moving = env.commands.ne(0).any(-1)
        self.assertGreater(int(moving.sum()), 0)
        torch.testing.assert_close(env.commands[moving], drawn[moving], atol=0, rtol=0)

    def test_command_lines_start_ppo_training_and_evaluation_from_the_recipe(self):
        recipe = RECIPES['c48_long']
        for script in (train, play):
            self.assertEqual(parsed_options(script, ENTRY), recipe)
        self.assertEqual(parsed_options(train, ENTRY + ['--steps-per-env', '24']), {**recipe, 'steps_per_env': 24})
        self.assertEqual(parsed_options(train, ENTRY + ['--recipe', 'none']), {**OPTION_DEFAULTS, 'gait_clock': 60})
        self.assertEqual(parsed_options(diagnose, ENTRY), OPTION_DEFAULTS)
        with tempfile.TemporaryDirectory() as directory:
            for index, (options, expected) in enumerate((
                    ({}, recipe), ({'recipe': 'none'}, {**OPTION_DEFAULTS, 'gait_clock': 60}),
                    ({'action_mean': 'unbounded'}, {**recipe, 'action_mean': 'unbounded'}))):
                with self.subTest(options=options):
                    binding = prepare(Path(directory)/f'pack{index}', REMOTE, **options)
                    # The pack's script resolves the pack's command line to the options the pack declares.
                    script = importlib.import_module(binding['module'])
                    self.assertEqual(parsed_options(script, binding['command_args'] + ['--output', 'o', '--preflight-only']),
                                     expected)


if __name__ == '__main__':
    unittest.main()
