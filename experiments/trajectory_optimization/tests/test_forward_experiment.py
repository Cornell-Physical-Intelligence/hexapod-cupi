"""Check pairing boundaries, demonstration identity and the real PPO adapter."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.trajectory_optimization.forward_experiment import (
    forward_task, initialize, load_protocol, sha, state_digest,
)
from experiments.trajectory_optimization.prepare_forward import create_protocol, prepare_arm
from locomotion.ppo import ppo_config


class ForwardProtocolTests(unittest.TestCase):
    def test_bound_data_and_budget_reject_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'experiment'
            protocol = create_protocol(output)
            path = output/'PROTOCOL.json'
            digest = sha(path)
            _, arrays = load_protocol(path, digest, arm='example', seed=protocol['seed'], updates=1200)
            self.assertEqual(arrays['policy'].shape, (1000, 231))
            with self.assertRaises(ValueError):
                load_protocol(path, digest, arm='example', seed=protocol['seed'], updates=300)
            with (output/'demonstration.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                load_protocol(path, digest, arm='example', seed=protocol['seed'], updates=1200)

    def test_paired_packs_differ_only_in_arm_and_output_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_protocol(root/'protocol')
            declared = json.loads((Path(__file__).resolve().parents[3]/'configs/locomotion_spark.json').read_text())
            declared['extra_mounts'].append(['/recorded/standing', '/standing_one'])
            (root/'inputs.json').write_text(json.dumps(declared))
            bindings = []
            for arm in ('scratch', 'example'):
                remote = '/home/orionh/HEXAPOD_runs/restart_20260914/test_forward/'+arm
                bindings.append(prepare_arm(root/'protocol/PROTOCOL.json', root/arm, remote,
                                            arm=arm, inputs=root/'inputs.json'))
            a, b = bindings
            self.assertIn(['/recorded/standing', '/standing_one'], a['extra_mounts'])
            self.assertEqual(a['source_freeze_sha256'], b['source_freeze_sha256'])
            self.assertEqual(a['command_args'][:-1], b['command_args'][:-1])
            self.assertEqual(a['command_args'][-1], 'scratch')
            self.assertEqual(b['command_args'][-1], 'example')
            self.assertEqual((root/'scratch/experiment/demonstration.npz').read_bytes(),
                             (root/'example/experiment/demonstration.npz').read_bytes())
            from locomotion import launch, reservation
            paths = {key: Path(a[key]) for key in ('source', 'output', 'asset', 'prior', 'geometry_source')}
            with patch.object(reservation, 'canonical_path', side_effect=Path):
                command = launch.command(a, paths, 'test_container')
                self.assertIn(a['extra_mounts'][-1][0]+':/realized_prior:ro', command)
                invalid = copy.deepcopy(a)
                invalid['extra_mounts'][-1][1] = '/unbound'
                with self.assertRaises(ValueError):
                    launch.command(invalid, paths, 'test_container')

    def test_forward_sampler_preserves_unselected_rows(self):
        class Base:
            def declaration(self):
                return {'reward_version': 'preserved'}
        task = forward_task(Base)()
        task.env = SimpleNamespace(commands=torch.full((3, 3), -7.))
        task.remaining_controls = torch.tensor([8, 9, 10])
        task.command_draws = 4
        task._resample(torch.tensor([0, 2]))
        self.assertTrue(torch.equal(task.env.commands[1], torch.full((3,), -7.)))
        self.assertEqual(task.remaining_controls.tolist(), [1000, 9, 1000])
        self.assertEqual(task.command_draws, 6)
        self.assertEqual(task.declaration()['reward_version'], 'preserved')


@unittest.skipUnless(importlib.util.find_spec('rsl_rl'), 'Run with --with rsl-rl-lib==5.0.1 for upstream integration')
class ForwardUpstreamTests(unittest.TestCase):
    def test_actor_only_initialization_and_real_ppo_updates_reload(self):
        from rsl_rl.runners import OnPolicyRunner
        from tensordict import TensorDict
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = create_protocol(root/'protocol')
                path = root/'protocol/PROTOCOL.json'
                _, arrays = load_protocol(path, sha(path), arm='scratch', seed=protocol['seed'], updates=1200)
                class ReplayAdapter:
                    """Exercise learner plumbing with recorded observations, without simulated physics."""
                    num_envs, num_actions, device, max_episode_length = 128, 18, 'cpu', 1000
                    cfg = {'test_fixture': 'recorded observations; no locomotion claim'}
                    def __init__(self):
                        self.step_index = 0
                        self.episode_length_buf = torch.zeros(128, dtype=torch.long)
                    def get_observations(self):
                        index = (torch.arange(128)+self.step_index) % 1000
                        return TensorDict({k: torch.from_numpy(arrays[k])[index] for k in ('policy', 'critic')}, [128])
                    def step(self, action):
                        self.step_index += 1
                        self.episode_length_buf += 1
                        return self.get_observations(), -action.square().mean(-1), torch.zeros(128, dtype=torch.long), {}
                receipts = []
                for arm in ('scratch', 'example'):
                    torch.manual_seed(protocol['seed'])
                    runner = OnPolicyRunner(ReplayAdapter(), copy.deepcopy(ppo_config(protocol['seed'])), device='cpu')
                    rng = torch.get_rng_state().clone()
                    receipt = initialize(runner, arrays, protocol, arm)
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                    self.assertFalse(runner.alg.optimizer.state)
                    if arm == 'example':
                        self.assertLess(receipt['after']['validation_action_mse'], receipt['before']['validation_action_mse'])
                    initial_checkpoint = root/(arm+'_zero.pt')
                    runner.save(str(initial_checkpoint), infos={'updates': 0})
                    initial = torch.load(initial_checkpoint, weights_only=False)
                    self.assertEqual(initial['iter'], 0)
                    self.assertEqual(initial['infos'], {'updates': 0})
                    self.assertEqual(state_digest(initial['actor_state_dict']), receipt['final_actor_sha256'])
                    runner.learn(2, init_at_random_ep_len=False)
                    steps = [int(state['step']) for state in runner.alg.optimizer.state.values()]
                    self.assertTrue(steps and all(step == 40 for step in steps))
                    checkpoint = root/(arm+'.pt')
                    runner.save(str(checkpoint), infos={'scope': 'upstream interface test'})
                    saved = state_digest(runner.alg.actor.state_dict())
                    reloaded = OnPolicyRunner(ReplayAdapter(), copy.deepcopy(ppo_config(protocol['seed'])), device='cpu')
                    reloaded.load(str(checkpoint), strict=True)
                    self.assertEqual(saved, state_digest(reloaded.alg.actor.state_dict()))
                    receipts.append(receipt)
                for key in ('initial_actor_sha256', 'shared_actor_non_mlp_sha256', 'shared_critic_sha256', 'before'):
                    self.assertEqual(receipts[0][key], receipts[1][key])
                self.assertNotEqual(receipts[0]['final_actor_sha256'], receipts[1]['final_actor_sha256'])
                print('FORWARD_PAIR_INTERFACE_RESULT='+json.dumps(receipts, sort_keys=True))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
