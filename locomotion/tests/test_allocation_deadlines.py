"""Check pilot deadline binding and checkpoint retention without native simulation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from locomotion import prepare, reservation, train
from locomotion.tests import test_launch


class AllocationDeadlineTests(unittest.TestCase):
    def test_frozen_amp_tree_binds_the_bank_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            for networks in ('mlp', 'paper'):
                pack = Path(directory)/networks
                binding = prepare.prepare(pack, str(prepare.REMOTE_ROOT/'deadline_fixture'),
                    learner='amp', networks=networks, reward_version='2')
                source = pack/'source'
                reservation.verify_tree(source, binding['source_freeze_sha256'])
                bank = source/'locomotion/priors/datasets/amp_demonstrations_001'
                for name in ('manifest.json', 'review.json', 'transitions.npz'):
                    path = bank/name
                    before = path.read_bytes()
                    path.write_bytes(before+b'changed')
                    with self.subTest(networks=networks, name=name), self.assertRaisesRegex(ValueError, 'frozen tree'):
                        reservation.verify_tree(source, binding['source_freeze_sha256'])
                    path.write_bytes(before)

    def test_prepare_propagates_pilot_deadline_and_preserves_standard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = str(prepare.REMOTE_ROOT/'deadline_fixture')
            standard = prepare.prepare(root/'standard', remote)
            self.assertEqual(standard['max_seconds'], 6600)
            for learner, networks in (('ppo', 'mlp'), ('amp', 'mlp'), ('amp', 'paper')):
                pack = root/(learner+'_'+networks)
                binding = prepare.prepare(pack, remote, updates=2000, reward_version='2',
                    learner=learner, networks=networks, allocation_profile='flat_pilot_v1',
                    max_wall_seconds=21600)
                self.assertEqual(binding['max_seconds'], 22000)
                args = binding['command_args']
                self.assertEqual(args[args.index('--max-wall-seconds')+1], '21600')
                self.assertEqual(args[args.index('--allocation-profile')+1], 'flat_pilot_v1')
                record = json.loads((pack/'PACK.json').read_text())
                self.assertEqual(record['max_wall_seconds'], 21600)
                self.assertEqual(record['allocation_profile'], 'flat_pilot_v1')
                test_launch.OwnedLaunchTests().verify_without_filesystem(binding)

    def test_profile_bounds_reject_invalid_deadlines_before_preparation(self):
        cases = [('standard', 'train', value) for value in (True, 0, -1, 6601, float('nan'), float('inf'))]
        cases += [('flat_pilot_v1', 'train', 21601), ('flat_pilot_v1', 'diagnostic', 6200),
                  ('flat_pilot_v1', 'throughput', 6200), ('unknown', 'train', 6200)]
        with tempfile.TemporaryDirectory() as directory:
            for profile, mode, seconds in cases:
                output = Path(directory)/'invalid'
                with self.subTest(profile=profile, mode=mode, seconds=seconds), self.assertRaises(ValueError):
                    prepare.prepare(output, str(prepare.REMOTE_ROOT/'deadline_fixture'), mode=mode,
                        allocation_profile=profile, max_wall_seconds=seconds)
                self.assertFalse(output.exists())

    def test_supervisor_rejects_deadline_tampering_and_lost_cleanup_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = prepare.prepare(Path(directory)/'pack', str(prepare.REMOTE_ROOT/'deadline_fixture'),
                allocation_profile='flat_pilot_v1', max_wall_seconds=21600)
            fixture = test_launch.OwnedLaunchTests()
            mutations = []
            for field, value in (('max_seconds', 21600), ('max_seconds', 22001),
                                 ('allocation_profile', 'standard'), ('mode', 'tripod')):
                binding = copy.deepcopy(valid)
                binding[field] = value
                mutations.append(binding)
            for option in ('--max-wall-seconds', '--allocation-profile'):
                for action in ('remove', 'change', 'duplicate', 'equals'):
                    binding = copy.deepcopy(valid)
                    args = binding['command_args']
                    index = args.index(option)
                    if action == 'remove':
                        del args[index:index+2]
                    elif action == 'change':
                        args[index+1] = 'standard' if option == '--allocation-profile' else '22000'
                    elif action == 'duplicate':
                        args += args[index:index+2]
                    else:
                        args += [option+'='+args[index+1]]
                    mutations.append(binding)
            for binding in mutations:
                with self.subTest(binding=binding), self.assertRaises(ValueError):
                    fixture.verify_without_filesystem(binding)

    def test_timeout_retains_last_complete_update_and_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []

            def checkpoint(update):
                events.append(('checkpoint', update))
                train.save(root/'checkpoint.json', {'updates': update})

            def loads():
                events.append(('loads',))
                train.save(root/'loads.json', {'samples': 800})

            train.finish_training_update(36, 2000, 21599, 21600, checkpoint, loads)
            self.assertEqual(events, [])
            with self.assertRaises(TimeoutError):
                train.finish_training_update(37, 2000, 21601, 21600, checkpoint, loads)
            self.assertEqual(events, [('checkpoint', 37), ('loads',)])
            self.assertEqual(json.loads((root/'checkpoint.json').read_text())['updates'], 37)
            self.assertEqual(json.loads((root/'loads.json').read_text())['samples'], 800)

    def test_timeout_at_save_interval_saves_once_and_final_update_completes(self):
        for update in (50, 2000):
            events = []
            call = lambda: train.finish_training_update(update, 2000, 21601, 21600,
                lambda number: events.append(number), lambda: events.append('loads'))
            if update < 2000:
                with self.assertRaises(TimeoutError):
                    call()
            else:
                call()
            self.assertEqual(events, [update, 'loads'])


if __name__ == '__main__':
    unittest.main()
