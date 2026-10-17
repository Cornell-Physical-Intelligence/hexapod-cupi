"""Check pack approval and the test-only parity pack against the launcher and comparison rules."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from locomotion.spark import pack
from locomotion.spark.safety import HOME_ROOT, verify_tree
from locomotion.spark.prepare import prepare
from locomotion.tests import native_parity_pack

CHECKPOINT = HOME_ROOT/'retained/checkpoint_update002500.pt'
WEIGHTS, DECLARATION = 'a'*64, 'b'*64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PackApprovalTests(unittest.TestCase):
    def test_approval_marks_the_review_and_keeps_the_pack_consistent(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'standing'
            prepare(output, HOME_ROOT/'attempt', mode='diagnostic', num_envs=1)
            before = json.loads((output/'binding.json').read_text())
            self.assertFalse(before['root_review_complete'])
            packed = pack.approve(output)
            binding = json.loads((output/'binding.json').read_text())
            self.assertTrue(binding['root_review_complete'])
            self.assertEqual(binding['source_freeze_sha256'], before['source_freeze_sha256'])
            self.assertEqual(packed['binding_sha256'], sha(output/'binding.json'))
            self.assertEqual(packed['files']['binding.json'], packed['binding_sha256'])
            self.assertNotIn('PACK.json', packed['files'])
            verify_tree(output/'source', binding['source_freeze_sha256'])


class ParityPackTests(unittest.TestCase):
    def test_candidate_pack_carries_the_harness_and_pins_the_ranges(self):
        with tempfile.TemporaryDirectory() as temp:
            ranges = Path(temp)/'ranges.json'
            ranges.write_text('{"limits": {}}\n')
            output = Path(temp)/'candidate'
            binding = native_parity_pack.candidate(
                output, HOME_ROOT/'attempt', source_commit='c'*40, ranges=ranges,
                baseline_sources={'env.py': 'd'*64}, checkpoint=CHECKPOINT,
                checkpoint_sha=WEIGHTS, checkpoint_declaration_sha=DECLARATION)
            self.assertEqual(binding['module'], 'locomotion.scripts.play')
            self.assertFalse(binding['root_review_complete'])
            source = output/'source'
            runtime = json.loads((source/'RUNTIME_SOURCES.json').read_text())['files']
            for name in native_parity_pack.HARNESS:
                self.assertEqual(runtime['locomotion/tests/'+name], sha(Path(__file__).parent/name))
            self.assertEqual(runtime['locomotion/tests/__init__.py'], hashlib.sha256(b'').hexdigest())
            proof = json.loads((source/'PARITY.json').read_text())
            self.assertEqual(proof['role'], 'candidate')
            self.assertEqual(proof['baseline_commit'], native_parity_pack.BASELINE_COMMIT)
            self.assertEqual(proof['ranges_sha256'], sha(ranges))
            self.assertEqual(proof['source_provenance_sha256'], sha(output/'source_provenance.json'))
            provenance = json.loads((output/'source_provenance.json').read_text())
            self.assertIn('omniwalk/env.py', provenance['candidate_physics_sources'])
            self.assertEqual(provenance['checkpoint_sha256'], WEIGHTS)
            packed = json.loads((output/'PACK.json').read_text())
            self.assertTrue(packed['test_only_parity'])
            self.assertEqual(packed['binding_sha256'], sha(output/'binding.json'))
            self.assertEqual(packed['source_freeze_sha256'], binding['source_freeze_sha256'])
            args = binding['command_args']
            self.assertEqual(args[args.index('--source-freeze-sha256')+1], binding['source_freeze_sha256'])
            verify_tree(source, binding['source_freeze_sha256'])
            self.assertEqual({WEIGHTS, DECLARATION} & set(binding['input_files'].values()), {WEIGHTS, DECLARATION})

    def test_harness_patches_the_names_the_scripts_call(self):
        import locomotion.scripts.cli as cli
        import locomotion.scripts.play as play
        self.assertTrue(callable(cli.require_checkpoint_identity))
        self.assertTrue(callable(play.require_learner_configuration))


if __name__ == '__main__':
    unittest.main()
