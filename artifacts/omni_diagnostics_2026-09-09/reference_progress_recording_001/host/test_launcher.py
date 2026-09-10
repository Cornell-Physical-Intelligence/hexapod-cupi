import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import launch_recording_spark as launcher


class RecordingHostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.p = Path(self.temp.name)
        self.identity = {'source_manifest_sha256': launcher.SOURCE_MAP, 'variant': 'f050_t060'}
        self.state = dict(status='completed', identity=self.identity, gate={'passed': True},
                          control_steps=2400, runtime_binding={'runtime_tree_sha256': launcher.RUNTIME_TREE})
        (self.p / 'rollout.mp4').write_bytes(b'fixture bytes only; not a playable video')
        self.video = dict(complete=True, frames=1200, fps=25, recorded_control_steps=2400,
                          planned_control_steps=2400, source_manifest_sha256=launcher.SOURCE_MAP,
                          source_and_inputs_reverified_after_recording=True, stage2_complete=False,
                          policy_training_started=False, pose_forcing=False,
                          video_sha256=launcher.sha(self.p / 'rollout.mp4'))

    def store(self):
        (self.p / 'state.json').write_text(json.dumps(self.state))
        (self.p / 'video.json').write_text(json.dumps(self.video))

    def test_complete_metadata_and_byte_receipt_accept_without_claiming_decode(self):
        self.store()
        self.assertEqual(launcher.validate_result(self.p, self.identity), self.video)

    def test_failed_physics_and_source_mismatch_cannot_be_promoted_by_video(self):
        for changed in ({'gate': {'passed': False}}, {'control_steps': 2399},
                        {'identity': {'source_manifest_sha256': 'different'}}, {'status': 'rejected'}):
            original = self.state.copy()
            self.state.update(changed)
            self.store()
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                launcher.validate_result(self.p, self.identity)
            self.state = original

    def test_partial_or_wrong_lineage_video_remains_rejected(self):
        for key, value in [('complete', False), ('frames', 1199), ('fps', 50),
                           ('source_manifest_sha256', 'wrong'), ('pose_forcing', True),
                           ('policy_training_started', True), ('stage2_complete', True),
                           ('source_and_inputs_reverified_after_recording', False)]:
            original = self.video.copy()
            self.video[key] = value
            self.store()
            with self.subTest(key=key), self.assertRaises(ValueError):
                launcher.validate_result(self.p, self.identity)
            self.video = original

    def test_changed_or_missing_video_bytes_rejected(self):
        self.store()
        (self.p / 'rollout.mp4').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            launcher.validate_result(self.p, self.identity)
        (self.p / 'rollout.mp4').unlink()
        with self.assertRaises(ValueError):
            launcher.validate_result(self.p, self.identity)

    def test_native_command_keeps_every_input_read_only_and_uses_no_actor(self):
        a = SimpleNamespace(source=Path('/frozen source'), run=Path('/qualified run'),
                            adapter=Path('/adapter source'), output=Path('/new output'))
        command = launcher.command(a, 'owned-exact')
        mounts = [command[i + 1] for i, token in enumerate(command) if token == '-v']
        self.assertEqual(mounts, ['/frozen source:/workspace/hexapod:ro', '/new output:/outputs:rw',
                                 '/qualified run:/qualified:ro', '/adapter source:/recording:ro'])
        self.assertIn('/recording/record_reference_video.py', command)
        self.assertIn('--enable_cameras', command)
        self.assertNotIn('--checkpoint', command)
        self.assertNotIn('--privileged', command)

    def test_wrong_qualified_source_rejected_before_any_other_input(self):
        source = self.p / 'source'
        source.mkdir()
        (source / 'campaign_source_hashes.json').write_text('{}')
        args = SimpleNamespace(source=source)
        with self.assertRaisesRegex(ValueError, 'Exact reviewed009 source'):
            launcher.validate_inputs(args)


if __name__ == '__main__':
    unittest.main()
