"""CPU contracts and fake-API rollout; no Isaac, GPU or walking qualification."""
import ast
import contextlib
import json
import math
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'omni_velocity_launch_003/source_003'
sys.path.insert(0, str(SOURCE / 'tools'))
import recording_contract as c
import record_candidate_video as recorder
from omni_flat_math import slew_commands


class ContractTests(unittest.TestCase):
    def admitted_fixture(self, directory):
        base = Path(directory); probe = base / 'probe'; pilot = base / 'pilot'
        (probe / 'inputs/study').mkdir(parents=True)
        (probe / 'flat').mkdir(); (probe / 'probe').mkdir(); (pilot / 'train').mkdir(parents=True)
        for i in range(550):
            (probe / 'inputs/study' / f'fixture_{i:03}.usd').write_text(f'admitted file {i}')
        evidence = {}
        for key, relative in (('admission', 'flat/admission.json'), ('calibration', 'probe/calibration.json'), ('runner_smoke', 'probe/runner_smoke.json')):
            p = probe / relative; p.write_text(json.dumps({**c.IDENTITY, 'passed': True})); evidence[key] = p
        shutil.copytree(probe / 'inputs/study', pilot / 'inputs/study')
        for key, p in evidence.items():
            shutil.copyfile(p, pilot / 'inputs' / (key + '.json'))
        study_receipt = probe / 'inputs/study_after_flat.sha256.json'
        study_receipt.write_text(json.dumps(c.tree_hashes(probe / 'inputs/study')))
        pilot_receipt = pilot / 'inputs_before.sha256.json'
        pilot_receipt.write_text(json.dumps(c.tree_hashes(pilot / 'inputs')))
        campaign = {'status': 'completed', 'source_unchanged': True, 'admitted_asset_unchanged': True,
                    'source_manifest_sha256': c.SOURCE_MANIFEST_SHA256,
                    'flat_admission_sha256': c.digest(evidence['admission']),
                    'calibration_sha256': c.digest(evidence['calibration']),
                    'runner_smoke_sha256': c.digest(evidence['runner_smoke'])}
        (probe / 'campaign.json').write_text(json.dumps(campaign))
        (pilot / 'comparison.json').write_text('{}')
        (pilot / 'campaign.json').write_text(json.dumps({'status': 'completed_needs_review', 'identity': c.IDENTITY,
            'source_unchanged': True, 'admitted_inputs_unchanged': True,
            'accepted_probe_campaign_sha256': c.digest(probe / 'campaign.json'),
            'comparison_sha256': c.digest(pilot / 'comparison.json')}))
        return SimpleNamespace(package=pilot / 'inputs/study', pilot_state=pilot / 'train/state.json',
            probe_campaign=probe / 'campaign.json', study_tree_receipt=study_receipt,
            pilot_inputs_receipt=pilot_receipt, **evidence)

    def test_full_admitted_package_and_both_receipts_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.admitted_fixture(directory)
            with patch.object(c, 'PROBE_CAMPAIGN_SHA256', c.digest(args.probe_campaign)):
                result = c.verify_admitted_package(args)
                self.assertEqual(result['study_file_count'], 550)
                self.assertTrue(result['complete_admitted_package_tree_verified'])
                (args.package / 'fixture_100.usd').write_text('changed payload')
                with self.assertRaisesRegex(ValueError, '550-file'):
                    c.verify_admitted_package(args)

    def test_original_source_package_or_incomplete_pilot_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.admitted_fixture(directory)
            with patch.object(c, 'PROBE_CAMPAIGN_SHA256', c.digest(args.probe_campaign)):
                admitted = args.package; args.package = SOURCE / 'robot/hexapod_mkii_length_study'
                with self.assertRaisesRegex(ValueError, 'exact pilot'):
                    c.verify_admitted_package(args)
                args.package = admitted
                expected = json.loads(args.pilot_inputs_receipt.read_text())
                del expected['study/fixture_001.usd']
                args.pilot_inputs_receipt.write_text(json.dumps(expected))
                with self.assertRaisesRegex(ValueError, 'pilot inputs tree'):
                    c.verify_admitted_package(args)

    def test_probe_receipt_and_unlisted_pilot_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.admitted_fixture(directory)
            with self.assertRaisesRegex(ValueError, 'campaign receipt'):
                c.verify_admitted_package(args)
            with patch.object(c, 'PROBE_CAMPAIGN_SHA256', c.digest(args.probe_campaign)):
                (args.package.parent / 'unlisted.txt').write_text('extra')
                with self.assertRaisesRegex(ValueError, 'pilot inputs tree'):
                    c.verify_admitted_package(args)
    def test_exact_frozen_source_digest_matches_actual_candidate_algorithm(self):
        from candidate_runner import source_digest
        self.assertEqual(c.verify_source(SOURCE), SOURCE.resolve())
        self.assertEqual(c.source_digest(SOURCE), source_digest(SOURCE))

    def test_checkpoint_bytes_sidecar_and_source_binding_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'final.pt'
            p.write_bytes(b'CPU fixture: not an actual checkpoint')
            sha = c.digest(p)
            sidecar = p.with_suffix('.pt.json')
            sidecar.write_text(json.dumps({**c.IDENTITY, 'checkpoint_sha256': sha}))
            c.verify_checkpoint_bytes(p, sha)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                c.verify_checkpoint_bytes(p, sha)
            p.write_bytes(b'CPU fixture: not an actual checkpoint')
            sidecar.write_text(json.dumps({**c.IDENTITY, 'source_sha256': 'a'*64, 'checkpoint_sha256': sha}))
            with self.assertRaises(ValueError):
                c.verify_checkpoint_bytes(p, sha)

    def test_calibration_and_smoke_must_match_same_passed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / 'calibration.json'
            smoke = Path(directory) / 'runner_smoke.json'
            calibration.write_text(json.dumps({**c.IDENTITY, 'passed': True, 'initial_std': .005}))
            data = {**c.IDENTITY, 'passed': True, 'calibration_sha256': c.digest(calibration)}
            smoke.write_text(json.dumps(data))
            c.verify_pilot_prerequisites(calibration, smoke)
            data['passed'] = False
            smoke.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                c.verify_pilot_prerequisites(calibration, smoke)

    def test_xyzw_rotation_and_heading_match_scipy(self):
        for angles in [(0, 0, 0), (.1, -.2, .3), (.3, .4, -2.4), (0, 0, math.pi / 2)]:
            rotation = Rotation.from_euler('xyz', angles)
            q = rotation.as_quat()
            np.testing.assert_allclose(c.xyzw_matrix(q, convention='xyzw'), rotation.as_matrix(), atol=1e-12)
            forward = rotation.apply([0, -1, 0])
            pose = c.actual_pose_xy_heading([1, 2, .14], q, convention='xyzw')
            self.assertAlmostEqual(pose[2], math.atan2(forward[1], forward[0]))
        with self.assertRaises(ValueError):
            c.xyzw_matrix([1, 0, 0, 0], convention='wxyz')
        with self.assertRaises(ValueError):
            c.xyzw_matrix([0, 0, 0, 2], convention='xyzw')

    def test_completed_pilot_state_proves_final_update_and_checkpoint_label(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'state.json'
            state = {**c.IDENTITY, 'status': 'completed', 'mode': 'train', 'iterations': 50, 'checkpoint_sha256': 'a'*64}
            p.write_text(json.dumps(state))
            c.verify_pilot_run(p, 'a'*64, 'scratch_50_update_final')
            with self.assertRaises(ValueError):
                c.verify_pilot_run(p, 'b'*64, 'scratch_50_update_final')
            state['iterations'] = 2
            p.write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                c.verify_pilot_run(p, 'a'*64, 'scratch_50_update_final')

    def test_schedule_is_continuous_bounded_and_explicit(self):
        rows = c.schedule()
        self.assertEqual(len(rows), 1700)
        self.assertEqual(len(rows) * c.DT, 34.)
        self.assertEqual(rows[0]['segment'], 'QUIET START')
        self.assertEqual(rows[-1]['segment'], 'QUIET END')
        self.assertEqual(rows[-1]['requested_command'], [0., 0., 0.])
        self.assertEqual({r['segment_index'] for r in rows}, set(range(10)))
        self.assertLessEqual(max(np.linalg.norm(r['requested_command'][:2]) for r in rows), .2)
        self.assertLessEqual(max(abs(r['requested_command'][2]) for r in rows), .4)
        np.testing.assert_allclose(np.diff([r['time_start_s'] for r in rows]), .02)

    def test_actor_command_reference_obeys_real_one_step_ramp_timing(self):
        rows = c.schedule()
        poses, commands = recorder.reference_schedule([0., 0., -math.pi/2], rows, 'cpu')
        # The first forward request is made at index150. The existing environment
        # updates its ramp after that action, so it first reaches actor at151.
        np.testing.assert_allclose(commands[150], 0)
        self.assertAlmostEqual(commands[151, 0], .005, places=7)
        self.assertAlmostEqual(poses[150, 1], 0., places=7)
        self.assertLess(poses[151, 1], 0.)

    def test_pre_reset_snapshot_and_quaternion_conventions_are_explicit(self):
        s = {'terminated': np.array([True]), 'truncated': np.array([False]),
             'quaternion_world_xyzw': np.array([[0., 0., 0., 1.]]),
             'quaternion_world_wxyz': np.array([[1., 0., 0., 0.]])}
        self.assertTrue(c.assert_event_snapshot(s, np.array([True]), np.array([False])))
        with self.assertRaises(ValueError):
            c.assert_event_snapshot(s, np.array([False]), np.array([False]))
        s['quaternion_world_wxyz'] = s['quaternion_world_xyzw']
        with self.assertRaises(ValueError):
            c.assert_event_snapshot(s, np.array([True]), np.array([False]))

    def test_observation_schema_and_finiteness(self):
        obs = {'policy': torch.zeros(1, 495), 'critic': torch.zeros(1, 498)}
        c.check_observations(obs)
        obs['policy'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            c.check_observations(obs)
        with self.assertRaises(ValueError):
            c.check_observations({'policy': torch.zeros(1, 315), 'critic': torch.zeros(1, 318)})

    def test_shared_physical_cfg_assignments_match_frozen_entrypoint(self):
        def assignments(path):
            tree = ast.parse(path.read_text())
            result = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    key = ast.unparse(node.targets[0])
                    if key.startswith('cfg.'):
                        result.setdefault(key, []).append(ast.dump(node.value, include_attributes=False))
            return result
        original = assignments(SOURCE / 'tools/train_velocity_candidate.py')
        actual = assignments(HERE / 'recording_env.py')
        deliberate = {'cfg.scene.num_envs', 'cfg.episode_length_s', 'cfg.video_recorder.window_width', 'cfg.video_recorder.window_height'}
        for key in set(original) - deliberate:
            self.assertEqual(actual.get(key), original[key], key)


class FakeEnv:
    def __init__(self, fail_step=None):
        self.device = 'cpu'; self.num_envs = 1; self.step_dt = .02
        self.episode_length_buf = torch.zeros(1)
        self._commands = torch.zeros(1, 3)
        self.target = torch.zeros(1, 3)
        self.steps = self.reset_count = self.render_count = 0
        self.fail_step = fail_step
        self.position = np.array([0., 0., .1365])
        self.action_history = []
        self._robot = SimpleNamespace(joint_names=['joint'+str(i) for i in range(18)], data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.tensor(self.position).reshape(1, 3)),
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0., 0., 0., 1.]]))))

    def reset(self, seed):
        self.reset_count += 1

    def set_evaluation_targets(self, target):
        self.target.copy_(target)

    def _get_observations(self):
        return {'policy': torch.zeros(1, 495), 'critic': torch.zeros(1, 498)}

    def step(self, actions):
        self.steps += 1
        self.action_history.append(actions.clone())
        terminated = torch.tensor([self.steps == self.fail_step])
        truncated = torch.tensor([False])
        self.position += np.array([float(self._commands[0, 1]), -float(self._commands[0, 0]), 0.]) * .02
        self.omni_diagnostic_sample = {
            'terminated': terminated.numpy(), 'truncated': truncated.numpy(),
            'quaternion_world_xyzw': np.array([[0., 0., 0., 1.]]),
            'quaternion_world_wxyz': np.array([[1., 0., 0., 0.]]),
            'position_world_m': self.position[None].copy(),
            'velocity_navigation_mps': self._commands.numpy().copy(),
            'gyro_navigation_rad_s': np.array([[0., 0., float(self._commands[0, 2])]]),
        }
        self._commands.copy_(slew_commands(self._commands, self.target, .02))
        return None, None, terminated, truncated, None

    def render(self):
        if self.steps == self.fail_step:
            raise AssertionError('Recorder rendered the automatically reset terminal state')
        self.render_count += 1
        frame = np.zeros((240, 800, 3), dtype=np.uint8)
        frame[:, 300:, 1] = 80
        return frame


class FakeDrawing:
    def reference(self, *args, **kwargs): pass
    def line(self, *args, **kwargs): pass
    def arrows(self, *args, **kwargs): pass
    def text(self, *args, **kwargs): pass


class RolloutContractTests(unittest.TestCase):
    def run_fake(self, directory, *, fail_step=None):
        modules = {}
        for name in ('omni_path_demo', 'tensordict', 'isaaclab_physx.renderers.kit_viewport_utils'):
            modules[name] = ModuleType(name)
        modules['omni_path_demo'].GroundDrawing = FakeDrawing
        modules['tensordict'].TensorDict = lambda value, batch_size: value
        modules['isaaclab_physx.renderers.kit_viewport_utils'].set_kit_renderer_camera_view = lambda **kwargs: None
        output = Path(directory)
        actions = torch.arange(18).reshape(1, 18).float() / 100
        runner = SimpleNamespace(alg=SimpleNamespace(eval_mode=lambda: None),
                                 get_inference_policy=lambda device: lambda obs: actions.clone())
        env = FakeEnv(fail_step)
        provenance = {'checkpoint_label': 'CPU_FAKE_API_ONLY', 'checkpoint_sha256': 'f'*64,
                      'recording_source_hashes': c.external_source_hashes(HERE), 'stage2_complete': False,
                      'admitted_package': {'CPU_fixture': True}}
        args = SimpleNamespace(output=output, source_root=SOURCE, checkpoint=output/'fixture.pt', checkpoint_sha256='f'*64, seed=1)
        class Writer:
            def __enter__(self):
                (output/'rollout.mp4').write_bytes(b'CPU fake writer, not a video')
                self.frames = 0
                return self
            def __exit__(self, *unused): pass
            def append_data(self, frame):
                self.frames += 1
                assert frame.ndim == 3
        with patch.dict(sys.modules, modules), patch('imageio.v2.get_writer', return_value=Writer()), \
             patch.object(recorder, 'verify_source', return_value=SOURCE), \
             patch.object(recorder, 'verify_admitted_package', return_value=provenance['admitted_package']), \
             patch.object(recorder, 'verify_checkpoint_bytes', return_value={}):
            result = recorder.record(env, runner, args, provenance)
        self.assertEqual(env.reset_count, 1)
        self.assertTrue(all(torch.equal(action, actions) for action in env.action_history))
        return env, result

    def test_full_sequence_runs_real_policy_output_without_resets_or_pose_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            env, result = self.run_fake(directory)
            self.assertTrue(result['complete'])
            self.assertEqual(env.steps, 1700)
            self.assertEqual(result['frames'], 850)
            self.assertEqual(result['playback_duration_s'], 34.)
            self.assertFalse(result['stage2_complete'])
            trace = np.load(Path(directory)/'trace.npz')
            self.assertEqual(trace['deterministic_actor_action'].shape, (1700, 1, 18))

    def test_termination_retains_pre_reset_evidence_and_stops_before_render(self):
        with tempfile.TemporaryDirectory() as directory:
            env, result = self.run_fake(directory, fail_step=211)
            self.assertFalse(result['complete'])
            self.assertEqual(env.steps, 211)
            self.assertTrue(result['terminal_event']['terminated'])
            self.assertTrue(result['telemetry'][-1]['terminated'])
            self.assertEqual(result['recorded_control_steps'], 211)


if __name__ == '__main__':
    unittest.main()
