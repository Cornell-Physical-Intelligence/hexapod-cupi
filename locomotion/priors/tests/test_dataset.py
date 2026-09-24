"""Exercise dataset rejection paths with synthetic records, without native claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from locomotion.amp import feature_contract
from locomotion.priors import dataset
from locomotion.priors.commands import motion_cases
from locomotion.env_config import JOINT_NAMES, MODEL_SHA256, USD_SHA256
from locomotion.force_metrics import report_for
from locomotion.tests.test_evaluation import fixture as scoring_fixture


def trace_fixture(command=(.025, 0., 0.), heading=0.):
    root = np.zeros((1000, 1, 7)); root[..., 2] = .1
    root[..., 5] = np.sin(heading/2); root[..., 6] = np.cos(heading/2)
    rotation = dataset.rotations(root[..., 3:])
    linear = np.tile([command[1], -command[0], 0.], (1000, 1, 1))
    angular = np.tile([0, 0, command[2]], (1000, 1, 1))
    offset = np.array([[.01, .02, .03]])
    trace = {'root_pose_xyzw': root, 'joint_position_rad': np.zeros((1000, 1, 18)),
        'joint_velocity_rad_s': np.zeros((1000, 1, 18)), 'gyro_body_rad_s': angular,
        'velocity_world_mps': np.einsum('...ij,...j->...i', rotation, linear+np.cross(angular, offset)),
        'toe_xyz_world_m': root[..., None, :3]+np.einsum('...ij,...lj->...li', rotation,
            np.broadcast_to(np.arange(18).reshape(6, 3)/100, (1000, 1, 6, 3))),
        'time_s': (np.arange(1000)+1)*.02, 'command': np.tile(command, (1000, 1, 1)),
        'distal_contact': np.ones((1000, 1, 6), dtype=bool),
        **{key: np.zeros((1000, 1), dtype=bool) for key in ('terminated', 'truncated', 'reset')}}
    trace['amp_state_after'] = dataset.reconstruct_after(trace, offset)
    trace['amp_state_before'] = trace['amp_state_after'].copy()
    return trace, offset


def replay_fixture(root):
    """Create a synthetic file bundle to exercise the offline audit end to end."""
    evaluation = root/'evaluation'; native = evaluation/'native400hz'
    native.mkdir(parents=True)
    command = [.025, 0., 0.]
    trace, offset = trace_fixture(command)
    for name, value in scoring_fixture(command=tuple(command)).items():
        if name not in trace:
            trace[name] = value[:, None]
    np.savez_compressed(evaluation/'control_trace.npz', **trace)
    # Video bytes stand in for a reviewed file; this fixture makes no visual claim.
    (evaluation/'rollout.mp4').write_bytes(b'synthetic video fixture')
    case = {'case_id': 'test_fixture', 'command': command, 'controls': 1000}
    raw = {'time_s': (np.arange(8000)+1)*.0025, 'sequence': np.arange(8000),
        'explicit_counter': np.arange(8000)+1, 'command': np.tile(command, (8000, 1, 1)),
        'distal_force_world_n': np.zeros((8000, 1, 6, 3)),
        'nonfoot_force_world_n': np.zeros((8000, 1, 4, 3)),
        'computed_torque_nm': np.zeros((8000, 1, 18)), 'applied_torque_nm': np.zeros((8000, 1, 18))}
    np.savez_compressed(native/'substeps_000.npz', **raw)
    dataset.save(native/'capture.json', {'steps': 8000, 'initial_counter': 0, 'failure': None,
        'joint_names': list(JOINT_NAMES), 'substep_files': ['substeps_000.npz'],
        'files': {'substeps_000.npz': dataset.sha(native/'substeps_000.npz')}})
    declaration = {'cases': [case], 'model_sha256': MODEL_SHA256, 'contact_classification': 'exact_distal_points'}
    dataset.save(evaluation/'declaration.json', declaration)
    dataset.save(evaluation/'force_metrics.json', report_for(evaluation))
    report = {**declaration, 'acquisition_complete': True, 'recorded_physics_steps': 8000,
        'results': [{'pass': True, 'native_motor_and_joint_checks_pass': True,
                     'native_contact_screen': {'pass': True}}], 'force_metrics': {'status': 'available'},
        'files': {p.name: dataset.sha(p) for p in evaluation.iterdir() if p.is_file()}}
    dataset.save(evaluation/'report.json', report)
    dataset.save(root/'state.json', {'status': 'completed', 'eligible_for_motion_prior': True,
        'replay': report, 'identity': {'controller_kind': 'optimized_periodic_motor_targets',
        'learned_policy': False, 'model_sha256': MODEL_SHA256, 'usd_sha256': USD_SHA256,
        'amp_feature_contract': feature_contract(), 'start_phase': 0., 'command': command,
        'root_com_local_m': offset.tolist(), 'physics_source_files': {'test_fixture': '0'*64},
        'optimization_config': {'period_s': 1.2}, 'optimization_source_files': {'test_fixture': '0'*64},
        'standing_admission': {'test_fixture': True}, 'trajectory_sha256': '0'*64,
        'optimization_input_sha256': '1'*64, 'optimization_result_sha256': '2'*64}})


class DatasetTests(unittest.TestCase):
    def test_file_audit_recomputes_features_motion_and_raw_load_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); replay_fixture(root)
            record, trace = dataset.audit_replay(root)
            self.assertEqual(record['command_index'], 0)
            self.assertLess(record['feature_parity_max_abs'], 1e-12)
            (root/'evaluation/native400hz/substeps_000.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'native force chunk'):
                dataset.audit_replay(root)

    def test_reconstruction_uses_origin_velocity_and_is_yaw_invariant(self):
        command = (.04, 0., .15)
        trace, offset = trace_fixture(command)
        rotated, rotated_offset = trace_fixture(command, heading=1.)
        self.assertLess(dataset.validate_trace(trace, command, offset), 1e-12)
        np.testing.assert_allclose(trace['amp_state_after'], rotated['amp_state_after'], atol=1e-12)
        np.testing.assert_allclose(trace['amp_state_after'][0, 0, 36:39], [0, -.04, 0])
        changed = copy.deepcopy(trace); changed['root_pose_xyzw'][..., :2] += 10
        changed['toe_xyz_world_m'][..., :2] += 10
        np.testing.assert_allclose(dataset.reconstruct_after(changed, offset), trace['amp_state_after'], atol=1e-12)

    def test_reset_terminal_and_time_gaps_cannot_hide_in_startup(self):
        for key in ('reset', 'terminated', 'truncated'):
            trace, offset = trace_fixture(); trace[key][1] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                dataset.validate_trace(trace, (.025, 0, 0), offset)
        trace, offset = trace_fixture(); trace['time_s'][101] += .02
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            dataset.validate_trace(trace, (.025, 0, 0), offset)
        trace, offset = trace_fixture(); offset[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'COM offset'):
            dataset.validate_trace(trace, (.025, 0, 0), offset)
        trace, offset = trace_fixture(); trace['root_pose_xyzw'] = trace['root_pose_xyzw'][:, 0]
        with self.assertRaisesRegex(ValueError, 'shape'):
            dataset.validate_trace(trace, (.025, 0, 0), offset)

    def test_permuted_features_and_reset_pairs_are_rejected(self):
        trace, offset = trace_fixture()
        trace['amp_state_after'][500, 0, 0] = .1
        with self.assertRaisesRegex(ValueError, 'discontinuity'):
            dataset.validate_trace(trace, (.025, 0, 0), offset)
        trace, offset = trace_fixture()
        trace['amp_state_after'][..., [36, 37]] = trace['amp_state_after'][..., [37, 36]]
        trace['amp_state_before'] = trace['amp_state_after'].copy()
        with self.assertRaisesRegex(ValueError, 'measurements'):
            dataset.validate_trace(trace, (.025, 0, 0), offset)

    def test_stationary_low_speed_is_exposed_for_human_review(self):
        trace, offset = trace_fixture()
        trace['amp_state_after'][..., 36:39] = 0
        result = dataset.motion_diagnostics(trace, (.025, 0, 0))
        self.assertEqual(result['signed_translation_fraction'], 0.)
        self.assertEqual(result['per_foot_contact_changes'], [0]*6)

    def test_export_requires_human_review_and_both_phases_for_each_command(self):
        records, traces, decisions = {}, {}, []
        for i, case in enumerate(motion_cases()):
            for phase in (0., .5):
                name = f'{i}_{phase}'
                records[name] = {'command_index': i, 'command': case['command'], 'start_phase': phase,
                    'report_sha256': name, 'video_sha256': name, 'native_state_sha256': name,
                    'identity': {**dict.fromkeys(('physics_source_files', 'physics_config',
                        'geometry_sha256', 'geometry_extrema_sha256', 'stance_sha256', 'trajectory_sha256',
                        'optimization_input_sha256', 'optimization_result_sha256'), 'test_fixture'),
                        'optimization_config': {'period_s': 1.2}}}
                traces[name] = trace_fixture(case['command'])[0]
                decisions.append({'report_sha256': name, 'video_sha256': name,
                                  'direction_accepted': True, 'tripod_accepted': True})
        def audited(path):
            return records[str(path)], traces[str(path)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(dataset, 'audit_replay', side_effect=audited):
            root = Path(tmp); review = root/'review.json'
            dataset.save(review, {'schema': 'hexapod_amp_motion_review_v1', 'reviewer': 'palerdr', 'clips': decisions})
            with self.assertRaisesRegex(ValueError, 'coverage'):
                dataset.export(list(records)[:-1], review, root/'incomplete')
            manifest = dataset.export(list(records), review, root/'complete')
            arrays, _ = dataset.load(root/'complete')
            self.assertEqual(manifest['transitions'], 18000)
            self.assertEqual(manifest['processing_identity']['source_files']['locomotion/priors/dataset.py'],
                             dataset.sha(dataset.__file__))
            altered = copy.deepcopy(manifest); del altered['processing_identity']
            dataset.save(root/'complete/manifest.json', altered)
            with self.assertRaisesRegex(ValueError, 'processing source'):
                dataset.load(root/'complete')
            dataset.save(root/'complete/manifest.json', manifest)
            pairs = dataset.sample(arrays, 16, np.random.default_rng(1))
            self.assertEqual(pairs.shape, (16, 122))
            changed = copy.deepcopy(list(records.values()))
            changed[0]['identity']['optimization_config']['period_s'] = 1.4
            with self.assertRaisesRegex(ValueError, 'gait'):
                dataset.require_bank_identity(changed)
            changed = copy.deepcopy(list(records.values()))
            changed[0]['identity']['trajectory_sha256'] = 'different_trajectory'
            with self.assertRaisesRegex(ValueError, 'different optimized trajectories'):
                dataset.require_bank_identity(changed)
            changed = copy.deepcopy(list(records.values()))
            changed[1]['native_state_sha256'] = changed[0]['native_state_sha256']
            with self.assertRaisesRegex(ValueError, 'second start phase'):
                dataset.require_bank_identity(changed)
            (root/'complete/transitions.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes'):
                dataset.load(root/'complete')
            decisions[0]['tripod_accepted'] = False
            dataset.save(review, {'schema': 'hexapod_amp_motion_review_v1', 'reviewer': 'palerdr', 'clips': decisions})
            with self.assertRaisesRegex(ValueError, 'acceptance'):
                dataset.export(list(records), review, root/'unreviewed')


if __name__ == '__main__':
    unittest.main()
