"""Check load definitions, time windows and recorded-input integrity."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from locomotion import evaluate
from locomotion.evaluate import run_batch
from locomotion.env_config import JOINT_NAMES, MODEL_SHA256
from locomotion.force_metrics import report_for, sha, summarize, write_summary
from locomotion.tests.test_evaluate import FakeCapture, FakeNative
from experiments.trajectory_optimization.prepare import prepare
from experiments.trajectory_optimization.model import ROOT


def fixture():
    n = 8
    feet = np.zeros((n, 2, 6, 3))
    feet[:, 0, 0, 2] = [0, 20]*4
    feet[:, 1, :, 2] = 10
    torque = np.ones((n, 2, 18))
    torque[::2] *= -1
    command = np.ones((n, 2, 3))
    command[6:, 0] = 0
    return {'time_s': (np.arange(n)+1)*.0025, 'sequence': np.arange(n),
        'explicit_counter': np.arange(n)+11, 'command': command,
        'distal_force_world_n': feet, 'nonfoot_force_world_n': np.zeros((n, 2, 4, 3)),
        'applied_torque_nm': torque, 'computed_torque_nm': torque*2}


CASES = [{'case_id': 'forward', 'controls': 1}, {'case_id': 'yaw', 'controls': 1}]


def recording(root):
    raw = fixture()
    native = root/'native400hz'
    native.mkdir()
    path = native/'substeps_000.npz'
    np.savez_compressed(path, **raw)
    (native/'capture.json').write_text(json.dumps({'steps': 8, 'initial_counter': 10,
        'failure': None, 'joint_names': list(JOINT_NAMES), 'substep_files': [path.name],
        'files': {path.name: sha(path)}}))
    (root/'declaration.json').write_text(json.dumps({'cases': CASES, 'model_sha256': MODEL_SHA256}))
    return path


class ForceMetricsTests(unittest.TestCase):
    def test_fresh_pack_binds_and_imports_load_reporter_without_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'pack'
            prepare(ROOT/'artifacts/trajectory_optimizer_20260917/solve_002', output,
                    '/home/orionh/HEXAPOD_runs/restart_20260914/test_force_metrics')
            source = output/'source'
            manifest = json.loads((source/'FREEZE_SHA256.json').read_text())
            self.assertEqual(manifest['locomotion/force_metrics.py'], sha(source/'locomotion/force_metrics.py'))
            result = subprocess.run([sys.executable, '-c',
                'from locomotion.force_metrics import MODEL_SHA256; print(MODEL_SHA256)'],
                cwd=source, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.strip(), MODEL_SHA256)

    def test_swing_zeros_stance_load_and_motor_signs_have_distinct_means(self):
        cases = summarize(fixture(), CASES, settle_seconds=.01)
        full = cases[0]['windows']['full_trial']
        foot = full['per_foot']['lf']
        self.assertEqual(foot['normal_resultant_magnitude_n']['mean'], 10.)
        self.assertEqual(foot['normal_resultant_magnitude_n']['rms'], np.sqrt(200.))
        self.assertEqual(foot['normal_resultant_magnitude_n']['peak'], 20.)
        self.assertEqual(foot['mean_magnitude_during_contact_n'], 20.)
        self.assertEqual(foot['contact_fraction'], .5)
        self.assertIsNone(full['per_foot']['lm']['mean_magnitude_during_contact_n'])
        self.assertEqual(full['motor_torque']['applied']['mean_abs_across_joints_nm'], 1.)
        self.assertEqual(full['motor_torque']['requested']['worst_joint_rms_nm'], 2.)
        self.assertEqual(cases[1]['windows']['full_trial']['total_vertical_support_n']['mean'], 60.)

    def test_startup_and_commanded_motion_keep_exact_sample_boundaries(self):
        row = summarize(fixture(), CASES, settle_seconds=.01)[0]
        self.assertEqual(row['windows']['startup']['samples'], 4)
        walking = row['windows']['commanded_locomotion_after_settle']
        self.assertEqual(walking['samples'], 2)
        self.assertEqual(walking['first_sample_s'], .0125)
        self.assertEqual(walking['last_sample_s'], .015)
        raw = fixture(); raw['command'][:] = 0
        empty = summarize(raw, CASES)[0]['windows']['commanded_locomotion_after_settle']
        self.assertEqual(empty, {'status': 'empty', 'samples': 0, 'duration_s': 0., 'metrics': None})

    def test_force_vector_cancellation_does_not_hide_individual_foot_load(self):
        raw = fixture(); raw['distal_force_world_n'][:] = 0
        raw['distal_force_world_n'][:, 0, 0, 0] = 3
        raw['distal_force_world_n'][:, 0, 1, 0] = -3
        raw['nonfoot_force_world_n'][:, 0, 0, 2] = 5
        row = summarize(raw, CASES)[0]['windows']['full_trial']
        self.assertEqual(row['total_normal_resultant_magnitude_n']['mean'], 5.)
        self.assertEqual(row['sum_foot_normal_magnitudes_n']['mean'], 6.)
        self.assertEqual(row['total_vertical_support_n']['mean'], 5.)

    def test_failed_prefix_retains_numbers_without_claiming_complete_window(self):
        cases = [{'case_id': 'forward', 'controls': 1000}]
        row = summarize(fixture(), cases)[0]
        self.assertFalse(row['requested_window_complete'])
        self.assertEqual(row['recorded_samples'], 8)
        self.assertEqual(row['windows']['full_trial']['samples'], 8)

    def test_missing_nonfinite_and_discontinuous_channels_are_rejected(self):
        for key in ('time_s', 'sequence', 'explicit_counter', 'applied_torque_nm'):
            raw = fixture(); raw[key] = raw[key].astype(float); raw[key].flat[0] += .5
            with self.subTest(key=key):
                if key == 'applied_torque_nm': raw[key].flat[0] = np.nan
                with self.assertRaises(ValueError): summarize(raw, CASES)
        raw = fixture(); del raw['nonfoot_force_world_n']
        with self.assertRaises(KeyError): summarize(raw, CASES)

    def test_reanalysis_hashes_inputs_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = recording(root)
            report = report_for(root, settle_seconds=.01)
            self.assertEqual(report['source_files']['native400hz/'+path.name], sha(path))
            self.assertEqual(report['cases'][0]['windows']['full_trial']['total_vertical_support_n']['mean'], 10.)
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'changed native force chunk'): report_for(root)

    def test_writer_preserves_failure_and_rejects_output_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'metrics.json'
            self.assertEqual(write_summary(Path(tmp), path)['status'], 'unavailable')
            with self.assertRaises(FileExistsError): write_summary(Path(tmp), path)

    def test_evaluator_always_publishes_hash_bound_metrics_status(self):
        case = {'case_id': 'forward', 'profile': 'omni_static', 'command': [.05, 0, 0], 'controls': 1}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); geometry = root/'geometry'; geometry.write_bytes(b'fixture')
            output = root/'evaluation'
            with patch.object(evaluate, 'ExactEvaluationCapture', FakeCapture):
                result = run_batch(FakeNative(), lambda _: torch.zeros(1, 18), [case], output, geometry,
                    checkpoint_sha256=None, source_sha256='fixture')
            self.assertEqual(result['force_metrics']['status'], 'unavailable')
            self.assertEqual(result['files']['force_metrics.json'], sha(output/'force_metrics.json'))
            self.assertEqual(result['controls'], 1)

    def test_evaluator_saves_available_metrics_from_full_native_channels(self):
        class Capture(FakeCapture):
            def __init__(self, env, output, geometry):
                super().__init__(env, output, geometry)
                self.output = output
                output.mkdir()

            def close(self):
                receipt = super().close()
                raw = {k: v[:, :1] if v.ndim > 1 else v for k, v in fixture().items()}
                path = self.output/'substeps_000.npz'
                np.savez_compressed(path, **raw)
                receipt.update(initial_counter=10, joint_names=list(JOINT_NAMES),
                    substep_files=[path.name], files={path.name: sha(path)})
                (self.output/'capture.json').write_text(json.dumps(receipt))
                return receipt
        case = {'case_id': 'forward', 'profile': 'omni_static', 'command': [.05, 0, 0], 'controls': 1}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); geometry = root/'geometry'; geometry.write_bytes(b'fixture')
            output = root/'evaluation'
            with patch.object(evaluate, 'ExactEvaluationCapture', Capture):
                result = run_batch(FakeNative(), lambda _: torch.zeros(1, 18), [case], output, geometry,
                    checkpoint_sha256=None, source_sha256='fixture')
            self.assertEqual(result['force_metrics']['status'], 'available')
            self.assertEqual(result['files']['force_metrics.json'], sha(output/'force_metrics.json'))
            summary = json.loads((output/'force_metrics.json').read_text())
            self.assertEqual(summary['cases'][0]['windows']['full_trial']['total_vertical_support_n']['mean'], 10.)


if __name__ == '__main__':
    unittest.main()
