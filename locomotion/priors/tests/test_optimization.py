"""Exercise the saved feasible solution and reject corrupted motion inputs."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from locomotion.priors.model import ROOT, RobotModel, digest
from locomotion.priors.optimize import Config, audit
from locomotion.priors.prepare import prepare
from locomotion.priors.replay_native import validate_trajectory

FIXTURE = ROOT/'locomotion/priors/tests/fixtures/solve'
TRAJECTORY_SHA = 'a76e093075ce471210a66bf4ae3835a1dca94406b930e4a791a028941a502ec1'


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel()
        if digest(FIXTURE/'trajectory.npz') != TRAJECTORY_SHA:
            raise ValueError('Recorded forward solve fixture changed')
        with np.load(FIXTURE/'trajectory.npz', allow_pickle=False) as data:
            cls.arrays = {k: data[k].copy() for k in data.files}
        cls.config = Config(**json.loads((FIXTURE/'INPUT.json').read_text())['config'])

    def test_recorded_cycle_satisfies_dynamics_and_actuator_constraints(self):
        report = audit(self.model, self.config, self.arrays, self.arrays['desired_feet'])
        self.assertTrue(report['passed'], report)
        self.assertLess(report['maximum_motor_torque_nm'], 1.6)
        self.assertLess(report['maximum_target_step_rad'], .04)

    def test_changed_acceleration_rejects_force_balance(self):
        arrays = {k: v.copy() for k, v in self.arrays.items()}
        arrays['a'][10, 2] += 1.
        report = audit(self.model, self.config, arrays, arrays['desired_feet'])
        self.assertFalse(report['passed'])
        self.assertGreater(report['constraint_violation']['root_force_moment'], 7.)

    def test_cyclic_target_jump_rejects_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name in ('INPUT.json', 'RESULT.json'):
                shutil.copy2(FIXTURE/name, output/name)
            arrays = {k: v.copy() for k, v in self.arrays.items()}
            arrays['target'][0, 0] = arrays['target'][-1, 0] + .08
            np.savez_compressed(output/'trajectory.npz', **arrays)
            changed = digest(output/'trajectory.npz')
            result = json.loads((output/'RESULT.json').read_text())
            result['trajectory_sha256'] = changed
            (output/'RESULT.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'action range or target slew'):
                validate_trajectory(output/'trajectory.npz', changed, self.model.identity()['model_sha256'])

    def test_smooth_target_offset_cannot_replace_dynamics_consistent_target(self):
        arrays = {k: v.copy() for k, v in self.arrays.items()}
        arrays['target'][:, 0] += .005
        report = audit(self.model, self.config, arrays, arrays['desired_feet'])
        self.assertFalse(report['passed'])
        self.assertGreater(report['constraint_violation']['pd_target_identity'], .0049)

    def test_nonfinite_state_is_rejected(self):
        arrays = {k: v.copy() for k, v in self.arrays.items()}
        arrays['q'][3, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            audit(self.model, self.config, arrays, arrays['desired_feet'])

    def test_changed_bytes_and_failed_solve_cannot_reach_replay(self):
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            validate_trajectory(FIXTURE/'trajectory.npz', '0'*64, self.model.identity()['model_sha256'])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name in ('INPUT.json', 'RESULT.json', 'trajectory.npz'):
                shutil.copy2(FIXTURE/name, output/name)
            result = json.loads((output/'RESULT.json').read_text())
            result['status'] = 'solver_failed'
            (output/'RESULT.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'solved, model-bound'):
                validate_trajectory(output/'trajectory.npz', TRAJECTORY_SHA, self.model.identity()['model_sha256'])

    def test_native_pack_retains_physics_and_named_entry_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'pack'
            declared = json.loads((ROOT/'configs/locomotion_spark.json').read_text())
            declared['extra_mounts'].append(['/recorded/batch', '/standing_batch'])
            inputs = Path(tmp)/'inputs.json'
            inputs.write_text(json.dumps(declared))
            binding = prepare(FIXTURE, output, '/home/orionh/HEXAPOD_runs/restart_20260914/test_trajectory', inputs=inputs)
            self.assertIn(['/recorded/batch', '/standing_batch'], binding['extra_mounts'])
            for name in ('env.py', 'env_config.py', 'reservation.py', 'launch.py', 'evaluate.py'):
                self.assertEqual((output/'source/locomotion'/name).read_bytes(), (ROOT/'locomotion'/name).read_bytes())
            self.assertEqual((output/'source/locomotion/priors/replay_native.py').read_bytes(), (ROOT/'locomotion/priors/replay_native.py').read_bytes())
            with self.assertRaises(FileExistsError):
                prepare(FIXTURE, output, '/home/orionh/HEXAPOD_runs/restart_20260914/test_trajectory')


if __name__ == '__main__':
    unittest.main()
