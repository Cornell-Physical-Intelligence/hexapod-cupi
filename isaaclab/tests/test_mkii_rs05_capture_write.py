"""Check that the diagnostic reads the native force before physics advances."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


class NativeWriteCaptureTests(unittest.TestCase):
    def test_readback_follows_command_write_and_preserves_pre_step_bytes(self):
        path = ROOT/'packages/hexapod_env/hexapod_env/tasks/mkii_rs05/env.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HexapodMkiiRs05Env')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_write_capture_forces')
        namespace = {'native_numpy': lambda value: np.array(value, copy=True)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        force = np.array([[0., 0.]])
        events = []
        def write():
            events.append('write')
            force[:] = [[3., 5.]]
        def read():
            events.append('read')
            return force
        env = SimpleNamespace(_capture_original_write=write, _capture=object(), _pending_row={},
            _robot=SimpleNamespace(root_view=SimpleNamespace(get_dof_actuation_forces=read)),
            _canonical_index=torch.tensor([1, 0]))
        namespace['_write_capture_forces'](env)
        force[:] = [[7., 11.]]
        self.assertEqual(events, ['write', 'read'])
        self.assertEqual(env._pending_row['native_input_pre_nm'].tolist(), [[5., 3.]])
        with self.assertRaisesRegex(RuntimeError, 'Repeated command write'):
            namespace['_write_capture_forces'](env)


if __name__ == '__main__':
    unittest.main()
