"""Independent CPU transport checks; only import the frozen owner read-only."""
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import Mock

import torch

OWNER = Path(__file__).resolve().parent.parent / 'phase3_sensor_transport_002'
EXPECTED_FREEZE = 'c6aab3c11cd64c3cde22950e546df4dc9970c8500f3ec8ddeed258f69a1ff781'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

new = load('_independent_transport_new', OWNER / 'sensor_model.py')
old = load('_independent_transport_old', OWNER / 'original/sensor_model.py')

class IndependentChecks(unittest.TestCase):
    def test_exact_freeze_and_inventory(self):
        freeze = OWNER / 'FREEZE_SHA256.json'
        self.assertEqual(hashlib.sha256(freeze.read_bytes()).hexdigest(), EXPECTED_FREEZE)
        files = json.loads(freeze.read_text())['files']
        self.assertEqual(len(files), 9)
        actual = {p.relative_to(OWNER).as_posix() for p in OWNER.rglob('*') if p.is_file()}
        self.assertEqual(actual, set(files) | {'FREEZE_SHA256.json'})
        for path, digest in files.items():
            self.assertEqual(hashlib.sha256((OWNER / path).read_bytes()).hexdigest(), digest, path)

    def test_default_configuration_and_portable_test_ast(self):
        self.assertEqual(asdict(old.DEFAULT_PHASE3_SENSOR_MODEL_CFG), asdict(new.DEFAULT_PHASE3_SENSOR_MODEL_CFG))
        trees = []
        for path in ('test_transport.py', 'integration/test_phase3_sensor_transport.py'):
            tree = ast.parse((OWNER / path).read_text())
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('old', 'new'):
                    node.value = ast.Constant('SOURCE_PATH_ONLY')
            trees.append(ast.dump(tree, include_attributes=False))
        self.assertEqual(*trees)

    def test_randomized_latest_valid_oracle_with_partial_reset(self):
        # Independent oracle retains the complete received history. The runtime
        # retains only latest values. Six rows, latency inversions, ties, masks,
        # and an in-flight reset exercise distinct histories in the same batch.
        for seed in range(24):
            rng = random.Random(seed)
            s = new._DelayedStream(new.TransportModelCfg(.01, .02, .02, 0., .2), seed)
            records = []
            for step in range(100):
                now = step * .005
                if step == 41:
                    s.reset([1, 4])
                    for record in records:
                        record['valid'][1] = record['valid'][4] = False
                if step < 80 and step % 2 == 0:
                    for _ in range(2 if step % 14 == 0 else 1):
                        jitter = rng.uniform(-.02, .02)
                        valid = [rng.random() > .28 for _ in range(6)]
                        values = [len(records) * 100 + row + 1 for row in range(6)]
                        record = dict(time=now, ready=now + .02 + jitter, sequence=len(records), valid=valid.copy(), values=values)
                        records.append(record)
                        s._jitter_rng.uniform = Mock(return_value=jitter)
                        with torch.inference_mode():
                            s.enqueue({'x': torch.tensor(values, dtype=torch.float32).reshape(6, 1)}, now, torch.tensor(valid))
                with torch.inference_mode():
                    frame = s.read(now)
                if frame is None:
                    self.assertFalse(any(r['ready'] <= now + 1e-9 for r in records))
                    continue
                for row in range(6):
                    eligible = [r for r in records if r['valid'][row] and r['ready'] <= now + 1e-9 and r['time'] <= now]
                    if not eligible:
                        self.assertFalse(frame.valid[row])
                        self.assertEqual(frame.values['x'][row].item(), 0.)
                        continue
                    best = max(eligible, key=lambda r: (r['time'], r['sequence']))
                    self.assertEqual(frame.values['x'][row].item(), best['values'][row])
                    self.assertEqual(s._order_sequence[row].item(), best['sequence'])
                    self.assertEqual(s._order_time_s[row].item(), best['time'])
                    age = torch.tensor(now, dtype=torch.float32) - torch.tensor(best['time'], dtype=torch.float32)
                    self.assertEqual(frame.age_s[row].item(), age.item())
                    self.assertEqual(frame.valid[row].item(), bool(0 <= age <= .2))
                self.assertFalse(torch.is_inference(s._latest_values['x']))

    def test_all_rows_partial_reset_retains_epoch_and_full_reset_changes_layout(self):
        s = new._DelayedStream(new.TransportModelCfg(.01, 0., 0., 0., .2), 0)
        with torch.inference_mode():
            s.enqueue({'x': torch.ones(2, 1)}, 2., torch.ones(2, dtype=torch.bool))
            s.read(2.)
        s.reset([0, 1])
        with self.assertRaises(ValueError):
            s.read(0.)
        with self.assertRaises(ValueError):
            s.enqueue({'x': torch.ones(2, 1)}, 0., torch.ones(2, dtype=torch.bool))
        s.reset(None)
        s.enqueue({'different_field': torch.ones(3, 2, dtype=torch.float64)}, 0., torch.ones(3, dtype=torch.bool))
        frame = s.read(0.)
        self.assertEqual(frame.values['different_field'].shape, (3, 2))
        self.assertEqual(frame.values['different_field'].dtype, torch.float64)
        self.assertTrue(frame.valid.all())

if __name__ == '__main__':
    unittest.main()
