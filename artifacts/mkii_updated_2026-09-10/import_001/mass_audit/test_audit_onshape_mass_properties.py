"""Focused checks of the audit's risky parsing and mass transformation math."""
import io
import pickle
import unittest
import numpy as np
from audit_onshape_mass_properties import RestrictedCadUnpickler, aggregate, physical_tensor, rounded_check

class AuditMathTests(unittest.TestCase):
    def test_forbidden_pickle_callable_is_rejected(self):
        # A valid protocol-0 global/reduce payload; eval itself must never load.
        payload = b"cbuiltins\neval\n(S'1+1'\ntR."
        with self.assertRaisesRegex(pickle.UnpicklingError, 'Unsupported pickle global'):
            RestrictedCadUnpickler(io.BytesIO(payload)).load()

    def test_allowlisted_numpy_roundtrip(self):
        expected = np.arange(9, dtype=np.float64).reshape(3, 3)
        restored = RestrictedCadUnpickler(io.BytesIO(pickle.dumps(expected, protocol=4))).load()
        np.testing.assert_array_equal(restored, expected)

    def test_parallel_axis_sum_preserves_translated_tensor(self):
        rows = [{'mass_kg': 2., 'com_export_m': [1., 2., 3.], 'inertia_about_com_export_kg_m2': np.diag([.1, .2, .3]).tolist()},
                {'mass_kg': 2., 'com_export_m': [-1., 2., 3.], 'inertia_about_com_export_kg_m2': np.diag([.1, .2, .3]).tolist()}]
        result = aggregate(rows)
        self.assertEqual(result['mass_kg'], 4.)
        np.testing.assert_allclose(result['com_export_m'], [0., 2., 3.])
        np.testing.assert_allclose(result['inertia_about_com_export_kg_m2'], np.diag([.2, 4.4, 4.6]))

    def test_nonphysical_triangle_and_zero_mass_inertia_rejected(self):
        self.assertFalse(physical_tensor(np.diag([1., 1., 3.]), 1.)['physically_consistent'])
        self.assertFalse(physical_tensor(np.eye(3), 0.)['physically_consistent'])
        self.assertTrue(physical_tensor(np.zeros((3, 3)), 0.)['physically_consistent'])

    def test_urdf_decimal_precision_limits(self):
        self.assertTrue(rounded_check(.12345644, '.123456')['within_written_precision'])
        self.assertFalse(rounded_check(.1234566, '.123456')['within_written_precision'])
        self.assertTrue(rounded_check(1.233e-5, '1.23e-5')['within_written_precision'])

if __name__ == '__main__':
    unittest.main()
