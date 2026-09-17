"""Keep replica offsets and mixed toe/shaft patches visible to admission."""
import unittest

import numpy as np

from experiments.paper_walk.env import _classify_patches
from hexapod_env.tasks.mkii_rs05.capture import DiagnosticGeometry
from hexapod_env.tasks.mkii_rs05.contacts import classify_patches

LEGS = ['lf', 'lm', 'lr', 'rf', 'rm', 'rr']


def fixture(origins=None):
    origins = np.zeros((2, 3)) if origins is None else np.asarray(origins, dtype=float)
    names = ['body'] + [leg+'_'+part for leg in LEGS for part in ('coxa', 'femur', 'tibia')]
    meta = {'body_names': names, 'shapes': [{'body': 'lf_tibia',
        'shape_to_link': np.eye(4).tolist(), 'cap_bounds_m': [[.8, -.1, -.1], [1., .1, .1]],
        'cap_lower_x_m': .8, 'contact_offset_m': .01}]}
    geometry = DiagnosticGeometry(meta, {}, meta['body_names'])
    geometry.names = geometry.body_names
    poses = np.zeros((2, 19, 7)); poses[:, :, 6] = 1
    force = np.zeros((5, 1)); force[:4, 0] = [10, 10, 10, 10]
    points = np.zeros((5, 3)); points[:4, 0] = [.9, .7, .9, .7]
    points[:2] += origins[0]; points[2:4] += origins[1]
    normals = np.zeros((5, 3)); normals[:4, 2] = 1
    counts, starts = np.zeros((38, 1), int), np.zeros((38, 1), int)
    counts[[3, 22], 0] = 2; starts[22, 0] = 2
    data = [force, points, normals, np.zeros((5, 1)), counts, starts]
    return data, [(replica, body) for replica in range(2) for body in names], poses, origins, geometry


class PatchContactTests(unittest.TestCase):
    def test_replica_offset_preserves_toe_and_shaft_classification(self):
        args = fixture([[0, 0, 0], [2, -4, 0]])
        result = classify_patches(*args, LEGS)
        np.testing.assert_array_equal(result['distal_force_world_n'][:, 0, 2], [10, 10])
        np.testing.assert_array_equal(result['nonfoot_force_world_n'][:, 3, 2], [10, 10])
        self.assertTrue(result['nonfoot_contact'].all())
        self.assertEqual([p['category'] for p in result['patches']], ['toe', 'shaft', 'toe', 'shaft'])

    def test_mixed_patches_match_frozen_scorer_before_aggregation(self):
        data, mapping, poses, origins, geometry = fixture([[0, 0, 0], [3, 5, 0]])
        world = poses.copy(); world[:, :, :3] += origins[:, None]
        expected = _classify_patches(data, mapping, world, geometry, 2)
        actual = classify_patches(data, mapping, poses, origins, geometry, LEGS)
        for new, old in [('distal_force_world_n', 'distal_force_world'),
                         ('nonfoot_force_world_n', 'nonfoot_force_world'),
                         ('nonfoot_contact', 'nonfoot_contact')]:
            np.testing.assert_array_equal(actual[new], expected[old])
        self.assertEqual(actual['patches'], expected['patches'])

    def test_opposing_nonfoot_patches_cannot_cancel_the_failure(self):
        data, mapping, poses, origins, geometry = fixture()
        data[1][:4, 0] = .7
        data[2][1::2, 2] = -1
        result = classify_patches(data, mapping, poses, origins, geometry, LEGS)
        np.testing.assert_array_equal(result['nonfoot_force_world_n'], 0.)
        self.assertTrue(result['nonfoot_contact'].all())

    def test_corrupt_or_exhausted_contact_buffers_fail_closed(self):
        for corruption in ('overlap', 'overflow', 'nonfinite', 'normal', 'full'):
            data, mapping, poses, origins, geometry = fixture()
            if corruption == 'overlap': data[5][22] = 1
            if corruption == 'overflow': data[4][22] = 10
            if corruption == 'nonfinite': data[1][0] = np.nan
            if corruption == 'normal': data[2][0, 2] = .5
            if corruption == 'full': data[:4] = [value[:4] for value in data[:4]]
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                classify_patches(data, mapping, poses, origins, geometry, LEGS)


if __name__ == '__main__':
    unittest.main()
