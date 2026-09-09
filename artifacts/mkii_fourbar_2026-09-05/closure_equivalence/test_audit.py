"""Independent Rodrigues tests of the global redundant-row bounds."""
import unittest

import numpy as np

from audit import bounds


def rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis, axis)+np.sin(angle)*skew


def normalized(v):
    return np.asarray(v)/np.linalg.norm(v)


def planar():
    return {"pA": np.array([0., 0., 0.]), "pB": np.array([.03, 0., 0.]),
            "pD": np.array([0., .0775, 0.]), "pC0": np.array([.03, .0775, 0.]),
            "pC1": np.array([.03, .0775, 0.]),
            **{key: np.array([0., 0., 1.]) for key in ("nA", "nB", "nD", "nC0", "nC1")}}


class BoundTests(unittest.TestCase):
    def test_exact_planar_geometry_has_zero_redundant_rows(self):
        result = bounds(planar())
        self.assertEqual(result["all_angle_c0_local_axial_gap_bound_m"], 0.)
        self.assertEqual(result["all_angle_c_axis_chord_bound"], 0.)
        self.assertTrue(result["equivalence_within_1e_minus_8"])

    def test_one_mm_axial_offset_is_not_equivalent(self):
        g = planar()
        g["pC1"][2] = .001
        result = bounds(g)
        self.assertAlmostEqual(result["all_angle_c0_local_axial_gap_bound_m"], .001)
        self.assertFalse(result["equivalence_within_1e_minus_8"])

    def test_tilted_axis_cannot_be_silently_ignored(self):
        g = planar()
        g["nC1"] = normalized([.001, 0., 1.])
        result = bounds(g)
        self.assertGreater(result["all_angle_c_axis_chord_bound"], .00099)
        self.assertFalse(result["equivalence_within_1e_minus_8"])

    def test_global_bounds_cover_independent_rotations_with_imperfect_axes(self):
        g = planar()
        g["nB"] = normalized([.003, -.001, 1.])
        g["nD"] = normalized([-.002, .004, 1.])
        g["nC0"] = normalized([.0035, -.002, 1.])
        g["nC1"] = normalized([-.001, .003, 1.])
        g["pC1"][2] = .0002
        bound = bounds(g)
        for angles in np.random.default_rng(7).uniform(-20*np.pi, 20*np.pi, (200, 3)):
            a, b, d = [rotation(g[key], angle) for key, angle in zip(("nA", "nB", "nD"), angles)]
            p0 = g["pA"]+a@(g["pB"]-g["pA"]+b@(g["pC0"]-g["pB"]))
            p1 = g["pD"]+d@(g["pC1"]-g["pD"])
            n0, n1 = a@b@g["nC0"], d@g["nC1"]
            self.assertLessEqual(abs(n0@(p0-p1)), bound["all_angle_c0_local_axial_gap_bound_m"]+1e-12)
            self.assertLessEqual(np.linalg.norm(n0-n1), bound["all_angle_c_axis_chord_bound"]+1e-12)


if __name__ == "__main__":
    unittest.main()
