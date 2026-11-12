"""Keep native contact validation and per-patch classification across source adapters."""

import unittest
from types import SimpleNamespace

import numpy as np

from locomotion.evaluation.capture import classify_patches


class ContactTests(unittest.TestCase):
    def fixture(self):
        force = np.array([[3.0], [-3.0], [2.0], [0.0]])
        point = np.zeros((4, 3))
        point[2, 0] = 0.15
        normal = np.tile([0.0, 0.0, 1.0], (4, 1))
        separation = np.zeros((4, 1))
        counts, starts = np.zeros((19, 1), dtype=int), np.zeros((19, 1), dtype=int)
        counts[0, 0], counts[1, 0], starts[1, 0] = 2, 1, 2
        sensor_map = [(0, "body"), (0, "lf_tibia")]
        poses = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (1, 2, 1))
        geometry = SimpleNamespace(
            names=["body", "lf_tibia"],
            shapes={
                "lf_tibia": {
                    "shape_to_link": np.eye(4),
                    "cap_bounds_m": [[0.1, -0.02, -0.02], [0.2, 0.02, 0.02]],
                    "contact_offset_m": 0.01,
                    "cap_lower_x_m": 0.1,
                }
            },
        )
        return [force, point, normal, separation, counts, starts], sensor_map, poses, geometry, 1

    def test_actual_patches_keep_cancellation_and_distal_point_classification(self):
        fixture = self.fixture()
        result = classify_patches(*fixture)
        self.assertEqual(result["used_patch_count"], 3)
        self.assertTrue(result["nonfoot_contact"][0])
        self.assertEqual(float(abs(result["nonfoot_force_world"]).sum()), 0.0)
        self.assertTrue(result["distal_contact"][0, 0])
        self.assertEqual([p["category"] for p in result["patches"]], ["body", "body", "toe"])
        fixture[0][1][2, 0] = 0.05
        self.assertEqual(classify_patches(*fixture)["patches"][-1]["category"], "shaft")

    def test_native_layout_indexing_capacity_and_patch_validity_remain_required(self):
        def change(data, index, row, value):
            data[index][row] = value

        mutations = (
            ("Incomplete sensor", lambda data: data.__setitem__(4, np.zeros((18, 1), dtype=int))),
            ("Negative contact", lambda data: change(data, 5, 0, -1)),
            ("overflow/range", lambda data: change(data, 4, 0, 5)),
            ("Overlapping", lambda data: change(data, 5, 1, 1)),
            ("capacity exhausted", lambda data: change(data, 4, 1, 2)),
            ("Nonfinite used", lambda data: change(data, 0, 0, np.nan)),
            ("Invalid patch normal", lambda data: change(data, 2, 0, [0.0, 0.0, 0.5])),
        )
        for message, mutate in mutations:
            with self.subTest(message=message):
                fixture = self.fixture()
                mutate(fixture[0])
                with self.assertRaisesRegex(ValueError, message):
                    classify_patches(*fixture)

    def test_inactive_zero_normal_patch_stays_valid(self):
        fixture = self.fixture()
        fixture[0][0][0] = 0.0
        fixture[0][2][0] = 0.0
        self.assertTrue(classify_patches(*fixture)["patches"][0]["inactive_zero_normal"])


if __name__ == "__main__":
    unittest.main()
