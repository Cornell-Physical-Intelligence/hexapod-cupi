"""Check the approved feature interpretation against native observation fields."""

import unittest

import numpy as np
import torch

from locomotion.amp.features import extract_features, feature_contract


class AMPFeatureTests(unittest.TestCase):
    def state(self):
        return {
            "q": np.arange(36, dtype=np.float32).reshape(2, 18),
            "dq": np.full((2, 18), 2.0, np.float32),
            "linear": np.full((2, 3), 3.0, np.float32),
            "angular": np.full((2, 3), 4.0, np.float32),
            "root": np.tile(np.array([1, 2, 5, 0, 0, 0, 1], np.float32), (2, 1)),
            "toe_body": np.arange(36, dtype=np.float32).reshape(2, 6, 3),
        }

    def test_native_policy_and_numpy_export_preserve_historical_layout(self):
        state = self.state()
        expected = np.concatenate(
            (
                state["q"],
                state["dq"],
                state["linear"],
                state["angular"],
                state["root"][:, 2:3],
                state["toe_body"].reshape(2, 18),
            ),
            axis=1,
        )
        tensors = {name: torch.from_numpy(value) for name, value in state.items()}
        observed = extract_features(tensors)
        np.testing.assert_array_equal(observed.numpy(), expected)
        np.testing.assert_array_equal(extract_features(state), expected)
        self.assertEqual(observed.dtype, torch.float32)
        self.assertEqual(observed.shape, (2, 61))

    def test_contract_has_absolute_joints_and_all_toe_coordinates(self):
        contract = feature_contract()
        self.assertEqual(contract["fields"][0]["frame"], "native_joint_absolute")
        self.assertEqual(contract["fields"][-1]["slice"], [43, 61])
        self.assertEqual(contract["leg_order"], ["lf", "lm", "lr", "rf", "rm", "rr"])
        contract["fields"].clear()
        self.assertEqual(len(feature_contract()["fields"]), 6)

    def test_six_heights_cannot_substitute_for_six_positions(self):
        state = self.state()
        state["toe_body"] = state["toe_body"][:, :, 2]
        with self.assertRaisesRegex(ValueError, "toe_body"):
            extract_features(state)


if __name__ == "__main__":
    unittest.main()
