"""Recorded command histories must distinguish actual ramps from step delivery."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("ramped_summary", Path(__file__).with_name("summarize.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RampTraceTests(unittest.TestCase):
    def fixture(self):
        initial = np.array([[.1, -.55], [-.2, .3]])
        endpoints = np.array([initial+.04, initial, initial-.02])
        start = np.concatenate((initial[None], endpoints[:-1]), axis=0)
        delivered = start[:,None]+((np.arange(16)+1)/16.)[None,:,None,None]*(endpoints-start)[:,None]
        return delivered.astype(np.float32).astype(float), np.repeat(endpoints.astype(np.float32).astype(float)[:,None],16,axis=1), initial.astype(np.float32).astype(float)

    def test_float32_linear_delivery_passes_both_directions(self):
        delivered, endpoints, initial = self.fixture()
        stats = module.ramp_metrics(delivered, endpoints, initial)
        self.assertTrue(stats["pass"])
        self.assertEqual(stats["control_environment_intervals"], 6)
        self.assertEqual(stats["target_values_checked"], 192)
        self.assertGreater(stats["max_delivered_minus_endpoint_rad"], .03)

    def test_zero_order_hold_and_delayed_endpoint_are_detected(self):
        delivered, endpoints, initial = self.fixture()
        step = module.ramp_metrics(endpoints, endpoints, initial)
        self.assertFalse(step["pass"])
        self.assertGreater(step["max_linear_interpolation_residual_rad"], .03)
        delivered[:, -1] = delivered[:, -2]
        self.assertFalse(module.ramp_metrics(delivered, endpoints, initial)["pass"])

    def test_endpoint_drift_and_wrong_initial_history_are_detected(self):
        delivered, endpoints, initial = self.fixture()
        endpoints[0, 1, 0, 0] += .0001
        self.assertFalse(module.ramp_metrics(delivered, endpoints, initial)["pass"])
        delivered, endpoints, initial = self.fixture()
        initial[0, 1] += .01
        self.assertFalse(module.ramp_metrics(delivered, endpoints, initial)["pass"])

    def test_missing_substep_and_nonfinite_data_are_rejected(self):
        delivered, endpoints, initial = self.fixture()
        with self.assertRaises(ValueError):
            module.ramp_metrics(delivered[:, :-1], endpoints[:, :-1], initial)
        delivered[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            module.ramp_metrics(delivered, endpoints, initial)


if __name__ == "__main__":
    unittest.main()
