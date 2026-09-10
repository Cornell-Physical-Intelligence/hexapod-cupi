import unittest
from unittest.mock import Mock
import numpy as np
from record_candidate_video import initial_rgb_frame
class RenderWarmupTests(unittest.TestCase):
    def test_lazy_empty_then_ready_without_physics(self):
        env=Mock();ready=np.arange(48,dtype=np.uint8).reshape(4,4,3)
        env.render.side_effect=[np.zeros((0,),dtype=np.uint8),np.zeros((4,4,3),dtype=np.uint8),ready]
        frame,report=initial_rgb_frame(env)
        self.assertTrue(np.array_equal(frame,ready));self.assertEqual(len(report['render_attempts']),3)
        self.assertEqual(env.sim.render.call_count,2);env.step.assert_not_called();self.assertEqual(report['physics_steps'],0)
    def test_persistent_blank_has_finite_failure(self):
        env=Mock();env.render.return_value=np.zeros((4,4,3),dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError,'bounded render-only'):initial_rgb_frame(env,max_attempts=3)
        self.assertEqual(env.render.call_count,3);env.step.assert_not_called()
