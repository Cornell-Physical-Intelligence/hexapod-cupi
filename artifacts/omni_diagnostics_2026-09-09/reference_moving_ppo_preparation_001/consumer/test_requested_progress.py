import ast
import math
from pathlib import Path
import types
import unittest
import torch
from test_moving_reward import fixture
from moving_reward import phase,tracking
from reward_adapter import build_reward_callable


class Tests(unittest.TestCase):
    def test_governor_cannot_redefine_requested_motion_as_successful_stillness(self):
        a=fixture([[.005,0.,0.]])
        a[1]['factor'].zero_();a[1]['command_target'].zero_();a[1]['command'].zero_()
        scope=phase(*a)
        paused=tracking(torch.zeros(1,2,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),scope)
        moving=tracking(torch.tensor([[.005,0.]],dtype=torch.float64),torch.zeros(1,dtype=torch.float64),scope)
        self.assertAlmostEqual(float(paused['linear_tracking']),math.exp(-1))
        self.assertEqual(float(paused['linear_progress']),0.)
        # Complete controlled weighted tracking/progress/torque/power comparison.
        # Other assigned components are zero; these are not measured episodes.
        paused_total=4*float(paused['linear_tracking'])+2-.03*.64
        moving_total=4*float(moving['linear_tracking'])+2+1.5-.03*.81-.015*4
        self.assertGreater(moving_total,paused_total)
        self.assertTrue(scope['commanded_motion'][0]);self.assertFalse(scope['settled_quiet'][0])

    def test_governed_pauses_both_yaw_signs_and_finite_stop(self):
        for yaw in (-.015,.015):
            a=fixture([[0.,0.,yaw]]);a[1]['factor'].zero_();a[1]['command_target'].zero_();a[1]['command'].zero_()
            t=tracking(torch.zeros(1,2,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),phase(*a))
            self.assertAlmostEqual(float(t['yaw_tracking']),math.exp(-1));self.assertEqual(float(t['yaw_progress']),0)
        a=fixture([[0.,0.,0.]]);a[1]['command'][0,0]=.002
        self.assertEqual(float(phase(*a)['tracking_target'][0,0]),.002)

    def test_exact_source_reward_formula_binding(self):
        path=Path(__file__).resolve().parent.parent/'reference_physics_adapter_009/source_009/tools/omni_flat_env.py'
        module=types.SimpleNamespace(__file__=str(path))
        call=build_reward_callable(module)
        self.assertEqual(call.__name__,'_get_rewards')
        self.assertIn('_moving_score',call.__code__.co_names)
        self.assertIn('_episode_elapsed_s',call.__code__.co_names)
        self.assertIn('slew_commands',call.__code__.co_names)

if __name__=='__main__':unittest.main()
