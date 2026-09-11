import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from experiments.c_length_study.tools.launch_omni_repair_spark import continuation_screen


class RepairAllocationTests(unittest.TestCase):
    def setUp(self):
        self.base=dict(terminations=0,truncations=0,saturation=.06,planar_error=.04,
                      yaw_error=.09,power=2.3,stand_joint_velocity_rms=.74,
                      scenarios={'forward':dict(terminations=0,planar_error_mps=.04,yaw_error_rad_s=.09,torque_saturation_fraction=.06)})

    def test_measurable_improvement_allows_short_continuation(self):
        self.assertTrue(continuation_screen({**self.base,'saturation':.04},self.base)['continue'])

    def test_unchanged_run_does_not_auto_extend(self):
        self.assertFalse(continuation_screen(self.base,self.base)['continue'])

    def test_direction_regression_and_plateau_stop_continuation(self):
        better={**self.base,'saturation':.03}
        bad={**better,'scenarios':{'forward':{**self.base['scenarios']['forward'],'planar_error_mps':.1}}}
        self.assertFalse(continuation_screen(bad,self.base)['continue'])
        self.assertFalse(continuation_screen(better,self.base,better)['continue'])

    def test_safety_and_tracking_regressions_cannot_buy_lower_torque(self):
        for regression in ({'terminations':1},{'planar_error':.09},{'yaw_error':.2},
                           {'truncations':1},{'stand_joint_velocity_rms':1.2}):
            self.assertFalse(continuation_screen({**self.base,'saturation':.03,**regression},self.base)['continue'])


if __name__=='__main__':unittest.main()
