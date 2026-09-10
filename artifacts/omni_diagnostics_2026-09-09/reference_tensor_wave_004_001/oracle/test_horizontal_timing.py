"""Timing intervention contracts; frozen measured failure remains a failure."""
import unittest
import numpy as np
from wave_math import Swing
from wave_reference import AdvancedHorizontalSwing


class TimingTests(unittest.TestCase):
    def test_same_vertical_curve_and_original_endpoint_for_entire_swing(self):
        p0=np.array([.2,-.25,-.0005]);end=np.array([.19,-.32,-.0005])
        old=Swing(1.,2.,p0,end,.007);new=AdvancedHorizontalSwing(1.,2.,p0,end,.007,.8)
        for t in np.linspace(1.,3.,201):
            for a,b in zip(old.sample(t),new.sample(t)):
                self.assertEqual(a[2],b[2])
        np.testing.assert_array_equal(new.end,old.end)
        np.testing.assert_array_equal(new.sample(1.)[0],p0)
        for t in (2.6,2.61,2.8,3.):
            p,v,a=new.sample(t);np.testing.assert_array_equal(p[:2],end[:2])
            np.testing.assert_array_equal(v[:2],0);np.testing.assert_array_equal(a[:2],0)

    def test_early_endpoint_is_c2_and_derivatives_match_finite_difference(self):
        new=AdvancedHorizontalSwing(0.,2.,np.zeros(3),np.array([.01,-.07,0]),.007,.8)
        h=1e-5
        for t in (.1,.5,1.2,1.5999,1.6,1.8):
            p,v,a=new.sample(t);left=new.sample(t-h);right=new.sample(t+h)
            np.testing.assert_allclose((right[0]-left[0])/(2*h),v,atol=1e-9)
            np.testing.assert_allclose((right[1]-left[1])/(2*h),a,atol=1e-5)
        p,v,a=new.sample(1.6-1e-7)
        self.assertLess(abs(v[:2]).max(),1e-10);self.assertLess(abs(a[:2]).max(),1e-6)

    def test_fraction_one_matches_original_oracle_and_invalid_reconstruction_rejected(self):
        p0=np.array([.2,-.25,-.0005]);end=np.array([.19,-.32,-.0005])
        old=Swing(1.,2.,p0,end,.007);same=AdvancedHorizontalSwing(1.,2.,p0,end,.007,1.)
        for t in np.linspace(.9,3.1,223):
            for a,b in zip(old.sample(t),same.sample(t)):
                np.testing.assert_allclose(a,b,atol=1e-14,rtol=0)
        for fraction in (0.,.49,1.01,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):AdvancedHorizontalSwing(1.,2.,p0,end,.007,fraction)
        for duration in (0.,-1.,float('inf'),float('nan')):
            with self.assertRaises(ValueError):AdvancedHorizontalSwing(1.,duration,p0,end,.007,.8)

    def test_actual004_bound_is_not_relaxed_by_new_timing(self):
        # Exact returned measured RF toe, preload and original endpoint. Timing
        # changes future physics; it cannot make this existing datum pass.
        actual=np.array([-.18279841717074696,-.30394952889067245,-.00008546297431095973])
        preload=np.array([-.00349463412528761,-.0020781300893079613,-.0002884626545858471])
        endpoint=np.array([-.1808860745861631,-.31700126792352856,-.000539601443438037])
        self.assertGreater(np.linalg.norm(endpoint-actual-preload),.012)
        self.assertAlmostEqual(np.linalg.norm(endpoint-actual-preload),.012234497899895053)


if __name__=='__main__':unittest.main()
