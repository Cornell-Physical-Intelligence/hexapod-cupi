"""Independent pad-frame geometry and exact float32 CSV encoding checks."""
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import numpy as np

spec=importlib.util.spec_from_file_location('first_event_checks',Path(__file__).with_name('first_event.py'))
event=importlib.util.module_from_spec(spec);spec.loader.exec_module(event)


class FirstEventChecks(unittest.TestCase):
    def test_native_xyzw_rotated_foot_sphere_height(self):
        legs=('lf','lm','lr','rf','rm','rr')
        root=ET.fromstring('<robot>'+''.join(f'<link name="{leg}_tibia"><collision><origin xyz=".1 0 0"/><geometry><sphere radius=".01"/></geometry></collision><collision><origin xyz="0 0 .2"/><geometry><sphere radius=".005"/></geometry></collision></link>' for leg in legs)+'</robot>')
        class Traces:
            def block(self,values,key,names):
                if key=='body_link_pos_w':return np.broadcast_to([10.,-20.,.2],(3,2,3))
                return np.broadcast_to([0.,np.sqrt(.5),0.,np.sqrt(.5)],(3,2,4))
        heights=event.minimum_pad_height(None,Traces(),root,legs)
        np.testing.assert_allclose(heights,np.full((3,2,6),.09),atol=1e-15,rtol=0)

    def test_nine_significant_digits_roundtrip_float32_bits(self):
        values=np.concatenate([np.random.default_rng(62).normal(size=10000).astype(np.float32),
                               np.array([0.,-0.,1e-38,-1e-38,3.4028235e38],np.float32)])
        decoded=np.array([float(format(float(value),'.9g')) for value in values],dtype=np.float32)
        np.testing.assert_array_equal(values.view(np.uint32),decoded.view(np.uint32))


if __name__=='__main__':unittest.main()
