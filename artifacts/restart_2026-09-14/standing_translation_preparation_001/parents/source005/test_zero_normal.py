from pathlib import Path
import importlib.util,json,unittest,hashlib
import numpy as np
from standing_math import Geometry,classify_contacts
HERE=Path(__file__).resolve().parent
class ExactInactiveTuple(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  d=HERE/'contact_failure_inputs';cls.meta=json.loads((d/'contact_view.json').read_text());cls.partial=json.loads((d/'failed_partial_step.json').read_text());cls.pose=np.asarray(cls.partial['link_pose_xyzw']);cls.names=['body','lf_coxa','lm_coxa','lr_coxa','rf_coxa','rm_coxa','rr_coxa','lf_femur','lm_femur','lr_femur','rf_femur','rm_femur','rr_femur','lf_tibia','lm_tibia','lr_tibia','rf_tibia','rm_tibia','rr_tibia']
  cls.sensors=[(0,p.rsplit('/',1)[1])for p in cls.meta['sensor_paths']]
  with np.load(d/'failed_contact_buffer.npz')as z:cls.data=[z[k]for k in ['force','point','normal','separation','counts','starts']]
  with np.load(HERE/'geometry/geometry_extrema.npz')as z:cls.geo=Geometry(json.loads((HERE/'geometry/geometry.json').read_text()),{k:z[k]for k in z.files},cls.names)
 def run_data(self,data):return classify_contacts(data,self.sensors,self.pose,self.geo,1)
 def test_actual_all153_records_preserved_and_exact127_inactive(self):
  spec=importlib.util.spec_from_file_location('_frozen_standing_math001',HERE/'history/standing_math_001.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  with self.assertRaisesRegex(ValueError,'Invalid patch normal'):old.classify_contacts(self.data,self.sensors,self.pose,self.geo,1)
  raw=[x.copy()for x in self.data];r=self.run_data(raw)
  self.assertEqual(r['used_patch_count'],153);self.assertEqual(len(r['patches']),153)
  marked=[p for p in r['patches']if p['inactive_zero_normal']];self.assertEqual([p['buffer_index']for p in marked],list(range(17,144)))
  self.assertTrue(all(p['normal_force_n']==0 and p['separation_m']==0 and p['normal_world']==[0.,0.,0.]for p in marked))
  self.assertTrue(r['distal_contact'].all());self.assertFalse(r['nonfoot_contact'].any())
  for a,b in zip(raw,self.data):np.testing.assert_array_equal(a,b)
  expected=np.zeros((1,6,3));legs=['lf','lm','lr','rf','rm','rr']
  for i,(count,start)in enumerate(zip(raw[4][:,0],raw[5][:,0])):
   if count:
    v=sum((float(raw[0][k,0])*raw[2][k] for k in range(int(start),int(start+count))),np.zeros(3));expected[0,legs.index(self.sensors[i][1][:2])]=v
  np.testing.assert_array_equal(r['distal_force_world'],expected)
 def test_nonzero_force_any_invalid_normal_still_rejects(self):
  for force in [float(np.nextafter(np.float32(0),np.float32(1))),1.,-1.]:
   d=[x.copy()for x in self.data];d[0][17,0]=force
   with self.assertRaisesRegex(ValueError,'Invalid patch normal'):self.run_data(d)
 def test_exact_zero_rule_has_no_tolerance_or_nonfinite_exception(self):
  for array,index,value in [(2,(17,0),np.float32(1e-30)),(3,(17,0),np.float32(1e-30)),(0,(17,0),np.nan),(1,(17,0),np.nan),(2,(17,0),np.nan)]:
   d=[x.copy()for x in self.data];d[array][index]=value
   with self.assertRaises(ValueError):self.run_data(d)
 def test_inactive_rule_does_not_bypass_buffer_ranges(self):
  d=[x.copy()for x in self.data];d[5][15,0]=17
  with self.assertRaisesRegex(ValueError,'Overlapping'):self.run_data(d)
  d=[x.copy()for x in self.data];d[4][12,0]=1024
  with self.assertRaisesRegex(ValueError,'overflow'):self.run_data(d)
if __name__=='__main__':unittest.main()
