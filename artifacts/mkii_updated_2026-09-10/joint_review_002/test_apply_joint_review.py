"""Numerical regression coverage for the new coordinate/limit revision."""
import copy,math,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from apply_joint_review import apply_review,compare_revision,fk

def fixture():
 names=['body','lf_coxa','lf_femur','lf_tibia'];q=[.2,.4,-.8];types=['coxa_yaw','femur_pitch','tibia_pitch']
 joints=[]
 for i,typ in enumerate(types):
  R=Rotation.from_euler('xyz',[0,0,-.7]) if i==0 else Rotation.from_euler('xyz',[math.pi/2,0,.1])
  joints.append({'name':'lf_'+typ,'parent':names[i],'child':names[i+1],'xyz':[.1,.02,.03],'quaternion_xyzw':R.as_quat().tolist(),'axis':[0,0,1],
   'lower':-.2,'upper':.2,'cad_value':q[i],'default_value':0.})
 return {'links':[{'name':n,'mass':1+i*.1,'com':[.01,.02,.03],'inertia':np.diag([.2,.3,.4]).tolist()} for i,n in enumerate(names)],'joints':joints,
  'cad_pose':{j['name']:j['cad_value'] for j in joints},'parts':[{'id':i,'link':n,'xyz':[.02,.03,.04],'quaternion_xyzw':Rotation.from_euler('xyz',[.2,-.4,.3]).as_quat().tolist()} for i,n in enumerate(names)],'warnings':[]}

def review(model,shift=.42):
 d={'schema':1,'revision':'test_revision','joints':{}}
 for j in model['joints']:
  yaw=j['name'].endswith('coxa_yaw');limits=(-1.,1.5) if yaw else tuple(math.radians(x) for x in ((-120,80) if 'femur' in j['name'] else (-5,180)))
  d['joints'][j['name']]={'lower_rad':limits[0],'upper_rad':limits[1],'zero_shift_rad':shift if yaw else 0.,'bounds_frame':'previous_viewer_zero','provenance':{'test':True}}
 return d

class JointReviewTests(unittest.TestCase):
 def test_nonzero_yaw_rebase_preserves_every_CAD_pose_and_tensor(self):
  old=fixture();new,_=apply_review(old,review(old));check,_=compare_revision(old,new)
  self.assertTrue(check['passes']);self.assertEqual(new['links'],old['links']);self.assertEqual(new['parts'],old['parts'])
  j=new['joints'][0];self.assertAlmostEqual(j['cad_value'],.2-.42);self.assertAlmostEqual(j['lower'],-1-.42)
  self.assertAlmostEqual(j['absolute_zero_azimuth_deg'],math.degrees(-.7+.42))

 def test_pitch_zero_cannot_be_shifted(self):
  old=fixture();r=review(old);r['joints']['lf_femur_pitch']['zero_shift_rad']=.001
  with self.assertRaisesRegex(ValueError,'pitch zeros'):apply_review(old,r)

 def test_user_pitch_travel_survives_without_clamping_to_old_envelope(self):
  old=fixture();new,_=apply_review(old,review(old,-.2));byname={j['name']:j for j in new['joints']}
  self.assertEqual(byname['lf_tibia_pitch']['upper'],math.pi)
  self.assertAlmostEqual(math.degrees(byname['lf_femur_pitch']['lower']),-120)
  for j in old['joints'][1:]:
   n=byname[j['name']];self.assertEqual(j['quaternion_xyzw'],n['quaternion_xyzw']);self.assertEqual(j['xyz'],n['xyz']);self.assertEqual(j['cad_value'],n['cad_value'])

 def test_absolute_interval_can_be_unwrapped_across180_degrees(self):
  old=fixture();r=review(old);r['joints']['lf_coxa_yaw']['absolute_previous_zero_azimuth_deg']=math.degrees(-.7)+360
  new,_=apply_review(old,r);j=new['joints'][0]
  self.assertGreater(j['absolute_upper_azimuth_deg'],360);self.assertLess(j['absolute_lower_azimuth_deg'],j['absolute_upper_azimuth_deg'])

if __name__=='__main__':unittest.main()
