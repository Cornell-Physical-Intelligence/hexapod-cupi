import copy,hashlib,json,math,struct,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import analyze_remote as a
class Checks(unittest.TestCase):
 def test_actual_report_original_failure(self):
  audit=a.read(HERE/'inputs/audit.json');self.assertEqual(a.sha(HERE/'inputs/audit.json'),a.AUDIT_SHA)
  r=audit['standing_report']['replicas'];self.assertEqual(sum(x['pass']for x in r),10);self.assertEqual(sum(not x['quiet']['failed_bounds']for x in r),24);self.assertEqual(sum(bool(x['physical']['post_settle_missing_six_toe_substeps'])for x in r),22)
 def test_product_rounding_exact_source(self):
  import numpy as np
  p={'env':0,'body':'lm_tibia','category':'toe','buffer_index':0,'normal_force_n':float(np.float32(1.234567)),'normal_world':np.array([.1234,.4567,.7891],np.float32).tolist()}
  expected=p['normal_force_n']*np.asarray(p['normal_world'],np.float32)
  self.assertEqual(a.products(p),expected.tolist());self.assertNotEqual(a.products(p),[p['normal_force_n']*x for x in p['normal_world']])
  toe,body,counts=a.aggregate([p],1);self.assertEqual(toe[0][1],expected.tolist());self.assertEqual(counts[0,'lm_tibia'],1)
  with self.assertRaises(ValueError):a.aggregate([p,p],1)
 def test_zero_force_evidence_retained(self):
  p={'env':0,'body':'rr_tibia','category':'toe','buffer_index':4,'normal_force_n':0.,'normal_world':[0.,0.,0.]}
  toe,body,counts=a.aggregate([p],1);self.assertEqual(counts[0,'rr_tibia'],1);self.assertEqual(toe[0][-1],[0.,0.,0.])
 def test_sensor_mapping_by_names_not_index(self):
  c={'sensor_paths':['/Robot_003/rr_tibia','/Robot/body','/Robot_003/body','/Robot/rr_tibia'],'filter_paths':[['/Ground']]*4}
  m=a.sensor_indices(c,['/Robot','/Robot_003'],['body','rr_tibia']);self.assertEqual(m[1,'rr_tibia'],0);self.assertEqual(m[0,'body'],1)
  c['sensor_paths'][0]=c['sensor_paths'][1]
  with self.assertRaises(ValueError):a.sensor_indices(c,['/Robot','/Robot_003'],['body','rr_tibia'])
 def test_com_position_rotation_and_translation(self):
  k=math.sqrt(.5);self.assertAlmostEqual(a.body_com_position([2,3,4,0,0,k,k],[1,0,0])[0],2);self.assertAlmostEqual(a.body_com_position([2,3,4,0,0,k,k],[1,0,0])[1],4)
 def test_real_single_numeric_channels(self):
  p=HERE.parent/'canonical_native_standing_terminal_006/run/standing/substeps_002.npz'
  if not p.exists():self.skipTest('Actual single raw fixture external; original hash remains audit-bound')
  audit=a.read(HERE/'inputs/audit.json');self.assertEqual(a.sha(p),audit['standing_one_admission_inventory'][p.name]['sha256'])
  q=a.numeric(p,'joint_position_rad',(800,1,18));pre=a.numeric(p,'pre_joint_position_rad',(800,1,18));iv=a.numeric(p,'interval_angle_rate_rad_s',(800,1,18))
  self.assertTrue(all(y==(x-z)/.0025 for x,z,y in zip(q,pre,iv)));self.assertEqual(len(a.numeric(p,a.LINK,(800,1,19,6))),91200);self.assertEqual(len(a.numeric(p,a.FLOOR,(800,19,1,3))),45600)
 def test_runtime_is_stdlib_read_only(self):
  text=(HERE/'analyze_remote.py').read_text()
  for token in ['import numpy','subprocess','ssh','open(\'w','write_text','write_bytes']:self.assertNotIn(token,text)
 def test_full_synthetic_stream_event_matrix_disagreement(self):
  import numpy as np
  audit=a.read(HERE/'inputs/audit.json');names=audit['session']['joint_names'];bodies=audit['session']['body_names']
  with tempfile.TemporaryDirectory()as td:
   root=Path(td);source=root/'source';source.mkdir();run=root/'run';d=run/'standing';d.mkdir(parents=True)
   for i in range(109):(source/str(i)).write_text('fixture')
   sm={p.name:a.sha(p)for p in source.iterdir()};(source/'FREEZE_SHA256.json').write_text(json.dumps(sm))
   def js(name,value):(d/name).write_text(json.dumps(value))
   session=copy.deepcopy(audit['session']);session.update(root_paths=['/Robot']);session['solver_diagnostics']['channels']={a.LINK:[1,19,6],a.FLOOR:[19,1,3]}
   report=copy.deepcopy(audit['standing_report']);report['replicas']=report['replicas'][:1];report['replicas'][0]['physical']['post_settle_missing_six_toe_substeps']=1;report['replicas'][0]['pass']=False
   js('session.json',session);js('standing_report.json',report);js('contact_view.json',{'sensor_paths':['/Robot/'+b for b in reversed(bodies)],'filter_paths':[['/Ground']]*19,'capacity':1024})
   js('native_readback.json',{'body_names':bodies,'joint_names':names,'masses':[[1.]*19],'coms':[[[0.,0.,0.]for _ in bodies]]});js('initial_reset.json',{});js('solver_readback.json',{});js('legacy_friction_readback.json',{})
   pose=np.zeros((800,1,19,7),np.float32);pose[:,:,:,6]=1.;zeros=np.zeros((800,1,18),np.float32)
   for c in range(10):
    s=np.arange(c*800,(c+1)*800);f=np.zeros((800,1,6,3));f[:,:,:,2]=2.;floor=np.zeros((800,19,1,3),np.float32)
    for l in a.LEGS:floor[:,list(reversed(bodies)).index(l+'_tibia'),0,2]=2.
    if c==2:f[0,0,1,2]=float(np.float32(.9));floor[0,list(reversed(bodies)).index('lm_tibia'),0,2]=3.
    np.savez_compressed(d/f'substeps_{c:03d}.npz',sequence=s,explicit_counter=s+3,control_index=s//8,time_s=(s+1)*.0025,distal_contact=np.linalg.norm(f,axis=-1)>1,distal_force_world_n=f,joint_position_rad=zeros,pre_joint_position_rad=zeros,joint_velocity_rad_s=zeros,interval_angle_rate_rad_s=zeros.astype(float),link_pose_xyzw=pose,link_com_velocity_world=np.zeros((800,1,19,6),np.float32),floor_contact_force_matrix_world_n=floor,root_com_velocity=np.zeros((800,1,6),np.float32),applied_torque_nm=zeros)
   with(d/'contacts.jsonl').open('w')as stream:
    for s in range(8000):
     ps=[{'env':0,'body':l+'_tibia','category':'toe','buffer_index':i,'normal_force_n':float(np.float32(.9))if s==1600 and i==1 else 2.,'normal_world':[0.,0.,1.],'inactive_zero_normal':False}for i,l in enumerate(a.LEGS)]
     stream.write(json.dumps({'sequence':s,'explicit_counter':s+3,'patches':ps})+'\n')
   fake={'audit_verified':True,'expected_invocation':a.INV,'raw_acquisition_completed':True,'native_state':{'standing_pass':False,'explicit_steps_completed':8000},'session':session,'standing_report':report,'raw_inventory':{'run/standing/'+p.name:{'sha256':a.sha(p),'size_bytes':p.stat().st_size}for p in d.iterdir()}}
   ap=root/'audit.json';ap.write_text(json.dumps(fake))
   with patch.object(a,'N',1),patch.object(a,'AUDIT_SHA',a.sha(ap)),patch.object(a,'SOURCE_SHA',a.sha(source/'FREEZE_SHA256.json')):result=a.analyze(run,source,ap)
   self.assertEqual(len(result['events']),1);event=result['events'][0];self.assertEqual(event['sequence'],1600);self.assertEqual(event['missing_legs'],['lm']);self.assertEqual(len(event['window']),3)
   now=event['window'][1];self.assertEqual(now['missing_leg_channel_comparison']['lm']['tibia_floor_matrix_norm_n'],3.);self.assertEqual(now['used_patch_count_env'],6);self.assertEqual(result['pressure_summary_all']['maximum'],6);self.assertGreater(result['maximum_all_body_floor_minus_patch_vector_norm_n_in_selected_rows'],2.);self.assertTrue(result['raw_and_source_reverified_after_analysis'])
if __name__=='__main__':unittest.main()
