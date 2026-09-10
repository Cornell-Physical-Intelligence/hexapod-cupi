exec(open('/tmp/hexapod-urdf-joint-recovery-20260910/inspect_geometry.py').read().split('for f,n in')[0])
fs=[i for i in instances if i['file']=='femur_first_stage.stl'];cs=[i for i in instances if i['file']=='first_joint_spacer.stl'];ts=[i for i in instances if i['file']=='tibia.stl']
for i in instances:
 if i['file'] in ['femur_first_stage.stl','tibia_attatchment_plate.stl','tibia.stl','motor_flange.stl','motor_bearing_holder.stl','bearing_insert.stl'] or i['file'].startswith('1_1_06_eb463_507'):
  # middle right/negative x assembly
  if i['center'][0]<-.1 and abs(i['center'][1])<.1:
   print(i['index'],i['file'],'C',np.round((np.linalg.inv(cs[0]['T'])@i['T'])[:3,3]*1000,3).tolist(),'F',np.round((np.linalg.inv(fs[0]['T'])@i['T'])[:3,3]*1000,3).tolist(),'T',np.round((np.linalg.inv(ts[0]['T'])@i['T'])[:3,3]*1000,3).tolist())
