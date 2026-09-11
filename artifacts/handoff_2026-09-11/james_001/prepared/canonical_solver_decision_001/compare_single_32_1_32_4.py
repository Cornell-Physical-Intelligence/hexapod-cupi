"""Hash-verified actual single-robot predecessor comparison. CPU only, no raw copies."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()

def summarize(base,selection):
 inventory=json.loads((base/'RAW_SHA256.json').read_text());consumed={}
 def read(rel):
  path=base/rel;expected=inventory[rel]
  assert sha(path)==expected['sha256'] and path.stat().st_size==expected['size_bytes'],rel
  consumed[rel]=expected
  return path
 prefix='run/standing/'
 state=json.loads(read(prefix+'state.json').read_text());session=json.loads(read(prefix+'session.json').read_text());reset=json.loads(read(prefix+'initial_reset.json').read_text());report=json.loads(read(prefix+'standing_report.json').read_text())
 names=session['joint_names'];rows={};previous=np.asarray(reset['post_reset']['joint_position_rad'],dtype=float)
 for filename in session['substep_files']:
  with np.load(read(prefix+filename))as z:
   q=z['joint_position_rad'].astype(float);interval=np.diff(np.concatenate([previous[None],q]),axis=0)/.0025
   assert np.array_equal(interval,z['interval_angle_rate_rad_s']),filename
   previous=q[-1]
   for key in ['joint_position_rad','joint_velocity_rad_s','interval_angle_rate_rad_s','computed_torque_nm','control_index','sequence']:
    rows.setdefault(key,[]).append(z[key].copy())
 arrays={key:np.concatenate(parts)for key,parts in rows.items()};assert np.array_equal(arrays['sequence'],np.arange(8000))
 post=arrays['control_index']>=200;q=arrays['joint_position_rad'][post,0].astype(float);sdk=arrays['joint_velocity_rad_s'][post,0].astype(float);fd=arrays['interval_angle_rate_rad_s'][post,0].astype(float)
 allj=[]
 for j,name in enumerate(names):
  allj.append({'name':name,'samples_400hz':len(q),'position_range_rad':float(np.ptp(q[:,j])),'sdk_mean_rad_s':float(sdk[:,j].mean()),'sdk_rms_rad_s':float(np.sqrt(np.mean(sdk[:,j]**2))),'interval_mean_rad_s':float(fd[:,j].mean()),'interval_rms_rad_s':float(np.sqrt(np.mean(fd[:,j]**2))),'sdk_minus_interval_integral_rad':float(np.sum(sdk[:,j]-fd[:,j])*.0025),'sdk_centered_rms_rad_s':float(np.sqrt(np.mean((sdk[:,j]-sdk[:,j].mean())**2)))})
 return {'selection':selection,'state_sha256':consumed[prefix+'state.json']['sha256'],'source_freeze_sha256':state['identity']['inspector_freeze_sha256'],'raw_inventory_sha256':sha(base/'RAW_SHA256.json'),'consumed_raw':consumed,'all_8000_interval_recurrences_exact':True,'post_settle_400hz_samples':int(post.sum()),'actual_report_pass':report['all_pass'],'actual_quiet_50hz':report['replicas'][0]['quiet'],'actual_physical':report['replicas'][0]['physical'],'joint_400hz':allj}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source003-run',type=Path,default=Path('tmp/canonical_native_standing_terminal_004'));p.add_argument('--source004-run',type=Path,default=Path('tmp/canonical_native_standing_terminal_005'));p.add_argument('--output',type=Path,default=Path(__file__).with_name('SINGLE_COMPARISON.json'));a=p.parse_args()
 result={'schema':'canonical_actual_solver_single_comparison_v1','scope':'Both actual8,000-step single runs, before any32/0 result; no native changes or velocity-channel substitution. Official gate remains SDK50Hz endpoint RMS;400Hz metrics below are separately named.','runs':[summarize(a.source003_run,'32/1'),summarize(a.source004_run,'32/4')],'limitations':['Finite differences are average angle rate over each2.5ms interval, not the instantaneous end-of-step velocity.','These constrained-pose/SDK comparisons do not independently identify friction, joint sleep or solver internals.','No assertion that a pass under a different solver setting establishes hardware fidelity.']}
 a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 for run in result['runs']:
  print(run['selection'],'pass',run['actual_report_pass'],'quietSDK50max',run['actual_quiet_50hz']['max_joint_velocity_rms_rad_s'])
  for row in run['joint_400hz']:
   if row['name']in ['lm_tibia_pitch','rm_tibia_pitch']:print(row)
