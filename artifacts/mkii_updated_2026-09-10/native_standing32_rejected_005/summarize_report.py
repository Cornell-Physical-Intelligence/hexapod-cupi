"""Derive summaries from the saved source-bound report, not from unavailable raw traces."""
from pathlib import Path
import json,csv,io

def summarize(report):
 rows=report['replicas'];out=[]
 for r in rows:
  p=r['physical'];q=r['quiet']
  out.append({'env':r['env'],'combined_pass':r['pass'],'physical_pass':not bool(r['failed_physical_bounds']),
    'quiet_pass':q['pass'],'missing_six_toe_substeps':p['post_settle_missing_six_toe_substeps'],
    'max_quiet_sdk_rate_rms_rad_s':q['max_joint_velocity_rms_rad_s'],
    'max_quiet_joint_range_rad':q['max_joint_position_range_rad'],'max_quiet_planar_excursion_m':q['max_planar_excursion_m'],
    'max_requested_all_substeps_nm':p['max_requested_all_substeps_nm']})
 return {'scope':'Derived from original standing report only; no local full raw numerical replay',
  'num_envs':report['num_envs'],'controls':report['controls'],'substeps':report['substeps'],'combined_passes':sum(x['combined_pass']for x in out),
  'physical_passes':sum(x['physical_pass']for x in out),'quiet_passes':sum(x['quiet_pass']for x in out),
  'support_failure_envs':[x['env']for x in out if x['missing_six_toe_substeps']>0],
  'quiet_failure_envs':[x['env']for x in out if not x['quiet_pass']],
  'missing_support_samples_total':sum(x['missing_six_toe_substeps']for x in out),
  'missing_support_samples_range_nonzero':[min(x['missing_six_toe_substeps']for x in out if x['missing_six_toe_substeps']),max(x['missing_six_toe_substeps']for x in out)],
  'max_quiet_sdk_rate_rms_rad_s':max(x['max_quiet_sdk_rate_rms_rad_s']for x in out),
  'max_quiet_joint_range_rad':max(x['max_quiet_joint_range_rad']for x in out),'max_quiet_planar_excursion_m':max(x['max_quiet_planar_excursion_m']for x in out),
  'max_requested_all_substeps_nm':max(x['max_requested_all_substeps_nm']for x in out),
  'max_requested_saturation_fraction_400hz':max(r['physical']['max_requested_saturation_fraction_400hz']for r in rows),
  'all_controlled_nonfoot_substeps':sum(r['physical']['all_controlled_nonfoot_substeps']for r in rows),
  'per_environment':out,'physics_admission':False,'PPO_admission':False}

def csv_text(summary):
 f=io.StringIO();rows=summary['per_environment'];w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows);return f.getvalue()
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,default=Path(__file__).resolve().parent/'terminal/run/standing/standing_report.json');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=summarize(json.loads(a.report.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n')
