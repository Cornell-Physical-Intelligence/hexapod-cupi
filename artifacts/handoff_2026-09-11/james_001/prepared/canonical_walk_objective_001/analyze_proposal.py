"""Reproducible synthetic mathematics; does not ingest or qualify a native rollout."""
from pathlib import Path
import json
import numpy as np
from commands import program, held_out_cases, LEVELS
from objective import reward, terminal_margin, Weights


def fixture(command, fraction):
 c=np.asarray(command,float)[None];n=1
 return dict(command=c,linear_body=np.tile([c[0,0]*fraction,c[0,1]*fraction,0.],(n,8,1)),angular_body=np.tile([0.,0.,c[0,2]*fraction],(n,8,1)),gravity_body=np.tile([0.,0.,-1.],(n,8,1)),computed_torque=np.zeros((n,8,18)),applied_torque=np.zeros((n,8,18)),target_delta=np.zeros((n,18)),target_previous_delta=np.zeros((n,18)),requested_minus_executed=np.zeros((n,18)),interval_angle_rate=np.zeros((n,18)),contact_tangent_speed=np.zeros((n,8,6)),contact_active=np.ones((n,8,6),bool),valid_interval=np.ones(n,bool),terminated=np.zeros(n,bool),truncated=np.zeros(n,bool))


def analyze():
 plans=[program(i,0,0) for i in range(25)]
 zero=sum(s.controls for p in plans for s in p['segments'] if not any(s.command))
 rows=[]
 for command in [[.02,0,0],[.03,0,0],[0,0,.12],[.025,-.025,-.12]]:
  rows.append({'command':command,'rewards_by_measured_fraction':{str(f):float(reward(**fixture(command,f))['reward'][0])for f in [-1,0,.5,1,2]}})
 base=fixture([.02,0,0],0);epsilon=1e-6
 baseline=reward(**base)['reward'][0];base['linear_body'][:,:,0]=epsilon
 derivative=(reward(**base)['reward'][0]-baseline)/epsilon
 return {'schema':'canonical_omni_proposal_math_v1','scope':'Synthetic mathematics only; no native trajectory or hardware evidence',
  'control_dt_s':.02,'command_levels_requested_not_admitted':LEVELS,'zero_controls_in_balanced_cycle':zero,'balanced_cycle_controls':25*600,'zero_time_fraction':zero/(25*600),
  'unbalanced32row_first_epoch_zero_time_fraction':sum(s.controls for i in range(32)for s in program(i,0,0)['segments']if not any(s.command))/(32*600),
  'exact_balance_scope':'Twenty-five equal-duration strata, or25 epochs per row;32 finite rows need not be exactly20percent in one epoch',
  'reward_weights':vars(Weights()),'terminal_discount_bound_gamma_0_99':terminal_margin(),'stationary_positive_direction_finite_difference_reward_per_m_s':float(derivative),
  'stationary_is_not_proven_optimum':bool(derivative>0),'tracking_sensitivity':rows,'held_out_cases':[{**p,'segments':[vars(s)for s in p['segments']]}for p in held_out_cases(0)],
  'native_admitted':False,'automatic_curriculum_promotion':False,'Stage2_complete':False}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.write_text(json.dumps(analyze(),indent=2,sort_keys=True,allow_nan=False)+'\n')
