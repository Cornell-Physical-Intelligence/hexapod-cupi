"""Bounded new fine-tune configuration; no automatic train allocation."""
import copy
SCHEMA='direct315_stand_stop_caps_v1'
def options(branch):
 if branch not in ('curriculum','caps'):raise ValueError('Only the declared two ablations exist')
 return {'schema':SCHEMA,'branch':branch,'maximum_updates':10,'controls_per_update':256,'replicas':128,
 'caps':{'temporal_weight':.1 if branch=='caps' else 0.,'spatial_weight':.1 if branch=='caps' else 0.,'noise_scale':1.,'noise_seed':1157},
 'command_schedule':'25percent_quiet_other_rows_8s_motion_8s_stop','checkpoint_sha256':'1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'}
def configure(cfg,value,iterations):
 from caps import validated_options
 if value!=options(value.get('branch')):raise ValueError('Unexpected declared ablation options')
 if iterations not in (2,10) or isinstance(iterations,bool):raise ValueError('Only explicit2-update smoke or10-update bounded comparison')
 c=copy.deepcopy(cfg)
 if c['obs_groups']!={'actor':['policy'],'critic':['critic']} or c.get('clip_actions') is not None:raise ValueError('Exact315/318 observation groups and unclipped wrapper required')
 c['num_steps_per_env']=256;c['save_interval']=1;c['max_iterations']=iterations
 c['algorithm']['class_name']='caps_ppo:CapsPPO';c['algorithm']['caps_options']=validated_options(value['caps'])
 # The checkpoint loader retains the original learned network/normalizers,
 # explicitly resets std to0.10 and freshAdam at5e-5; adaptive schedule retained.
 return c
