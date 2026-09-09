#!/usr/bin/env python3
"""CPU arithmetic on the exact frozen reward; no simulator or policy evaluation."""
import hashlib
import json
import math
from pathlib import Path
import torch
from hexapod_core import fourbar_v1 as contract
from hexapod_env.tasks.mkii_fourbar_v1.math import MotorCoordinates, reward_terms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_COMMIT = "c2af43ca0f384a4c2c7ab8f1d627f309dc78a683"
SOURCE_IDENTITY = "c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc"
KIN = contract.load_kinematics(ROOT / contract.KINEMATICS_PATH)
COORD = MotorCoordinates(contract.TREE_JOINT_NAMES, KIN, dtype=torch.float64)

def scenario(command=(0.,0.,0.), linear=(0.,0.,0.), yaw=0., **overrides):
    z = torch.zeros(1,18,dtype=torch.float64)
    values = dict(command=torch.tensor([command],dtype=torch.float64),
        linear_navigation=torch.tensor([linear],dtype=torch.float64),
        angular_navigation=torch.tensor([[0.,0.,yaw]],dtype=torch.float64),
        gravity_body=torch.tensor([[0.,0.,-1.]],dtype=torch.float64),
        active_torque=z.clone(), active_velocity=z.clone(), active_acceleration=z.clone(),
        active_position=COORD.default[None],soft_limits=COORD.soft_limits,
        action=z.clone(),previous_action=z.clone(),
        height=torch.tensor([KIN['nominal_height_m']],dtype=torch.float64),nominal_height=KIN['nominal_height_m'],
        nonfoot_contacts=torch.tensor([0]), support_count=torch.tensor([6]),
        foot_slip=torch.tensor([0.],dtype=torch.float64),clipping_nm=z.clone(),overload_nm=z.clone())
    values.update(overrides)
    terms={k:float(v[0]) for k,v in reward_terms(**values).items()}
    return dict(terms_per_second=terms, reward_per_control=sum(terms.values())*contract.POLICY_DT_S)

cases={}
for command in ((0.,0.,0.),(.05,0.,0.),(.10,0.,0.),(.15,0.,0.),(0.,0.,.30),(.15,.15,.30)):
    cases[str(command)]={'stationary':scenario(command),
        'ideal_tracking_only':scenario(command,linear=(*command[:2],0.),yaw=command[2])}
I=math.sqrt(math.pi)*math.erf(1)/2
stand_expectation=3*(.2+.8*I**2)+.6*(.2+.8*I)
penalty_cases={}
for accel in (100.,300.,1000.):
    penalty_cases[f'all_18_acceleration_{accel}_rad_s2']=scenario(active_acceleration=torch.full((1,18),accel,dtype=torch.float64))
for torque in (1.2,5.5):
    penalty_cases[f'all_18_applied_torque_{torque}_nm']=scenario(active_torque=torch.full((1,18),torque,dtype=torch.float64))
penalty_cases['six_pads_sliding_0.15_mps']=scenario(foot_slip=torch.tensor([6*.15**2],dtype=torch.float64))
penalty_cases['height_error_20_mm']=scenario(height=torch.tensor([KIN['nominal_height_m']+.02],dtype=torch.float64))
penalty_cases['one_nonfoot_contact']=scenario(nonfoot_contacts=torch.tensor([1]))
penalty_cases['two_support_feet']=scenario(support_count=torch.tensor([2]))
penalty_cases['all_actions_flip_minus1_plus1']=scenario(action=torch.ones(1,18,dtype=torch.float64),previous_action=-torch.ones(1,18,dtype=torch.float64))
qlo=COORD.default-.3
qhi=COORD.default+.3
assert bool((qlo >= COORD.soft_limits[:,0]).all() and (qhi<=COORD.soft_limits[:,1]).all())
p_slew=math.erfc(.04/(.3*.15*math.sqrt(2)))
source_files=['isaaclab/train_mkii_fourbar.py','isaaclab/deploy/run-mkii-fourbar-campaign',
    'packages/hexapod_core/hexapod_core/fourbar_v1.py',
    'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/config.py',
    'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py',
    'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/math.py',
    'packages/hexapod_env/hexapod_env/ppo_cfg.py',
    'packages/hexapod_env/hexapod_env/actuators/rs05_v2_model.py',contract.KINEMATICS_PATH]
result=dict(schema='hexapod.ppo_feasibility_cpu_arithmetic.v1',simulation_or_policy_evaluation=False,
    source_commit=SOURCE_COMMIT,source_identity=SOURCE_IDENTITY,
    files_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files},
    cases=cases,penalty_cases=penalty_cases,
    stationary_tracking_distribution=dict(uniform_integral=I,expected_per_second=stand_expectation,
        ideal_per_second=3.6,fraction_of_ideal=stand_expectation/3.6,
        assumption='Ideal upright stationary state; command draw distribution, not observed occupancy; other penalties excluded'),
    action_envelope=dict(all_18_default_plus_minus_0_3_inside_soft_limits=True,
        minimal_soft_limit_margin_rad=float(torch.minimum(qlo-COORD.soft_limits[:,0],COORD.soft_limits[:,1]-qhi).min()),
        target_slew_max_rad_s=.04/.02,substep_max_increment_rad=.04/32,
        initial_unslewed_offset_noise_std_rad=.3*.15,
        centered_first_step_slew_probability_per_motor=p_slew,
        centered_first_step_probability_any_motor_slew=1-(1-p_slew)**18,
        exploration_assumption='Independent zero-mean Gaussian at reset target; not measured initial actor means'),
    workload=dict(full_iterations=1000,num_envs=512,rollout_steps=24,
        transitions=1000*512*24,global_control_steps=1000*24,global_physics_steps=1000*24*32,
        simulated_seconds_per_environment=1000*24*.02,
        aggregate_simulated_hours=1000*512*24*.02/3600,
        minibatch_size=512*24//4,gradient_minibatches=1000*5*4,
        scratch_transitions=3*64*24,scratch_seconds_per_environment=3*24*.02,
        discount_time_constant_s=-.02/math.log(.99),gae_time_constant_s=-.02/math.log(.99*.95)))
(HERE/'reward_arithmetic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('stationary_tracking_distribution','action_envelope','workload')},indent=2))
