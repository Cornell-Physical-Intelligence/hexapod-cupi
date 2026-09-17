"""Audit the improved forward replay without changing its verdict."""
from pathlib import Path
import ast
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.paper_walk.analyze import load_recording
from experiments.paper_walk.env import _diagnostic_servo, _diagnostic_rotation, emitted_target
from experiments.paper_walk.env_config import KD, LEGS
from experiments.paper_walk.evaluation import score_recording
from experiments.trajectory_optimization.model import RobotModel
from experiments.trajectory_optimization.optimize import Config, audit

A = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def arrays(p):
    with np.load(p, allow_pickle=False) as d: return {k:d[k].copy() for k in d.files}

# Preserve the earlier independent patch classifier as an explicit audit input.
helper = ROOT/'artifacts/restart_2026-09-14/paper_walk_execution_001/verification_evaluate_015/verify.py'
nodes = [n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ('classify', 'support_summary')]
assert len(nodes) == 2
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(helper), 'exec'))

base = A/'replay_pack_001/replay_001/standing/evaluation'
report, capture, c, n, integrity, binding = load_recording(base)
remote = read(A/'replay_pack_001/REMOTE_INVENTORY.json')
local = {str(p.relative_to(A/'replay_pack_001/replay_001')):sha(p) for p in sorted((A/'replay_pack_001/replay_001').rglob('*')) if p.is_file()}
assert remote['files'] == local
assert report['acquisition_complete'] and report['failure'] is None and capture['failure'] is None
assert report['controls'] == 1000 and capture['steps'] == 8000
assert report['pose_forcing'] is False and report['resets_during_trial'] == 0
assert report['checkpoint_sha256'] is None
state = read(A/'replay_pack_001/replay_001/standing/state.json')
freeze = read(A/'replay_pack_001/source/FREEZE_SHA256.json')
assert state['identity']['source_files'] == freeze
assert all(sha(A/'replay_pack_001/source'/name) == digest for name,digest in freeze.items())
assert state['identity']['learned_policy'] is False
assert state['eligible_for_motion_prior'] and not state['stage2_complete']
assert np.array_equal(n['explicit_counter'], np.arange(1,8001)+capture['initial_counter'])
assert np.array_equal(n['pre_joint_position_rad'][1:], n['joint_position_rad'][:-1])
assert np.array_equal(n['pre_joint_velocity_rad_s'][1:], n['joint_velocity_rad_s'][:-1])
for key in ('joint_position_rad','joint_velocity_rad_s','root_pose_xyzw','joint_target_rad','distal_contact','toe_xyz_world_m'):
    assert np.array_equal(n[key][7::8], c[key]), key
expected = _diagnostic_servo(n['pre_joint_position_rad'][:,0], n['pre_joint_velocity_rad_s'][:,0], n['joint_target_rad'][:,0], np.full(18,12.,np.float32), np.asarray(KD,np.float32))
for key,value in zip(('computed_torque_nm','applied_torque_nm','effort_ceiling_nm'),expected):
    assert np.array_equal(n[key][:,0],value), key
assert np.array_equal(n['native_input_pre_nm'],n['applied_torque_nm'])
assert np.array_equal(n['joint_target_rad'],np.repeat(c['joint_target_rad'],8,axis=0))
initial=read(base/'native400hz/initial_state.json')
neutral=torch.tensor(initial['q'],dtype=torch.float32);held=neutral.clone()
native=read(A/'replay_pack_001/replay_001/standing/native/native_readback.json')
limits=torch.tensor(native['limits'],dtype=torch.float32)
solved=arrays(A/'solve_smooth_001/trajectory.npz')
target=torch.tensor(solved['target'],dtype=torch.float32)
action_error=0.
for i,action in enumerate(c['policy_action']):
    expected_action=((target[i%len(target)]-neutral[0])/.35)[None,:]
    action_error=max(action_error,float(abs(expected_action.numpy()-action).max()))
    assert action_error < 1e-7
    held=emitted_target(torch.from_numpy(action),held,neutral,limits[:,:,0],limits[:,:,1],.35,.04)
    assert np.array_equal(held.numpy(),c['joint_target_rad'][i])
transitions=arrays(A/'replay_pack_001/replay_001/standing/recorded_transitions.npz')
for stored,key in [('states','amp_state_before'),('next_states','amp_state_after'),('commands','command')]:
    assert np.array_equal(transitions[stored],c[key][:,0])
geometry_path=ROOT/'artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry.json'
geometry={row['body']:row for row in read(geometry_path)['shapes']}
packets=patches=0
with (base/'native400hz/contacts.jsonl').open() as stream:
    for line in stream:
        packet=json.loads(line)
        assert packet['sequence']==packets and packet['explicit_counter']==int(n['explicit_counter'][packets])
        feet,other,nonfoot,_=classify(packet,n['link_pose_xyzw'][packets,0],capture['body_names'],geometry)
        assert np.array_equal(feet,n['distal_force_world_n'][packets,0])
        assert np.array_equal(other,n['nonfoot_force_world_n'][packets,0])
        assert np.array_equal(np.linalg.norm(feet,axis=-1)>1,n['distal_contact'][packets,0])
        assert nonfoot==bool(n['nonfoot_contact'][packets,0])
        packets+=1;patches+=len(packet['patches'])
assert packets==8000
case=report['cases'][0]
scored=score_recording(c,{**read(base/'declaration.json'),'profile':case['profile'],'env_index':0,'command':case['command'],'target_slew_rad':.04,'control_dt_s':.02,'seed':report['seed']})
assert scored['checks']==report['results'][0]['checks']
assert scored['failed_bounds']==report['results'][0]['failed_bounds']==[]
physical={
    'joint_limits':bool(np.all(n['joint_position_rad']>=limits[:,:,0].numpy()-2e-6) and np.all(n['joint_position_rad']<=limits[:,:,1].numpy()+2e-6)),
    'joint_speed':bool(np.all(abs(n['joint_velocity_rad_s'])<=np.asarray(native['native_max_velocity'])+2e-6)),
    'applied_torque':bool(abs(n['applied_torque_nm']).max()<=1.60001),
    'nonfoot_contact':bool(n['nonfoot_contact'].mean()<=.001),
    'non_toe_clearance':bool(n['minimum_non_toe_floor_m'].min()>=-.001),
    'plate_height':bool(n['root_pose_xyzw'][:,:,2].min()>=.055)}
assert all(physical.values()) and report['results'][0]['native_motor_and_joint_checks_pass']
cleanup=read(A/'replay_pack_001/replay_001/cleanup.json')
assert cleanup['cleanup_checked'] and cleanup['inspections'][-1]['absent']
assert not cleanup['reservation_released'] and not any(cleanup['resources'].values())
assert 'MainPID=0' in remote['service'] and 'ExecMainStatus=0' in remote['service']
config=Config(**read(A/'solve_smooth_001/INPUT.json')['config'])
expanded_audit=audit(RobotModel(),config,solved,solved['desired_feet'])
assert expanded_audit['passed']
velocity=c['velocity_navigation_mps'][100:,0,0]
result={'schema':'optimized_trajectory_replay_audit_v1','utc':datetime.now(timezone.utc).isoformat(),
    'source_base':'7eea4c905f93ba61df00d5afbd090584fc0fffdc','trajectory_sha256':sha(A/'solve_smooth_001/trajectory.npz'),
    'remote_files_verified':len(local),'physics_steps':packets,'contact_patches_reclassified':patches,
    'servo_and_targets_bitexact':True,'action_cpu_gpu_max_abs':action_error,'scorer_checks_exact':True,'transitions_exact':True,
    'physical_checks':physical,'expanded_cpu_audit':expanded_audit,'original_result':report['results'][0],
    'root_displacement_world_m':(c['root_pose_xyzw'][-1,0,:3]-np.asarray(initial['root'])[0,:3]).tolist(),
    'forward_velocity_range_after_settle_mps':[float(velocity.min()),float(velocity.max())],
    'maximum_actual_target_step_rad':float(abs(np.diff(np.concatenate([neutral.numpy()[None],c['joint_target_rad']]),axis=0)).max()),
    'support':support_summary(n,0,8000),'physics_binding':binding,
    'helper_source':str(helper.relative_to(ROOT)),'helper_sha256':sha(helper),'audit_source_sha256':sha(Path(__file__)),
    'video_sha256':sha(base/'rollout.mp4'),'cleanup_verified':True,'learned_policy':False,'eligible_for_motion_prior':True,'stage2_complete':False}
with (Path(__file__).parent/'RESULT.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({k:result[k] for k in ('remote_files_verified','physics_steps','contact_patches_reclassified','root_displacement_world_m','forward_velocity_range_after_settle_mps','maximum_actual_target_step_rad')}))
