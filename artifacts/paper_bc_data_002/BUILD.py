"""Recorded artifact builder: actual native rows only; no training or physics."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paper_walk.env import inverse_rotate, navigation
from experiments.paper_walk.env_config import JOINT_NAMES

RAW = ROOT/'artifacts/restart_2026-09-14/paper_walk_execution_001/results_replay_001/standing/native_replay'
AMP_SHA = '22f7b04b0c7540f7a27cbe4d13e23c09dbd3a2363ee0792f292e6293b16b9515'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(name):
    with np.load(RAW/name, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def main():
    torch.set_num_threads(2)
    report = json.loads((RAW/'report.json').read_text())
    verified = {name: sha(RAW/name) for name in report['raw_files']}
    assert verified == report['raw_files']
    assert verified['realized_prior.npz'] == AMP_SHA
    original, raw, native = load('realized_prior.npz'), load('replay_controls.npz'), load('replay_substeps.npz')
    assert report['complete'] and report['all_commands_accepted'] and report['failure'] is None
    accepted = [x['env'] for x in report['replicas'] if x['accepted']]
    assert accepted == list(range(31))
    assert not raw['terminated'].any() and not raw['truncated'].any()
    # Predeclared onset screens copy the existing replay numbers; row selection
    # applies to complete cycle-1 windows, never individual successful instants.
    onset = []
    for e in range(32):
        c = raw['commands'][200,e]; state = raw['next_states'][200:260,e]
        v = np.array([-state[:,37].mean(), state[:,36].mean(), state[:,41].mean()])
        speed = np.linalg.norm(c[:2])
        linear = float(v[:2]@c[:2]/speed**2) if speed > 1e-8 else None
        yaw = float(v[2]/c[2]) if abs(c[2]) > 1e-8 else None
        sl = slice(1600,2080)
        rms = float(np.sqrt(np.mean((native['joint_position_rad'][sl,e].astype(float)-native['joint_target_rad'][sl,e])**2)))
        sat = float((np.abs(native['requested_torque_nm'][sl,e])>1.6).mean(0).max())
        failures = []
        if e not in accepted: failures.append('excluded_original_whole_replica')
        if rms >= .05: failures.append('actual_target_joint_rms')
        if sat >= .02: failures.append('requested_saturation_fraction')
        if any(x is not None and x < .1 for x in (linear,yaw)): failures.append('commanded_motion_not_realized')
        if native['nonfoot_contact'][sl,e].any(): failures.append('nonfoot_contact')
        if native['fall_or_joint_violation'][sl,e].any(): failures.append('fall_or_joint_violation')
        if np.abs(native['applied_torque_nm'][sl,e]).max() > 1.60001: failures.append('applied_torque_cap')
        onset.append({'env':e,'command_index':report['command_index_by_env'][e], 'command':c.tolist(),
                      'controls':[200,259], 'accepted':not failures, 'failed':failures,
                      'actual_target_joint_rms_rad':rms,'requested_saturation_fraction':sat,
                      'linear_projection_fraction':linear,'yaw_projection_fraction':yaw})
    onset_env = np.array([x['env'] for x in onset if x['accepted']], dtype=np.int64)
    commands = sorted({x['command_index'] for x in onset if x['accepted']})
    assert len(onset_env)==25 and commands==[0,1,2,3,4,5,6,7,8,9,13,14,17,18,19]
    zero = []
    for e in (0,21):
        assert np.all(raw['commands'][:,e] == 0) and np.all(raw['actions'][:200,e] == 0)
        prefix = slice(0,1600); settled = slice(1200,1600)
        saturation = float((np.abs(native['requested_torque_nm'][prefix,e])>1.6).mean(0).max())
        rms = float(np.sqrt(np.mean(native['joint_velocity_rad_s'][settled,e].astype(float)**2,axis=0)).max())
        support = int(native['distal_contact'][settled,e].sum(-1).min())
        assert saturation==0 and not native['nonfoot_contact'][prefix,e].any()
        assert not native['fall_or_joint_violation'][prefix,e].any()
        assert np.abs(native['applied_torque_nm'][prefix,e]).max() <= 1.60001
        assert support==6 and rms<.03
        zero.append({'env':e,'controls':[0,199], 'phase':'actual_reset_and_settling_then_settled_hold',
                     'requested_saturation_fraction_full_4s':saturation,'final_1s_minimum_support':support,
                     'final_1s_worst_joint_velocity_rms_rad_s':rms,
                     'scope':'Only final1s is labeled settled; neither1s nor4s substitutes for a20s/32s quiet gate.'})
    # Keep the first 1860 rows byte-identical in their original order.
    t0 = 260+original['phase_index']; e0 = original['env_index']
    t1 = np.repeat(np.arange(200,260),len(onset_env)); e1 = np.tile(onset_env,60)
    t2 = np.repeat(np.arange(200),2); e2 = np.tile([0,21],200)
    times = np.concatenate([t0,t1,t2]); envs = np.concatenate([e0,e1,e2])
    categories = np.repeat(np.array([0,1,2],np.int8),[1860,1500,400])
    assert len(set(zip(times.tolist(),envs.tolist())))==3760 and times.max()==319
    fields = ['states','next_states','observations','actions','commands','requested_joint_target_rad','applied_joint_target_rad']
    data = {k:raw[k][times,envs].copy() for k in fields}
    for k in fields: assert np.array_equal(data[k][:1860],original[k]),k
    data.update(env_index=envs, control_index=times, cycle_index=raw['cycle_index'][times],
                phase_index=raw['phase_index'][times], source_kind=categories,
                physics_counter_after=raw['physics_counter'][times], physics_counter_before=raw['physics_counter'][times]-8,
                original_steady_row=np.concatenate([np.arange(1860),np.full(1900,-1)]),
                zero_phase=np.where(categories!=2,-1,np.where(times<150,0,1)).astype(np.int8))
    data['velocity_targets_navigation_mps'] = data['states'][:,[37,36,38]].copy()
    data['velocity_targets_navigation_mps'][:,0] *= -1
    assert all(np.isfinite(v).all() for v in data.values())
    assert np.array_equal(data['observations'][:,210:213],data['commands'])
    neutral = np.array(report['neutral_joint_position_rad'],np.float32)
    prev = np.zeros((len(times),18),np.float32); positive = times>0
    prev[positive]=(raw['applied_joint_target_rad'][times[positive]-1,envs[positive]]-neutral)/np.float32(.35)
    previous_error = float(np.abs(prev-data['observations'][:,213:]).max())
    target_error = float(np.abs(neutral+np.float32(.35)*data['actions']-data['requested_joint_target_rad']).max())
    assert previous_error < 2e-7 and target_error < 1e-7
    assert np.array_equal(data['next_states'][:,:18],native['joint_position_rad'][(times+1)*8-1,envs])
    assert np.array_equal(data['next_states'][:,18:36],native['joint_velocity_rad_s'][(times+1)*8-1,envs])
    assert np.array_equal(data['states'][positive],raw['next_states'][times[positive]-1,envs[positive]])
    history_times = np.maximum(times[:,None]-np.arange(4,-1,-1),0)
    history = data['observations'][:,:210].reshape(-1,5,42)
    assert np.array_equal(history,raw['observations'][history_times,envs[:,None],168:210])
    states = raw['states'][history_times,envs[:,None]]
    omega = states[:,:,39:42][:,:,[1,0,2]].copy(); omega[:,:,0] *= -1
    q = states[:,:,:18]-neutral; dq=states[:,:,18:36]
    assert np.array_equal(history[:,:,:3],omega)
    assert np.array_equal(history[:,:,6:24],q) and np.array_equal(history[:,:,24:],dq)
    mask = history_times>0
    quat = native['root_pose_xyzw'][history_times[mask]*8-1,np.broadcast_to(envs[:,None],history_times.shape)[mask],3:]
    gravity = navigation(inverse_rotate(torch.from_numpy(quat),torch.tensor([0.,0.,-1.]).expand(len(quat),-1))).numpy()
    gravity_error=float(np.abs(history[:,:,3:6][mask]-gravity).max())
    assert gravity_error<1e-7
    np.savez_compressed(OUT/'bc_dataset.npz',**data)
    # Independent file readback validates every stored field, not only a hash.
    with np.load(OUT/'bc_dataset.npz',allow_pickle=False) as stored:
        assert set(stored.files)==set(data)
        assert all(np.array_equal(stored[k],data[k]) for k in data)
    provenance = []
    for i,(t,e) in enumerate(zip(times,envs)):
        provenance.append({'dataset_row':i,'source_kind':int(categories[i]),'raw_control':int(t),'env':int(e),
                           'physics_counter_before':int(data['physics_counter_before'][i]),
                           'physics_counter_after':int(data['physics_counter_after'][i]),
                           'original_steady_row':int(data['original_steady_row'][i])})
    (OUT/'ROW_PROVENANCE.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in provenance))
    count_zero=int(np.all(data['commands']==0,axis=-1).sum())
    selection = {'schema':'canonical_bc_only_dataset_selection_v1', 'selection_approved_in_principle_by_root':True,
                 'scope':'BC-only candidate, no fitting or native action. Original AMP prior stays unchanged.',
                 'original_steady_rows':1860,'onset_rows':1500,'zero_reset_settle_rows':400,'total_rows':3760,
                 'zero_command_rows':count_zero,'zero_fraction':count_zero/3760,
                 'source_kind_labels':{'0':'original_accepted_steady','1':'screened_cycle1_command_start_or_zero_continuation','2':'actual_zero_reset_settle'},
                 'zero_phase_labels':{'-1':'not_added_zero_prefix','0':'reset_and_settling_controls0_149','1':'settled_final1s_controls150_199'},
                 'onset_command_indices':commands,'onset_replica_screen':onset,'zero_trajectory_screen':zero,
                 'all_commands_preserved_in_steady_rows':True,'onset_after_4s_settling_not_reset_start':True,
                 'no_walking_to_stop_examples':True,'cycle3_excluded':True,'independent_validation_claim':False,
                 'original_amp_path':str(RAW/'realized_prior.npz'),'original_amp_sha256':AMP_SHA,
                 'bc_dataset_sha256':sha(OUT/'bc_dataset.npz'),'row_provenance_sha256':sha(OUT/'ROW_PROVENANCE.jsonl'),
                 'source_inputs':{**{str((RAW/k).relative_to(ROOT)):v for k,v in verified.items()},
                                  str((RAW/'report.json').relative_to(ROOT)):sha(RAW/'report.json')},
                 'model_sha256':report['model_sha256'],'urdf_sha256':report['urdf_sha256'],
                 'physics_identity':report['physics_identity'],'joint_names':list(JOINT_NAMES),
                 'velocity_target_scope':'Actual pre-hold AMP native body-origin linear velocity rotated to navigation [-body_y,body_x,body_z]; not COM velocity, not a policy input.',
                 'future_velocity_loss_adopted':False,'physical_admission':False,'stage2_complete':False}
    save(OUT/'SELECTION.json',selection)
    save(OUT/'RECONSTRUCTION.json',{'schema':'canonical_bc_dataset_reconstruction_v1','verified':True,
        'original_1860_rows_exact':True,'distinct_raw_control_env_pairs':3760,'heldout_cycle3_rows':0,
        'full_file_readback_exact':True,'finite_all_arrays':True,'commands_exact':True,
        'recorded_proprio_history_across_rows_exact':True,'angular_q_dq_from_amp_history_exact':True,
        'initial_history':'Actual recorded reset history repeated; initial gravity copied from that native observation, not independently recoverable from AMP61.',
        'subsequent_gravity_from_native_quaternion_max_error':gravity_error,
        'previous_actual_action_max_float_error':previous_error,'normalized_target_reconstruction_max_rad':target_error,
        'q_dq_actual_native_endpoints_exact':True,'before_amp_equals_previous_native_endpoint_for_noninitial_rows':True,
        'amp_prior_after_build_sha256':sha(RAW/'realized_prior.npz'),'amp_prior_unchanged':sha(RAW/'realized_prior.npz')==AMP_SHA,
        'bc_dataset_sha256':sha(OUT/'bc_dataset.npz'),'no_fitting':True,'no_native_action':True})
    print(json.dumps({'bc_dataset':str(OUT/'bc_dataset.npz'),'sha256':sha(OUT/'bc_dataset.npz'),
                      'rows':3760,'zero_rows':count_zero,'zero_fraction':count_zero/3760,'reconstruction_verified':True}),flush=True)


if __name__=='__main__':
    main()
