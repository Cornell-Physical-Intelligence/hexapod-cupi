"""Independent read-only CPU audit of recorded BC timing and fitted inference.

Writes only this fresh artifact. Does not run fit scripts, create a simulator,
change weights, import native SDKs, or change a qualification gate.
"""
from pathlib import Path
import ast
import datetime
import hashlib
import importlib.util
import json
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
A = HERE.parent
ROOT = A.parents[2]
PINS = {}


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def pin(path):
    PINS[str(path.relative_to(ROOT))] = sha(path)
    return path


def load(path):
    with np.load(pin(path), allow_pickle=False) as source:
        return {k: source[k].copy() for k in source.files}


def delta(x, y):
    return float(np.max(np.abs(np.asarray(x, dtype=float)-np.asarray(y, dtype=float))))


def function(path, name):
    matches = [n for n in ast.walk(ast.parse(path.read_text()))
               if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
    assert len(matches) == 1
    return ast.dump(matches[0], include_attributes=False)


def emit(action, held, neutral, lower, upper):
    # Independent NumPy rendition, float32 desired target and directed-rounded
    # float64 slew envelope, preserving original named joint/action clamps.
    desired = np.clip(neutral + np.float32(.35)*np.clip(action, -1, 1), lower, upper)
    lo = np.maximum(lower.astype(float), held.astype(float)-.04)
    hi = np.minimum(upper.astype(float), held.astype(float)+.04)
    target = np.maximum(np.minimum(desired.astype(float), hi), lo).astype(np.float32)
    target = np.where(target.astype(float)>hi, np.nextafter(target, np.float32(-np.inf)), target)
    target = np.where(target.astype(float)<lo, np.nextafter(target, np.float32(np.inf)), target)
    return target


def main():
    torch.set_num_threads(2)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    pin(Path(__file__).resolve())
    data_dir = ROOT/'artifacts/paper_bc_data_002'
    raw_dir = A/'results_replay_001/standing/native_replay'
    for path in (data_dir/'BUILD.py', data_dir/'RECONSTRUCTION.json', data_dir/'SELECTION.json',
                 ROOT/'artifacts/paper_bc_fit_003/REPORT.json', ROOT/'artifacts/paper_bc_fit_004/REPORT.json',
                 A/'bc_onset_distribution_diagnostic_001/REVIEW.md',
                 A/'bc_onset_distribution_diagnostic_001/INITIAL_ACTION_DETAIL.json'):
        pin(path)
    old = pin(A/'source_017/learner.py')
    current = pin(ROOT/'experiments/paper_walk/learner.py')
    parity = {name:function(old,name)==function(current,name)
              for name in ('ActorCritic','RunningMeanStd','pretrain_bc')}
    assert all(parity.values()), parity
    for name in ('env.py','env_config.py','replay.py'):
        pin(ROOT/'experiments/paper_walk'/name)
    spec = importlib.util.spec_from_file_location('bc_learning_audit_frozen017', old)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cp = torch.load(pin(ROOT/'artifacts/paper_bc_fit_004/candidate_checkpoint_update000000.pt'),
                    weights_only=False, map_location='cpu')
    assert cp['schema'] == module.SCHEMA and cp['learner_sha256'] == sha(old)
    assert cp['counters']['bc_steps']==1000 and cp['counters']['updates']==cp['counters']['transitions']==0
    model = module.ActorCritic(module.Config(**cp['config'])).eval()
    model.load_state_dict(cp['model'], strict=True)
    data, raw, native = load(data_dir/'bc_dataset.npz'), load(raw_dir/'replay_controls.npz'), load(raw_dir/'replay_substeps.npz')
    report = json.loads(pin(raw_dir/'report.json').read_text())
    assert sha(ROOT/'experiments/paper_walk/replay.py')==report['source_sha256']
    assert all(sha(ROOT/'experiments/paper_walk'/name)==value
               for name,value in report['physics_identity']['physics_source_files'].items())
    for name in ('replay_controls.npz','replay_substeps.npz'):
        assert sha(raw_dir/name)==report['raw_files'][name]
    provenance = [json.loads(line) for line in pin(data_dir/'ROW_PROVENANCE.jsonl').read_text().splitlines()]
    t, e = data['control_index'], data['env_index']
    assert len(t)==len(provenance)==3760 and len(set(zip(t.tolist(),e.tolist())))==3760
    assert all(p['dataset_row']==i and p['raw_control']==int(t[i]) and p['env']==int(e[i])
               and p['physics_counter_before']==int(data['physics_counter_before'][i])
               and p['physics_counter_after']==int(data['physics_counter_after'][i])
               for i,p in enumerate(provenance))
    fields = ('states','next_states','observations','commands','actions','requested_joint_target_rad','applied_joint_target_rad')
    assert all(np.array_equal(data[k],raw[k][t,e]) for k in fields)
    assert not raw['terminated'].any() and not raw['truncated'].any()
    assert np.array_equal(data['physics_counter_after'],raw['physics_counter'][t])
    assert np.all(data['physics_counter_after']-data['physics_counter_before']==8)
    assert np.array_equal(native['physics_counter'],np.arange(1,3041))
    neutral = np.array(report['neutral_joint_position_rad'],np.float32)
    held = np.where((t>0)[:,None],raw['applied_joint_target_rad'][np.maximum(t-1,0),e],neutral)
    prev = (held-neutral)/np.float32(.35)
    assert delta(prev,data['observations'][:,213:])<=1e-7
    assert np.array_equal(data['commands'],data['observations'][:,210:213])
    reconstructed = neutral + np.float32(.35)*data['actions']
    assert delta(reconstructed,data['requested_joint_target_rad'])<=1e-7
    assert np.array_equal(data['next_states'][:,:18],native['joint_position_rad'][(t+1)*8-1,e])
    assert np.array_equal(data['next_states'][:,18:36],native['joint_velocity_rad_s'][(t+1)*8-1,e])
    assert np.array_equal(data['states'][t>0],raw['next_states'][t[t>0]-1,e[t>0]])
    assert np.array_equal(data['velocity_targets_navigation_mps'],data['states'][:,[37,36,38]]*np.array([-1,1,1],np.float32))
    history_t = np.maximum(t[:,None]+np.arange(-4,1),0)
    history = data['observations'][:,:210].reshape(-1,5,42)
    assert np.array_equal(history,raw['observations'][history_t,e[:,None],168:210])
    history_states = raw['states'][history_t,e[:,None]]
    expected = np.concatenate((history_states[..., [40,39,41]]*np.array([-1,1,1],np.float32),
                              history_states[...,:18]-neutral,history_states[...,18:36]),axis=-1)
    assert np.array_equal(history[...,[0,1,2,*range(6,42)]],expected)
    for offset in range(8):
        assert np.array_equal(native['joint_target_rad'][t*8+offset,e],data['applied_joint_target_rad'])
    geometry = json.loads(pin(ROOT/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json').read_text())
    names = [f'{leg}_{joint}' for leg in ('lf','lm','lr','rf','rm','rr') for joint in ('coxa_yaw','femur_pitch','tibia_pitch')]
    named = {j['name']:j for j in geometry['joints']}
    lower,upper = (np.array([named[n][key] for n in names],np.float32) for key in ('lower','upper'))
    emitted = emit(data['actions'],held,neutral,lower,upper)
    assert delta(emitted,data['applied_joint_target_rad'])<=1e-7
    obs = torch.from_numpy(data['observations'])
    independently_updated = module.RunningMeanStd(231)
    independently_updated.update(obs)
    rms_equal = {k:torch.equal(v,model.obs_normalizer.state_dict()[k])
                 for k,v in independently_updated.state_dict().items()}
    assert all(rms_equal.values())
    with torch.inference_mode():
        action, velocity = model.actor(obs)
        detached_action, detached_velocity = model.actor(obs,detach_velocity=True)
        assert torch.equal(action,detached_action) and torch.equal(velocity,detached_velocity)
        standardized = ((obs-model.obs_normalizer.mean)/torch.sqrt(model.obs_normalizer.var+1e-6)).numpy()
    predicted = action.numpy()
    error = (predicted.astype(float)-data['actions'].astype(float))*.35
    limited_error = emit(predicted,held,neutral,lower,upper).astype(float)-data['applied_joint_target_rad']
    targets = data['velocity_targets_navigation_mps'].astype(float)
    zero = np.all(data['commands']==0,axis=1)
    moving = ~zero
    groups = {'all':np.ones(len(t),bool),'steady':data['source_kind']==0,'cycle1':data['source_kind']==1,
              'zero_prefix':data['source_kind']==2,'moving_first_control':(t==200)&moving,
              'moving_first_five':(t>=200)&(t<205)&moving,
              'moving_rest_cycle1':(t>=205)&(t<260)&moving,
              'forward_first_control':(t==200)&np.all(data['commands']==np.array([.05,0,0],np.float32),axis=1)}
    measurements = {}
    for name,mask in groups.items():
        measurements[name] = {'rows':int(mask.sum()),'row_fraction':float(mask.mean()),
            'requested_target_error_rms_rad':float(np.sqrt(np.mean(error[mask]**2))),
            'requested_target_squared_error_share':float(np.sum(error[mask]**2)/np.sum(error**2)),
            'post_slew_target_error_rms_rad':float(np.sqrt(np.mean(limited_error[mask]**2))),
            'velocity_estimator_rmse_mps':float(np.sqrt(np.mean((velocity.numpy()[mask]-targets[mask])**2))),
            'normalized_input_clamp_coordinate_fraction':float(np.mean(np.abs(standardized[mask])>10)),
            'clamped_input_rows':int(np.any(np.abs(standardized[mask])>10,axis=1).sum())}
    command_onsets = sorted(set(report['command_index_by_env'][int(x)] for x in e[data['source_kind']==1]))
    raw_moving = np.any(raw['commands']!=0,axis=-1)
    moving_to_zero = int(np.sum(raw_moving[:-1]&~raw_moving[1:]))
    assert moving_to_zero==0
    result = {'schema':'canonical_bc_source_data_audit_v1','started_utc':started,
        'scope':'CPU artifact-only examination of existing native demonstrations and fitted teacher-input predictions; no fitting, native run, new gate or causal intervention.',
        'source_function_ast_equal_017_to_current':parity,
        'current_physics_and_replay_source_match_native_replay_declaration':True,
        'alignment':{'rows':len(t),'distinct_raw_rows':len(set(zip(t.tolist(),e.tolist()))),
            'raw_fields_and_provenance_exact':True,'prehold_history_exact':True,
            'prehold_angular_position_velocity_exact':True,'prehold_amp_previous_endpoint_exact':True,
            'next_q_dq_native_endpoint_exact':True,'held_target_all_eight_substeps_exact':True,
            'command_and_velocity_target_exact':True,'previous_held_normalized_action_max_error':delta(prev,data['observations'][:,213:]),
            'desired_target_reconstruction_max_error_rad':delta(reconstructed,data['requested_joint_target_rad']),
            'independent_numpy_emitted_target_max_error_rad':delta(emitted,data['applied_joint_target_rad']),
            'independent_numpy_emitted_target_exact':np.array_equal(emitted,data['applied_joint_target_rad']),
            'initial_gravity_limit':'Initial gravity is copied recorded reset observation; AMP61 alone has no quaternion. Other history fields independently reconstructed; subsequent gravity reconstruction is pinned in original builder receipt, not rerun here.'},
        'normalizer':{'single_full_dataset_update_buffers_bitwise_equal':rms_equal,'count':float(model.obs_normalizer.count),
            'raw_dataset_normalized_once_in_actor':True,'bc_detach_inference_outputs_exact':True,
            'standardized_absmax_before_clamp':float(np.abs(standardized).max()),
            'rows_with_any_clamp':int(np.any(np.abs(standardized)>10,axis=1).sum()),
            'constant_or_floor_variance_channels':int((model.obs_normalizer.var<=1e-7).sum())},
        'group_inference':measurements,
        'dataset_coverage':{'onset_command_indices':command_onsets,'missing_onset_command_indices':sorted(set(range(21))-set(command_onsets)),
            'moving_raw_reset_rows':int(((t==0)&moving).sum()),'moving_to_zero_transitions':moving_to_zero,
            'max_raw_control':int(t.max()),'cycle3_training_rows':int((data['cycle_index']==2).sum()),
            'history_sample_span_s':.08,'all_zero_prefix_controls':[0,199],'moving_onset_control':200},
        'confirmed_alignment_bug':False,'stage2_complete':False,'physical_admission':False,
        'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'torch_threads':torch.get_num_threads()}}
    assert all(sha(ROOT/name)==value for name,value in PINS.items())
    result['all_pinned_inputs_unchanged_after_analysis']=True
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (HERE/'SOURCE_PINS.json').write_text(json.dumps(PINS,indent=2)+'\n')
    print(json.dumps({'alignment':result['alignment'],'normalizer':result['normalizer'],'group_inference':measurements},indent=2))


if __name__=='__main__':
    main()
