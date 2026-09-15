"""Read-only CPU analysis of actual012/013 traces; no policy/simulator updates."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
A=OUT.parent
ROOT=A.parents[2]
INPUTS={}
torch.set_num_threads(2)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p,expected=None):
    actual=sha(p)
    if expected is not None and actual!=expected:raise ValueError('Hash mismatch: '+str(p))
    INPUTS[str(p.relative_to(ROOT))]={'sha256':actual,'bytes':p.stat().st_size,'expected_sha256':expected}
    return actual
def read(p):
    bind(p);return json.loads(p.read_text())
def checkpoint(p,expected):
    bind(p,expected);return torch.load(p,map_location='cpu',weights_only=False)
def same_bytes(a,b):
    return a.keys()==b.keys() and all(a[k].numpy().tobytes()==b[k].numpy().tobytes() for k in a)

bc=checkpoint(ROOT/'artifacts/paper_bc_fit_004/candidate_checkpoint_update000000.pt','6b169d9a406b8a0d6208a3282f59111529b375461b2e86bc391dfb9fe0a2c8f9')
bc32=checkpoint(ROOT/'artifacts/paper_bc_fit_003/candidate_checkpoint_update000000.pt','a48fddf1f73fcaf5b93db44241390184dd36c60e031daaad87d3a63bfd2bd81a')
assert same_bytes(bc['model'],bc32['model']) and same_bytes(bc['amp'],bc32['amp'])
trained=checkpoint(A/'results_train_006/standing/learner/checkpoint_000200.pt','eb2847e9bce7db0b090debdbc2b0b3395679ff3e2b78f8ecf7407aaa4a8b2d1a')
metrics_path=A/'results_train_006/standing/learner/metrics.jsonl';bind(metrics_path)
metrics=[json.loads(s) for s in metrics_path.read_text().splitlines()]
assert len(metrics)==200 and metrics[-1]['update']==200
result={'scope':'Actual recorded deterministic controls and front-end reconstruction; no new trajectories, fitting, or physical counterfactual.',
 'training_final':{k:metrics[-1][k] for k in ('update','transitions','raw_action_clip_fraction','raw_action_abs_mean','action_std')},
 'training_per_joint_mean_clipping':'Unavailable: original training raw observations/means/actions were not saved. Recorded final training metric counts stochastic raw samples, not deterministic means.',
 'bc004_model_and_amp_byte_equal_actual_eval012_bc003':True,'traces':[]}

for run,payload in [('012',bc32),('013',trained)]:
    directory=A/('results_evaluate_'+run)/'standing'
    native=read(directory/'native/native_readback.json')
    names=native['canonical_joint_names'];limits=np.asarray(native['limits'][0],np.float32)
    std=payload['model']['log_std'].clamp(-5.,1.).exp().numpy().astype(np.float64)
    for index in range(3):
        folder=directory/'evaluation'/f'batch_{index:03d}'
        report=read(folder/'report.json');capture=read(folder/'native400hz/capture.json')
        trace_path=folder/'control_trace.npz';bind(trace_path,report['files']['control_trace.npz'])
        initial_path=folder/'native400hz/initial_state.json';bind(initial_path,capture['files']['initial_state.json'])
        initial=json.loads(initial_path.read_text());neutral=torch.tensor(initial['q'][0],dtype=torch.float32)
        with np.load(trace_path,allow_pickle=False) as raw:d={k:raw[k].copy() for k in raw.files}
        assert report['acquisition_complete'] and report['failure'] is None and not d['reset'].any()
        assert report['checkpoint_sha256']==('a48fddf1f73fcaf5b93db44241390184dd36c60e031daaad87d3a63bfd2bd81a' if run=='012' else 'eb2847e9bce7db0b090debdbc2b0b3395679ff3e2b78f8ecf7407aaa4a8b2d1a')
        a=d['policy_action'][:,0];obs=d['policy_observation'][:,0];target=d['joint_target_rad'][:,0]
        cmd=d['command'][:,0];q=d['joint_position_rad'][:,0]
        assert np.isfinite(a).all() and np.isfinite(obs).all() and np.array_equal(obs[:,210:213],cmd)
        previous=np.concatenate((neutral.numpy()[None],target[:-1]))
        expected_feature=(torch.from_numpy(previous)-neutral)/.35
        assert np.array_equal(expected_feature.numpy(),obs[:,213:])
        action=torch.from_numpy(a);held=torch.from_numpy(previous)
        lower=torch.from_numpy(limits[:,0]);upper=torch.from_numpy(limits[:,1])
        nominal=neutral+.35*action.clamp(-1,1)
        requested=nominal.clamp(lower,upper)
        lo=torch.maximum(lower.double(),held.double()-.04)
        hi=torch.minimum(upper.double(),held.double()+.04)
        reproduced=torch.maximum(torch.minimum(requested.double(),hi),lo).float()
        reproduced=torch.where(reproduced.double()>hi,torch.nextafter(reproduced,torch.full_like(reproduced,-torch.inf)),reproduced)
        reproduced=torch.where(reproduced.double()<lo,torch.nextafter(reproduced,torch.full_like(reproduced,torch.inf)),reproduced)
        assert np.array_equal(reproduced.numpy(),target)
        nominal_endpoints=np.stack(((neutral-.35).numpy(),(neutral+.35).numpy()),axis=-1)
        assert np.all(nominal_endpoints[:,0]>limits[:,0]) and np.all(nominal_endpoints[:,1]<limits[:,1])
        r={'evaluation':run,'case':report['cases'][0]['case_id'],'controls':len(a),'joint_names':names,
           'command_in_policy_obs_exact':True,'previous_executed_target_feature_exact':True,
           'all_emitted_targets_bit_exact_frontend_reconstruction':True,
           'all_nominal_action_endpoints_strictly_inside_native_joint_limits':True,
           'joint_position_clamp_changed_fraction':float((nominal!=requested).float().mean()),
           'full_mean_action_clip_fraction':float(np.mean(abs(a)>1)),
           'action_scale_rad':.35,'slew_cap_rad':.04,'normal_std_per_joint':std.tolist(),'segments':[]}
        segments=[('all',0,len(a)),('after_first100_controls',100,len(a))]
        if index==2:segments += [('moving_after100_before400',100,400),('stopped_after100_zero_controls',500,len(a))]
        for label,start,end in segments:
            aa=a[start:end].astype(np.float64);tt=target[start:end];pp=previous[start:end]
            c=cmd[start:end];qq=q[start:end]
            inside=np.array([[.5*(math.erfc((abs(mu)-1)/(s*math.sqrt(2)))-math.erfc((abs(mu)+1)/(s*math.sqrt(2)))) for mu,s in zip(row,std)] for row in aa])
            at_edge=np.isclose(tt,nominal_endpoints[:,0],atol=1e-7,rtol=0)|np.isclose(tt,nominal_endpoints[:,1],atol=1e-7,rtol=0)
            row={'name':label,'control_slice':[start,end],'controls':end-start,
                 'commands':np.unique(c,axis=0).tolist(),'mean_clip_fraction':float(np.mean(abs(aa)>1)),
                 'mean_at_frontend_bound_fraction':float(at_edge.mean()),
                 'mean_forward_actual_mps':float(d['velocity_navigation_mps'][start:end,0,0].mean()),
                 'per_joint':[]}
            for j,name in enumerate(names):
                row['per_joint'].append({'name':name,'mean_action':float(aa[:,j].mean()),'action_min':float(aa[:,j].min()),'action_max':float(aa[:,j].max()),
                    'positive_clip_fraction':float(np.mean(aa[:,j]>1)),'negative_clip_fraction':float(np.mean(aa[:,j]<-1)),
                    'mean_outside_margin_in_std':float(np.maximum(abs(aa[:,j])-1,0).mean()/std[j]),
                    'mean_probability_gaussian_sample_inside_action_bounds':float(inside[:,j].mean()),
                    'actual_target_at_frontend_bound_fraction':float(at_edge[:,j].mean()),
                    'actual_target_mean_rad':float(tt[:,j].mean()),'actual_target_range_rad':float(np.ptp(tt[:,j])),
                    'actual_target_step_at_slew_fraction':float(np.mean(abs(tt[:,j]-pp[:,j])>=.04-2e-7)),
                    'actual_q_target_rms_rad':float(np.sqrt(np.mean((qq[:,j]-tt[:,j])**2)))})
            r['segments'].append(row)
        result['traces'].append(r)
result['interpretation_limits']=[
 'Gaussian interior probabilities use saved per-joint standard deviation and actual recorded means; they describe the policy distribution on those recorded states, not fresh stochastic or physical rollouts.',
 'The input clamp creates regions where further changes in already-outside raw samples give identical desired joint targets. PPO still has raw-Gaussian likelihood gradients and can move means; these data do not prove a zero gradient or an irrecoverable policy.',
 'The exact number of training means beyond bounds per joint/command remains missing. The deterministic three-probe evaluation supports widespread mean saturation on its measured states only.',
 'Frontend action/target bounds are distinct from joint hard limits and motor torque saturation. No physical gates or existing verdicts are changed.',
 'Eval012 actually used Config32 BCfit003. Its model/AMP state is byte-identical to Config128 BCfit004; future rollout batch count differs.']
(OUT/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
(OUT/'INPUTS.json').write_text(json.dumps(INPUTS,indent=2)+'\n')
print(json.dumps([{'eval':r['evaluation'],'case':r['case'],'clip':r['segments'][1]['mean_clip_fraction'],'target_at_bound':r['segments'][1]['mean_at_frontend_bound_fraction']} for r in result['traces']],indent=2))
