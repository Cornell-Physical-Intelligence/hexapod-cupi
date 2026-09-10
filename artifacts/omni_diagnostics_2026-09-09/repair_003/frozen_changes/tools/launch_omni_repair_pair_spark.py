#!/usr/bin/env python3
"""Exactly two independent 50-update comparisons; no automatic continuation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace
from launch_length_training_spark import run_job, save, verified_source
from launch_omni_repair_spark import metrics, continuation_screen


def assert_plan_pair(plans, expected):
    if len(plans)!=2: raise ValueError('Exactly two source plans required')
    for plan in plans:
        o=plan['omni']; overrides=o['overrides']
        if (plan['training_iterations']!=50 or plan['training_num_envs']!=1024 or plan['evaluation_num_envs']!=48
            or o['repair_training']['checkpoint_sha256']!=expected
            or overrides['target_slew_rad_per_20ms']!=.03
            or overrides['observation_noise_scale']!=1 or overrides['target_filter_time_constant_s']!=0
            or o['diagnostics']['controller']!='policy'):
            raise ValueError('Unexpected matched-comparison plan')
    left=json.loads(json.dumps(plans[0]));right=json.loads(json.dumps(plans[1]))
    if left['omni']['overrides']['reward_weights'].pop('stand_raw_action')!=0 or right['omni']['overrides']['reward_weights'].pop('stand_raw_action')!=-2:
        raise ValueError('Expected A raw cost 0 and B raw cost -2')
    if left!=right: raise ValueError('Branch plans differ beyond the isolated reward')


def checked_report(path, expected):
    report=json.loads((path/'diagnostics.json').read_text())
    if report['checkpoint_sha256']!=expected or not report['complete']:
        raise ValueError('Incomplete or wrong-checkpoint diagnostics')
    audit=report['observation_audit']
    if audit['actor_width']!=315 or audit['critic_width']!=318 or any(audit[k]!=0 for k in
        ('max_same_step_repeat_difference','max_command_slice_difference','max_history_shift_difference')):
        raise RuntimeError('Observation history/command audit failed')
    if any(r['windows']['all']['applied_torque_abs_max_nm']>1.60001 for r in report['scenarios']):
        raise RuntimeError('Applied motor torque cap exceeded')
    result=metrics(report)
    if result['terminations'] or result['truncations']:
        raise RuntimeError('Diagnostic reset detected; stop bounded campaign for inspection')
    # Nonfinite metrics cannot be interpreted as improvement.
    import math
    def check(value):
        if isinstance(value,dict): return all(check(v) for v in value.values())
        if isinstance(value,(int,float)): return math.isfinite(value)
        return True
    if not check(result): raise ValueError('Nonfinite diagnostic metrics')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,nargs=2,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'))
    a=p.parse_args();a.output=a.output.resolve();a.sources=[s.resolve() for s in a.sources]
    if a.output.exists():p.error('Fresh output required')
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=a.checkpoint_sha256:p.error('Checkpoint mismatch')
    plans=[]
    for source in a.sources:
        verified_source(source);plans.append(json.loads((source/'robot/hexapod_mkii_length_study/training_plan.json').read_text()))
    assert_plan_pair(plans,a.checkpoint_sha256)
    a.output.mkdir(parents=True);stop=a.output/'stop.request';deadline=time.monotonic()+3600
    signal.signal(signal.SIGTERM,lambda *_:stop.touch());signal.signal(signal.SIGINT,lambda *_:stop.touch())
    status={'status':'starting','stage2_complete':False,'started_unix':time.time(),'branches':[],
            'checkpoint_sha256':a.checkpoint_sha256,'maximum_updates':100,'scope':'Matched short allocation experiment only'}
    def job(target,mode,**kw):
        while True:
            if stop.exists():raise InterruptedError('Stop requested')
            if time.monotonic()>deadline:raise TimeoutError('One-hour campaign bound')
            status.update(status='running',current_mode=mode,current_output=str(target.output));save(a.output/'repair_pair.json',status)
            try:return run_job(target,'f050_t060',mode,0,**kw)
            except BlockingIOError:
                status['status']='waiting_for_shared_gpu';save(a.output/'repair_pair.json',status);time.sleep(15)
    try:
        for index,source in enumerate(a.sources):
            directory=a.output/('branch_a' if index==0 else 'branch_b');inputs=directory/'inputs';inputs.mkdir(parents=True)
            cp=inputs/'original.pt';shutil.copy2(a.checkpoint,cp)
            if hashlib.sha256(cp.read_bytes()).hexdigest()!=a.checkpoint_sha256:raise ValueError('Copied checkpoint mismatch')
            target=SimpleNamespace(source=source,output=directory,isaaclab=a.isaaclab,stop_request=stop)
            entry={'status':'validating','source':str(source),'output':str(directory),'updates':50};status['branches'].append(entry)
            admission,state=job(target,'validate')
            if state['status']!='completed':raise RuntimeError('Fresh standing gate rejected')
            admission=admission/'admission.json'
            baseline_target=SimpleNamespace(source=source,output=directory/'baseline',isaaclab=a.isaaclab,stop_request=stop)
            (baseline_target.output/'inputs').mkdir(parents=True)
            base_cp=baseline_target.output/'inputs/original.pt';shutil.copy2(cp,base_cp)
            base_adm=baseline_target.output/'inputs/admission.json';shutil.copy2(admission,base_adm)
            evaluated,_=job(baseline_target,'evaluate',admission=base_adm,checkpoint=base_cp)
            baseline=checked_report(evaluated,a.checkpoint_sha256);entry['baseline']=baseline
            trained,state=job(target,'train',admission=admission,checkpoint=cp,iterations=50)
            initialization=json.loads((trained/'repair_initialization.json').read_text())
            if not initialization['actor_and_critic_preserved_except_std'] or not initialization['observation_normalizers_preserved']:
                raise RuntimeError('Checkpoint preservation failed')
            checkpoint=trained/'policy/final.pt';sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if state['checkpoint_sha256']!=sha:raise ValueError('Training checkpoint mismatch')
            evaluated,_=job(target,'evaluate',admission=admission,checkpoint=checkpoint)
            current=checked_report(evaluated,sha)
            entry.update(status='completed',metrics=current,screen=continuation_screen(current,baseline),
                         checkpoint=str(checkpoint),checkpoint_sha256=sha,initialization=initialization)
            save(a.output/'repair_pair.json',status)
        status.update(status='completed_needs_review',stage2_complete=False,reason='No further training allocated automatically')
    except Exception as exc:
        status.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc));raise
    finally:
        status['finished_unix']=time.time();save(a.output/'repair_pair.json',status)

if __name__=='__main__':main()
