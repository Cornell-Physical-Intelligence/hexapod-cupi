#!/usr/bin/env python3
"""Bounded PPO repair: matched baseline, short pilots, regression screen, full review."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
from statistics import median
import time
from types import SimpleNamespace
from launch_length_training_spark import run_job, save, verified_source


def metrics(report):
    rows=[r['windows']['post_settle_nonterminal'] for r in report['scenarios']]
    if any(not r.get('samples') for r in rows):raise ValueError('A diagnostic scenario has no settled samples')
    stand=next(r['windows']['post_settle_nonterminal'] for r in report['scenarios'] if r['name']=='stand')
    return {'terminations':sum(r['terminations'] for r in report['scenarios']),
            'truncations':sum(r['truncations'] for r in report['scenarios']),
            'saturation':median(r['torque_saturation_fraction'] for r in rows),
            'planar_error':median(r['planar_error_mps'] for r in rows),
            'yaw_error':median(r['yaw_error_rad_s'] for r in rows),
            'power':median(r['positive_mechanical_power_w'] for r in rows),
            'stand_joint_velocity_rms':math.sqrt(sum(j['velocity_rms_rad_s']**2 for j in stand['joints'].values())/len(stand['joints'])),
            'scenarios':{r['name']:{'terminations':r['terminations'],
                **{k:r['windows']['post_settle_nonterminal'][k] for k in
                   ('planar_error_mps','yaw_error_rad_s','torque_saturation_fraction')}} for r in report['scenarios']}}


def continuation_screen(current, baseline, previous=None):
    """A short-run allocation screen, never a qualification decision."""
    stable=(current['terminations']<=baseline['terminations'] and current['truncations']==0
            and current['planar_error']<=baseline['planar_error']*1.25+.005
            and current['yaw_error']<=baseline['yaw_error']*1.25+.005
            and current['power']<=baseline['power']*1.25
            and current['saturation']<=baseline['saturation']*1.2
            and current['stand_joint_velocity_rms']<=baseline['stand_joint_velocity_rms']*1.2)
    if set(current['scenarios'])!=set(baseline['scenarios']):raise ValueError('Diagnostic scenario mismatch')
    direction_failures=[]
    for name,c in current['scenarios'].items():
        b=baseline['scenarios'][name]
        if (c['terminations']>b['terminations'] or c['planar_error_mps']>b['planar_error_mps']*1.25+.005
            or c['yaw_error_rad_s']>b['yaw_error_rad_s']*1.25+.005
            or c['torque_saturation_fraction']>b['torque_saturation_fraction']*1.3+.002):
            direction_failures.append(name)
    stable=stable and not direction_failures
    improved=(current['saturation']<=baseline['saturation']*.9
              or current['stand_joint_velocity_rms']<=baseline['stand_joint_velocity_rms']*.85)
    marginal=True
    if previous is not None:
        previous_screen=continuation_screen(current,previous)
        marginal=previous_screen['stable'] and (current['saturation']<=previous['saturation']*.95
                  or current['stand_joint_velocity_rms']<=previous['stand_joint_velocity_rms']*.95)
    return {'stable':bool(stable),'improved':bool(improved),
            'marginal_improvement':bool(marginal),'continue':bool(stable and improved and marginal),
            'direction_regressions':direction_failures,
            'scope':'Compute allocation screen only; all-direction smoothness and full gates still required'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--qualification-source',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'))
    p.add_argument('--updates',type=int,nargs='+',default=[100,300,300])
    a=p.parse_args();a.output=a.output.resolve();a.source=a.source.resolve();a.qualification_source=a.qualification_source.resolve()
    if a.output.exists():p.error('Use a fresh output directory')
    if not a.updates or min(a.updates)<=0 or sum(a.updates)>1000:p.error('Positive updates, total<=1000 required')
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=a.checkpoint_sha256:p.error('Checkpoint mismatch')
    verified_source(a.source);verified_source(a.qualification_source)
    diagnostic_plan=json.loads((a.source/'robot/hexapod_mkii_length_study/training_plan.json').read_text())
    full_plan=json.loads((a.qualification_source/'robot/hexapod_mkii_length_study/training_plan.json').read_text())
    if 'diagnostics' not in diagnostic_plan['omni'] or 'diagnostics' in full_plan['omni']:
        p.error('Expected separate diagnostic and full qualification plans')
    if diagnostic_plan['omni']['overrides']!=full_plan['omni']['overrides']:p.error('Controller mismatch between plans')
    a.output.mkdir(parents=True);stop=a.output/'stop.request'
    signal.signal(signal.SIGTERM,lambda *_:stop.touch());signal.signal(signal.SIGINT,lambda *_:stop.touch())
    status={'status':'starting','started_unix':time.time(),'checkpoint_sha256':a.checkpoint_sha256,
            'scope':'Stage2 repair; completion requires all-direction visual review and robustness','stages':[]}
    deadline=time.monotonic()+4*3600
    def job(target,mode,**kw):
        while True:
            if stop.exists():raise InterruptedError('Stop requested')
            if time.monotonic()>deadline:raise TimeoutError('Four-hour bound')
            status.update(status='running',current_mode=mode,current_output=str(target.output));save(a.output/'repair.json',status)
            try:return run_job(target,'f050_t060',mode,0,**kw)
            except BlockingIOError:
                status['status']='waiting_for_shared_gpu';save(a.output/'repair.json',status);time.sleep(15)
    def target(name,source):
        output=a.output/name;(output/'inputs').mkdir(parents=True)
        return SimpleNamespace(source=source,output=output,isaaclab=a.isaaclab,stop_request=stop)
    try:
        base=target('baseline',a.source);checkpoint=base.output/'inputs/policy.pt';shutil.copy2(a.checkpoint,checkpoint)
        validated,state=job(base,'validate')
        if state['status']!='completed':raise RuntimeError('Repair standing gate rejected')
        admission=validated/'admission.json'
        evaluated,_=job(base,'evaluate',admission=admission,checkpoint=checkpoint)
        baseline=metrics(json.loads((evaluated/'diagnostics.json').read_text()));status['baseline']=baseline
        previous=checkpoint;selected=checkpoint;best_score=1.;total=0;previous_metrics=None
        for index,count in enumerate(a.updates):
            stage=target(f'stage_{index:03d}',a.source)
            previous_copy=stage.output/'inputs/previous.pt';shutil.copy2(previous,previous_copy)
            admitted=stage.output/'inputs/admission.json';shutil.copy2(admission,admitted)
            entry={'status':'training','updates':count,'output':str(stage.output)};status['stages'].append(entry)
            trained,_=job(stage,'train',admission=admitted,checkpoint=previous_copy,iterations=count)
            previous=trained/'policy/final.pt';total+=count
            evaluated,_=job(stage,'evaluate',admission=admitted,checkpoint=previous)
            current=metrics(json.loads((evaluated/'diagnostics.json').read_text()));screen=continuation_screen(current,baseline,previous_metrics)
            score=.5*current['saturation']/max(baseline['saturation'],1e-6)+.5*current['stand_joint_velocity_rms']/max(baseline['stand_joint_velocity_rms'],1e-6)
            if screen['stable'] and screen['improved'] and score<best_score:selected=previous;best_score=score
            entry.update(status='completed',metrics=current,screen=screen,checkpoint=str(previous))
            status.update(updates_this_campaign=total,selected_checkpoint=str(selected));save(a.output/'repair.json',status)
            previous_metrics=current
            if not screen['continue']:break
        if selected==checkpoint:
            status.update(status='needs_repair',stage2_complete=False,reason='No improved checkpoint passed per-direction allocation screen')
            return
        full=target('full_review',a.qualification_source)
        checkpoint=full.output/'inputs/policy.pt';shutil.copy2(selected,checkpoint)
        validated,state=job(full,'validate')
        if state['status']!='completed':raise RuntimeError('Full review standing gate rejected')
        evaluated,_=job(full,'evaluate',admission=validated/'admission.json',checkpoint=checkpoint)
        video,_=job(full,'video',admission=validated/'admission.json',checkpoint=checkpoint)
        evaluation=json.loads((evaluated/'evaluation.json').read_text())
        status.update(status='needs_review',selected_checkpoint=str(checkpoint),video=str(video/'rollout.mp4'),
                      static_passed=sum(r['pass'] for r in evaluation['static']),static_total=len(evaluation['static']),
                      transitions_passed=sum(r['pass'] for r in evaluation['transitions']),
                      numerical_gates_pass=evaluation['all_scenarios_pass'],stage2_complete=False)
    except Exception as exc:
        status.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc));raise
    finally:
        status['finished_unix']=time.time();save(a.output/'repair.json',status)


if __name__=='__main__':main()
