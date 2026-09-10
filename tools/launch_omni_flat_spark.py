#!/usr/bin/env python3
"""Guarded, resumable omni experiment; short pilot exercises all runtime paths."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace
from launch_length_training_spark import run_job, save, verified_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--isaaclab', type=Path, default=Path('/home/orionh/IsaacLab'))
    p.add_argument('--initial-updates', type=int, default=100)
    p.add_argument('--chunk-updates', type=int, default=700)
    p.add_argument('--max-updates', type=int, default=1500)
    p.add_argument('--resume-checkpoint', type=Path)
    args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    if args.output.exists():p.error('Use a fresh output directory')
    args.output.mkdir(parents=True);verified_source(args.source)
    args.stop_request=args.output/'stop.request'
    signal.signal(signal.SIGTERM,lambda *_:args.stop_request.touch())
    signal.signal(signal.SIGINT,lambda *_:args.stop_request.touch())
    report={'status':'starting','source':str(args.source),'started_unix':time.time(),
            'scope':'flat omni only; Benchmark 1 is immutable','stages':[]}
    deadline=time.monotonic()+8*3600
    def job(target,mode,**kwargs):
        while True:
            if args.stop_request.exists():raise InterruptedError('User stop requested')
            if time.monotonic()>deadline:raise TimeoutError('Eight-hour campaign bound')
            report.update(status='running',current_mode=mode,current_output=str(target.output))
            save(args.output/'omni.json',report)
            try:return run_job(target,'f050_t060',mode,0,**kwargs)
            except BlockingIOError:
                report.update(status='waiting_for_shared_gpu');save(args.output/'omni.json',report);time.sleep(20)
    try:
        validated,state=job(args,'validate')
        if state['status']!='completed':raise RuntimeError('Full standing gate failed')
        admission=validated/'admission.json'
        previous=args.resume_checkpoint
        total=0
        while total<args.max_updates:
            count=min(args.initial_updates if total==0 else args.chunk_updates,args.max_updates-total)
            stage=args.output/f'stage_{len(report["stages"]):03d}';stage.mkdir()
            inputs=stage/'inputs';inputs.mkdir();shutil.copy2(admission,inputs/'admission.json')
            checkpoint=None
            if previous:
                checkpoint=inputs/'previous.pt';shutil.copy2(previous,checkpoint)
            target=SimpleNamespace(source=args.source,output=stage,isaaclab=args.isaaclab,stop_request=args.stop_request)
            entry={'new_updates':count,'status':'training','output':str(stage)};report['stages'].append(entry)
            trained,_=job(target,'train',admission=inputs/'admission.json',checkpoint=checkpoint,iterations=count)
            previous=trained/'policy/final.pt';total+=count
            entry.update(status='evaluating',total_updates_this_campaign=total)
            evaluated,_=job(target,'evaluate',admission=inputs/'admission.json',checkpoint=previous)
            evaluation=json.loads((evaluated/'evaluation.json').read_text())
            entry.update(static_passed=sum(r['pass'] for r in evaluation['static']),
                         static_total=len(evaluation['static']),
                         transitions_passed=sum(r['pass'] for r in evaluation['transitions']),
                         all_scenarios_pass=evaluation['all_scenarios_pass'],status='recording')
            video,_=job(target,'video',admission=inputs/'admission.json',checkpoint=previous)
            entry.update(status='completed',video=str(video/'rollout.mp4'),checkpoint=str(previous))
            save(args.output/'omni.json',report)
            if evaluation['all_scenarios_pass']:
                report.update(status='omni_nominal_pass_pending_robustness');break
        else:report.update(status='training_budget_complete_needs_review')
        report.update(checkpoint=str(previous),updates_this_campaign=total)
    except Exception as exc:
        report.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc))
        raise
    finally:
        report['finished_unix']=time.time();save(args.output/'omni.json',report)


if __name__=='__main__':main()
