#!/usr/bin/env python3
"""Run isolated short comparisons with exact-plan standing admissions and GPU locks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace
from launch_length_training_spark import run_job, save, verified_source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,nargs='+',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'))
    args=parser.parse_args()
    args.output=args.output.resolve();args.checkpoint=args.checkpoint.resolve()
    args.sources=[p.resolve() for p in args.sources]
    if args.output.exists():parser.error('Use a fresh output directory')
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=args.checkpoint_sha256:
        parser.error('Checkpoint identity mismatch')
    for source in args.sources:
        verified_source(source)
        plan=json.loads((source/'robot/hexapod_mkii_length_study/training_plan.json').read_text())
        if 'diagnostics' not in plan.get('omni',{}):parser.error('Every source must explicitly request diagnostics')
    args.output.mkdir(parents=True)
    stop=args.output/'stop.request'
    signal.signal(signal.SIGTERM,lambda *_:stop.touch())
    signal.signal(signal.SIGINT,lambda *_:stop.touch())
    status={'status':'starting','started_unix':time.time(),'checkpoint_sha256':args.checkpoint_sha256,
            'scope':'diagnostics only; no qualification or training claims','comparisons':[]}
    deadline=time.monotonic()+90*60
    def job(target,mode,**kw):
        while True:
            if stop.exists():raise InterruptedError('Stop requested')
            if time.monotonic()>deadline:raise TimeoutError('Diagnostic campaign exceeded90minutes')
            status.update(status='running',current_source=str(target.source),current_mode=mode)
            save(args.output/'diagnostic_campaign.json',status)
            try:return run_job(target,'f050_t060',mode,0,**kw)
            except BlockingIOError:
                status.update(status='waiting_for_shared_gpu');save(args.output/'diagnostic_campaign.json',status)
                time.sleep(15)
    try:
        for index,source in enumerate(args.sources):
            output=args.output/f'comparison_{index:02d}'
            inputs=output/'inputs';inputs.mkdir(parents=True)
            checkpoint=inputs/'policy.pt';shutil.copy2(args.checkpoint,checkpoint)
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=args.checkpoint_sha256:
                raise RuntimeError('Copied checkpoint mismatch')
            target=SimpleNamespace(source=source,output=output,isaaclab=args.isaaclab,stop_request=stop)
            entry={'source':str(source),'output':str(output),'status':'validating'}
            status['comparisons'].append(entry)
            validated,state=job(target,'validate')
            if state['status']!='completed':
                entry.update(status='standing_rejected',gate=state.get('gate'));continue
            entry['status']='evaluating'
            evaluated,_=job(target,'evaluate',admission=validated/'admission.json',checkpoint=checkpoint)
            report=json.loads((evaluated/'diagnostics.json').read_text())
            if report['checkpoint_sha256']!=args.checkpoint_sha256:raise RuntimeError('Diagnostic checkpoint mismatch')
            entry.update(status='completed',diagnostics=str(evaluated/'diagnostics.json'),
                         terminations=sum(row['terminations'] for row in report['scenarios']),
                         overrides=report['overrides'])
            save(args.output/'diagnostic_campaign.json',status)
        status['status']='completed_needs_comparison'
    except Exception as exc:
        status.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc))
        raise
    finally:
        status['finished_unix']=time.time();save(args.output/'diagnostic_campaign.json',status)


if __name__=='__main__':main()
