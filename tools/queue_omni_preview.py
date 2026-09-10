#!/usr/bin/env python3
"""Render a corrected preview at the next free GPU interval, without interrupting PPO."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from launch_length_training_spark import run_job,save,verified_source

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--campaign',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
verified_source(a.source)
(a.output/'inputs').mkdir()
target=SimpleNamespace(source=a.source,output=a.output,isaaclab=Path('/home/orionh/IsaacLab'),stop_request=a.output/'stop.request')
status={'status':'waiting_for_gpu','source':str(a.source),'created_unix':time.time()}
save(a.output/'preview.json',status)
deadline=time.monotonic()+90*60
try:
 while time.monotonic()<deadline:
    if target.stop_request.exists():raise InterruptedError('Preview stop requested')
    ready=[]
    for d in sorted(a.campaign.glob('stage_*')):
        t=d/'f050_t060/train/state.json';e=d/'f050_t060/evaluate/evaluation.json'
        if t.exists() and e.exists():
            train=json.loads(t.read_text());evaluation=json.loads(e.read_text())
            if train.get('status')=='completed' and evaluation.get('complete') and train['checkpoint_sha256']==evaluation['checkpoint_sha256']:
                ready.append((d,train))
    if not ready:time.sleep(20);continue
    stage,train=ready[-1]
    checkpoint=a.output/'inputs/policy.pt';admission=a.output/'inputs/admission.json'
    shutil.copy2(stage/'f050_t060/train/policy/final.pt',checkpoint)
    shutil.copy2(a.campaign/'f050_t060/validate_0/admission.json',admission)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==train['checkpoint_sha256']
    status.update(selected_stage=str(stage),checkpoint_sha256=train['checkpoint_sha256'])
    save(a.output/'preview.json',status)
    try:
        out,state=run_job(target,'f050_t060','video',0,admission=admission,checkpoint=checkpoint)
    except BlockingIOError:time.sleep(20);continue
    status.update(status='completed',video=str(out/'rollout.mp4'),finished_unix=time.time())
    save(a.output/'preview.json',status)
    break
 else:raise TimeoutError('No preview GPU slot within90minutes')
except Exception as exc:
 status.update(status='failed',error=repr(exc),finished_unix=time.time());save(a.output/'preview.json',status)
 raise
