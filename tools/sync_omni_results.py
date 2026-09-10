#!/usr/bin/env python3
"""Fetch completed omni evidence; never present an unfinished or mismatched clip."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--campaign',default='omni_flat_001')
p.add_argument('--stage',type=int,default=0)
a=p.parse_args()
if not a.campaign.replace('_','').isalnum() or a.stage<0:p.error('Invalid campaign/stage')
root='/home/orionh/HEXAPOD_runs/mock_length_study_20260909/'+a.campaign
code='''
from pathlib import Path
import json
b=Path(ROOT);s=b/STAGE/'f050_t060'
r={'campaign':json.loads((b/'omni.json').read_text()),'ready':False}
v=s/'video/video.json';e=s/'evaluate/evaluation.json';t=s/'train/state.json'
if all(p.exists() for p in (v,e,t)):
 video=json.loads(v.read_text());evaluation=json.loads(e.read_text());train=json.loads(t.read_text())
 r['ready']=bool(video.get('complete') and evaluation.get('complete') and train.get('status')=='completed' and video['checkpoint_sha256']==evaluation['checkpoint_sha256']==train['checkpoint_sha256'])
 r['checkpoint_sha256']=train.get('checkpoint_sha256')
print(json.dumps(r))
'''.replace('ROOT',repr(root)).replace('STAGE',repr(f'stage_{a.stage:03d}'))
r=json.loads(subprocess.check_output(['ssh','spark','python3','-'],input=code,text=True))
local=Path(__file__).resolve().parents[1]/'artifacts/omni_flat_2026-09-09'/a.campaign
local.mkdir(parents=True,exist_ok=True)
(local/'campaign_snapshot.json').write_text(json.dumps(r['campaign'],indent=2)+'\n')
if not r['ready']:
 print(json.dumps({'ready':False,'status':r['campaign']['status'],'mode':r['campaign'].get('current_mode')}));raise SystemExit(75)
out=local/f'stage_{a.stage:03d}';out.mkdir(exist_ok=True)
remote=root+f'/stage_{a.stage:03d}/f050_t060/'
for rel,name in [('video/rollout.mp4','rollout.mp4'),('video/video.json','video.json'),
                 ('evaluate/evaluation.json','evaluation.json'),('evaluate/state.json','evaluation_state.json'),
                 ('train/state.json','train_state.json'),('train/environment.yaml','environment.yaml'),
                 ('train/agent.yaml','agent.yaml'),('train/policy/final.pt','policy.pt')]:
 subprocess.run(['scp','-q','spark:'+remote+rel,str(out/name)],check=True)
assert hashlib.sha256((out/'policy.pt').read_bytes()).hexdigest()==r['checkpoint_sha256']
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(out/'rollout.mp4')],text=True))
video=json.loads((out/'video.json').read_text());evaluation=json.loads((out/'evaluation.json').read_text())
stream=next(s for s in probe['streams'] if s['codec_type']=='video')
assert int(stream['nb_frames'])==video['frames']
assert abs(float(probe['format']['duration'])-video['duration_s'])<.05
for seconds in (5,9,13,21,33,38):
 subprocess.run(['ffmpeg','-v','error','-y','-ss',str(seconds),'-i',str(out/'rollout.mp4'),'-frames:v','1',str(out/f'frame_{seconds:02d}.png')],check=True)
audit={'checkpoint_sha256':r['checkpoint_sha256'],'duration_s':video['duration_s'],
       'frames':video['frames'],'width':stream['width'],'height':stream['height'],
       'video_terminations':sum(t['terminated'] for t in video['trajectory']),
       'static_passed':sum(s['pass'] for s in evaluation['static']),
       'static_total':len(evaluation['static']),
       'transitions_passed':sum(s['pass'] for s in evaluation['transitions']),
       'all_scenarios_pass':evaluation['all_scenarios_pass'],'video':str(out/'rollout.mp4')}
(out/'recording_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
