#!/usr/bin/env python3
"""Fetch only a completed, identity-checked C checkpoint recording from Spark."""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--campaign",default="candidate_c_001")
p.add_argument("--stage",type=int)
args=p.parse_args()
if not args.campaign.replace("_","").isalnum():p.error("Invalid campaign name")
remote_root="/home/orionh/HEXAPOD_runs/mock_length_study_20260909/"+args.campaign
code="""
import json
from pathlib import Path
p=Path(ROOT)
r=json.loads((p/'candidate.json').read_text())
ready=[]
for d in sorted(p.glob('stage_*')):
 v=d/'f050_t060/video/video.json';t=d/'f050_t060/train/state.json'
 if v.exists() and t.exists():
  video=json.loads(v.read_text());train=json.loads(t.read_text())
  if video.get('complete') and train.get('status')=='completed' and video['checkpoint_sha256']==train['checkpoint_sha256']:
   ready.append({'stage':int(d.name.split('_')[-1]),'path':str(d),'train':train,'video':{k:v for k,v in video.items() if k!='trajectory'}})
print(json.dumps({'campaign':r,'ready':ready}))
""".replace("ROOT",repr(remote_root))
snapshot=json.loads(subprocess.check_output(["ssh","spark","python3","-"],input=code,text=True))
root=Path(__file__).resolve().parents[3]/"artifacts/length_study_2026-09-09"/args.campaign
root.mkdir(exist_ok=True)
(root/"candidate_snapshot.json").write_text(json.dumps(snapshot["campaign"],indent=2)+"\n")
ready=[r for r in snapshot["ready"] if args.stage is None or r["stage"]==args.stage]
if not ready:
    print(json.dumps({"status":"recording_not_ready","current_mode":snapshot["campaign"].get("current_mode")}));raise SystemExit(75)
selected=ready[-1];destination=root/f"stage_{selected['stage']:03d}";destination.mkdir(exist_ok=True)
remote=Path(selected["path"])
if not remote.is_relative_to(Path(remote_root)):raise RuntimeError("Unexpected remote artifact path")
for relative,name in (("video/rollout.mp4","rollout.mp4"),("video/video.json","video.json"),
                      ("train/policy/final.pt","policy.pt"),("train/environment.yaml","environment.yaml"),
                      ("train/agent.yaml","agent.yaml")):
    subprocess.run(["scp","-q","spark:"+str(remote/"f050_t060"/relative),str(destination/name)],check=True)
assert hashlib.sha256((destination/"policy.pt").read_bytes()).hexdigest()==selected["video"]["checkpoint_sha256"]
probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(destination/"rollout.mp4")],text=True))
stream=next(s for s in probe["streams"] if s["codec_type"]=="video")
if int(stream["nb_frames"])!=selected["video"]["frames"]:raise RuntimeError("Video frame count mismatch")
for seconds in (1,5,10):
    subprocess.run(["ffmpeg","-v","error","-y","-ss",str(seconds),"-i",str(destination/"rollout.mp4"),"-frames:v","1",str(destination/f"frame_{seconds:02d}.png")],check=True)
video=json.loads((destination/"video.json").read_text())
steady=[t for t in video["trajectory"] if t["time_s"]>=2]
audit={"stage":selected["stage"],"checkpoint_sha256":selected["video"]["checkpoint_sha256"],
       "training_updates_this_chunk":selected["train"]["iterations"],"duration_s":float(probe["format"]["duration"]),
       "width":stream["width"],"height":stream["height"],"frames":int(stream["nb_frames"]),
       "mean_forward_velocity_after_2s_mps":sum(t["forward_velocity_mps"] for t in steady)/len(steady),
       "falls":sum(t["terminated"] for t in video["trajectory"]),"path":str(destination/"rollout.mp4")}
(destination/"recording_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps(audit,indent=2))
