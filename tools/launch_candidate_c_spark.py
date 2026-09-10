#!/usr/bin/env python3
"""Single-C validation, stepping probe, resumable PPO chunks and real videos."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace
from launch_length_training_spark import run_job,save,verified_source

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
p.add_argument("--isaaclab",type=Path,default=Path("/home/orionh/IsaacLab"))
p.add_argument("--first-updates",type=int,default=300)
p.add_argument("--chunk-updates",type=int,default=600)
p.add_argument("--max-updates",type=int,default=6300)
args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
if args.output.exists():p.error("Use fresh output to preserve all prior runs")
args.output.mkdir(parents=True);verified_source(args.source)
args.stop_request=args.output/"stop.request"
signal.signal(signal.SIGTERM,lambda signum,frame:args.stop_request.touch())
signal.signal(signal.SIGINT,lambda signum,frame:args.stop_request.touch())
report={"variant":"f050_t060","status":"starting","source":str(args.source),"started_unix":time.time(),"stages":[]}
deadline=time.monotonic()+30*3600

def job(target,mode,**kwargs):
    while True:
        if args.stop_request.exists():raise InterruptedError("User stop requested")
        if time.monotonic()>deadline:raise TimeoutError("Candidate campaign time bound reached")
        report.update(status="running",current_mode=mode,current_output=str(target.output));save(args.output/"candidate.json",report)
        try:return run_job(target,"f050_t060",mode,0,**kwargs)
        except BlockingIOError:
            report.update(status="waiting_for_shared_gpu");save(args.output/"candidate.json",report);time.sleep(20)

def scores(path):return json.loads((path/"evaluation.json").read_text())["speeds"]

def acceptable(rows):
    return len(rows)==2 and all(r["fall_fraction"]==0 and r["nonfoot_contact_fraction"]<=.001
        and r["saturation_fraction"]<=.005 and r["abs_forward_error_mps"]<=max(.03,.25*r["command_mps"])
        and r["abs_lateral_velocity_mps"]<=.03 and r["abs_yaw_rate_rad_s"]<=.15
        and r["tilt_rms_deg"]<=5 and r["vertical_velocity_rms_mps"]<=.04 for r in rows)

try:
    validated,state=job(args,"validate")
    if state["status"]!="completed":raise RuntimeError("C lower stance failed full standing gate")
    admission=validated/"admission.json"
    probed,_=job(args,"probe",admission=admission)
    probe=scores(probed);report["untrained_reference_probe"]=probe
    if any(r["fall_fraction"]>.25 or r["nonfoot_contact_fraction"]>.1 for r in probe):
        raise RuntimeError("Reference controller pilot is unstable; correct it before training")
    previous=None;total=0;consecutive=0
    while total<args.max_updates:
        count=min(args.first_updates if total==0 else args.chunk_updates,args.max_updates-total)
        stage=args.output/f"stage_{len(report['stages']):03d}";stage.mkdir()
        target=SimpleNamespace(source=args.source,output=stage,isaaclab=args.isaaclab,stop_request=args.stop_request)
        inputs=stage/"inputs";inputs.mkdir()
        shutil.copy2(admission,inputs/"admission.json")
        checkpoint=None
        if previous:
            checkpoint=inputs/"previous.pt";shutil.copy2(previous,checkpoint)
        entry={"output":str(stage),"new_updates":count,"status":"training"}
        report["stages"].append(entry)
        trained,_=job(target,"train",admission=inputs/"admission.json",checkpoint=checkpoint,iterations=count)
        previous=trained/"policy/final.pt";total+=count
        entry.update(total_updates=total,status="recording")
        video,_=job(target,"video",admission=inputs/"admission.json",checkpoint=previous)
        entry.update(video=str(video/"rollout.mp4"),status="evaluating")
        save(args.output/"candidate.json",report)
        evaluated,_=job(target,"evaluate",admission=inputs/"admission.json",checkpoint=previous)
        rows=scores(evaluated);passed=acceptable(rows);consecutive=consecutive+1 if passed else 0
        entry.update(status="completed",evaluation=rows,walking_gate_pass=passed)
        save(args.output/"candidate.json",report)
        if consecutive>=2:
            report.update(status="walking_gate_passed_pending_robustness",checkpoint=str(previous),total_updates=total)
            break
    else:report.update(status="training_budget_complete_needs_review",total_updates=total,checkpoint=str(previous))
except Exception as exc:
    report.update(status="stopped" if isinstance(exc,InterruptedError) else "failed",error=repr(exc))
    raise
finally:
    report["finished_unix"]=time.time();save(args.output/"candidate.json",report)
