#!/usr/bin/env python3
"""Record and evaluate a completed baseline; never enqueue the other sizes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from launch_length_training_spark import run_job, save, verified_source

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--source",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--training-campaign",type=Path,required=True)
p.add_argument("--isaaclab",type=Path,default=Path("/home/orionh/IsaacLab"))
args=p.parse_args()
if args.output.exists():p.error("Use fresh pilot output")
args.output.mkdir(parents=True)
verified_source(args.source)
report={"status":"waiting_for_training","source":str(args.source),"started_unix":time.time()}
save(args.output/"pilot.json",report)
try:
    deadline=time.monotonic()+1800
    while json.loads((args.training_campaign/"campaign.json").read_text())["status"] not in ("stopped","failed","completed"):
        if time.monotonic()>deadline:raise TimeoutError("Waiting for baseline hold")
        time.sleep(5)
    prior=args.training_campaign/"f100_t100"
    state=json.loads((prior/"train/state.json").read_text())
    checkpoint=prior/"train/policy/final.pt"
    if state["status"]!="completed" or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=state["checkpoint_sha256"]:
        raise RuntimeError("Baseline training did not complete with a verified checkpoint")
    for relative in ("train/policy/final.pt","train/state.json","validate_0/admission.json"):
        destination=args.output/"f100_t100"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(prior/relative,destination)
    for mode in ("video","evaluate"):
        report.update(status="running",mode=mode)
        save(args.output/"pilot.json",report)
        launch_deadline=time.monotonic()+600
        while True:
            try:
                run_job(args,"f100_t100",mode,state["stance_index"],admission=args.output/"f100_t100/validate_0/admission.json",checkpoint=args.output/"f100_t100/train/policy/final.pt")
                break
            except BlockingIOError:
                if time.monotonic()>launch_deadline:raise
                time.sleep(5)
    report.update(status="completed",finished_unix=time.time())
except Exception as exc:
    report.update(status="failed",error=repr(exc),finished_unix=time.time())
    raise
finally:
    save(args.output/"pilot.json",report)
