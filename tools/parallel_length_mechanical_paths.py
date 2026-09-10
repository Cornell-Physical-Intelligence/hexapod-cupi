#!/usr/bin/env python3
"""Run independent CPU geometry shards, preserving checked completed results."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--static",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
p.add_argument("--prior",type=Path);p.add_argument("--workers",type=int,default=4)
p.add_argument("--package",type=Path,default=Path("robot/hexapod_mkii_length_study"))
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
names=list(json.loads((args.static/"mechanical_screen.json").read_text())["variants"])
report={"status":"running","variants":{},"started_unix":time.time()}
if args.prior:
    report=json.loads((args.prior/"path_screen.json").read_text())
    for name in report["variants"]:
        shutil.copy2(args.prior/(name+"_trials.json"),args.output/(name+"_trials.json"))
remaining=[name for name in names if name not in report["variants"]]
jobs=[];logs=[];start=time.time()
try:
    for i in range(args.workers):
        subset=remaining[i::args.workers]
        if not subset:continue
        out=args.output/f"shard_{i}";log=(args.output/f"shard_{i}.log").open("w");logs.append(log)
        env={**os.environ,"VECLIB_MAXIMUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
        cmd=[sys.executable,"-B",str(Path(__file__).with_name("run_length_mechanical_paths.py")),"--static",str(args.static),"--output",str(out),"--package",str(args.package),"--variants",*subset]
        jobs.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env),out))
    previous=-1
    while True:
        for process,out in jobs:
            file=out/"path_screen.json"
            if file.exists():
                try:data=json.loads(file.read_text())
                except json.JSONDecodeError:continue
                if "source_sha256" in report and report["source_sha256"]!=data["source_sha256"]:raise RuntimeError("Shard source mismatch")
                for key,value in data.items():
                    if key not in ("status","variants","started_unix","elapsed_s"):report[key]=value
                for name,value in data["variants"].items():
                    report["variants"][name]=value
                    if not (args.output/(name+"_trials.json")).exists():shutil.copy2(out/(name+"_trials.json"),args.output/(name+"_trials.json"))
            if process.poll() not in (None,0):raise RuntimeError("Failed CPU shard: "+str(out))
        report["wall_elapsed_s"]=time.time()-start
        (args.output/"path_screen.json").write_text(json.dumps(report,indent=2))
        if len(report["variants"])!=previous:
            previous=len(report["variants"]);print(f"MECHANICS {previous}/{len(names)}",flush=True)
        if all(proc.poll()==0 for proc,_ in jobs):break
        time.sleep(5)
    if set(report["variants"])!=set(names):raise RuntimeError("Incomplete geometry set")
    report["status"]="completed"
except BaseException as exc:
    report.update(status="failed",error=repr(exc));raise
finally:
    for process,_ in jobs:
        if process.poll() is None:process.terminate();process.wait()
    for log in logs:log.close()
    (args.output/"path_screen.json").write_text(json.dumps(report,indent=2))
