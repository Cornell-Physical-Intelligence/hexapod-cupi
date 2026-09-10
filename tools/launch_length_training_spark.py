#!/usr/bin/env python3
"""Bounded 49-size campaign; waits for the shared GPU and owns only its jobs.

No lock is retained between actual jobs. A stop.request file stops the campaign
and only its currently owned immutable Docker ID. Failed standing poses never
train. Infrastructure errors stop the campaign for inspection rather than
being mistaken for morphology rejections.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from launch_length_study_spark import preflight, resources
from rank_length_study import summarize


def save(path, value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2)+"\n")
    temp.replace(path)


def verified_source(source):
    hashes=json.loads((source/"campaign_source_hashes.json").read_text())
    for relative, expected in hashes.items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError("Frozen source changed: "+relative)


def live_competitors(processes, owned, identity, proc_root=Path("/proc")):
    """Reconcile nvidia-smi and Docker snapshots taken at different times.

    A CUDA PID may exit between the GPU query and docker top during Kit
    shutdown. Ignore vanished processes and independently confirm the cgroup
    for live processes omitted by the later Docker snapshot. All other live
    CUDA processes remain competitors, including unreadable cgroups.
    """
    competing=[]
    for line in processes.splitlines():
        pid=line.split(",")[0].strip()
        if pid in owned:
            continue
        try:
            cgroup=(proc_root/pid/"cgroup").read_text()
        except FileNotFoundError:
            continue
        except PermissionError:
            cgroup="unreadable"
        if identity not in cgroup:
            competing.append({"process":line,"cgroup":cgroup.strip()})
    return competing


def run_job(args, variant, mode, stance, admission=None, checkpoint=None, iterations=None):
    output=args.output/variant/(f"validate_{stance}" if mode=="validate" else mode)
    if output.exists():
        raise RuntimeError("Refusing to overwrite prior job: "+str(output))
    locks=[]
    identity=None
    process=None
    try:
        for path in ("/opt/wx/gpu.lock","/tmp/hexapod-isaac-gpu.lock"):
            fd=os.open(path,os.O_RDONLY)
            locks.append(fd)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot=preflight()
        verified_source(args.source)
        output.mkdir(parents=True)
        name="hexapod-length-policy-"+uuid.uuid4().hex[:12]
        out_in=Path("/workspace/length-training-output")/variant/output.name
        cli=["docker","compose","--env-file","docker/.env.base","-f","docker/docker-compose.yaml","--profile","base",
             "run","--rm","--no-deps","--name",name,"-w",str(out_in),
             "-e","PYTHONDONTWRITEBYTECODE=1",
             "-e","PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/length-study/isaaclab:/workspace/length-study/tools",
             "-v",f"{args.source}:/workspace/length-study:rw",
             "-v",f"{args.output}:/workspace/length-training-output:rw",
             "--entrypoint","/workspace/isaaclab/_isaac_sim/python.sh","isaac-lab-base",
             "/workspace/length-study/tools/train_length_study.py","--headless","--device","cuda:0",
             "--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry",
             "--package","/workspace/length-study/robot/hexapod_mkii_length_study",
             "--output",str(out_in),"--variant",variant,"--mode",mode,"--stance-index",str(stance)]
        for flag,path in (("--admission",admission),("--checkpoint",checkpoint)):
            if path:
                cli += [flag,str(Path("/workspace/length-training-output")/path.relative_to(args.output))]
        if iterations is not None:cli += ["--iterations",str(iterations)]
        job={"status":"starting","container_name":name,"preflight":snapshot,"started_unix":time.time(),"mode":mode}
        save(output/"launcher.json",job)
        # Hard bounds include Kit startup. A timed-out learner is not ranked.
        deadline=time.monotonic()+(7200 if mode=="train" else 1800)
        with (output/"run.log").open("w") as log:
            process=subprocess.Popen(cli,cwd=args.isaaclab,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if identity is None:
                    found=subprocess.run(["docker","inspect","--format","{{.Id}}",name],text=True,capture_output=True,timeout=20)
                    if found.returncode==0:
                        identity=found.stdout.strip()
                        job.update(container_id=identity,status="running")
                        save(output/"launcher.json",job)
                if (args.output/"stop.request").exists() or (getattr(args,"stop_request",args.output/"stop.request")).exists():
                    raise InterruptedError("User/campaign stop requested")
                if time.monotonic()>=deadline:
                    raise TimeoutError("Per-job deadline exceeded")
                processes,available=resources()
                if available<16*1024**3:
                    raise MemoryError("Less than 16 GiB available; stopping only this campaign job")
                if identity:
                    top=subprocess.run(["docker","top",identity,"-eo","pid"],text=True,capture_output=True,timeout=20)
                    if top.returncode==0:
                        owned={p.strip() for p in top.stdout.splitlines()[1:]}
                        competing=live_competitors(processes,owned,identity)
                        if competing:
                            job.update(status="interrupted",competing=competing,gpu_snapshot=processes,
                                       docker_owned_pids=sorted(owned),interrupted_unix=time.time())
                            save(output/"launcher.json",job)
                            raise RuntimeError("Competing CUDA process appeared; stopping only this job")
                time.sleep(5)
        state_path=output/"state.json"
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        # Kit sometimes returns zero after Python exceptions; require artifacts.
        if (process.returncode!=0 or (output/"failure.json").exists()
            or state.get("status") not in ("completed","rejected")):
            raise RuntimeError("Job failed; inspect "+str(output/"run.log"))
        if mode=="train" and not (output/"policy/final.pt").is_file():
            raise RuntimeError("Training completion has no final checkpoint")
        if mode in ("evaluate","probe"):
            plan=json.loads((args.source/"robot/hexapod_mkii_length_study/training_plan.json").read_text())
            diagnostic=mode=="evaluate" and "diagnostics" in plan.get("omni",{})
            report_path=output/("diagnostics.json" if diagnostic else "evaluation.json")
            report=json.loads(report_path.read_text())
            if not report.get("complete"):
                raise RuntimeError("Evaluation is incomplete")
            if diagnostic and (report.get("kind")!="diagnostic_not_qualification" or not (output/"diagnostic_trace.npz").is_file()):
                raise RuntimeError("Diagnostic evidence is incomplete")
        if mode=="video" and not (json.loads((output/"video.json").read_text()).get("complete") and (output/"rollout.mp4").stat().st_size>10000):
            raise RuntimeError("Video is incomplete")
        job.update(status=state["status"],finished_unix=time.time(),exit_code=process.returncode)
        save(output/"launcher.json",job)
        return output,state
    finally:
        if identity:
            running=subprocess.run(["docker","inspect","--format","{{.State.Running}}",identity],text=True,capture_output=True,timeout=20)
            if running.returncode==0 and running.stdout.strip()=="true":
                subprocess.run(["docker","stop","--time","20",identity],timeout=30,capture_output=True)
        if process is not None and process.poll() is None:
            if not identity:
                process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
        for fd in reversed(locks):os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--isaaclab",type=Path,default=Path("/home/orionh/IsaacLab"))
    p.add_argument("--campaign-hours",type=float,default=48)
    args=p.parse_args()
    args.source=args.source.resolve();args.output=args.output.resolve()
    if args.output.exists():p.error("Use a fresh campaign output directory")
    args.output.mkdir(parents=True)
    verified_source(args.source)
    plan=json.loads((args.source/"robot/hexapod_mkii_length_study/training_plan.json").read_text())
    order=list(plan["variants"])
    # Baseline first diagnoses the adapter; then reproducibly shuffle sizes to
    # avoid confounding leg length with host temperature or wall-clock order.
    import random
    order.remove("f100_t100");random.Random(57).shuffle(order);order.insert(0,"f100_t100")
    report={"status":"starting","pid":os.getpid(),"source":str(args.source),"started_unix":time.time(),
            "variants":{},"order":order,"plan":{k:v for k,v in plan.items() if k!="variants"}}
    deadline=time.monotonic()+args.campaign_hours*3600
    def job(*values,**kwargs):
        while True:
            if time.monotonic()>=deadline:raise TimeoutError("Campaign deadline exceeded")
            if (args.output/"stop.request").exists():raise InterruptedError("Stop requested")
            try:
                report.update(status="running",current_variant=values[0],current_mode=values[1])
                save(args.output/"campaign.json",report)
                return run_job(args,*values,**kwargs)
            except BlockingIOError as exc:
                report.update(status="waiting_for_shared_gpu",reason=str(exc),last_checked_unix=time.time())
                save(args.output/"campaign.json",report)
                time.sleep(30)
    try:
        for variant in order:
            entry=report["variants"][variant]={"status":"validating","stances":[]}
            admitted=None
            for index in range(len(plan["variants"][variant]["stances"])):
                out,state=job(variant,"validate",index)
                entry["stances"].append(state.get("gate"))
                if state["status"]=="completed":
                    admitted=out/"admission.json";break
            if admitted is None:
                entry["status"]="standing_rejected"
                save(args.output/"campaign.json",report)
                continue
            entry.update(status="training",stance_index=index)
            trained,_=job(variant,"train",index,admission=admitted)
            entry["status"]="evaluating"
            job(variant,"evaluate",index,admission=admitted,checkpoint=trained/"policy/final.pt")
            entry["status"]="completed"
            summarize(args.output)
            save(args.output/"campaign.json",report)
        report.update(status="completed",finished_unix=time.time())
        summarize(args.output)
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc,InterruptedError) else "failed",reason=repr(exc),finished_unix=time.time())
        raise
    finally:save(args.output/"campaign.json",report)


if __name__=="__main__":main()
