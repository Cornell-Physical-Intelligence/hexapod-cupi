#!/usr/bin/env python3
"""Contingent 50-update scratch pilot plus matched initial/final diagnostics.

Run OUTSIDE the immutable admitted source tree. This launcher never resumes old
weights, automatically continues training, or marks Stage 2 complete. The host
must use the established identified forecasting pause/restoration wrapper.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
sys.dont_write_bytecode=True

VARIANT="f050_t060"
RUNTIME_TREE="abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280"
COORDINATION=Path("/home/orionh/SPARK_COMPUTE_COORDINATION.md")
LINEAGE="c_serial_omni_target_velocity_v1"
PROBE_SOURCE_ID="fb5e472bb710daa50d1d7f16a6f3812d2ed845e2400fcc6769859b3fb0b9519e"
EXPECTED_PROBE_NAME="omni_velocity_probe_003"
IDENTITY_KEYS=("variant","urdf_sha256","plan_sha256","stance_index","source_sha256")

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def tree_hashes(path):return {str(p.relative_to(path)):digest(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}
def write(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n");temporary.replace(path)

def executable_source_digest(root):
    paths=set();root=Path(root)
    for folder in ("tools","isaaclab","packages","experiments/c_length_study/runtime"):
        paths.update(p for p in (root/folder).rglob("*.py") if "__pycache__" not in p.parts)
    paths.update((root/"experiments/c_length_study/runtime").glob("*.json"))
    encoded={str(p.relative_to(root)):digest(p) for p in sorted(paths)}
    return hashlib.sha256(json.dumps(encoded,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def check_source(source):
    source=Path(source)
    manifest=read(source/"campaign_source_hashes.json")
    actual={key:value for key,value in tree_hashes(source).items() if key!="campaign_source_hashes.json"}
    if actual!=manifest:raise ValueError("Frozen source has changed or unlisted files; use exact admitted source")
    if executable_source_digest(source)!=PROBE_SOURCE_ID:raise ValueError("This launcher is bound to reviewed candidate source_003")
    plan=read(source/"robot/hexapod_mkii_length_study/training_plan.json")
    expected={"training_num_envs":1024,"training_iterations":50,"evaluation_num_envs":48,"validation_num_envs":32,"validation_control_steps":1000}
    if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError("Pilot allocation differs from admitted explicit plan")
    omni=plan["omni"]
    if (omni.get("architecture")!=LINEAGE or omni.get("velocity_candidate")!={"profile":"formal_004","max_acceleration_rad_s2":8.0}
        or "repair_training" in omni or "reference_controller" in plan):raise ValueError("Wrong controller lineage/profile or old checkpoint repair path")
    if omni.get("initial_exploration_std") != .005 or omni.get("sampled_calibration_control_steps") != 1000:
        raise ValueError("Candidate 003 requires std 0.005 and 20 s sampled calibration")
    return {"source_sha256":PROBE_SOURCE_ID,"plan_sha256":digest(source/"robot/hexapod_mkii_length_study/training_plan.json")}

def accepted_probe(source,probe,expected_campaign_sha256):
    """Evidence must all match; no acceptance inferred from an existing directory."""
    source=Path(source);probe=Path(probe);binding=check_source(source)
    if probe.name != EXPECTED_PROBE_NAME:
        raise ValueError("This launcher requires the exact reviewed probe 003 output")
    if (not isinstance(expected_campaign_sha256,str) or len(expected_campaign_sha256)!=64
        or any(c not in "0123456789abcdef" for c in expected_campaign_sha256)
        or digest(probe/"campaign.json") != expected_campaign_sha256):
        raise ValueError("Accepted probe campaign SHA256 differs from the explicit reviewed receipt")
    campaign=read(probe/"campaign.json")
    if campaign.get("status")!="completed" or campaign.get("walking_training_started") is not False:
        raise ValueError("A completed standing-only probe is required")
    if (campaign.get("source_manifest_sha256")!=digest(source/"campaign_source_hashes.json")
        or campaign.get("runtime_binding",{}).get("runtime_tree_sha256")!=RUNTIME_TREE
        or campaign.get("source_unchanged") is not True or campaign.get("admitted_asset_unchanged") is not True):
        raise ValueError("Probe source/runtime/admitted-asset binding mismatch")
    admission=read(probe/"flat/admission.json");calibration=read(probe/"probe/calibration.json");smoke=read(probe/"probe/runner_smoke.json")
    for key,path in [("flat_admission_sha256","flat/admission.json"),("calibration_sha256","probe/calibration.json"),("runner_smoke_sha256","probe/runner_smoke.json")]:
        if campaign.get(key)!=digest(probe/path):raise ValueError("Probe evidence hashes differ from completed campaign")
    identity={k:admission[k] for k in IDENTITY_KEYS}
    if (identity["variant"]!=VARIANT or identity["stance_index"]!=0
        or any(identity[k]!=value for k,value in binding.items())
        or any(calibration.get(k)!=v or smoke.get(k)!=v for k,v in identity.items())):
        raise ValueError("Source/plan/asset/stance identity differs across accepted evidence")
    gate=admission.get("gate",{})
    if gate.get("passed") is not True or gate.get("num_envs")!=32 or gate.get("control_steps")!=1000:
        raise ValueError("Missing full 32-environment standing admission")
    if (calibration.get("passed") is not True or calibration.get("initial_std")!=.005
        or calibration.get("stage2_complete") is not False or smoke.get("passed") is not True
        or smoke.get("learning_updates")!=2 or smoke.get("stage2_complete") is not False
        or smoke.get("calibration_sha256")!=digest(probe/"probe/calibration.json")):
        raise ValueError("Actual exploration and two-update runner smoke must pass with matching calibration")
    trials=calibration.get("trials",[])
    if (len(trials)!=2 or trials[0].get("name")!="zero_mean" or trials[0].get("steps")!=250
        or trials[1].get("name")!="sampled_std_0p005" or trials[1].get("steps")!=1000
        or any(t.get("passed") is not True or len(t.get("per_environment",[]))!=32
               or any(r.get("passed") is not True for r in t["per_environment"]) for t in trials)):
        raise ValueError("Every replica must pass the exact zero-mean and 20 s std-0.005 trials")
    if tree_hashes(probe/"inputs/study")!=read(probe/"inputs/study_after_flat.sha256.json"):
        raise ValueError("Accepted study package changed after standing")
    return identity

CHECKPOINT_SHA="88f8e60a7b1536229f51524cdc646f06d53e4f33cabb74aa89ebde73d06f7247"
PROBE_RECEIPT="ca436a5439526cfd9b742143650fa1a6b9f132e812fa02f38415bf0324327155"

def recording_arguments():
    return ["--source-root","/workspace/hexapod","--package","/pilot/inputs/study",
        "--checkpoint","/pilot/train/policy/final.pt","--checkpoint-sha256",CHECKPOINT_SHA,
        "--checkpoint-label","scratch_50_update_final","--admission","/probe/flat/admission.json",
        "--calibration","/probe/probe/calibration.json","--runner-smoke","/probe/probe/runner_smoke.json",
        "--pilot-state","/pilot/train/state.json","--probe-campaign","/probe/campaign.json",
        "--study-tree-receipt","/probe/inputs/study_after_flat.sha256.json",
        "--pilot-inputs-receipt","/pilot/inputs_before.sha256.json","--output","/outputs/recording"]

def command(args,name):
    return ["docker","compose","--env-file","docker/.env.base","-f","docker/docker-compose.yaml",
        "--profile","base","run","--rm","--no-deps","--name",name,"-w","/outputs",
        "-e","PYTHONDONTWRITEBYTECODE=1","-e",
        "PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/hexapod/tools:/workspace/hexapod/isaaclab",
        "-v",f"{args.source}:/workspace/hexapod:ro","-v",f"{args.output}:/outputs:rw",
        "-v",f"{args.probe}:/probe:ro","-v",f"{args.pilot}:/pilot:ro","-v",f"{args.adapter}:/recording:ro",
        "--entrypoint","/workspace/isaaclab/_isaac_sim/python.sh","isaac-lab-base",
        "/recording/record_candidate_video.py",*recording_arguments(),"--headless","--enable_cameras","--device","cuda:0","--info",
        "--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry"]

def validate_inputs(args):
    identity=accepted_probe(args.source,args.probe,PROBE_RECEIPT)
    campaign=read(args.pilot/"campaign.json")
    if (campaign.get("status")!="completed_needs_review" or campaign.get("identity")!=identity
        or campaign.get("source_unchanged") is not True or campaign.get("admitted_inputs_unchanged") is not True
        or campaign.get("checkpoint_shas",{}).get("final")!=CHECKPOINT_SHA
        or campaign.get("comparison_sha256")!=digest(args.pilot/"comparison.json")):
        raise ValueError("Exact pilot must finish both matched evaluations before recording")
    if tree_hashes(args.pilot/"inputs")!=read(args.pilot/"inputs_before.sha256.json"):
        raise ValueError("Pilot admitted inputs changed")
    if tree_hashes(args.pilot/"inputs/study")!=read(args.probe/"inputs/study_after_flat.sha256.json"):
        raise ValueError("Pilot package differs from actual standing-admitted package")
    if digest(args.pilot/"train/policy/final.pt")!=CHECKPOINT_SHA:
        raise ValueError("Final checkpoint bytes changed")
    manifest=read(args.adapter/"FREEZE_SHA256.json")
    actual={k:v for k,v in tree_hashes(args.adapter).items() if k!="FREEZE_SHA256.json"}
    if actual!=manifest["files"]:raise ValueError("Recorder code changed or has unlisted files")
    return identity

def validate_recording(output,returncode):
    output=Path(output)
    if returncode!=0 or (output/"failure.json").exists():raise RuntimeError("Recorder failed; preserve output/logs")
    video=read(output/"video.json")
    if (video.get("checkpoint_sha256")!=CHECKPOINT_SHA or video.get("strict_tensor_readback_completed") is not True
        or video.get("stage2_complete") is not False or video.get("qualification_performed") is not False
        or video.get("runtime_binding",{}).get("runtime_tree_sha256")!=RUNTIME_TREE):
        raise ValueError("Missing exact runtime/checkpoint/strict-load recording evidence")
    if video.get("frames",0)<=0 or not (output/"rollout.mp4").is_file():raise ValueError("No actual video frames")
    if video.get("video_sha256")!=digest(output/"rollout.mp4"):raise ValueError("Video bytes differ from recorder receipt")
    if video.get("source_and_checkpoint_reverified_after_recording") is not True:raise ValueError("Missing final source verification")
    if video.get("complete") and (video.get("recorded_control_steps")!=1700 or video.get("planned_control_steps")!=1700):raise ValueError("Complete clip must cover the entire command sequence")
    if not video.get("complete") and not video.get("terminal_event"):raise ValueError("Incomplete recording lacks terminal evidence")
    return video

def owned_container(name, identity=None):
    """Recover immutable identity even if Docker client exits before first poll."""
    found = subprocess.run(["docker", "inspect", "--format", "{{.Id}} {{.Name}} {{.State.Running}}",
                            identity or name], text=True, capture_output=True, timeout=20)
    if found.returncode:
        return None
    fields = found.stdout.strip().split()
    if len(fields) != 3 or fields[1] != "/" + name or (identity and fields[0] != identity):
        raise RuntimeError("Container identity mismatch; do not signal it")
    return fields[0], fields[2] == "true"


def run_owned(args, phase):
    locks, process, identity = [], None, None
    name = "hexapod-velocity-recording-" + uuid.uuid4().hex
    report = dict(status="starting", phase=phase, container_name=name, started_unix=time.time(),
                  deadline_seconds=600, existing_checkpoint_loaded=True, walking_training_started=False, walking_update_cap=0)
    report_path = args.output / "jobs" / (phase + ".json")
    try:
        if (args.output / "stop.request").exists():
            raise InterruptedError("Stop requested before acquiring a new job")
        if digest(COORDINATION) != args.coordination_sha256:
            raise InterruptedError("Coordination note changed; return ownership for review")
        report["coordination_sha256"] = args.coordination_sha256
        for path in ("/opt/wx/gpu.lock", "/tmp/hexapod-isaac-gpu.lock"):
            fd = os.open(path, os.O_RDONLY)
            locks.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report["preflight"] = preflight()
        verified_source(args.source)
        save(report_path, report)
        deadline = time.monotonic() + 600
        with (args.output / "logs" / (phase + ".log")).open("w") as log:
            process = subprocess.Popen(command(args, name), cwd=args.isaaclab,
                                       stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                owned = owned_container(name, identity)
                if owned:
                    identity = owned[0]
                    report.update(status="running", container_id=identity)
                    save(report_path, report)
                if (args.output / "stop.request").exists():
                    raise InterruptedError("Stop requested")
                if digest(COORDINATION) != args.coordination_sha256:
                    raise InterruptedError("Coordination changed during the bounded pilot; yielding owned job")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Bounded phase exceeded ten-minute limit")
                processes, available = resources()
                if available < 16 * 1024**3:
                    raise MemoryError("Less than 16 GiB available")
                if identity:
                    top = subprocess.run(["docker", "top", identity, "-eo", "pid"], text=True,
                                         capture_output=True, timeout=20)
                    if top.returncode == 0:
                        owned_pids = {v.strip() for v in top.stdout.splitlines()[1:]}
                        competitors = live_competitors(processes, owned_pids, identity)
                        if competitors:
                            report["competitors"] = competitors
                            raise RuntimeError("Unrelated CUDA process appeared; yielding this owned job")
                time.sleep(5)
        video = validate_recording(args.output / "recording", process.returncode)
        report.update(status="completed" if video["complete"] else "stopped_at_terminal_event", exit_code=process.returncode)
        return video
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc, InterruptedError) else "failed", error=repr(exc))
        raise
    finally:
        # No Docker client process-state condition: client exit does not mean
        # its container exited. Name recovery is bounded and identity-checked.
        try:
            owned = owned_container(name, identity) if process is not None else None
            if owned and owned[1]:
                identity = owned[0]
                subprocess.run(["docker", "stop", "--time", "20", identity], timeout=30, check=True, capture_output=True)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            # Catch container creation racing the Docker-client termination.
            remaining = owned_container(name, identity) if process is not None else None
            if remaining and remaining[1]:
                subprocess.run(["docker", "stop", "--time", "20", remaining[0]], timeout=30, check=True, capture_output=True)
            report["cleanup_checked"] = True
        finally:
            report.update(finished_unix=time.time(), container_id=identity)
            save(report_path, report)
            for fd in reversed(locks):
                os.close(fd)


def main():
    parser=argparse.ArgumentParser(description="One bounded actual PPO RGB recording; no training or qualification")
    for name in ("source","probe","pilot","adapter","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--isaaclab",type=Path,default=Path("/home/orionh/IsaacLab"))
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    for name in ("source","probe","pilot","adapter","output"):setattr(args,name,getattr(args,name).resolve())
    identity=validate_inputs(args)
    if args.output.exists():parser.error("Fresh recording output required")
    if args.preflight_only:print(json.dumps({"passed":True,"identity":identity,"checkpoint_sha256":CHECKPOINT_SHA}));return
    sys.path.insert(0,str(args.source/"tools"))
    global preflight,resources,live_competitors,verified_source,save
    resource_module=importlib.import_module("launch_length_study_spark")
    training_module=importlib.import_module("launch_length_training_spark")
    preflight=resource_module.preflight;resources=resource_module.resources
    live_competitors=training_module.live_competitors;verified_source=training_module.verified_source;save=training_module.save
    args.coordination_sha256=digest(COORDINATION)
    args.output.mkdir(parents=True)
    for directory in ("logs","jobs"):(args.output/directory).mkdir()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:(args.output/"stop.request").touch())
    report={"status":"recording","started_unix":time.time(),"stage2_complete":False,"identity":identity,
        "source_manifest_sha256":digest(args.source/"campaign_source_hashes.json"),"launcher_sha256":digest(__file__),
        "adapter_manifest_sha256":digest(args.adapter/"FREEZE_SHA256.json"),"pilot_campaign_sha256":digest(args.pilot/"campaign.json"),
        "checkpoint_sha256":CHECKPOINT_SHA,"walking_training_started":False}
    save(args.output/"campaign.json",report)
    try:
        video=run_owned(args,"recording")
        validate_inputs(args)
        report.update(status="completed" if video["complete"] else "stopped_at_terminal_event",
            source_and_admitted_inputs_unchanged=True,video_sha256=digest(args.output/"recording/rollout.mp4"),
            video_report_sha256=digest(args.output/"recording/video.json"))
    except Exception as exc:report.update(status="failed",error=repr(exc));raise
    finally:report["finished_unix"]=time.time();save(args.output/"campaign.json",report)

if __name__=="__main__":main()
