from pathlib import Path
base=Path('tmp/omni_velocity_train_003/launch_velocity_train_spark.py').read_text()
head=base[:base.index('def command(')]
owned=base[base.index('def owned_container('):base.index('def checkpoint_contract(')]
owned=owned.replace('hexapod-velocity-pilot-', 'hexapod-velocity-recording-')
start=owned.index('        state_path = args.output / phase / "state.json"')
end=owned.index('    except Exception as exc:',start)
owned=owned[:start]+'''        video = validate_recording(args.output / "recording", process.returncode)
        report.update(status="completed" if video["complete"] else "stopped_at_terminal_event", exit_code=process.returncode)
        return video
'''+owned[end:]
owned=owned.replace('command(args.source, args.output, name, phase)', 'command(args, name)')
owned=owned.replace('existing_checkpoint_loaded=(phase!="train"), walking_training_started=(phase=="train"),\n                  walking_update_cap=(50 if phase=="train" else 0)', 'existing_checkpoint_loaded=True, walking_training_started=False, walking_update_cap=0')
command='''CHECKPOINT_SHA="88f8e60a7b1536229f51524cdc646f06d53e4f33cabb74aa89ebde73d06f7247"
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

'''
main='''def main():
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
'''
Path('tmp/omni_velocity_recording_launch_004/launch_recording_spark.py').write_text(head+command+owned+main)
