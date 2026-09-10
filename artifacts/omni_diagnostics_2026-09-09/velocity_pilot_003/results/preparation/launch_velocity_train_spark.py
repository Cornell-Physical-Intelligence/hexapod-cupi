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

def command(source,output,name,phase):
    if phase not in ("train","initial","final"):raise ValueError("Only one 50-update scratch pilot and its matched evaluations are allowed")
    argv=["docker","compose","--env-file","docker/.env.base","-f","docker/docker-compose.yaml",
        "--profile","base","run","--rm","--no-deps","--name",name,"-w","/outputs",
        "-e","PYTHONDONTWRITEBYTECODE=1","-e",
        "PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/hexapod/tools:/workspace/hexapod/isaaclab",
        "-v",f"{source}:/workspace/hexapod:ro","-v",f"{output}:/outputs:rw",
        "-v",f"{output/'inputs/study'}:/study:ro",
        "--entrypoint","/workspace/isaaclab/_isaac_sim/python.sh","isaac-lab-base",
        "/workspace/hexapod/tools/train_velocity_candidate.py","--mode","train" if phase=="train" else "evaluate",
        "--package","/study","--variant",VARIANT,"--stance-index","0","--admission","/outputs/inputs/admission.json",
        "--output",f"/outputs/{phase}","--headless","--device","cuda:0","--info",
        "--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry"]
    if phase=="train":argv += ["--calibration","/outputs/inputs/calibration.json","--runner-smoke","/outputs/inputs/runner_smoke.json","--iterations","50"]
    else:argv += ["--checkpoint",f"/outputs/train/policy/{phase}.pt"]
    return argv

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
    name = "hexapod-velocity-pilot-" + uuid.uuid4().hex
    report = dict(status="starting", phase=phase, container_name=name, started_unix=time.time(),
                  deadline_seconds=600, existing_checkpoint_loaded=(phase!="train"), walking_training_started=(phase=="train"),
                  walking_update_cap=(50 if phase=="train" else 0))
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
            process = subprocess.Popen(command(args.source, args.output, name, phase), cwd=args.isaaclab,
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
        state_path = args.output / phase / "state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if process.returncode != 0 or state.get("status") != "completed" or (args.output / phase / "failure.json").exists():
            raise RuntimeError(f"{phase} did not complete; inspect its state and log")
        if state.get("runtime_binding", {}).get("runtime_tree_sha256") != RUNTIME_TREE:
            raise RuntimeError("Runtime identity missing from candidate evidence")
        report.update(status="completed", exit_code=process.returncode)
        return state
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


def checkpoint_contract(path,identity):
    path=Path(path);meta=read(path.with_suffix(path.suffix+".json"))
    expected={**identity,"lineage":LINEAGE,"profile":"formal_004","actor_frame_width":99,
        "actor_width":495,"critic_width":498,"max_position_step_rad_per_20ms":.04,
        "max_target_velocity_rad_s":2.,"max_target_acceleration_rad_s2":8.,"max_acceleration_rad_s2":8.,
        "action_semantics":"normalized_joint_target_velocity","integration":"semi_implicit_discrete",
        "physics_admitted":False,"motor_speed_limit_claim":False}
    if {k:v for k,v in meta.items() if k!="checkpoint_sha256"}!=expected or meta.get("checkpoint_sha256")!=digest(path):
        raise ValueError("Checkpoint sidecar/hash/action/source identity mismatch")
    return meta

def finite_report(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError("Nonfinite evaluation report")
    if isinstance(value,dict):
        for child in value.values():finite_report(child)
    elif isinstance(value,list):
        for child in value:finite_report(child)

def compare_evaluations(initial,final,identity,checkpoint_shas):
    """Keep every direction/replica/quiet trial; never promote an aggregate median."""
    reports=[];quiet_reports=[]
    for phase,path in [("initial",Path(initial)),("final",Path(final))]:
        report=read(path/"diagnostics.json");quiet=read(path/"quiet_review/quiet_stand.json")
        completion=read(path/"evaluation.json")
        finite_report(report);finite_report(quiet)
        if (report.get("complete") is not True or report.get("stage2_complete") is not False
            or report.get("checkpoint_sha256")!=checkpoint_shas[phase]
            or report.get("controller",{}).get("lineage")!=LINEAGE
            or report.get("controller",{}).get("profile")!="formal_004"
            or report.get("observation_audit",{}).get("actor_width")!=495
            or report.get("observation_audit",{}).get("critic_width")!=498
            or quiet.get("complete") is not True or quiet.get("checkpoint_sha256")!=checkpoint_shas[phase]
            or completion.get("complete") is not True or completion.get("stage2_complete") is not False):
            raise ValueError("Incomplete or wrong-lineage direction/quiet evaluation")
        audit=report["observation_audit"]
        for key in ("max_same_step_repeat_difference","max_command_slice_difference","max_history_shift_difference","max_executable_state_difference"):
            if audit.get(key)!=0:raise ValueError("Candidate observation/state/history audit failed")
        if len(report.get("scenarios",[]))!=12 or len(quiet.get("static",[]))!=96:
            raise ValueError("Expected 12 direction cases and 48 quiet plus 48 stop replicas")
        if any(row.get("replicas")!=4 or len(row.get("per_replica",[]))!=4 for row in report["scenarios"]):
            raise ValueError("Missing diagnostic replica evidence")
        reports.append(report);quiet_reports.append(quiet)
    if (reports[0]["joint_names"]!=reports[1]["joint_names"] or reports[0]["options"]!=reports[1]["options"]
        or reports[0]["controller"]!=reports[1]["controller"] or reports[0]["overrides"]!=reports[1]["overrides"]
        or reports[0]["reward_weights"]!=reports[1]["reward_weights"]):raise ValueError("Initial/final dynamics, diagnostics or runtime ordering differ")
    directions=[]
    metrics=("planar_error_mps","yaw_error_rad_s","torque_saturation_fraction","computed_torque_abs_max_nm",
             "applied_torque_abs_max_nm","positive_mechanical_power_w","tilt_rms_deg",
             "finite_difference_planar_error_mps","reported_minus_finite_difference_velocity_rms_mps")
    for before,after in zip(reports[0]["scenarios"],reports[1]["scenarios"]):
        if (before["name"],before["command"])!=(after["name"],after["command"]):raise ValueError("Diagnostic command ordering mismatch")
        rows=[]
        for i in range(4):
            a=before["per_replica"][i]["windows"]["post_settle_nonterminal"]
            b=after["per_replica"][i]["windows"]["post_settle_nonterminal"]
            if not a.get("samples") or not b.get("samples"):raise ValueError("Missing steady per-replica samples")
            rows.append({"replica":i,"initial":{k:a[k] for k in metrics},"final":{k:b[k] for k in metrics},
                "planar_error_change_mps":b["planar_error_mps"]-a["planar_error_mps"],
                "yaw_error_change_rad_s":b["yaw_error_rad_s"]-a["yaw_error_rad_s"],
                "initial_joints":a["joints"],"final_joints":b["joints"]})
        directions.append({"name":before["name"],"command":before["command"],"replicas":rows,
            "initial_terminations":before["terminations"],"final_terminations":after["terminations"],
            "initial_truncations":before["truncations"],"final_truncations":after["truncations"]})
    quiet=[]
    for a,b in zip(quiet_reports[0]["static"],quiet_reports[1]["static"]):
        key=lambda row:(row["name"],row["preceding_case"],row["environment_index"],row["replica"])
        if key(a)!=key(b):raise ValueError("Quiet/stop trial ordering differs")
        quiet.append({"name":a["name"],"preceding_case":a["preceding_case"],"environment_index":a["environment_index"],
            "replica":a["replica"],"initial":a,"final":b})
    return {"complete":True,"kind":"50_update_new_action_matched_comparison_not_qualification","stage2_complete":False,
        **identity,"checkpoint_shas":checkpoint_shas,"directions":directions,"quiet_and_stop_replicas":quiet,
        "initial_quiet_all_pass":quiet_reports[0]["all_scenarios_pass"],"final_quiet_all_pass":quiet_reports[1]["all_scenarios_pass"],
        "next_action":"Root review of every direction, joint, torque and quiet/stop replica; no automatic continuation",
        "limits":["This is one scratch seed and 50 updates, not a mature-policy ranking.",
            "Full 154 scenarios, uninterrupted transitions, pathing and Benchmark1 visual smoothness are still required.",
            "0.04/8 controller profile only; do not pool with the separate 0.03 diagnostic profile."]}

def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--accepted-probe",type=Path,required=True)
    parser.add_argument("--accepted-probe-campaign-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--isaaclab",type=Path,default=Path("/home/orionh/IsaacLab"))
    args=parser.parse_args()
    args.source=args.source.resolve();args.accepted_probe=args.accepted_probe.resolve();args.output=args.output.resolve()
    if args.output.exists():parser.error("Fresh output required; no retry or continuation in place")
    if Path(__file__).resolve().is_relative_to(args.source):parser.error("Run this host overlay outside the immutable admitted source")
    identity=accepted_probe(args.source,args.accepted_probe,args.accepted_probe_campaign_sha256)
    # Import only the just-verified frozen helpers; no simulator initialization.
    sys.path.insert(0,str(args.source/"tools"))
    global preflight,resources,live_competitors,verified_source,save
    resource_module=importlib.import_module("launch_length_study_spark")
    training_module=importlib.import_module("launch_length_training_spark")
    preflight=resource_module.preflight;resources=resource_module.resources
    live_competitors=training_module.live_competitors;verified_source=training_module.verified_source;save=training_module.save
    args.coordination_sha256=digest(COORDINATION)
    args.output.mkdir(parents=True)
    for directory in ("inputs","logs","jobs"):(args.output/directory).mkdir()
    signal.signal(signal.SIGTERM,lambda *_:(args.output/"stop.request").touch())
    signal.signal(signal.SIGINT,lambda *_:(args.output/"stop.request").touch())
    report={"status":"preparing","started_unix":time.time(),"stage2_complete":False,
        "source":str(args.source),"source_manifest_sha256":digest(args.source/"campaign_source_hashes.json"),
        "accepted_probe":str(args.accepted_probe),"accepted_probe_campaign_sha256":digest(args.accepted_probe/"campaign.json"),
        "launcher_sha256":digest(__file__),"identity":identity,"maximum_walking_updates":50,"automatic_continuation":False}
    save(args.output/"campaign.json",report)
    try:
        shutil.copytree(args.accepted_probe/"inputs/study",args.output/"inputs/study")
        for source,target in [("flat/admission.json","admission.json"),("probe/calibration.json","calibration.json"),("probe/runner_smoke.json","runner_smoke.json")]:
            shutil.copyfile(args.accepted_probe/source,args.output/"inputs"/target)
        inputs_before=tree_hashes(args.output/"inputs")
        save(args.output/"inputs_before.sha256.json",inputs_before)
        report.update(status="scratch_training",walking_training_started=True)
        save(args.output/"campaign.json",report)
        trained=run_owned(args,"train")
        initialization=read(args.output/"train/initialization.json")
        if (trained.get("iterations")!=50 or trained.get("architecture")!=LINEAGE
            or initialization.get("initialization")!="scratch_zero_actor_mean_head"
            or initialization.get("initial_action_std")!=.005 or initialization.get("optimizer_state_entries")!=0
            or initialization.get("learning_rate")!=1e-4 or initialization.get("entropy_coef")!=0):
            raise ValueError("Training did not follow reviewed scratch/exploration/optimizer settings")
        learned_std=trained.get("learned_std",[])
        if len(learned_std)!=18 or not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in learned_std):
            raise ValueError("Invalid learned exploration distribution")
        checkpoint_shas={phase:checkpoint_contract(args.output/"train/policy"/(phase+".pt"),identity)["checkpoint_sha256"] for phase in ("initial","final")}
        report.update(status="initial_evaluation",checkpoint_shas=checkpoint_shas)
        save(args.output/"campaign.json",report)
        for phase in ("initial","final"):
            check_source(args.source)
            if tree_hashes(args.output/"inputs")!=inputs_before:raise ValueError("Admitted inputs changed during the campaign")
            report["status"]=phase+"_evaluation";save(args.output/"campaign.json",report)
            run_owned(args,phase)
        comparison=compare_evaluations(args.output/"initial",args.output/"final",identity,checkpoint_shas)
        save(args.output/"comparison.json",comparison)
        check_source(args.source)
        if tree_hashes(args.output/"inputs")!=inputs_before:raise ValueError("Admitted input bytes changed")
        report.update(status="completed_needs_review",source_unchanged=True,admitted_inputs_unchanged=True,
            comparison_sha256=digest(args.output/"comparison.json"),
            scope="Exactly 50 scratch walking updates and two matched diagnostic/quiet evaluations; no continuation or qualification")
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc,InterruptedError) else "failed",error=repr(exc))
        raise
    finally:
        report["finished_unix"]=time.time();save(args.output/"campaign.json",report)

if __name__=="__main__":main()
