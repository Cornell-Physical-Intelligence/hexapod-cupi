#!/usr/bin/env python3
"""One exact campaign → one guarded video; optionally bind a reviewed wrapper by hash."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TERMINAL = {"no_video", "video_complete", "video_failed", "launch_uncertain"}


def durable_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def process_identity(pid, proc=Path("/proc")):
    """Linux PID reuse protection; zombies have finished executing."""
    try:
        folder = proc / str(pid)
        first = (folder/"stat").read_text().rsplit(")", 1)[1].split()
        argv = (folder/"cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        second = (folder/"stat").read_text().rsplit(")", 1)[1].split()
        if first[19] != second[19] or second[0] == "Z":
            return None
        return {"pid":pid, "start_ticks":int(second[19]),
                "argv":[v.decode("utf-8") for v in argv],
                "boot_id":(proc/"sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected, actual):
    return actual is not None and expected == actual


def flag(argv, name):
    if argv.count(name) != 1 or argv.index(name)+1 >= len(argv):
        raise ValueError(f"Missing/duplicate process flag: {name}")
    return argv[argv.index(name)+1]


def campaign_entrypoint(binding):
    """A wrapper is accepted only by its exact path and immutable reviewed bytes."""
    wrapper, expected = binding.get("campaign_wrapper"), binding.get("campaign_wrapper_sha256")
    if wrapper is None and expected is None:
        return Path(binding["source"])/"isaaclab/deploy/run-mkii-fourbar-campaign"
    if (not isinstance(wrapper, str) or not isinstance(expected, str)
            or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("Campaign wrapper requires an exact path and SHA-256")
    path = Path(wrapper).resolve(strict=True)
    if str(path) != wrapper or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Reviewed campaign wrapper bytes or path changed")
    return path


def verify_campaign_process(identity, binding):
    if (identity is None or identity["pid"] != binding["campaign_pid"]
            or identity["start_ticks"] != binding["campaign_start_ticks"]):
        raise ValueError("Exact campaign process is absent or its PID was reused")
    argv = identity["argv"]
    if (len(argv)<2 or Path(argv[1]).resolve() != campaign_entrypoint(binding)
            or Path(flag(argv,"--source-dir")).resolve() != Path(binding["source"])
            or flag(argv,"--source-commit") != binding["source_commit"]
            or Path(flag(argv,"--output-root")).resolve() != Path(binding["campaign_file"]).parent.parent):
        raise ValueError("Campaign PID arguments do not match the exact requested campaign/source")


def verify_campaign_header(campaign, binding):
    campaign_entrypoint(binding)
    if (campaign.get("schema") != "hexapod.fourbar_campaign.v1"
            or Path(campaign.get("campaign", "")).resolve() != Path(binding["campaign_file"]).parent
            or Path(campaign.get("source", "")).resolve() != Path(binding["source"])
            or campaign.get("pid") != binding["campaign_pid"]
            or campaign.get("source_commit") != binding["source_commit"]
            or campaign.get("contract",{}).get("sha256") != binding["source_sha256"]):
        raise ValueError("Campaign report identity changed")


def owned_path(path, root, *, name=None):
    result = Path(path).resolve(strict=True)
    if not result.is_relative_to(root.resolve()) or (name and result.name != name):
        raise ValueError("Selected training evidence escapes its own campaign phase")
    return result


def select_full_capture(campaign, campaign_file, common):
    """Only the completed 1000-update full phase can select a checkpoint."""
    if campaign.get("state") != "complete" or campaign.get("pass") is not True or campaign.get("errors") != []:
        raise ValueError("Campaign is not completely successful")
    if campaign.get("separate_process_resume_verified") is not True:
        raise ValueError("Campaign has not verified the separate full-training process")
    root = Path(campaign_file).resolve().parent
    phases = campaign.get("phases", [])
    full = [row for row in phases if row.get("name") == "full"]
    if len(full) != 1:
        raise ValueError("Expected exactly one full-training phase")
    row = full[0]
    request = row.get("requested", {})
    if (row.get("state") != "passed" or row.get("launcher_exit_code") != 0
            or request.get("name") != "full" or request.get("mode") != "train"
            or type(request.get("iterations")) is not int or request["iterations"] != 1000
            or type(request.get("num_envs")) is not int or request["num_envs"] != 512):
        raise ValueError("The 512-environment, 1000-iteration full phase has not passed")
    report = owned_path(row.get("report", ""), root/"full", name="report.json")
    if report.relative_to(root/"full").parts != (report.parent.name, "report.json"):
        raise ValueError("Unexpected full-training report nesting")
    if common.digest(report) != row.get("report_sha256"):
        raise ValueError("Full-training primary report bytes changed")
    checkpoint = owned_path(campaign.get("final_checkpoint", ""), root/"full", name="checkpoint.pt")
    if checkpoint != report.with_name("checkpoint.pt"):
        raise ValueError("Final checkpoint does not belong to the full-training report")
    entry = campaign.get("admission", {})
    admission = owned_path(entry.get("path", ""), root, name="admission.json")
    if admission != root/"admission.json" or entry.get("pass") is not True or common.digest(admission) != entry.get("sha256"):
        raise ValueError("Own-campaign admission identity failed")
    training = common.read_json(report)
    if (training.get("num_envs") != 512 or training.get("iterations_requested") != 1000
            or training.get("iterations_completed") != 1000 or training.get("paused") is not False):
        raise ValueError("Full training stopped short or paused")
    supervisor = common.read_json(report.with_name("supervisor.json"))
    if (supervisor.get("supervisor_exit_code") != 0 or supervisor.get("execution") != "finished"
            or supervisor.get("validator_report_status") != "passed" or supervisor.get("cleanup") != "removed_exact_id"
            or supervisor.get("source_identity_unchanged_at_finish") is not True
            or supervisor.get("contract") != campaign["contract"]):
        raise ValueError("Full-training supervisor did not verify completion and cleanup")
    return {"checkpoint":str(checkpoint), "admission":str(admission), "training_report":str(report)}


def capture_launch_command(command, binding):
    """Only the actual capture job owns the optional shared lock; never wait."""
    lock = binding.get("capture_shared_lock")
    if lock is None:
        return list(command)
    if lock != "/opt/wx/gpu.lock":
        raise ValueError("Unreviewed capture shared lock")
    return ["/usr/bin/flock", "--nonblock", "--no-fork", lock, *command]


def reserve_launch(state, save, argv):
    """Fsync before Popen: a restart never retries an ambiguous launch window."""
    if state.get("state") != "waiting" or "capture_argv" in state:
        raise ValueError("Capture was already reserved/launched; refusing a duplicate")
    state.update(state="launch_reserved", capture_argv=argv, launch_reserved_utc=time.time())
    save()


def require_unused_capture_output(output):
    """A failed child must never inherit a previous capture's successful report."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Capture output already exists; refusing to reuse earlier evidence")


def verify_completed_capture(folder, state, common):
    """Bind final artifacts to this dispatch, including the external capture code."""
    binding = state["binding"]
    campaign_entrypoint(binding)
    tools = Path(binding["capture_tools"])
    if common.tool_identity(tools) != binding["capture_tools_sha256"]:
        raise ValueError("Capture tools changed after follower launch")
    reports = list((Path(folder)/"capture_outputs").glob("hexapod-policy-capture-*/supervisor.json"))
    if len(reports) != 1:
        raise ValueError("Capture exited without exactly one final supervisor report")
    supervisor = common.read_json(reports[0])
    selected = state["selected_inputs"]
    verified = common.verify_inputs(Path(binding["source"]), selected["checkpoint"],
                                    selected["admission"], selected["training_report"])
    if (verified["input_sha256"] != state["input_sha256"]
            or verified["contract"]["sha256"] != binding["source_sha256"]
            or supervisor.get("pass") is not True or supervisor.get("supervisor_exit_code") != 0
            or supervisor.get("execution") != "finished" or supervisor.get("cleanup") != "removed_exact_id"
            or supervisor.get("decoded_video_verified") is not True
            or supervisor.get("source_and_inputs_unchanged") is not True
            or supervisor.get("contract") != verified["contract"]
            or supervisor.get("capture_tools_sha256") != binding["capture_tools_sha256"]
            or supervisor.get("input_sha256") != state["input_sha256"]):
        raise ValueError("Guarded capture did not complete with the exact bound source, inputs and tools")
    result = common.validate_capture_report(reports[0].with_name("report.json"), verified, 750)
    if result.get("capture_tools_sha256") != binding["capture_tools_sha256"]:
        raise ValueError("Final recorder report has different capture tools")
    return reports[0]


def forward_owned_signal(identity, signum):
    if not same_process(identity, process_identity(identity["pid"])):
        return False
    # pidfd binds the signal to this process, including across the final identity check.
    try:
        fd = os.pidfd_open(identity["pid"])
    except ProcessLookupError:
        return False
    try:
        if not same_process(identity, process_identity(identity["pid"])):
            return False
        try:
            signal.pidfd_send_signal(fd, signum)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(fd)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("campaign", "source-dir", "capture-tools", "state-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--campaign-pid", type=int, required=True)
    p.add_argument("--campaign-start-ticks", type=int, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--campaign-wrapper", type=Path)
    p.add_argument("--campaign-wrapper-sha256")
    p.add_argument("--capture-shared-lock", choices=("/opt/wx/gpu.lock",))
    p.add_argument("--wait-seconds", type=int, default=64800)
    args = p.parse_args(argv)
    if (args.campaign_wrapper is None) != (args.campaign_wrapper_sha256 is None):
        p.error("Reviewed wrapper path and SHA-256 must be supplied together")
    if not 1 <= args.wait_seconds <= 86400 or min(args.campaign_pid,args.campaign_start_ticks)<=0:
        p.error("Wait is bounded to 1–86400 seconds; PID/start ticks must be positive")
    campaign_file, source, tools = [path.resolve(strict=True) for path in (args.campaign,args.source_dir,args.capture_tools)]
    folder = args.state_dir.resolve()
    if folder.is_relative_to(source) or source.is_relative_to(folder):
        p.error("Follower state must be outside frozen source")
    sys.path.insert(0,str(tools))
    spec = importlib.util.spec_from_file_location("capture_common",tools/"capture_common.py")
    common = importlib.util.module_from_spec(spec)
    sys.modules["capture_common"] = common
    spec.loader.exec_module(common)
    api = common.source_api(source)
    binding = {"campaign_file":str(campaign_file), "campaign_pid":args.campaign_pid,
        "campaign_start_ticks":args.campaign_start_ticks, "source":str(source),
        "source_sha256":args.source_sha256, "source_commit":args.source_commit,
        "capture_tools":str(tools), "capture_tools_sha256":common.tool_identity(tools),
        "follower_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.campaign_wrapper is not None:
        binding.update(campaign_wrapper=str(args.campaign_wrapper.resolve(strict=True)),
                       campaign_wrapper_sha256=args.campaign_wrapper_sha256)
        campaign_entrypoint(binding)
    if args.capture_shared_lock is not None:
        binding["capture_shared_lock"] = args.capture_shared_lock
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/"follower.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = folder/"state.json"
        state = common.read_json(path) if path.exists() else {"schema":"hexapod.campaign_capture_followup.v2",
            "binding":binding,"state":"waiting","started_utc":time.time(),"deadline_utc":time.time()+args.wait_seconds}
        if state.get("binding") != binding:
            raise ValueError("Existing follower state is bound to a different campaign/tools/source")
        def save():
            state["updated_utc"] = time.time()
            durable_json(path,state)
        if state["state"] in TERMINAL:
            return state.get("exit_code",1)
        if state["state"] == "launch_reserved":
            state.update(state="launch_uncertain",exit_code=1,reason="Launch reservation exists without a confirmed PID; do not relaunch")
            save()
            return 1
        interrupted = []
        def stop(signum,_frame):
            if not interrupted:
                interrupted.append(signum)
        old = {s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        child = None
        try:
            campaign = common.read_json(campaign_file)
            verify_campaign_header(campaign,binding)
            if api.identity(source) != campaign["contract"]:
                raise ValueError("Frozen source differs from the bound campaign identity")
            if state["state"] == "waiting":
                current = process_identity(args.campaign_pid)
                if campaign.get("state") not in ("complete","failed","paused"):
                    verify_campaign_process(current,binding)
                if current is not None:
                    verify_campaign_process(current,binding)
                    if "campaign_process" in state and state["campaign_process"] != current:
                        raise ValueError("Campaign process identity changed since follower start")
                    state["campaign_process"] = current
                save()
            next_source_check = 0.
            while state["state"] == "waiting":
                if interrupted:
                    state.update(state="no_video",exit_code=1,reason="Follower interrupted before capture")
                    save(); return 1
                campaign = common.read_json(campaign_file)
                verify_campaign_header(campaign,binding)
                state["campaign_state"],state["campaign_phase"] = campaign.get("state"),campaign.get("active_phase")
                if campaign.get("state") in ("failed","paused"):
                    state.update(state="no_video",exit_code=0,reason="Campaign "+campaign["state"])
                    save(); return 0
                if time.time() >= state["deadline_utc"]:
                    state.update(state="no_video",exit_code=1,reason="Bounded campaign wait expired")
                    save(); return 1
                if campaign.get("state") == "complete":
                    selected = select_full_capture(campaign,campaign_file,common)
                    verified = common.verify_inputs(source,selected["checkpoint"],selected["admission"],selected["training_report"])
                    if verified["contract"] != campaign["contract"] or common.tool_identity(tools) != binding["capture_tools_sha256"]:
                        raise ValueError("Source/capture identity changed before launch")
                    state["selected_inputs"] = selected
                    state["campaign_complete_sha256"] = common.digest(campaign_file)
                    state["input_sha256"] = verified["input_sha256"]
                    require_unused_capture_output(folder/"capture_outputs")
                    cmd = [sys.executable,str(tools/"capture_policy.py"),"--source-dir",str(source),
                        "--checkpoint",selected["checkpoint"],"--admission",selected["admission"],
                        "--training-report",selected["training_report"],"--output-root",str(folder/"capture_outputs"),
                        "--seconds","15","--width","1280","--height","720","--timeout-seconds","1800"]
                    campaign_entrypoint(binding)
                    spawn_cmd = capture_launch_command(cmd,binding)
                    reserve_launch(state,save,spawn_cmd)
                    with (folder/"capture_launcher.log").open("x") as log:
                        child = subprocess.Popen(spawn_cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            start_new_session=True,close_fds=True,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})
                    # flock --no-fork exec keeps the PID and held lock; its argv
                    # becomes cmd after successful admission. Bind that exact Python
                    # process, never the transient flock invocation or a descendant.
                    owned = None
                    for _ in range(100):
                        observed = process_identity(child.pid)
                        if observed is not None and observed["argv"] == cmd:
                            owned = observed; break
                        if child.poll() is not None:
                            break
                        time.sleep(.05)
                    if owned is None:
                        state.update(state="launch_uncertain",capture_pid=child.pid,exit_code=1,
                            reason="Capture started but exact post-exec process identity was not recorded; never retry")
                        save(); return 1
                    state.update(state="capture_running",capture_process=owned,capture_pid=owned["pid"],
                                 capture_deadline_utc=time.time()+2160)
                    save()
                    break
                if not same_process(state.get("campaign_process"),process_identity(args.campaign_pid)):
                    raise ValueError("Campaign process disappeared without a terminal report")
                if time.monotonic() >= next_source_check:
                    if api.identity(source) != campaign["contract"]:
                        raise ValueError("Frozen source changed while awaiting full training")
                    next_source_check=time.monotonic()+60
                save()
                time.sleep(min(10,max(0,state["deadline_utc"]-time.time())))
            if state["state"] != "capture_running":
                raise ValueError("Unknown durable follower state")
            owned = state["capture_process"]
            while same_process(owned,process_identity(owned["pid"])):
                if interrupted or time.time() >= state["capture_deadline_utc"]:
                    state["signal_forwarded_to_owned_capture"] = forward_owned_signal(owned,signal.SIGTERM)
                    state.update(state="video_failed",exit_code=1,reason="Follower interrupted or capture deadline expired")
                    save(); return 1
                if child is not None and child.poll() is not None:
                    break
                time.sleep(2)
            if child is not None and child.wait(timeout=5) != 0:
                raise ValueError("Capture supervisor subprocess exited unsuccessfully")
            supervisor_path = verify_completed_capture(folder,state,common)
            state.update(state="video_complete",exit_code=0,supervisor=str(supervisor_path),
                supervisor_sha256=common.digest(supervisor_path),video=str(supervisor_path.with_name("policy.mp4")),
                video_sha256=common.digest(supervisor_path.with_name("policy.mp4")))
            save(); return 0
        except BaseException as error:
            state.update(state="video_failed" if "capture_argv" in state else "no_video",exit_code=1,
                         reason=f"{type(error).__name__}: {error}")
            save()
            return 1
        finally:
            for signum,handler in old.items():
                signal.signal(signum,handler)


if __name__ == "__main__":
    raise SystemExit(main())
