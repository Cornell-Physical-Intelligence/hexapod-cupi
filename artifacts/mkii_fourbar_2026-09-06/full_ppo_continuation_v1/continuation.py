#!/usr/bin/env python3
"""Bounded, exact-checkpoint continuation of one slow full PPO campaign.

External orchestration only. No source edits, gate changes or admission reuse
across functional identities. Preparation does not dispatch this coordinator.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

SOURCE_SHA = "c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc"
SOURCE_COMMIT = "c2af43ca0f384a4c2c7ab8f1d627f309dc78a683"
MANIFEST_SHA = "8d6ad0b053e2b2a73e6610a51871443efdeb1234d7c44165dbf27a67498e66fb"
HOST_SHA = "60dc7803aa962c2a1152fa00bb20ccafe5348890509a903d65bfb850fb42bfbc"
TARGET_UPDATES, NUM_ENVS, MAX_SEGMENTS = 1000, 512, 3
TOTAL_SECONDS, JOB_SECONDS, CAPTURE_SECONDS = 64800, 21600, 2160
FLOCK = ["/usr/bin/flock", "--nonblock", "--no-fork", "/opt/wx/gpu.lock"]
TERMINAL = {"original_completed", "original_failed", "video_complete", "failed", "launch_uncertain"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boot_seconds():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def verify_recorded_files(config):
    """Stdlib-only bootstrap: reject changed code before executing any of it."""
    source = Path(config['source_dir']).resolve(strict=True)
    if (config['source_sha256'], config['source_commit'], config['source_manifest_sha256']) != (SOURCE_SHA, SOURCE_COMMIT, MANIFEST_SHA):
        raise ValueError('Coordinator requires its exact reviewed physical source')
    targets = [(source, source / config['source_manifest'], MANIFEST_SHA)]
    for key, digest_key in (('follower_tools', 'follower_manifest_sha256'), ('capture_tools', 'capture_manifest_sha256')):
        root = Path(config[key]).resolve(strict=True)
        targets.append((root, root / 'SHA256SUMS', config[digest_key]))
    for root, manifest, expected in targets:
        if sha(manifest) != expected:
            raise ValueError('Bootstrap source/tool manifest changed')
        for line in manifest.read_text().splitlines():
            if not line or line.startswith('#'):
                continue
            digest, name = line.split(maxsplit=1)
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(root) or sha(path) != digest:
                raise ValueError('Bootstrap source/tool bytes changed: ' + name)
    if sha(source / 'isaaclab/deploy/run-mkii-fourbar') != HOST_SHA:
        raise ValueError('Bootstrap host source changed')
    wrapper = Path(config['campaign_wrapper']).resolve(strict=True)
    if str(wrapper) != config['campaign_wrapper'] or sha(wrapper) != config['campaign_wrapper_sha256']:
        raise ValueError('Bootstrap campaign wrapper changed')


def positive_int(value, label, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"Invalid integer {label}")
    return value


def finite_positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid positive {label}")
    return float(value)


def progress_sample(value, *, start_iteration, requested):
    count = positive_int(value.get("iterations_completed"), "progress count")
    elapsed = finite_positive(value.get("elapsed_seconds"), "progress elapsed")
    if count > requested or type(value.get("last_iteration")) is not int or value["last_iteration"] != start_iteration + count - 1:
        raise ValueError("Progress iteration identity differs from this segment")
    finite_positive(value.get("collect_seconds"), "collection duration")
    finite_positive(value.get("learn_seconds"), "learning duration")
    return {"completed": count, "elapsed": elapsed}


def update_timing(history, sample):
    """Use completed-update wall time, including callbacks/checkpoint overhead."""
    if history:
        last = history[-1]
        if sample == last:
            return history
        if sample["completed"] <= last["completed"] or sample["elapsed"] <= last["elapsed"]:
            raise ValueError("Training progress regressed or changed within one update")
    return [*history[-7:], sample]


def measured_iteration_seconds(history):
    if len(history) < 2 or history[-1]["completed"] - history[0]["completed"] < 3:
        return None
    rates = [(b["elapsed"] - a["elapsed"]) / (b["completed"] - a["completed"])
             for a, b in zip(history, history[1:])]
    return max(*rates, history[-1]["elapsed"] / history[-1]["completed"])


def pause_decision(history, requested, remaining_seconds):
    """Never pause solely for a guessed rate or before the bounded safety window."""
    rate = measured_iteration_seconds(history)
    if rate is None:
        return {"request_pause": False, "reason": "insufficient completed-update timing"}
    completed = history[-1]["completed"]
    eta = (requested - completed) * rate + (100 / 24) * rate
    margin = 300 + 2 * rate
    return {"request_pause": completed < requested and eta > remaining_seconds and remaining_seconds <= margin,
            "measured_iteration_seconds": rate, "remaining_job_seconds": remaining_seconds,
            "completion_eta_seconds": eta, "checkpoint_margin_seconds": margin}


def choose_chunk(remaining_updates, rate, remaining_total_seconds):
    positive_int(remaining_updates, "remaining updates")
    rate = finite_positive(rate, "measured iteration duration")
    # Reserve final capture/cleanup and startup/checkpoint/100-step inference.
    available = min(JOB_SECONDS, remaining_total_seconds - CAPTURE_SECONDS)
    overhead = 600 + (100 / 24) * rate + 2 * rate
    capacity = math.floor((available - overhead) / (rate * 1.15))
    if capacity < 1:
        raise ValueError("Remaining bounded time cannot admit another measured PPO update")
    return min(remaining_updates, capacity)


def segment_argv(config, output_root, admission, checkpoint, iterations):
    if not 1 <= positive_int(iterations, "chunk iterations") <= TARGET_UPDATES:
        raise ValueError("Invalid bounded segment size")
    source = Path(config["source_dir"])
    command = ["/usr/bin/python3", str(source / "isaaclab/deploy/run-mkii-fourbar"), "train",
               "--source-dir", str(source), "--source-commit", SOURCE_COMMIT,
               "--output-root", str(output_root), "--num-envs", str(NUM_ENVS),
               "--iterations", str(iterations), "--admission", str(admission),
               "--checkpoint", str(checkpoint), "--asset-model", "mkii_fourbar_v5",
               "--environment-layout", "coincident_flat_origin_v1", "--timeout-seconds", str(JOB_SECONDS)]
    return [*FLOCK, *command], command


def request_checkpoint(output, decision, token, *, verify_owned):
    """The ownership callback must recheck PID/start ticks and exact live container."""
    if decision.get("request_pause") is not True:
        raise ValueError("No measured deadline reason for a checkpoint request")
    verify_owned()
    path = Path(output) / "stop_requested"
    value = {"schema": "hexapod.owned_deadline_checkpoint.v1", "owner_token": token,
             "reason": "Completed-update ETA cannot fit the unchanged host deadline", "decision": decision}
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return {"path": str(path), "sha256": sha(path), "owner_token": token}


def verify_segment(report_path, *, requested, parent, contract, admission, source, common, host):
    """Validate an honest partial/full segment without changing its report fields."""
    path = Path(report_path).resolve(strict=True)
    report = common.read_json(path)
    supervisor_path = path.with_name("supervisor.json")
    supervisor = common.read_json(supervisor_path)
    if (supervisor.get("supervisor_exit_code") != 0 or supervisor.get("execution") != "finished"
            or supervisor.get("validator_report_status") != "passed" or supervisor.get("cleanup") != "removed_exact_id"
            or supervisor.get("source_identity_unchanged_at_finish") is not True or supervisor.get("contract") != contract
            or supervisor.get('source') != str(Path(source).resolve()) or supervisor.get('source_commit') != SOURCE_COMMIT):
        raise ValueError("Segment lacks successful source-bound supervisor and exact cleanup")
    host.validate_written_report(path, args=argparse.Namespace(mode="train", iterations=requested,
        num_envs=NUM_ENVS, asset_model="mkii_fourbar_v5", environment_layout="coincident_flat_origin_v1"), contract=contract)
    checkpoint = path.with_name("checkpoint.pt")
    sidecar = host.require_checkpoint(checkpoint, contract)
    count = positive_int(report.get("iterations_completed"), "completed segment updates")
    start = positive_int(report.get("start_iteration"), "start iteration", zero=True)
    nxt = positive_int(report.get("next_iteration"), "next iteration")
    if (report.get("task_id") != common.TASK_ID or report.get("contract") != contract
            or report.get("mode") != "provisional_physical_fourbar_ppo" or report.get("pass") is not True
            or report.get("errors") != [] or report.get("hardware_admission") is not False
            or type(report.get("paused")) is not bool or type(report.get("num_envs")) is not int or report["num_envs"] != NUM_ENVS
            or type(report.get("iterations_requested")) is not int or report["iterations_requested"] != requested
            or count > requested or (not report["paused"] and count != requested)
            or any(report.get(key) is not True for key in ("checkpoint_verified", "checkpoint_roundtrip_pass", "policy_changed"))
            or start != parent["next_iteration"] or nxt != start + count or nxt != sidecar["next_iteration"]
            or report.get("resumed_from_sha256") != parent["checkpoint_sha256"]
            or report.get("checkpoint_sha256") != sidecar["checkpoint_sha256"]
            or report.get("policy_before_sha256") != parent["policy_after_sha256"]
            or report.get("algorithm_before_sha256") != parent["algorithm_after_sha256"]
            or common.canonical(report.get("runtime_manifest")) != common.canonical(admission.get("runtime_manifest"))):
        raise ValueError("Segment update/checkpoint/policy/optimizer/normalizer handoff failed")
    for key in ("policy_after_sha256", "algorithm_after_sha256"):
        value = report.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Segment lacks a valid trained-state digest")
    if not report["paused"]:
        probe = report.get("inference_probe", {})
        if probe.get("finite") is not True or type(probe.get("steps")) is not int or probe["steps"] != 100:
            raise ValueError("Completed segment lacks finite checkpoint inference")
    return {"report": str(path), "report_sha256": sha(path), "supervisor": str(supervisor_path),
            "supervisor_sha256": sha(supervisor_path), "checkpoint": str(checkpoint),
            "checkpoint_sha256": sidecar["checkpoint_sha256"], "sidecar_sha256": sha(str(checkpoint) + ".json"),
            "requested": requested, "completed": count, "paused": report["paused"],
            "start_iteration": start, "next_iteration": nxt,
            "policy_after_sha256": report["policy_after_sha256"], "algorithm_after_sha256": report["algorithm_after_sha256"]}


def chain_totals(segments, scratch_next):
    if not 1 <= len(segments) <= MAX_SEGMENTS:
        raise ValueError("Continuation chain exceeds three total full-training segments")
    expected, total = scratch_next, 0
    for segment in segments:
        count = positive_int(segment["completed"], "chain completed updates")
        if segment["start_iteration"] != expected or segment["next_iteration"] != expected + count:
            raise ValueError("Continuation chain has a gap, overlap or repeated iteration")
        expected += count; total += count
    if total > TARGET_UPDATES:
        raise ValueError("Continuation chain exceeds 1000 full-training updates")
    return total


class Coordinator:
    def __init__(self, config, folder):
        self.config, self.folder = config, Path(folder).resolve()
        self.source = Path(config["source_dir"]).resolve(strict=True)
        if self.folder.is_relative_to(self.source) or self.source.is_relative_to(self.folder):
            raise ValueError("Coordinator state must be outside frozen source")
        if (config["source_sha256"], config["source_commit"], config["source_manifest_sha256"]) != (SOURCE_SHA, SOURCE_COMMIT, MANIFEST_SHA):
            raise ValueError("Coordinator is pinned to the reviewed c2af43c physical source")
        verify_recorded_files(config)
        sys.path[:0] = [config["follower_tools"], config["capture_tools"]]
        import follow_campaign
        import capture_common
        self.follower, self.common = follow_campaign, capture_common
        self.api = self.common.source_api(self.source)
        self.host = self.common.load_module(self.source / "isaaclab/deploy/run-mkii-fourbar", "continuation_frozen_host")
        self.contract = self.api.identity(self.source)
        self.binding = {"campaign_file": config["campaign"], "campaign_pid": config["campaign_pid"],
            "campaign_start_ticks": config["campaign_start_ticks"], "source": str(self.source),
            "source_sha256": SOURCE_SHA, "source_commit": SOURCE_COMMIT,
            "campaign_wrapper": config["campaign_wrapper"], "campaign_wrapper_sha256": config["campaign_wrapper_sha256"],
            "capture_tools": config["capture_tools"], "capture_tools_sha256": self.common.tool_identity(config["capture_tools"]),
            "capture_shared_lock": "/opt/wx/gpu.lock"}
        self.state, self.child, self.interrupted = None, None, False
        self.verify_source()

    def verify_source(self):
        c = self.config
        verify_recorded_files(c)
        if self.api.identity(self.source)["sha256"] != SOURCE_SHA or self.contract["sha256"] != SOURCE_SHA:
            raise ValueError("Frozen functional source changed")
        manifest = self.source / c["source_manifest"]
        if sha(manifest) != MANIFEST_SHA or sha(self.source / "isaaclab/deploy/run-mkii-fourbar") != HOST_SHA:
            raise ValueError("Source/host deployment manifest changed")
        for line in manifest.read_text().splitlines():
            if line and not line.startswith("#"):
                expected, name = line.split(maxsplit=1)
                if sha(self.source / name) != expected:
                    raise ValueError("Frozen deployment file changed: " + name)
        for key, manifest_key in (("follower_tools", "follower_manifest_sha256"), ("capture_tools", "capture_manifest_sha256")):
            root = Path(c[key]); recorded = root / "SHA256SUMS"
            if sha(recorded) != c[manifest_key]:
                raise ValueError("Bound external tool manifest changed")
            for line in recorded.read_text().splitlines():
                expected, name = line.split(maxsplit=1)
                if sha(root / name) != expected:
                    raise ValueError("Bound external tool bytes changed")
        self.follower.campaign_entrypoint(self.binding)
        from hexapod_core.fourbar_v1 import PHYSICS_DT_S, DECIMATION, POLICY_DT_S
        if (PHYSICS_DT_S, DECIMATION, POLICY_DT_S) != (.000625, 32, .02):
            raise ValueError("Unexpected actual imported physics/policy timing")

    def save(self):
        self.state["updated_utc"] = time.time()
        self.follower.durable_json(self.folder / "state.json", self.state)

    def check(self):
        if boot_id() != self.state['boot_id']:
            raise ValueError('Coordinator cannot resume across a different Linux boot')
        if self.interrupted or boot_seconds() >= self.state["deadline_boottime"]:
            raise ValueError("Coordinator interrupted or total 18-hour deadline expired")
        if self.state.get('admission'):
            self.pin_admission(self.state['admission']['path'], self.state['admission']['sha256'])
        self.host.require_coordination_none(self.host.read_coordination_control())
        if boot_seconds() >= self.state.get("next_source_check_boottime", 0):
            self.verify_source()
            self.state["next_source_check_boottime"] = boot_seconds() + 60

    def pin_admission(self, path, expected_sha):
        record = {'path': str(Path(path).resolve(strict=True)), 'sha256': expected_sha}
        if sha(record['path']) != expected_sha:
            raise ValueError('Original physical admission bytes changed')
        current = self.state.get('admission')
        if current is not None and current != record:
            raise ValueError('Original physical admission binding changed')
        if current is None:
            self.state['admission'] = record
            self.save()

    def campaign(self):
        value = self.common.read_json(self.config["campaign"])
        self.follower.verify_campaign_header(value, self.binding)
        return value

    def exact_process(self, expected):
        return self.follower.same_process(expected, self.follower.process_identity(expected["pid"]))

    def capture_original_job(self, campaign):
        rows = [p for p in campaign.get("phases", []) if p.get("name") == "full"]
        if not rows:
            return None
        if len(rows) != 1 or rows[0].get("requested", {}).get("iterations") != TARGET_UPDATES or rows[0]["requested"].get("num_envs") != NUM_ENVS:
            raise ValueError("Original campaign full phase differs from 512 by 1000")
        row = rows[0]
        if row.get("state") != "running":
            return None
        command = row["argv"]
        if command[:4] != FLOCK:
            raise ValueError("Original job lacks the reviewed per-job shared flock")
        process = self.follower.process_identity(row.get("supervisor_pid", -1))
        if process is None:
            return None
        if process["argv"] != command[4:] or process["boot_id"] != self.config["campaign_boot_id"]:
            raise ValueError("Original full supervisor PID/argv/boot identity mismatch")
        if (self.follower.flag(command, "--source-dir") != str(self.source)
                or self.follower.flag(command, "--source-commit") != SOURCE_COMMIT
                or self.follower.flag(command, "--timeout-seconds") != str(JOB_SECONDS)):
            raise ValueError("Original full job source/deadline mismatch")
        root = Path(self.config["campaign"]).parent / "full"
        if Path(self.follower.flag(command, "--output-root")).resolve() != root:
            raise ValueError("Original full output escaped campaign")
        return {"process": process, "output_root": str(root), "requested": TARGET_UPDATES,
                "origin": "original_campaign", "history": [], "pause_request": None}

    def discover_output(self, job):
        paths = list(Path(job["output_root"]).glob("hexapod-fourbar-train-*/supervisor.json"))
        if len(paths) > 1:
            raise ValueError("Ambiguous segment supervisor output")
        if not paths:
            return None
        path = paths[0].resolve()
        if not path.is_relative_to(Path(job["output_root"]).resolve()):
            raise ValueError("Segment output escaped owned root")
        value = self.common.read_json(path)
        if (value.get("source") != str(self.source) or value.get("contract") != self.contract
                or value.get("source_commit") != SOURCE_COMMIT):
            raise ValueError("Segment supervisor source identity differs")
        if job.get("supervisor") not in (None, str(path)):
            raise ValueError("Segment output changed")
        job["supervisor"], job["output"] = str(path), str(path.parent)
        if "container" in job and any(value.get(key) != job["container"][target]
                for key, target in (("container_id", "id"), ("container_name", "name"), ("owner", "owner"))):
            raise ValueError("Previously bound supervisor/container ownership changed")
        if value.get("container_id") and "container" not in job:
            owner = value.get("owner")
            if not isinstance(owner, str) or len(owner) != 32 or any(c not in "0123456789abcdef" for c in owner):
                raise ValueError("Supervisor lacks its exact recorded container owner")
            observed = self.host.inspect_container(value["container_id"])
            if observed is not None:
                if observed["id"] != value["container_id"] or not self.host.check_identity(observed, value["container_name"], owner):
                    raise ValueError("Cannot bind the exact segment container")
                job["container"] = {"id": observed["id"], "name": value["container_name"], "owner": owner}
        return value

    def verify_owned_live_job(self, job):
        self.check()
        self.discover_output(job)
        if not self.exact_process(job["process"]):
            raise ValueError("Checkpoint request refused: supervisor PID changed/exited")
        own = job.get("container")
        current = self.host.inspect_container(own["id"]) if own else None
        if not own or not self.host.check_identity(current, own["name"], own["owner"]) or not current["running"]:
            raise ValueError("Checkpoint request refused: exact live container is absent")

    def monitor_job(self, job, parent):
        while True:
            self.check()
            supervisor = self.discover_output(job)
            alive = self.exact_process(job["process"])
            if not alive:
                if self.child is not None:
                    if self.child.wait(timeout=5) != 0:
                        raise ValueError("Owned segment host exited unsuccessfully")
                    self.child = None
                if not supervisor or supervisor.get("cleanup") != "removed_exact_id":
                    raise ValueError("Segment exited without verified owned cleanup")
                return Path(job["output"]) / "report.json"
            if supervisor and supervisor.get("execution") == "running":
                progress = Path(job["output"]) / "progress.json"
                if progress.is_file():
                    sample = progress_sample(self.common.read_json(progress), start_iteration=parent["next_iteration"], requested=job["requested"])
                    job["history"] = update_timing(job["history"], sample)
                    # CLOCK_BOOTTIME/start ticks conservatively starts the budget
                    # before the host creates its actual monotonic deadline.
                    elapsed = boot_seconds() - job["process"]["start_ticks"] / os.sysconf("SC_CLK_TCK")
                    decision = pause_decision(job["history"], job["requested"], JOB_SECONDS - elapsed)
                    job["latest_decision"] = decision
                    if decision["request_pause"] and not job["pause_request"]:
                        job["pause_request"] = request_checkpoint(job["output"], decision, self.state["owner_token"],
                            verify_owned=lambda: self.verify_owned_live_job(job))
                    self.save()
            time.sleep(10)

    def require_owned_pause(self, segment, job):
        if not segment["paused"]:
            return
        request = job["pause_request"]
        if not request or sha(request["path"]) != request["sha256"]:
            raise ValueError("Refusing continuation of a pause not requested by this coordinator")

    def original_parent_and_admission(self, campaign):
        root = Path(self.config["campaign"]).parent
        entry = campaign.get("admission", {})
        admission = Path(entry.get("path", "")).resolve(strict=True)
        if admission != root / "admission.json" or entry.get("pass") is not True or sha(admission) != entry.get("sha256"):
            raise ValueError("Original qualified admission is absent or changed")
        self.pin_admission(admission, entry['sha256'])
        admitted = self.api.require_admission(admission, self.contract)
        rows = [p for p in campaign["phases"] if p.get("name") == "scratch"]
        if len(rows) != 1 or rows[0].get("state") != "passed":
            raise ValueError("Original three-update scratch phase is not verified")
        row = rows[0]; path = Path(row["report"]).resolve(strict=True)
        if not path.is_relative_to(root / "scratch") or sha(path) != row["report_sha256"]:
            raise ValueError("Original scratch report identity mismatch")
        report = self.common.read_json(path)
        if (report.get("iterations_requested") != 3 or report.get("iterations_completed") != 3
                or report.get("start_iteration") != 0 or report.get("next_iteration") != 3):
            raise ValueError("Only the verified three-update scratch checkpoint can precede the full chain")
        self.common.verify_inputs(self.source, path.with_name("checkpoint.pt"), admission, path)
        return report, admitted, admission

    def wait_previous_follower(self):
        deadline = min(self.state["deadline_boottime"], boot_seconds() + 120)
        while boot_seconds() < deadline:
            self.check()
            state = self.common.read_json(Path(self.config["state_dir"]) / "state.json")
            binding = state["binding"]
            if any(binding.get(key) != self.binding[key] for key in ("campaign_file", "source_sha256", "campaign_pid", "campaign_start_ticks")):
                raise ValueError("Original video follower binding changed")
            if "capture_argv" in state:
                raise ValueError("Original follower already reserved capture; never duplicate it")
            if state.get("state") == "no_video" and not self.exact_process(self.config["campaign_follower_process"]):
                self.state["previous_follower_exit"] = {"state": state, "verified_utc": time.time()}
                self.save(); return
            time.sleep(2)
        raise ValueError("Original follower did not reach no_video and exit within the bounded wait")

    def launch(self, spawn, expected, *, kind, output_root, requested=None):
        if self.state.get("launch_reserved"):
            raise ValueError("Unresolved launch reservation; never dispatch twice")
        if Path(output_root).exists() or Path(output_root).is_symlink():
            raise ValueError("Refusing an existing output root")
        self.verify_source(); self.check()
        reservation = {"kind": kind, "argv": spawn, "expected_post_exec_argv": expected,
                       "output_root": str(output_root), "requested": requested, "reserved_utc": time.time()}
        self.state["launch_reserved"] = reservation; self.save()
        log_path = self.folder / f"{kind}_{len(self.state['segments']) + 1}.log"
        with log_path.open("x") as log:
            self.child = subprocess.Popen(spawn, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.state['launch_reserved']['spawned_pid'] = self.child.pid
        self.save()
        process = None
        for _ in range(100):
            observed = self.follower.process_identity(self.child.pid)
            if observed and observed["argv"] == expected:
                process = observed; break
            if self.child.poll() is not None:
                break
            time.sleep(.05)
        if not process:
            self.state["state"] = "launch_uncertain"; self.save()
            raise ValueError("Child launch/PID confirmation uncertain; automatic retry forbidden")
        job = {"process": process, "output_root": str(output_root), "requested": requested,
               "origin": "coordinator", "history": [], "pause_request": None, "kind": kind}
        self.state["active_job"] = job
        self.state["launch_history"].append({**reservation, "process": process})
        self.state["launch_reserved"] = None; self.save()
        return job

    def accept_segment(self, path, job, parent, admitted, admission_path):
        self.pin_admission(admission_path, self.state['admission']['sha256'])
        record = verify_segment(path, requested=job["requested"], parent=parent, contract=self.contract,
            admission=admitted, source=self.source, common=self.common, host=self.host)
        self.require_owned_pause(record, job)
        record["owned_job"] = job
        self.state["segments"].append(record)
        self.state["active_job"] = None
        self.state["completed_full_updates"] = chain_totals(self.state["segments"], 3)
        self.save()
        return record

    def recheck_chain(self, scratch, admitted):
        parent = scratch
        for recorded in self.state["segments"]:
            for path_key, hash_key in (("report", "report_sha256"), ("supervisor", "supervisor_sha256"), ("checkpoint", "checkpoint_sha256")):
                if sha(recorded[path_key]) != recorded[hash_key]:
                    raise ValueError("Previously verified segment evidence changed")
            verified = verify_segment(recorded["report"], requested=recorded["requested"], parent=parent,
                contract=self.contract, admission=admitted, source=self.source, common=self.common, host=self.host)
            if any(verified[k] != recorded[k] for k in verified):
                raise ValueError("Continuation evidence no longer matches its attestation")
            parent = self.common.read_json(recorded["report"])
        return chain_totals(self.state["segments"], 3)

    def capture(self, final, scratch, admitted, admission_path):
        if self.recheck_chain(scratch, admitted) != TARGET_UPDATES or final["paused"]:
            raise ValueError("Only exactly 1000 full updates and unpaused final inference admit capture")
        self.wait_previous_follower()
        self.pin_admission(admission_path, self.state['admission']['sha256'])
        inputs = self.common.verify_inputs(self.source, final["checkpoint"], admission_path, final["report"])
        attestation = {"schema": "hexapod.exact_full_ppo_continuation.v1", "contract": self.contract,
            "pass": True, "full_updates_completed": TARGET_UPDATES, "scratch_updates_excluded": 3,
            "final_next_iteration": 1003, "segments": self.state["segments"],
            "admission": self.state["admission"], "final_input_sha256": inputs["input_sha256"],
            "interpretation": "Exact checkpoint/optimizer/normalizer and iteration continuity; environment trajectories restart at each segment. No terrain/hardware qualification."}
        path = self.folder / "chain_attestation.json"
        if path.exists():
            if self.common.read_json(path) != attestation:
                raise ValueError("Existing chain attestation differs")
        else:
            self.follower.durable_json(path, attestation)
        self.state.update(chain_attestation_sha256=sha(path), selected_inputs={"checkpoint": final["checkpoint"],
            "admission": str(admission_path), "training_report": final["report"]}, input_sha256=inputs["input_sha256"])
        self.save()
        job = self.state.get("active_job")
        if not job:
            command = ["/usr/bin/python3", str(Path(self.config["capture_tools"]) / "capture_policy.py"),
                "--source-dir", str(self.source), "--checkpoint", final["checkpoint"],
                "--admission", str(admission_path), "--training-report", final["report"],
                "--output-root", str(self.folder / "capture_outputs"), "--seconds", "15", "--width", "1280",
                "--height", "720", "--timeout-seconds", "1800"]
            if boot_seconds() + CAPTURE_SECONDS > self.state["deadline_boottime"]:
                raise ValueError("Total time budget cannot admit bounded capture")
            job = self.launch([*FLOCK, *command], command, kind="capture", output_root=self.folder / "capture_outputs")
            job["deadline_boottime"] = job['process']['start_ticks'] / os.sysconf('SC_CLK_TCK') + CAPTURE_SECONDS
            self.save()
        while self.exact_process(job["process"]):
            self.check()
            if boot_seconds() >= job["deadline_boottime"]:
                raise ValueError("Capture supervisor deadline expired")
            time.sleep(2)
        if self.child is not None and self.child.wait(timeout=5) != 0:
            raise ValueError("Capture child exited unsuccessfully")
        path = self.follower.verify_completed_capture(self.folder, {"binding": self.binding,
            "selected_inputs": self.state["selected_inputs"], "input_sha256": self.state["input_sha256"]}, self.common)
        self.state.update(state="video_complete", active_job=None, supervisor=str(path), supervisor_sha256=sha(path),
            video=str(path.with_name("policy.mp4")), video_sha256=sha(path.with_name("policy.mp4")), exit_code=0)
        self.save()

    def run(self):
        if self.state.get("launch_reserved"):
            self.state.update(state="launch_uncertain", exit_code=1); self.save(); return 1
        while not self.state["segments"]:
            self.check(); campaign = self.campaign()
            if campaign["state"] == "complete":
                self.verify_original_completion(campaign)
                self.state.update(state="original_completed", exit_code=0,
                    reason="Existing campaign follower owns original-success capture"); self.save(); return 0
            if campaign["state"] in ("failed", "paused") and not self.state.get("active_job"):
                self.state.update(state="original_failed", exit_code=1, reason="Original campaign failed or paused without our owned request")
                self.save(); return 1
            job = self.state.get("active_job")
            if job is None:
                self.follower.verify_campaign_process(self.follower.process_identity(self.config["campaign_pid"]), self.binding)
                job = self.capture_original_job(campaign)
                if job is None:
                    time.sleep(10); continue
                self.state["active_job"] = job; self.save()
            scratch, admitted, admission_path = self.original_parent_and_admission(campaign)
            path = self.monitor_job(job, scratch)
            report = self.common.read_json(path)
            deadline = min(self.state["deadline_boottime"], boot_seconds() + 120)
            while self.campaign()["state"] == "running" and boot_seconds() < deadline:
                self.check(); time.sleep(2)
            campaign = self.campaign()
            if report.get('paused') is not True:
                # A successful inner report cannot hide rejected supervision.
                self.verify_original_completion(campaign)
                self.state.update(state='original_completed', active_job=None, exit_code=0,
                    reason='Verified original full campaign completed; existing follower owns capture')
                self.save(); return 0
            if campaign["state"] != "paused":
                raise ValueError("Original campaign did not verify and record paused completion")
            full = next(p for p in campaign["phases"] if p["name"] == "full")
            if full.get("state") != "paused" or full.get("report_sha256") != sha(path):
                raise ValueError("Paused original full report is not campaign-bound")
            self.accept_segment(path, job, scratch, admitted, admission_path)
            self.wait_previous_follower()
        campaign = self.campaign()
        if campaign["state"] != "paused":
            raise ValueError("Original campaign pause identity changed")
        scratch, admitted, admission_path = self.original_parent_and_admission(campaign)
        if sha(admission_path) != self.state["admission"]["sha256"]:
            raise ValueError("Bound physical admission changed")
        while self.recheck_chain(scratch, admitted) < TARGET_UPDATES:
            self.check(); self.wait_previous_follower()
            if len(self.state["segments"]) >= MAX_SEGMENTS:
                raise ValueError("Three bounded full-training segments exhausted before 1000 updates")
            previous = self.state["segments"][-1]
            parent = self.common.read_json(previous["report"])
            job = self.state.get("active_job")
            if not job:
                rate = measured_iteration_seconds(previous["owned_job"]["history"])
                if rate is None:
                    rate = previous["owned_job"].get("planned_iteration_seconds")
                count = choose_chunk(TARGET_UPDATES - self.state["completed_full_updates"], rate,
                                     self.state["deadline_boottime"] - boot_seconds())
                root = self.folder / f"segment_{len(self.state['segments']) + 1:02d}"
                spawn, expected = segment_argv(self.config, root, admission_path, previous["checkpoint"], count)
                job = self.launch(spawn, expected, kind="train", output_root=root, requested=count)
                job["planned_iteration_seconds"] = rate; self.save()
            path = self.monitor_job(job, parent)
            self.accept_segment(path, job, parent, admitted, admission_path)
        self.capture(self.state["segments"][-1], scratch, admitted, admission_path)
        return 0

    def verify_original_completion(self, campaign):
        selected = self.follower.select_full_capture(campaign, self.config['campaign'], self.common)
        verified = self.common.verify_inputs(self.source, selected['checkpoint'], selected['admission'], selected['training_report'])
        if verified['contract'] != self.contract:
            raise ValueError('Completed original campaign differs from the bound source')
        self.state['original_completion'] = {'campaign_sha256': sha(self.config['campaign']),
            'selected_inputs': selected, 'input_sha256': verified['input_sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    coordinator = Coordinator(config, args.state_dir)
    if args.preflight_only:
        coordinator.follower.verify_campaign_process(coordinator.follower.process_identity(config["campaign_pid"]), coordinator.binding)
        coordinator.campaign()
        print(json.dumps({"pass": True, "dispatch": False, "source_sha256": SOURCE_SHA})); return 0
    coordinator.folder.mkdir(parents=True, exist_ok=True)
    with (coordinator.folder / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = coordinator.folder / "state.json"
        binding = {"config_sha256": sha(args.config), "coordinator_sha256": sha(__file__), "binding": coordinator.binding}
        coordinator.state = coordinator.common.read_json(path) if path.exists() else {
            "schema": "hexapod.full_ppo_continuation.v1", "identity": binding,
            "state": "monitoring", "owner_token": uuid.uuid4().hex, "started_utc": time.time(),
            "deadline_utc": time.time() + TOTAL_SECONDS, "boot_id": boot_id(),
            "deadline_boottime": boot_seconds() + TOTAL_SECONDS, "segments": [], "completed_full_updates": 0,
            "launch_history": [], "launch_reserved": None, "active_job": None}
        if coordinator.state["identity"] != binding:
            raise ValueError("Existing coordinator belongs to different source/config/code")
        if coordinator.state['boot_id'] != boot_id() or coordinator.state['boot_id'] != config['campaign_boot_id']:
            raise ValueError('Coordinator cannot resume across a different campaign/Linux boot')
        if coordinator.state["state"] in TERMINAL:
            return coordinator.state.get("exit_code", 1)
        coordinator.save()
        def stop(*_):
            coordinator.interrupted = True
        old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            return coordinator.run()
        except BaseException as error:
            job = coordinator.state.get("active_job")
            if job and job.get("origin") == "coordinator":
                coordinator.state["owned_signal_forwarded"] = coordinator.follower.forward_owned_signal(job["process"], signal.SIGTERM)
            if coordinator.state["state"] != "launch_uncertain":
                coordinator.state["state"] = "failed"
            coordinator.state.update(exit_code=1, error=f"{type(error).__name__}: {error}")
            coordinator.save(); return 1
        finally:
            for signum, handler in old.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
