"""Spark reservation checks copied unchanged from the recorded Sept14 ownership guard."""

from contextlib import contextmanager

from pathlib import Path

from types import SimpleNamespace

import argparse

import fcntl

import hashlib

import importlib.util

import json

import os

import re

import signal

import subprocess

import sys

import time

RESERVATION_ROOT = Path("/home/orionh/HEXAPOD_runs/restart_20260914/spark_ownership_001")

POLICY_SHA256 = "163f4163f04c6855acde6e014e1ba9a985def5f6ebb95e7323fff8e91996d581"

MARKER_SHA256 = "ad5550c0938e954410a2039f40d58f5529e7b49ccfff1dff1fac4e7dab22a2e1"

COORDINATION = Path("/home/orionh/SPARK_COMPUTE_COORDINATION.md")

COORDINATION_SHA256 = "c89c99ebdd16941e3f8e7f23ef3525aeb360c78361aa49aa04e766b381a4727c"

LOCKS = ("/opt/wx/gpu.lock", "/tmp/hexapod-isaac-gpu.lock")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def canonical_path(value):
    require(isinstance(value, str) and value, "Missing absolute input/output path")
    path = Path(value)
    require(path.is_absolute() and str(path) == value and ".." not in path.parts,
            "Path must be an exact absolute path")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symbolic input/output path")
    return path

def pinned_file(path, expected):
    path = Path(path)
    require(valid_hash(expected), "Root-reviewed binding pending")
    require(path.is_file() and not path.is_symlink(), "Missing or symbolic pinned input: " + str(path))
    require(sha(path) == expected, "Pinned input changed: " + str(path))

def verify_tree(root, manifest_hash):
    root = Path(root)
    pinned_file(root / "FREEZE_SHA256.json", manifest_hash)
    paths = list(root.rglob("*"))
    require(not root.is_symlink() and not any(p.is_symlink() for p in paths), "Symbolic frozen tree")
    actual = {p.relative_to(root).as_posix(): sha(p) for p in paths if p.is_file()}
    actual.pop("FREEZE_SHA256.json")
    require(actual == read(root / "FREEZE_SHA256.json"), "Changed or unlisted frozen tree: " + str(root))

def call(command):
    return subprocess.check_output(command, text=True, timeout=30).strip()

def unit_fields(unit, user=True, fields=()):
    command = ["systemctl"] + (["--user"] if user else []) + ["show", unit]
    for field in fields:
        command.extend(["-p", field])
    return dict(line.split("=", 1) for line in call(command).splitlines() if "=" in line)

def verify_policy_bytes():
    pinned_file(RESERVATION_ROOT / "reservation_policy.json", POLICY_SHA256)
    pinned_file(RESERVATION_ROOT / "ACTIVE", MARKER_SHA256)
    pinned_file(COORDINATION, COORDINATION_SHA256)
    policy = read(RESERVATION_ROOT / "reservation_policy.json")
    require(policy.get("schema") == "canonical_exclusive_mask_reservation_v2"
            and policy.get("release_only_on_user_instruction") is True
            and policy.get("marker_sha256") == MARKER_SHA256,
            "Different active reservation policy")
    require(read(RESERVATION_ROOT / "ACTIVE").get("exclusive") is True, "Reservation released")
    for name, expected in policy["files"].items():
        pinned_file(name, expected)
    for name in policy["blocked_entry_directories"]:
        path = Path(name)
        require(path.is_dir() and not path.is_symlink() and str(path / "__main__.py") in policy["files"],
                "Reconstruction blocker was replaced")
    return policy

def verify_reservation():
    """Adapt archived v2 checks to the new root policy, without old owner states."""
    policy = verify_policy_bytes()
    require(len(policy["mask_paths"]) == 32 and len(policy["system_masked_units"]) == 4,
            "Incomplete current scheduler policy")
    reload_flags = {}
    for name in policy["mask_paths"]:
        path = Path(name)
        require(path.is_symlink() and os.readlink(path) == "/dev/null", "Exact user mask missing: " + name)
        fields = unit_fields(path.name, fields=("LoadState", "UnitFileState", "ActiveState", "MainPID", "FragmentPath", "NeedDaemonReload"))
        require(fields.get("LoadState") == fields.get("UnitFileState") == "masked"
                and fields.get("ActiveState") == "inactive" and fields.get("FragmentPath") == name,
                "Loaded mask differs: " + name)
        require(fields.get("MainPID", "0") == "0", "Masked unit has a process: " + name)
        require(fields.get("NeedDaemonReload") in ("yes", "no"), "Reload metadata missing")
        reload_flags[path.name] = fields["NeedDaemonReload"]
    for name in policy["system_masked_units"]:
        fields = unit_fields(name, user=False, fields=("LoadState", "UnitFileState", "ActiveState", "MainPID"))
        require(fields.get("LoadState") == fields.get("UnitFileState") == "masked"
                and fields.get("ActiveState") == "inactive" and fields.get("MainPID", "0") == "0",
                "System mask differs: " + name)
    fields = unit_fields(policy["queue_unit"], fields=("ActiveState", "SubState", "MainPID", "UnitFileState", "FragmentPath", "NeedDaemonReload", "DropInPaths"))
    require(fields.get("ActiveState") == "active" and fields.get("SubState") == "running"
            and fields.get("UnitFileState") == "enabled" and fields.get("NeedDaemonReload") == "no"
            and fields.get("DropInPaths") == "" and fields.get("FragmentPath") == policy["queue_unit_path"],
            "Current queue lock helper unavailable or overridden")
    pid = fields.get("MainPID", "")
    require(pid.isdecimal() and int(pid) > 1, "Current queue owner PID missing")
    record = read(policy["queue_lock_record"])
    require(record.get("pid") == int(pid) and record.get("lock_path") == policy["queue_lock_path"]
            and record.get("reservation_path") == str(RESERVATION_ROOT / "ACTIVE")
            and record.get("reservation_sha256") == MARKER_SHA256 and record.get("queue_mutated") is False,
            "Current queue acquisition record differs")
    expected = ("/usr/bin/python3\0-B\0" + policy["queue_helper"] + "\0").encode()
    require((Path("/proc") / pid / "cmdline").read_bytes() == expected, "Queue helper process origin differs")
    info = Path(policy["queue_lock_path"]).stat()
    inode = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}"
    locks = [line.split() for line in Path("/proc/locks").read_text().splitlines()]
    require(any(len(row) >= 8 and row[1:5] == ["FLOCK", "ADVISORY", "WRITE", pid]
                and row[5] == inode and row[6:8] == ["0", "EOF"] for row in locks), "Queue lock lacks verified live owner")
    pinned_file(policy["original_units_before"], "15d866dc1a1bdd841049ed489cec4b9bf1e14bf3fe1f93a59f6158db4a84dc90")
    before = read(policy["original_units_before"])
    for unit, metadata in before["files"].items():
        path = Path(policy["original_units_backup"]) / unit
        if metadata["kind"] == "file":
            pinned_file(path, metadata["sha256"])
        elif metadata["kind"] == "symlink":
            require(path.is_symlink() and os.readlink(path) == metadata["target"], "Original unit symlink backup changed")
        else:
            require(metadata["kind"] == "absent_override", "Unknown original backup kind")
    return {"policy_sha256": POLICY_SHA256, "marker_sha256": MARKER_SHA256,
            "coordination_sha256": COORDINATION_SHA256, "queue_pid": int(pid),
            "masked_user_units": 32, "masked_system_units": 4, "mask_reload_flags": reload_flags,
            "reservation_released": False}

@contextmanager
def both_locks():
    descriptors = []
    try:
        for name in LOCKS:
            descriptor = os.open(name, os.O_RDONLY)
            descriptors.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)

def no_live_compute():
    gpu = call(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"])
    require(not gpu, "Unrelated CUDA compute remains; dispatcher must resolve it")
    containers = call(["docker", "ps", "--format", "{{.ID}} {{.Names}} {{.Image}}"])
    require(not containers, "Active containers remain; dispatcher must identify them")
    return {"cuda_processes": gpu, "active_containers": containers}

def cleanup_owned(output):
    """ExecStopPost fallback: signal only the identity recorded by this attempt."""
    report_path = output / "jobs/standing.json"
    if not report_path.exists():
        return {"no_owned_job_record": True, "reservation_released": False}
    require(not report_path.is_symlink(), "Symbolic owned job record")
    report = read(report_path)
    name, identity = report.get("container_name"), report.get("container_id")
    require(isinstance(name, str) and re.fullmatch(r"hexapod-reference-physics-[a-f0-9]{32}", name),
            "Unrecognized owned container name; no cleanup")
    require(identity is None or valid_hash(identity), "Invalid immutable owned container ID")
    inspections = []
    for _ in range(2):
        result = subprocess.run(["docker", "inspect", "--format", "{{.Id}} {{.Name}} {{.State.Running}}", identity or name],
                                text=True, capture_output=True, timeout=20)
        if result.returncode:
            require("no such object" in result.stderr.lower() or "no such container" in result.stderr.lower(),
                    "Owned container absence unknown; no unrelated cleanup")
            inspections.append({"absent": True})
            break
        fields = result.stdout.strip().split()
        require(len(fields) == 3 and valid_hash(fields[0]) and fields[1] == "/" + name
                and fields[2] in ("true", "false") and (identity is None or identity == fields[0]),
                "Owned container identity mismatch; do not signal it")
        identity = fields[0]
        inspections.append({"container_id": identity, "running": fields[2] == "true"})
        if fields[2] == "false":
            break
        subprocess.run(["docker", "stop", "--time", "20", identity], check=True, capture_output=True, timeout=30)
    else:
        raise RuntimeError("Owned container still running after exact stop; review required")
    return {"container_name": name, "container_id": identity, "inspections": inspections,
            "cleanup_checked": True, "reservation_released": False}
