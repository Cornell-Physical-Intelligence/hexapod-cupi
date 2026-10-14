"""Frozen-tree, pinned-input, GPU-lock and owned-container helpers kept from the Sept 14 launcher."""

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
