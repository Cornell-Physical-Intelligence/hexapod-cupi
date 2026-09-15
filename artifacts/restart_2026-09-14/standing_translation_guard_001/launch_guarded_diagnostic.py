"""Root-bound, single-placement diagnostic; reuse the frozen host supervisor."""
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

sys.dont_write_bytecode = True
SCHEMA = "canonical_single_placement_guard_v1"
RESERVATION_ROOT = Path("/home/orionh/HEXAPOD_runs/restart_20260914/spark_ownership_001")
POLICY_SHA256 = "163f4163f04c6855acde6e014e1ba9a985def5f6ebb95e7323fff8e91996d581"
MARKER_SHA256 = "ad5550c0938e954410a2039f40d58f5529e7b49ccfff1dff1fac4e7dab22a2e1"
COORDINATION = Path("/home/orionh/SPARK_COMPUTE_COORDINATION.md")
COORDINATION_SHA256 = "fc1da7bf07d1db3022b1b9a3ec41d261a69e7a19c562457ea4e4326a043893b0"
OLD_SOURCE_FREEZE = "c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131"
LOCKS = ("/opt/wx/gpu.lock", "/tmp/hexapod-isaac-gpu.lock")
ARMS = {"origin": [0.0, 0.0], "xy14_4": [14.0, 4.0]}
FALSE_FIELDS = ("standing_admission", "batch_admission", "stage2_complete",
                "training_allowed", "physical_admission", "physics_admitted")
FRESH_ROOTS = (Path("/home/orionh/HEXAPOD_runs/restart_20260914"),
               Path("/home/orionh/HEXAPOD_runs/canonical_restart_20260914"))


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


def check_binding(binding, placement, output):
    require(binding.get("schema") == SCHEMA, "Wrong diagnostic guard schema")
    require(binding.get("root_review_complete") is True, "Root review/bindings pending; no allocation")
    require(placement in ARMS and binding.get("placement") == placement, "Placement differs from reviewed arm")
    require(str(output) == binding.get("output"), "Output differs from reviewed arm")
    require(type(binding.get("num_envs")) is int and binding["num_envs"] == 1, "Single robot only")
    require(binding.get("max_seconds") == 1200, "Only the existing 1200-second diagnostic bound is permitted")
    require(binding.get("diagnostic_only") is True, "Diagnostic scope missing")
    require(all(binding.get(key) is False for key in FALSE_FIELDS), "Diagnostic cannot admit stages or training")
    require(binding.get("reservation_root") == str(RESERVATION_ROOT), "Historical or different reservation rejected")
    require(binding.get("policy_sha256") == POLICY_SHA256, "Old or unreviewed reservation policy rejected")
    require(binding.get("coordination_sha256") == COORDINATION_SHA256, "Old or unreviewed coordination rejected")
    for key in ("source_freeze_sha256", "host_freeze_sha256", "native_bindings_sha256"):
        require(valid_hash(binding.get(key)), "Root-reviewed binding pending: " + key)
    require(binding["source_freeze_sha256"] != OLD_SOURCE_FREEZE, "Historical admission source rejected")
    paths = {key: canonical_path(binding.get(key)) for key in
             ("source", "host", "asset", "admission", "supervisor_source", "native_bindings", "output", "isaaclab")}
    for key in ("source", "host", "native_bindings", "output"):
        require(any(parent in paths[key].parents for parent in FRESH_ROOTS), "Fresh restart path required: " + key)
    require(paths["isaaclab"] == Path("/home/orionh/IsaacLab"), "Different Isaac host tree rejected")
    for key in ("source", "host", "asset", "admission", "supervisor_source", "native_bindings"):
        path = paths[key]
        require(paths["output"] != path and path not in paths["output"].parents and paths["output"] not in path.parents,
                "Output overlaps immutable input: " + key)
    require(RESERVATION_ROOT not in paths["output"].parents, "Output overlaps reservation evidence")
    return paths


def check_identity(identity, placement, source_hash):
    require(identity.get("schema") == "canonical_single_placement_diagnostic_v1", "Historical or admitting native schema")
    require(identity.get("num_envs") == 1, "Native identity is not single robot")
    require(all(identity.get(key) is False for key in FALSE_FIELDS), "Native identity could admit training/batch/physics")
    require(identity.get("runtime_binding") == {
        "runtime_tree_sha256": source_hash, "scope": "canonical_single_placement_diagnostic_only"},
        "Native runtime binding differs from diagnostic source")
    location = identity.get("placement", {})
    require(location.get("placement") == placement and location.get("xy_m") == ARMS[placement],
            "Native placement differs from selected arm")
    require(location.get("root") == "/Robot", "Diagnostic must retain the single /Robot root")
    require(identity.get("steps") == 8000 and identity.get("controls") == 1000
            and identity.get("settle_controls") == 200 and identity.get("dt") == .0025
            and identity.get("control_dt") == .02 and identity.get("substeps_per_control") == 8,
            "Diagnostic timing differs from unchanged standing screen")


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


def load_host(binding, paths):
    verify_tree(paths["source"], binding["source_freeze_sha256"])
    verify_tree(paths["host"], binding["host_freeze_sha256"])
    pinned_file(paths["native_bindings"], binding["native_bindings_sha256"])
    native = read(paths["native_bindings"])
    require(native.get("schema") == "canonical_placement_host_binding_v1"
            and native.get("native_dispatch_authorized") is True, "Native host binding is not authorized")
    for key in ("source_freeze_sha256", "host_freeze_sha256", "coordination_sha256", "placement", "output"):
        require(native.get(key) == binding[key], "Native host binding mismatch: " + key)
    entry = paths["host"] / "launch_standing_spark.py"
    spec = importlib.util.spec_from_file_location("_root_bound_translation_host", entry)
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    args = SimpleNamespace(**{key: value for key, value in paths.items() if key not in ("host", "native_bindings")},
                           bindings=paths["native_bindings"], placement=binding["placement"], num_envs=1,
                           standing_one=None, host_freeze_sha256=binding["host_freeze_sha256"],
                           coordination_sha256=COORDINATION_SHA256)
    identity = host.verify_inputs(args)
    check_identity(identity, args.placement, binding["source_freeze_sha256"])
    return host, args, identity


def install_reservation_checks(parent):
    """Preserve run_owned, cleanup, deadlines, resource ownership and audit code."""
    original_preflight, original_resources = parent.preflight, parent.resources
    def preflight():
        reservation = verify_reservation()
        no_live_compute()
        result = original_preflight()
        return {**result, "current_reservation": reservation}
    def resources():
        verify_policy_bytes()
        return original_resources()
    parent.preflight, parent.resources = preflight, resources


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


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--bindings-sha256", required=True)
    parser.add_argument("--guard-freeze-sha256", required=True)
    parser.add_argument("--placement", choices=sorted(ARMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--cleanup-only", action="store_true")
    cli = parser.parse_args()
    verify_tree(Path(__file__).resolve().parent, cli.guard_freeze_sha256)
    pinned_file(cli.bindings, cli.bindings_sha256)
    binding = read(cli.bindings)
    paths = check_binding(binding, cli.placement, cli.output)
    if cli.cleanup_only:
        with both_locks():
            cleanup = cleanup_owned(paths["output"])
            cleanup["reservation"] = verify_reservation()
            cleanup["resources_after_cleanup"] = no_live_compute()
            if paths["output"].exists():
                save(paths["output"] / "guard_cleanup.json", cleanup)
        print(json.dumps(cleanup, indent=2))
        return
    host, args, identity = load_host(binding, paths)
    host.require_fresh_output(args)
    with both_locks():
        reservation = verify_reservation()
        resources = no_live_compute()
    if cli.preflight_only:
        print(json.dumps({"identity": identity, "reservation": reservation, "resources": resources,
                          "native_run_started": False, "output_created": False}, indent=2))
        return
    parent = host.load_supervisor(args, identity)
    install_reservation_checks(parent)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("jobs", "logs"):
        (args.output / name).mkdir()
    signal.signal(signal.SIGTERM, lambda *_: (args.output / "stop.request").touch())
    signal.signal(signal.SIGINT, lambda *_: (args.output / "stop.request").touch())
    guard = {"schema": SCHEMA, "status": "starting", "placement": args.placement,
             "started_unix": time.time(), "preflight": reservation, "resources_before": resources,
             "guard_freeze_sha256": cli.guard_freeze_sha256, "bindings_sha256": cli.bindings_sha256,
             "native_bindings_sha256": binding["native_bindings_sha256"],
             "supervisor_lifecycle_reused": True, "max_native_seconds": 1200,
             "automatic_continuation": False, **{key: False for key in FALSE_FIELDS}}
    save(args.output / "guard.json", guard)
    try:
        host.run_campaign(args, identity)
        guard["status"] = "completed"
    except Exception as exc:
        guard.update(status="stopped" if isinstance(exc, InterruptedError) else "failed", error=repr(exc))
        raise
    finally:
        try:
            with both_locks():
                guard["owned_cleanup"] = cleanup_owned(args.output)
                guard["terminal_reservation"] = verify_reservation()
                guard["terminal_resources"] = no_live_compute()
                verify_tree(paths["source"], binding["source_freeze_sha256"])
                verify_tree(paths["host"], binding["host_freeze_sha256"])
                pinned_file(cli.bindings, cli.bindings_sha256)
                pinned_file(paths["native_bindings"], binding["native_bindings_sha256"])
                require(host.verify_inputs(args) == identity, "Terminal diagnostic input identity changed")
                guard["terminal_inputs_unchanged"] = True
        except Exception as exc:
            guard.update(status="failed", terminal_error=repr(exc), owner_review_required=True)
        guard["finished_unix"] = time.time()
        save(args.output / "guard.json", guard)
    require(guard["status"] == "completed", "Terminal diagnostic guard verification failed")


if __name__ == "__main__":
    main()
