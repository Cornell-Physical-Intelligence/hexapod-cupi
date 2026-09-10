"""Exact reviewed source009 shared-GPU read-only preflight/competition helpers."""
from pathlib import Path
import subprocess

def command(args):
    return subprocess.check_output(args, text=True, timeout=20).strip()

def resources():
    processes = command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"])
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")}
    return processes, memory["MemAvailable"]

def preflight():
    processes, available = resources()
    containers = command(["docker", "ps", "--format", "{{.Names}} {{.Image}}"])
    if processes or any("isaac-lab" in line or "hexapod-rl" in line for line in containers.splitlines()):
        raise BlockingIOError(f"Shared GPU is occupied; no launch.\n{processes}\n{containers}")
    if available < 16 * 1024**3:
        raise BlockingIOError("Less than 16 GiB host memory available; no launch")
    return {"available_memory_bytes": available, "containers": containers, "compute_processes": processes}

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

