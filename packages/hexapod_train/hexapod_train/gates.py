"""Resource gates: is this machine clear to start an attempt right now?

These are resource gates, not acceptance gates. Acceptance gates decide whether
a policy is admitted and live in ``docs/TRAINING.md`` §6; nothing here has any
opinion about a policy.

Every predicate is pure over an injected :class:`ResourceSnapshot`, and the
parsers that turn raw command output into a snapshot are pure too, so the whole
gate can be exercised against recorded evidence — including the NSVA case from
``docs/incidents/2026-08-26-preappready-stall.md`` §5, where the producer script
``/root/nsva_dl.sh`` predated the preflight and spawned its GPU child afterward.

The rule that case teaches is the reason ``producers_absent`` exists separately
from ``gpu_idle``: a free GPU is not evidence that a producer has finished. A
producer process or any descendant of one keeps the machine unclear even while
the GPU reads idle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .contract import GateCheck, GateReport


__all__ = [
    "GpuProcess",
    "PRODUCER_PATTERNS",
    "ProcessEntry",
    "ProducerHit",
    "ResourceSnapshot",
    "evaluate",
    "parse_docker_names",
    "parse_gpu_compute_apps",
    "parse_process_table",
    "producer_hits",
    "service_state_from_status",
]


#: Unrelated shared-host producers and their workers, named so evidence reads
#: back to the incident record. ``train_model_only_resume`` and
#: ``evaluate_checkpoint`` are deliberately absent: those are ours.
PRODUCER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("nsva_dl", r"(?:^|[\s/])nsva_dl\.sh(?:\s|$)"),
    ("validate_nsva", r"(?:^|[\s/])validate_nsva\.py(?:\s|$)"),
    ("score_clip", r"(?:^|[\s/])score_clip\.py(?:\s|$)"),
    ("validate_b51", r"(?:^|[\s/])validate_b51\.py(?:\s|$)"),
    ("torch_compile_worker", r"torch.*compile"),
    ("clip_scoring_worker", r"clip[_-]?scor(?:e|ing)"),
)

_COMPILED_PRODUCERS = tuple(
    (name, re.compile(pattern)) for name, pattern in PRODUCER_PATTERNS
)

#: Ancestry walks stop here; nothing above these pids is a meaningful ancestor.
_ROOT_PIDS = frozenset({0, 1})
_MAX_ANCESTRY_DEPTH = 64


@dataclass(frozen=True)
class ProcessEntry:
    """One row of ``ps -eo pid=,ppid=,args=``."""

    pid: int
    ppid: int
    command: str


@dataclass(frozen=True)
class GpuProcess:
    """One row of ``nvidia-smi --query-compute-apps``."""

    pid: int
    process_name: str = ""
    used_memory_mib: int | None = None

    def render(self) -> str:
        memory = "unknown" if self.used_memory_mib is None else f"{self.used_memory_mib} MiB"
        return f"pid={self.pid} name={self.process_name or 'unknown'} memory={memory}"


@dataclass(frozen=True)
class ProducerHit:
    """A producer process, or a descendant of one, that is still alive."""

    process: ProcessEntry
    pattern: str
    relation: str
    ancestor_pid: int | None = None

    def render(self) -> str:
        ancestry = "" if self.ancestor_pid is None else f" ancestor={self.ancestor_pid}"
        return (
            f"pid={self.process.pid} ppid={self.process.ppid} "
            f"match={self.pattern} relation={self.relation}{ancestry} "
            f"cmd={self.process.command}"
        )


@dataclass(frozen=True)
class ResourceSnapshot:
    """Everything the gates are allowed to consult, captured at one moment."""

    processes: tuple[ProcessEntry, ...] = ()
    gpu_processes: tuple[GpuProcess, ...] = ()
    docker_containers: tuple[str, ...] = ()
    service_state: str = "unknown"
    gpu_lock_free: bool | None = None
    observed_at_utc: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


def parse_process_table(text: str) -> tuple[ProcessEntry, ...]:
    """Parse ``ps -eo pid=,ppid=,args=`` output; unparsable rows are dropped."""
    entries: list[ProcessEntry] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        command = parts[2] if len(parts) > 2 else ""
        entries.append(ProcessEntry(pid=int(parts[0]), ppid=int(parts[1]), command=command))
    return tuple(entries)


def parse_gpu_compute_apps(text: str) -> tuple[GpuProcess, ...]:
    """Parse ``nvidia-smi --query-compute-apps=pid,process_name,used_memory``."""
    processes: list[GpuProcess] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("no running"):
            continue
        fields = [part.strip() for part in stripped.split(",")]
        # A gpu_uuid column may lead; take the first purely numeric field as pid.
        pid_index = next(
            (index for index, value in enumerate(fields) if value.isdigit()), None
        )
        if pid_index is None:
            continue
        name = fields[pid_index + 1] if len(fields) > pid_index + 1 else ""
        memory: int | None = None
        if len(fields) > pid_index + 2:
            raw_memory = fields[pid_index + 2].split()[0] if fields[pid_index + 2] else ""
            memory = int(raw_memory) if raw_memory.isdigit() else None
        processes.append(
            GpuProcess(pid=int(fields[pid_index]), process_name=name, used_memory_mib=memory)
        )
    return tuple(processes)


def parse_docker_names(text: str) -> tuple[str, ...]:
    """Parse ``docker ps --format '{{.Names}}'``."""
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def service_state_from_status(status: int | None) -> str:
    """Map ``systemctl is-active --quiet`` exit status to a state name.

    The launchers treat anything other than 0 (active) and 3 (inactive) as
    unproven, and so does this.
    """
    if status == 0:
        return "active"
    if status == 3:
        return "inactive"
    return "unknown"


def match_producer(command: str) -> str | None:
    """The name of the producer pattern this command line matches, if any."""
    for name, pattern in _COMPILED_PRODUCERS:
        if pattern.search(command):
            return name
    return None


def producer_hits(snapshot: ResourceSnapshot) -> tuple[ProducerHit, ...]:
    """Every live producer process and every live descendant of one."""
    by_pid = {entry.pid: entry for entry in snapshot.processes}
    hits: list[ProducerHit] = []
    for entry in snapshot.processes:
        direct = match_producer(entry.command)
        if direct is not None:
            hits.append(ProducerHit(process=entry, pattern=direct, relation="self"))
            continue
        ancestor_pid = entry.ppid
        depth = 0
        seen: set[int] = {entry.pid}
        while (
            ancestor_pid not in _ROOT_PIDS
            and ancestor_pid in by_pid
            and ancestor_pid not in seen
            and depth < _MAX_ANCESTRY_DEPTH
        ):
            seen.add(ancestor_pid)
            ancestor = by_pid[ancestor_pid]
            inherited = match_producer(ancestor.command)
            if inherited is not None:
                hits.append(
                    ProducerHit(
                        process=entry,
                        pattern=inherited,
                        relation="descendant",
                        ancestor_pid=ancestor.pid,
                    )
                )
                break
            ancestor_pid = ancestor.ppid
            depth += 1
    return tuple(hits)


def _gpu_check(snapshot: ResourceSnapshot) -> GateCheck:
    evidence = tuple(process.render() for process in snapshot.gpu_processes)
    return GateCheck(name="gpu_idle", passed=not snapshot.gpu_processes, evidence=evidence)


def _producer_check(snapshot: ResourceSnapshot) -> GateCheck:
    hits = producer_hits(snapshot)
    evidence = tuple(hit.render() for hit in hits)
    if not hits:
        evidence = ("no producer script or descendant is alive",)
    return GateCheck(name="producers_absent", passed=not hits, evidence=evidence)


def _container_check(snapshot: ResourceSnapshot) -> GateCheck:
    return GateCheck(
        name="containers_absent",
        passed=not snapshot.docker_containers,
        evidence=tuple(f"container={name}" for name in snapshot.docker_containers),
    )


def _service_check(snapshot: ResourceSnapshot) -> GateCheck:
    return GateCheck(
        name="training_service_inactive",
        passed=snapshot.service_state == "inactive",
        evidence=(f"hexapod-rl-training.service={snapshot.service_state}",),
    )


def _lock_check(snapshot: ResourceSnapshot) -> GateCheck:
    if snapshot.gpu_lock_free is None:
        return GateCheck(
            name="gpu_lock_available",
            passed=False,
            evidence=("the shared GPU lock could not be probed",),
        )
    return GateCheck(
        name="gpu_lock_available",
        passed=snapshot.gpu_lock_free,
        evidence=(
            "shared GPU lock is free"
            if snapshot.gpu_lock_free
            else "another Isaac GPU launcher holds the shared lock",
        ),
    )


def evaluate(snapshot: ResourceSnapshot) -> GateReport:
    """Every resource gate for one launch decision.

    A snapshot that cannot prove a condition fails that check: unknown service
    state and an unprobed lock are both refusals, never assumptions.
    """
    return GateReport(
        checks=(
            _gpu_check(snapshot),
            _producer_check(snapshot),
            _container_check(snapshot),
            _service_check(snapshot),
            _lock_check(snapshot),
        ),
        observed_at_utc=snapshot.observed_at_utc,
    )


def snapshot_from_text(
    *,
    process_table: str = "",
    gpu_compute_apps: str = "",
    docker_names: str = "",
    service_status: int | None = None,
    gpu_lock_free: bool | None = None,
    observed_at_utc: str = "",
    notes: Sequence[str] | Iterable[str] = (),
) -> ResourceSnapshot:
    """Build a snapshot from raw command output. Parsing only; no side effects."""
    return ResourceSnapshot(
        processes=parse_process_table(process_table),
        gpu_processes=parse_gpu_compute_apps(gpu_compute_apps),
        docker_containers=parse_docker_names(docker_names),
        service_state=service_state_from_status(service_status),
        gpu_lock_free=gpu_lock_free,
        observed_at_utc=observed_at_utc,
        notes=tuple(notes),
    )
