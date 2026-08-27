"""The run contract: what a supervised attempt pins, observes, and records.

Everything here is data plus pure functions. The parent pin names the immutable
checkpoint an attempt resumes from; the gate report records why the machine was
or was not clear to launch; the milestone events record when startup crossed
each boundary; the attempt outcome records how it ended and carries exactly the
evidence the bounded-retry rule is allowed to consult.

``verify_parent`` hashes through an injected reader, so the contract can be
exercised without a filesystem.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Iterable


__all__ = [
    "AttemptOutcome",
    "GateCheck",
    "GateReport",
    "Milestone",
    "MilestoneEvent",
    "PROVEN_CLEANUP_STATES",
    "ParentPin",
    "ParentVerification",
    "Termination",
    "verify_parent",
]


# The launcher's own ``cleanup.status`` vocabulary for a cleanup that proved the
# exact owned container is gone or stopped. Anything else is unproven.
PROVEN_CLEANUP_STATES = ("stopped", "container-absent", "already-stopped")


class Milestone(IntEnum):
    """Startup progress, ordered. Comparisons are the point of the ordering."""

    LAUNCHED = 0
    CONFIG_LOADED = 1
    APP_READY = 2
    TRAINING = 3
    DONE = 4


class Termination(IntEnum):
    """How one attempt ended, in the vocabulary the retry rule understands."""

    SUCCESS = 0
    PRE_APPREADY_STALL = 1
    POST_APPREADY_FAILURE = 2
    USER_SIGNAL = 3
    DOCKER_ERROR = 4
    CLEANUP_UNVERIFIED = 5


@dataclass(frozen=True)
class ParentPin:
    """The immutable parent an attempt resumes from."""

    run_dir: str
    checkpoint: str
    sha256: str

    @property
    def path(self) -> str:
        return f"{self.run_dir.rstrip('/')}/{self.checkpoint}"


@dataclass(frozen=True)
class ParentVerification:
    """The result of hashing a parent checkpoint against its pinned hash."""

    path: str
    expected_sha256: str
    actual_sha256: str

    @property
    def matches(self) -> bool:
        return self.expected_sha256 == self.actual_sha256


def verify_parent(
    path: str,
    expected_sha256: str,
    read_chunks: Callable[[str], Iterable[bytes]],
) -> ParentVerification:
    """Hash ``path`` through the injected reader and compare with the pin."""
    digest = hashlib.sha256()
    for chunk in read_chunks(path):
        digest.update(chunk)
    return ParentVerification(
        path=path,
        expected_sha256=expected_sha256,
        actual_sha256=digest.hexdigest(),
    )


@dataclass(frozen=True)
class GateCheck:
    """One resource gate and the evidence behind its verdict."""

    name: str
    passed: bool
    evidence: tuple[str, ...] = ()

    def render(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        head = f"{verdict}: {self.name}"
        if not self.evidence:
            return head
        return "\n".join([head, *(f"  {line}" for line in self.evidence)])


@dataclass(frozen=True)
class GateReport:
    """Every resource gate for one launch decision, plus a single verdict."""

    checks: tuple[GateCheck, ...]
    observed_at_utc: str = ""

    @property
    def clear(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def check(self, name: str) -> GateCheck:
        for candidate in self.checks:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def render(self) -> str:
        lines = [f"clear={'yes' if self.clear else 'no'}"]
        if self.observed_at_utc:
            lines.append(f"observed_at_utc={self.observed_at_utc}")
        lines.extend(check.render() for check in self.checks)
        return "\n".join(lines)


@dataclass(frozen=True)
class MilestoneEvent:
    """One startup boundary crossed at a known offset from launch."""

    milestone: Milestone
    at_seconds: float
    detail: str = ""

    def render(self) -> str:
        suffix = f" {self.detail}" if self.detail else ""
        return f"{self.milestone.name}={self.at_seconds:.3f}s{suffix}"


@dataclass(frozen=True)
class AttemptOutcome:
    """How one attempt ended and the evidence the retry rule may consult.

    The evidence fields are deliberately explicit rather than derived: the retry
    rule must be able to say "there is no proof the parent is unchanged" and
    refuse, instead of inferring absence from a missing field.
    """

    label: str
    attempt_index: int
    classification: Termination
    reached: Milestone = Milestone.LAUNCHED
    events: tuple[MilestoneEvent, ...] = ()
    exit_status: int | None = None
    user_signal: str | None = None
    docker_status: str = "not-started"
    cleanup_required: bool = False
    cleanup_status: str = "not-required"
    run_directory_exists: bool = False
    checkpoint_exists: bool = False
    parent_unchanged: bool = False
    traceback_seen: bool = False
    oom_seen: bool = False
    diagnostics: tuple[str, ...] = ()
    started_at_utc: str = ""
    finished_at_utc: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def cleanup_proven(self) -> bool:
        if not self.cleanup_required:
            return True
        return self.cleanup_status in PROVEN_CLEANUP_STATES

    def render_lines(self) -> tuple[str, ...]:
        """Launcher-style ``key=value`` lines for the attempt artifact."""
        lines = [
            "schema_version=1",
            f"label={self.label}",
            f"attempt_index={self.attempt_index:02d}",
            f"classification={self.classification.name}",
            f"reached_milestone={self.reached.name}",
            f"exit_status={'unset' if self.exit_status is None else self.exit_status}",
            f"user_signal={self.user_signal or 'none'}",
            f"docker_status={self.docker_status}",
            f"cleanup_required={'true' if self.cleanup_required else 'false'}",
            f"cleanup_status={self.cleanup_status}",
            f"cleanup_proven={'true' if self.cleanup_proven else 'false'}",
            f"run_directory_exists={'true' if self.run_directory_exists else 'false'}",
            f"checkpoint_exists={'true' if self.checkpoint_exists else 'false'}",
            f"parent_unchanged={'true' if self.parent_unchanged else 'false'}",
            f"traceback_seen={'true' if self.traceback_seen else 'false'}",
            f"oom_seen={'true' if self.oom_seen else 'false'}",
            f"started_at_utc={self.started_at_utc or 'not-recorded'}",
            f"finished_at_utc={self.finished_at_utc or 'not-recorded'}",
        ]
        lines.extend(f"milestone[{index}]={event.render()}" for index, event in enumerate(self.events))
        lines.extend(f"diagnostics[{index}]={path}" for index, path in enumerate(self.diagnostics))
        lines.extend(f"note[{index}]={note}" for index, note in enumerate(self.notes))
        return tuple(lines)
