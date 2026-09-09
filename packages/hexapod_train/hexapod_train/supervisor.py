"""The attempt-aware startup supervisor specified in ``docs/OPERATIONS.md`` §6.

The bash launcher stays the executor. It holds the shared GPU flock, gates on
Docker and the systemd service, verifies the immutable parent, writes its atomic
artifacts, and cleans up by exact container ID. What it cannot do from inside a
single ``timeout``-wrapped ``docker compose run`` is watch its own startup: this
module supplies the milestone timing, the no-progress diagnostics trigger, the
terminal classification, and the bounded retry decision.

Every decision here is pure. The clock, the diagnostics capture, and the
artifact writer are injected, so the whole state machine and the whole retry
truth table are exercised with fakes.

The one thing this module never does is act on the machine. It does not remove
containers, delete caches or volumes, or signal anything it did not create; when
an attempt must end, the launcher's own signal handling and exact-ID cleanup do
the work.

Implemented to spec; not yet validated in vivo on the Spark.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol, Sequence

from . import labels
from .contract import (
    PROVEN_CLEANUP_STATES,
    AttemptOutcome,
    Milestone,
    MilestoneEvent,
    Termination,
)


__all__ = [
    "APP_READY_MARKER",
    "ArtifactWriter",
    "AttemptRecorder",
    "CONFIG_MARKER",
    "DEFAULT_DEADLINES",
    "MAX_ATTEMPTS",
    "MilestoneDeadlines",
    "MilestoneMachine",
    "RetryDecision",
    "ShardSequencer",
    "TRAINING_MARKERS",
    "classify_outcome",
    "retry_decision",
]


#: The two startup markers the incident record proves are load-bearing: a
#: healthy start reaches the first at about +2 s and the second at about +12 s.
CONFIG_MARKER = "Loading user config"
APP_READY_MARKER = "AppLauncher initialization complete"
#: PPO has begun; the run is past every startup boundary.
TRAINING_MARKERS = ("Learning iteration", "Computation:", "Mean reward:")

#: Docker/compose client failures, as distinct from a stalled Isaac startup.
DOCKER_ERROR_STATUSES = frozenset({125, 126, 127})

#: One attempt plus at most one retry. Never more.
MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class MilestoneDeadlines:
    """The spec's startup budget, in seconds since launch."""

    config_loaded: float = 45.0
    app_ready: float = 90.0
    overall: float = 420.0
    no_progress: float = 30.0


DEFAULT_DEADLINES = MilestoneDeadlines()


@dataclass(frozen=True)
class Stall:
    """Why the machine believes this attempt is over."""

    termination: Termination
    reason: str
    at_seconds: float


class MilestoneMachine:
    """LAUNCHED -> CONFIG_LOADED -> APP_READY -> TRAINING -> DONE.

    ``observe`` is fed whatever the child wrote; ``tick`` is called with the
    current clock reading and decides deadlines and the no-progress capture. A
    milestone that arrives *after* its deadline does not rescue the attempt: the
    45 s and 90 s budgets are boundaries, not hints.
    """

    def __init__(
        self,
        *,
        deadlines: MilestoneDeadlines = DEFAULT_DEADLINES,
        capture_diagnostics: Callable[[str, float], str] | None = None,
        started_at: float = 0.0,
        config_marker: str = CONFIG_MARKER,
        app_ready_marker: str = APP_READY_MARKER,
        training_markers: Sequence[str] = TRAINING_MARKERS,
    ) -> None:
        self.deadlines = deadlines
        self._capture = capture_diagnostics
        self._started_at = started_at
        self._config_marker = config_marker
        self._app_ready_marker = app_ready_marker
        self._training_markers = tuple(training_markers)
        self._state = Milestone.LAUNCHED
        self._events: list[MilestoneEvent] = [
            MilestoneEvent(Milestone.LAUNCHED, 0.0, "launcher started")
        ]
        self._last_progress = started_at
        self._capture_armed = True
        self._captures: list[str] = []
        self._stall: Stall | None = None
        self._traceback_seen = False
        self._oom_seen = False

    # -- observation -----------------------------------------------------

    @property
    def state(self) -> Milestone:
        return self._state

    @property
    def events(self) -> tuple[MilestoneEvent, ...]:
        return tuple(self._events)

    @property
    def captures(self) -> tuple[str, ...]:
        return tuple(self._captures)

    @property
    def stall(self) -> Stall | None:
        return self._stall

    @property
    def traceback_seen(self) -> bool:
        return self._traceback_seen

    @property
    def oom_seen(self) -> bool:
        return self._oom_seen

    def elapsed(self, now: float) -> float:
        return now - self._started_at

    def observe(self, text: str, now: float) -> None:
        """Feed one chunk of child output observed at ``now``."""
        if not text:
            return
        self._last_progress = now
        self._capture_armed = True
        elapsed = self.elapsed(now)

        if "Traceback (most recent call last)" in text:
            self._traceback_seen = True
        lowered = text.lower()
        if "out of memory" in lowered or "oom-kill" in lowered or "killed process" in lowered:
            self._oom_seen = True

        if self._state < Milestone.CONFIG_LOADED and self._config_marker in text:
            self._advance(Milestone.CONFIG_LOADED, elapsed, self._config_marker)
            if elapsed > self.deadlines.config_loaded:
                self._declare(
                    Termination.PRE_APPREADY_STALL,
                    f"{self._config_marker!r} arrived at {elapsed:.3f}s, after the "
                    f"{self.deadlines.config_loaded:g}s budget",
                    elapsed,
                )
        if self._state < Milestone.APP_READY and self._app_ready_marker in text:
            self._advance(Milestone.APP_READY, elapsed, self._app_ready_marker)
            if elapsed > self.deadlines.app_ready:
                self._declare(
                    Termination.PRE_APPREADY_STALL,
                    f"{self._app_ready_marker!r} arrived at {elapsed:.3f}s, after the "
                    f"{self.deadlines.app_ready:g}s budget",
                    elapsed,
                )
        if self._state < Milestone.TRAINING and any(
            marker in text for marker in self._training_markers
        ):
            if self._state < Milestone.APP_READY:
                # PPO output without an AppReady marker means the marker was
                # missed, not that startup was skipped; record it as reached.
                self._advance(Milestone.APP_READY, elapsed, "inferred from PPO output")
            self._advance(Milestone.TRAINING, elapsed, "PPO update output")

    def complete(self, now: float, detail: str = "launcher reported completion") -> None:
        """The launcher finished its own success path."""
        elapsed = self.elapsed(now)
        if self._state < Milestone.TRAINING:
            self._advance(Milestone.TRAINING, elapsed, "inferred from completion")
        self._advance(Milestone.DONE, elapsed, detail)

    def tick(self, now: float) -> Stall | None:
        """Evaluate the deadlines and the no-progress trigger at ``now``."""
        if self._stall is not None:
            return self._stall
        elapsed = self.elapsed(now)

        if (
            self._capture_armed
            and now - self._last_progress >= self.deadlines.no_progress
            and self._state < Milestone.TRAINING
        ):
            self._capture_armed = False
            reason = (
                f"no progress for {now - self._last_progress:.3f}s in state "
                f"{self._state.name}"
            )
            if self._capture is not None:
                self._captures.append(self._capture(reason, elapsed))
            else:
                self._captures.append(f"capture-not-configured: {reason}")

        if elapsed > self.deadlines.overall:
            return self._declare(
                Termination.PRE_APPREADY_STALL
                if self._state < Milestone.APP_READY
                else Termination.POST_APPREADY_FAILURE,
                f"overall {self.deadlines.overall:g}s cap exceeded in state "
                f"{self._state.name}",
                elapsed,
            )
        if self._state < Milestone.CONFIG_LOADED and elapsed > self.deadlines.config_loaded:
            return self._declare(
                Termination.PRE_APPREADY_STALL,
                f"{self._config_marker!r} not seen within "
                f"{self.deadlines.config_loaded:g}s",
                elapsed,
            )
        if self._state < Milestone.APP_READY and elapsed > self.deadlines.app_ready:
            return self._declare(
                Termination.PRE_APPREADY_STALL,
                f"{self._app_ready_marker!r} not seen within "
                f"{self.deadlines.app_ready:g}s",
                elapsed,
            )
        return None

    # -- internals -------------------------------------------------------

    def _advance(self, milestone: Milestone, elapsed: float, detail: str) -> None:
        if milestone <= self._state:
            return
        self._state = milestone
        self._events.append(MilestoneEvent(milestone, elapsed, detail))

    def _declare(self, termination: Termination, reason: str, elapsed: float) -> Stall:
        if self._stall is None:
            self._stall = Stall(termination=termination, reason=reason, at_seconds=elapsed)
        return self._stall


def classify_outcome(
    *,
    reached: Milestone,
    stall: Stall | None = None,
    exit_status: int | None = None,
    user_signal: str | None = None,
    docker_status: int | str | None = None,
    cleanup_required: bool = False,
    cleanup_status: str = "not-required",
    completed: bool = False,
) -> Termination:
    """Name the terminal class of one attempt.

    Order matters and encodes the spec's precedence: an operator signal is never
    reinterpreted as an infrastructure stall, and cleanup that could not be
    proven outranks whatever failure preceded it, because an unproven cleanup is
    the one state that forbids any further action.
    """
    if user_signal:
        return Termination.USER_SIGNAL
    if completed and exit_status == 0:
        return Termination.SUCCESS
    if cleanup_required and cleanup_status not in PROVEN_CLEANUP_STATES:
        return Termination.CLEANUP_UNVERIFIED
    if isinstance(docker_status, int) and docker_status in DOCKER_ERROR_STATUSES:
        return Termination.DOCKER_ERROR
    if reached >= Milestone.APP_READY:
        return Termination.POST_APPREADY_FAILURE
    if stall is not None:
        return stall.termination
    return Termination.PRE_APPREADY_STALL


@dataclass(frozen=True)
class RetryDecision:
    """Whether the one permitted identical retry is allowed, and why."""

    retry: bool
    reason: str
    next_label: str | None = None
    refusals: tuple[str, ...] = ()

    def render_lines(self) -> tuple[str, ...]:
        lines = [
            "schema_version=1",
            f"retry={'true' if self.retry else 'false'}",
            f"reason={self.reason}",
            f"next_label={self.next_label or 'none'}",
        ]
        lines.extend(f"refusal[{index}]={text}" for index, text in enumerate(self.refusals))
        return tuple(lines)


def retry_decision(
    outcome: AttemptOutcome, *, moment: datetime | None = None
) -> RetryDecision:
    """Spec item 7, verbatim.

    One identical retry under a new immutable attempt label is allowed only for
    a proven pre-AppReady stall with no run directory and no checkpoint, an
    unchanged parent, no traceback and no OOM, and proven cleanup. User signals,
    post-AppReady failures, Docker errors, and unverifiable cleanup are never
    retried, and a retry is never itself retried.
    """
    refusals: list[str] = []

    if outcome.classification is Termination.SUCCESS:
        refusals.append("the attempt succeeded; there is nothing to retry")
    elif outcome.classification is not Termination.PRE_APPREADY_STALL:
        refusals.append(
            f"{outcome.classification.name} is never retried"
        )
    if outcome.user_signal:
        refusals.append(f"a user signal ({outcome.user_signal}) is never retried")
    if outcome.attempt_index >= MAX_ATTEMPTS - 1:
        refusals.append(
            f"attempt {outcome.attempt_index:02d} is the last permitted attempt "
            f"(at most {MAX_ATTEMPTS} attempts)"
        )
    if outcome.run_directory_exists:
        refusals.append("a run directory exists; the attempt was not pre-AppReady")
    if outcome.checkpoint_exists:
        refusals.append("a checkpoint exists; the attempt produced evidence")
    if not outcome.parent_unchanged:
        refusals.append("the immutable parent was not proven unchanged")
    if outcome.traceback_seen:
        refusals.append("a traceback was observed; this is a code fault, not a stall")
    if outcome.oom_seen:
        refusals.append("an OOM was observed")
    if not outcome.cleanup_proven:
        refusals.append(
            f"cleanup was not proven (cleanup_status={outcome.cleanup_status})"
        )

    if refusals:
        return RetryDecision(
            retry=False,
            reason="; ".join(refusals),
            next_label=None,
            refusals=tuple(refusals),
        )

    next_label = (
        labels.retry_label(outcome.label, moment) if moment is not None else None
    )
    return RetryDecision(
        retry=True,
        reason=(
            "proven pre-AppReady stall with no run or checkpoint, unchanged parent, "
            "no traceback or OOM, and proven cleanup: one identical retry under a new "
            "immutable attempt label"
        ),
        next_label=next_label,
        refusals=(),
    )


class ArtifactWriter(Protocol):
    """The only filesystem surface the recorder is allowed to use."""

    def makedirs(self, path: Path) -> None: ...

    def write_text(self, path: Path, text: str) -> None: ...


class AttemptRecorder:
    """Writes ``attempts/NN``, ``retry.decision``, and the root outcome.

    Overall success must never hide a retry, so the root outcome always states
    how many attempts ran and whether a retry occurred.
    """

    def __init__(self, root: str | Path, writer: ArtifactWriter) -> None:
        self.root = Path(root)
        self._writer = writer

    def attempt_directory(self, attempt_index: int) -> Path:
        if attempt_index < 0:
            raise ValueError(f"attempt index must not be negative: {attempt_index}")
        return self.root / "attempts" / f"{attempt_index:02d}"

    def record_attempt(self, outcome: AttemptOutcome) -> Path:
        directory = self.attempt_directory(outcome.attempt_index)
        self._writer.makedirs(directory)
        self._write(directory / "outcome", outcome.render_lines())
        self._write(
            directory / "milestones",
            tuple(event.render() for event in outcome.events),
        )
        return directory

    def record_retry_decision(self, decision: RetryDecision) -> Path:
        path = self.root / "retry.decision"
        self._writer.makedirs(self.root)
        self._write(path, decision.render_lines())
        return path

    def record_root_outcome(
        self,
        outcomes: Sequence[AttemptOutcome],
        decision: RetryDecision | None = None,
    ) -> Path:
        if not outcomes:
            raise ValueError("a root outcome needs at least one attempt")
        final = outcomes[-1]
        retry_performed = len(outcomes) > 1
        lines = [
            "schema_version=1",
            f"attempts={len(outcomes)}",
            f"retry_performed={'true' if retry_performed else 'false'}",
            f"retry_allowed={'true' if decision is not None and decision.retry else 'false'}",
            f"final_classification={final.classification.name}",
            f"final_label={final.label}",
            f"overall={'succeeded' if final.classification is Termination.SUCCESS else 'failed'}",
        ]
        for index, outcome in enumerate(outcomes):
            lines.append(
                f"attempt[{index:02d}]={outcome.label} {outcome.classification.name} "
                f"reached={outcome.reached.name}"
            )
        if decision is not None:
            lines.append(f"retry_reason={decision.reason}")
        path = self.root / "outcome"
        self._writer.makedirs(self.root)
        self._write(path, tuple(lines))
        return path

    def _write(self, path: Path, lines: Sequence[str]) -> None:
        self._writer.write_text(path, "".join(f"{line}\n" for line in lines))


class ShardSequencer:
    """Spec item 10: shard B is admissible only after shard A reaches AppReady.

    The sharded screen launcher starts both shards itself, and this phase wraps
    that launcher rather than rewriting it, so this class *verifies* the
    ordering rather than imposing it: it watches both shard logs, records when
    shard A reached AppReady, and reports a violation if shard B produced output
    first. ``hexctl screen`` refuses to certify a screen whose ordering was
    violated.

    Enforcing the order at launch time requires changing
    ``isaaclab/deploy/screen-stage2c-probe-sharded``, which is out of scope here
    and is recorded as such.
    """

    def __init__(self, *, app_ready_marker: str = APP_READY_MARKER) -> None:
        self._marker = app_ready_marker
        self._shard_a_ready_at: float | None = None
        self._shard_b_first_output_at: float | None = None

    @property
    def shard_a_ready_at(self) -> float | None:
        return self._shard_a_ready_at

    @property
    def shard_b_first_output_at(self) -> float | None:
        return self._shard_b_first_output_at

    def observe(self, shard: str, text: str, now: float) -> None:
        if not text:
            return
        if shard == "shard_a":
            if self._shard_a_ready_at is None and self._marker in text:
                self._shard_a_ready_at = now
        elif shard == "shard_b":
            if self._shard_b_first_output_at is None:
                self._shard_b_first_output_at = now
        else:
            raise ValueError(f"unknown shard: {shard}")

    @property
    def admissible(self) -> bool:
        """True only once shard A is proven ready before shard B did anything."""
        if self._shard_a_ready_at is None:
            return False
        if self._shard_b_first_output_at is None:
            return True
        return self._shard_b_first_output_at >= self._shard_a_ready_at

    def violation(self) -> str | None:
        if self._shard_a_ready_at is None:
            return "shard A never reached AppReady"
        if (
            self._shard_b_first_output_at is not None
            and self._shard_b_first_output_at < self._shard_a_ready_at
        ):
            return (
                f"shard B produced output at {self._shard_b_first_output_at:.3f}s, "
                f"before shard A reached AppReady at {self._shard_a_ready_at:.3f}s"
            )
        return None
