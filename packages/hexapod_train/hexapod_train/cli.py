"""``hexctl`` — compose, check, and supervise one launcher invocation.

Four subcommands:

``compose``
    Print the exact launcher argv for a named experiment plus intervention
    deltas. No side effects, works anywhere, and refuses if the named config and
    the launcher's pinned baseline have diverged.
``doctor``
    Run the resource gates read-only and print the report.
``probe``
    Gate, compose, launch the probe launcher as a supervised child, and record
    the attempt outcome and the bounded retry decision.
``screen``
    The same wrapper around the formal sharded screen, watching the shard logs
    so a screen whose shard ordering was violated is never certified.

``probe`` and ``screen`` fail closed on any host that is not the Spark: they
refuse with the reasons rather than attempting a launch or raising a traceback.

Implemented to spec; not yet validated in vivo on the Spark.
"""

from __future__ import annotations

import argparse
import shlex
import signal
import sys
from pathlib import Path
from typing import Sequence

from . import compose as compose_module
from . import gates, labels, supervisor
from .configio import ConfigError, load_config
from .compose import CompositionError, ProbeInvocation
from .contract import AttemptOutcome, Milestone, Termination, verify_parent
from .supervisor import (
    AttemptRecorder,
    DEFAULT_DEADLINES,
    MilestoneMachine,
    ShardSequencer,
    retry_decision,
)
from .system import DiagnosticsCapture, SparkLayoutError, SystemInterface


__all__ = ["main"]


EX_OK = 0
EX_USAGE = 64
EX_DATAERR = 65
EX_UNAVAILABLE = 69
EX_SOFTWARE = 70
EX_TEMPFAIL = 75

_POLL_SECONDS = 1.0
_GRACE_SECONDS = 60.0


# ---------------------------------------------------------------- composing


def _compose_from_arguments(arguments: argparse.Namespace) -> ProbeInvocation:
    return compose_module.compose_probe(
        arguments.experiment,
        arguments.intervention or (),
        seed=arguments.seed,
        label=arguments.label,
        launcher=arguments.launcher,
    )


def _command_compose(arguments: argparse.Namespace) -> int:
    invocation = _compose_from_arguments(arguments)
    if arguments.json:
        import json

        print(json.dumps(invocation.argv))
    else:
        print(shlex.join(invocation.argv))
    return EX_OK


# ------------------------------------------------------------------ gating


def _command_doctor(arguments: argparse.Namespace) -> int:
    system = SystemInterface()
    system.require_spark_layout()
    report = gates.evaluate(system.snapshot())
    print(report.render())
    return EX_OK if report.clear else EX_TEMPFAIL


# ------------------------------------------------------------- supervising


def _artifact_facts(system: SystemInterface, artifact_dir: Path) -> dict[str, object]:
    """Read the launcher's own atomic artifacts back."""

    def first_line(name: str) -> str:
        path = artifact_dir / name
        if not path.is_file():
            return ""
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return lines[0].strip() if lines else ""

    run_path = first_line("child.run_path")
    checkpoint_exists = (artifact_dir / "child.checkpoints.sha256").is_file()
    if run_path and not checkpoint_exists:
        checkpoint_exists = any(Path(run_path).glob("model_*.pt")) if Path(run_path).is_dir() else False
    return {
        "docker_status": first_line("docker.status") or "not-started",
        "cleanup_status": first_line("cleanup.status") or "not-required",
        "overall_status": first_line("overall.status"),
        "run_path": run_path,
        "run_directory_exists": bool(run_path) and Path(run_path).is_dir(),
        "checkpoint_exists": checkpoint_exists,
        "signal_received": (artifact_dir / "signal.received").is_file(),
    }


def _run_attempt(
    system: SystemInterface,
    *,
    invocation: ProbeInvocation,
    label: str,
    attempt_index: int,
    logs_root: Path,
    recorder: AttemptRecorder,
    parent_path: Path,
    parent_sha256: str,
    deadlines: supervisor.MilestoneDeadlines,
) -> AttemptOutcome:
    """Launch one attempt under the milestone machine and classify it.

    The launcher is spawned as an owned child and is the only process this
    function ever signals. Nothing here removes a container, deletes a cache, or
    touches an unrelated workload.
    """
    artifact_dir = logs_root / label
    launcher_log = logs_root / f"{label}.launcher.log"
    train_log = artifact_dir / "train.log"
    attempt_dir = recorder.attempt_directory(attempt_index)
    system.writer.makedirs(attempt_dir)

    if launcher_log.exists() or artifact_dir.exists():
        raise CompositionError(
            f"attempt label {label} already has artifacts under {logs_root}; "
            "labels are immutable and never reused"
        )

    argv = list(invocation.argv)
    started_at_utc = system.utcnow_text()
    process = system.spawn(argv, cwd=system.workspace_host, stdout_path=launcher_log)
    started_at = system.now()
    container_name = f"hexapod-stage2c-single-{label}-{process.pid}"
    capture = DiagnosticsCapture(
        system,
        attempt_dir,
        container_name=container_name,
        child_pid=process.pid,
        log_path=train_log,
        run_dir=artifact_dir,
        parent_path=parent_path,
        parent_sha256=parent_sha256,
    )
    machine = MilestoneMachine(
        deadlines=deadlines, capture_diagnostics=capture, started_at=started_at
    )

    offsets = {launcher_log: 0, train_log: 0}
    notes: list[str] = [f"container_name={container_name}", f"launcher_pid={process.pid}"]
    user_signal: str | None = None
    rechecked_after_container = False
    rechecked_before_ppo = False
    stall: supervisor.Stall | None = None

    def recheck(reason: str) -> None:
        report = gates.evaluate(system.snapshot())
        notes.append(f"recheck[{reason}]=clear" if report.clear else f"recheck[{reason}]=contended")
        for failure in report.failures:
            notes.append(f"recheck[{reason}] {failure.name}: {'; '.join(failure.evidence)}")

    try:
        while True:
            now = system.now()
            for path in (launcher_log, train_log):
                text, offsets[path] = system.read_new(path, offsets[path])
                if text:
                    machine.observe(text, now)

            # Spec item 3: the prelaunch gate cannot see a producer that spawns
            # its GPU child afterwards, so recheck once the container exists and
            # again before PPO starts.
            if not rechecked_after_container and (artifact_dir / "docker.pid").is_file():
                rechecked_after_container = True
                recheck("after-container-creation")
            if not rechecked_before_ppo and machine.state >= Milestone.APP_READY:
                rechecked_before_ppo = True
                recheck("before-ppo")

            stall = machine.tick(now)
            if stall is not None:
                notes.append(f"supervisor_stall={stall.reason}")
                break
            if process.poll() is not None:
                break
            system.sleep(_POLL_SECONDS)
    except KeyboardInterrupt:
        user_signal = "INT"
        notes.append("operator interrupted the supervisor")

    if process.poll() is None:
        # The launcher's own INT handler forwards to its owned timeout child and
        # then runs the exact-container-ID cleanup. Never SIGKILL, never docker rm.
        process.send_signal(signal.SIGINT)
        deadline = system.now() + _GRACE_SECONDS
        while process.poll() is None and system.now() < deadline:
            system.sleep(_POLL_SECONDS)
        if process.poll() is None:
            notes.append("launcher did not exit within the grace period; left untouched")

    exit_status = process.poll()
    if exit_status is not None and exit_status == 0:
        machine.complete(system.now())

    for path in (launcher_log, train_log):
        text, offsets[path] = system.read_new(path, offsets[path])
        if text:
            machine.observe(text, system.now())

    facts = _artifact_facts(system, artifact_dir)
    if facts["signal_received"] and user_signal is None and stall is None:
        user_signal = "TERM"
    docker_status = str(facts["docker_status"])
    cleanup_required = docker_status not in ("0", "not-started")

    parent_unchanged = False
    if parent_path.is_file():
        parent_unchanged = verify_parent(
            str(parent_path), parent_sha256, system.read_chunks
        ).matches

    classification = supervisor.classify_outcome(
        reached=machine.state,
        stall=stall,
        exit_status=exit_status,
        user_signal=user_signal,
        docker_status=int(docker_status) if docker_status.isdigit() else docker_status,
        cleanup_required=cleanup_required,
        cleanup_status=str(facts["cleanup_status"]),
        completed=exit_status == 0,
    )

    outcome = AttemptOutcome(
        label=label,
        attempt_index=attempt_index,
        classification=classification,
        reached=machine.state,
        events=machine.events,
        exit_status=exit_status,
        user_signal=user_signal,
        docker_status=docker_status,
        cleanup_required=cleanup_required,
        cleanup_status=str(facts["cleanup_status"]),
        run_directory_exists=bool(facts["run_directory_exists"]),
        checkpoint_exists=bool(facts["checkpoint_exists"]),
        parent_unchanged=parent_unchanged,
        traceback_seen=machine.traceback_seen,
        oom_seen=machine.oom_seen,
        diagnostics=machine.captures,
        started_at_utc=started_at_utc,
        finished_at_utc=system.utcnow_text(),
        notes=tuple(notes),
    )
    recorder.record_attempt(outcome)
    return outcome


def _command_probe(arguments: argparse.Namespace) -> int:
    system = SystemInterface()
    system.require_spark_layout()

    logs_root = Path(arguments.logs_root)
    invocation = _compose_from_arguments(arguments)
    pin = compose_module.parent_pin(arguments.experiment)
    parent_path = (
        system.workspace_host
        / "isaaclab"
        / "logs"
        / "rsl_rl"
        / str(load_config(arguments.experiment).section("run")["experiment_name"])
        / pin.run_dir
        / pin.checkpoint
    )

    labels.assert_available(arguments.label, logs_root, lambda path: path.exists())

    report = gates.evaluate(system.snapshot())
    print(report.render())
    if not report.clear:
        print(
            "hexctl: resource gates are not clear; refusing to launch. Nothing "
            "belonging to another workload was signalled or modified.",
            file=sys.stderr,
        )
        return EX_TEMPFAIL

    if arguments.dry_run:
        print(shlex.join(invocation.argv))
        return EX_OK

    supervision_root = (
        Path(arguments.supervision_root)
        if arguments.supervision_root
        else logs_root / f"{arguments.label}.supervision"
    )
    recorder = AttemptRecorder(supervision_root, system.writer)
    outcomes: list[AttemptOutcome] = []

    outcome = _run_attempt(
        system,
        invocation=invocation,
        label=arguments.label,
        attempt_index=0,
        logs_root=logs_root,
        recorder=recorder,
        parent_path=parent_path,
        parent_sha256=pin.sha256,
        deadlines=DEFAULT_DEADLINES,
    )
    outcomes.append(outcome)

    decision = retry_decision(outcome, moment=system.utcnow())
    if arguments.no_retry and decision.retry:
        decision = supervisor.RetryDecision(
            retry=False,
            reason="--no-retry was requested by the operator",
            next_label=None,
            refusals=("--no-retry",),
        )
    recorder.record_retry_decision(decision)
    print(f"attempt 00: {outcome.classification.name} ({outcome.reached.name})")
    print(f"retry: {'yes' if decision.retry else 'no'} — {decision.reason}")

    if decision.retry and decision.next_label:
        retry_report = gates.evaluate(system.snapshot())
        if not retry_report.clear:
            print(retry_report.render())
            print(
                "hexctl: the retry is permitted but the machine is no longer clear; "
                "not launching.",
                file=sys.stderr,
            )
        else:
            labels.assert_available(
                decision.next_label,
                logs_root,
                lambda path: path.exists(),
                known_labels=[outcome.label for outcome in outcomes],
            )
            retry_invocation = compose_module.compose_probe(
                arguments.experiment,
                arguments.intervention or (),
                seed=arguments.seed,
                label=decision.next_label,
                launcher=arguments.launcher,
            )
            outcomes.append(
                _run_attempt(
                    system,
                    invocation=retry_invocation,
                    label=decision.next_label,
                    attempt_index=1,
                    logs_root=logs_root,
                    recorder=recorder,
                    parent_path=parent_path,
                    parent_sha256=pin.sha256,
                    deadlines=DEFAULT_DEADLINES,
                )
            )
            print(
                f"attempt 01: {outcomes[-1].classification.name} "
                f"({outcomes[-1].reached.name})"
            )

    recorder.record_root_outcome(outcomes, decision)
    final = outcomes[-1]
    print(f"supervision artifacts: {supervision_root}")
    return EX_OK if final.classification is Termination.SUCCESS else EX_DATAERR


# ------------------------------------------------------------------ screen


def _command_screen(arguments: argparse.Namespace) -> int:
    system = SystemInterface()
    system.require_spark_layout()

    launcher = arguments.launcher or compose_module.sharded_screen_path()
    try:
        relative = Path(launcher).resolve().relative_to(compose_module.repo_root())
        launcher_command = f"./{relative.as_posix()}"
    except ValueError:
        launcher_command = str(launcher)

    for name in (
        arguments.batch_label,
        arguments.run_name,
        arguments.parent_run,
        arguments.parent_checkpoint,
        arguments.output_label,
    ):
        labels.assert_launcher_safe(name)
    if len(arguments.parent_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in arguments.parent_sha256
    ):
        raise CompositionError("parent SHA-256 must be 64 lowercase hex characters")
    if not arguments.checkpoint:
        raise CompositionError("at least one checkpoint index is required")

    argv = [
        launcher_command,
        arguments.batch_label,
        arguments.run_name,
        arguments.parent_run,
        arguments.parent_checkpoint,
        arguments.parent_sha256,
        arguments.output_label,
        *(str(number) for number in arguments.checkpoint),
    ]
    if arguments.dry_run:
        print(shlex.join(argv))
        return EX_OK

    report = gates.evaluate(system.snapshot())
    print(report.render())
    if not report.clear:
        print("hexctl: resource gates are not clear; refusing to launch.", file=sys.stderr)
        return EX_TEMPFAIL

    evaluation_dir = (
        Path(arguments.logs_root)
        / arguments.batch_label
        / "evaluation"
        / arguments.output_label
    )
    shard_logs = {
        "shard_a": evaluation_dir / "shard_a.log",
        "shard_b": evaluation_dir / "shard_b.log",
    }
    screen_log = Path(arguments.logs_root) / f"{arguments.batch_label}.{arguments.output_label}.screen.log"
    if screen_log.exists():
        raise CompositionError(f"refusing to reuse screen log: {screen_log}")

    sequencer = ShardSequencer()
    process = system.spawn(argv, cwd=system.workspace_host, stdout_path=screen_log)
    started_at = system.now()
    offsets = {path: 0 for path in shard_logs.values()}
    while True:
        now = system.now()
        for shard, path in shard_logs.items():
            text, offsets[path] = system.read_new(path, offsets[path])
            if text:
                sequencer.observe(shard, text, now - started_at)
        if process.poll() is not None:
            break
        system.sleep(_POLL_SECONDS)

    status = process.poll()
    violation = sequencer.violation()
    print(f"screen exit status: {status}")
    print(
        "shard ordering: "
        + ("admissible" if sequencer.admissible else f"VIOLATED — {violation}")
    )
    if status != 0:
        return status if isinstance(status, int) and status else EX_SOFTWARE
    if not sequencer.admissible:
        print(
            "hexctl: shard B was not proven to start after shard A reached AppReady; "
            "this screen is not certified.",
            file=sys.stderr,
        )
        return EX_DATAERR
    return EX_OK


# -------------------------------------------------------------------- CLI


def _add_compose_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True, help="named experiment config")
    parser.add_argument(
        "--intervention",
        action="append",
        default=[],
        help="intervention delta config; repeatable",
    )
    parser.add_argument("--seed", type=int, required=True, help="training seed")
    parser.add_argument("--label", required=True, help="immutable batch label")
    parser.add_argument(
        "--launcher", default=None, help="launcher to compose for (defaults to this checkout)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hexctl",
        description=(
            "Compose, gate, and supervise the hardened Stage2C launchers. The "
            "launchers remain the executors; hexctl wraps them."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compose_parser = subparsers.add_parser(
        "compose", help="print the exact launcher argv; no side effects"
    )
    _add_compose_arguments(compose_parser)
    compose_parser.add_argument("--json", action="store_true", help="print argv as JSON")
    compose_parser.set_defaults(handler=_command_compose)

    doctor_parser = subparsers.add_parser(
        "doctor", help="run the resource gates read-only and print the report"
    )
    doctor_parser.set_defaults(handler=_command_doctor)

    probe_parser = subparsers.add_parser(
        "probe", help="gate, compose, and supervise one probe attempt"
    )
    _add_compose_arguments(probe_parser)
    probe_parser.add_argument(
        "--logs-root",
        default=str(Path("/home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c")),
        help="probe-batch artifact root on the Spark",
    )
    probe_parser.add_argument(
        "--supervision-root", default=None, help="where attempts/ and retry.decision are written"
    )
    probe_parser.add_argument(
        "--no-retry", action="store_true", help="never retry, even when the rule permits it"
    )
    probe_parser.add_argument(
        "--dry-run", action="store_true", help="gate and compose, but do not launch"
    )
    probe_parser.set_defaults(handler=_command_probe)

    screen_parser = subparsers.add_parser(
        "screen", help="supervise the formal sharded screen"
    )
    screen_parser.add_argument("--batch-label", required=True)
    screen_parser.add_argument("--run-name", required=True)
    screen_parser.add_argument("--parent-run", required=True)
    screen_parser.add_argument("--parent-checkpoint", required=True)
    screen_parser.add_argument("--parent-sha256", required=True)
    screen_parser.add_argument("--output-label", required=True)
    screen_parser.add_argument(
        "--checkpoint", type=int, action="append", default=[], help="checkpoint index; repeatable"
    )
    screen_parser.add_argument(
        "--logs-root",
        default=str(Path("/home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c")),
    )
    screen_parser.add_argument("--launcher", default=None)
    screen_parser.add_argument(
        "--dry-run", action="store_true", help="print the launcher argv without launching"
    )
    screen_parser.set_defaults(handler=_command_screen)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(arguments.handler(arguments))
    except SparkLayoutError as error:
        print(f"hexctl: {error}", file=sys.stderr)
        return EX_UNAVAILABLE
    except (CompositionError, ConfigError, labels.LabelError) as error:
        print(f"hexctl: {error}", file=sys.stderr)
        return EX_USAGE
    except KeyboardInterrupt:  # pragma: no cover - operator action
        print("hexctl: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
