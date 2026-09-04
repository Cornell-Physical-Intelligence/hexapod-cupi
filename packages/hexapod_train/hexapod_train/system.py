"""The one place this package touches the real machine.

Everything else in ``hexapod_train`` is pure and injectable; this module holds
the subprocess calls, the filesystem access, and the Spark layout constants, so
a unit test replaces exactly one object to run the whole supervisor against
fakes.

It is deliberately thin. It runs read-only probes, spawns the launcher as a
child with ``PYTHONUNBUFFERED=1``, tails logs, and writes diagnostics
atomically. It never removes a container, never deletes a cache or a Docker
volume, and never signals a process this package did not create: cleanup belongs
to the launcher's own exact-container-ID path.

Implemented to spec; not yet validated in vivo on the Spark.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from . import gates


__all__ = [
    "CommandResult",
    "DiagnosticsCapture",
    "DirectoryWriter",
    "GPU_LOCK",
    "ISAAC_LAB_DIR",
    "PROBE_BATCH_ROOT",
    "SparkLayoutError",
    "SystemInterface",
    "TRAINING_UNIT",
    "WORKSPACE_HOST",
]


WORKSPACE_HOST = Path("/home/orionh/HEXAPOD")
ISAAC_LAB_DIR = Path("/home/orionh/IsaacLab")
GPU_LOCK = Path("/tmp/hexapod-isaac-gpu.lock")
TRAINING_UNIT = "hexapod-rl-training.service"
PROBE_BATCH_ROOT = WORKSPACE_HOST / "isaaclab" / "logs" / "probe_batches" / "stage2c"

_READ_CHUNK = 1 << 20
_PROBE_TIMEOUT_SECONDS = 20


class SparkLayoutError(RuntimeError):
    """This host is not the Spark, so no launch action may be attempted."""


@dataclass(frozen=True)
class CommandResult:
    """One completed read-only probe."""

    argv: tuple[str, ...]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    available: bool = True

    @property
    def ok(self) -> bool:
        return self.available and self.returncode == 0


class DirectoryWriter:
    """Atomic ``ArtifactWriter``: write a temporary sibling, then rename."""

    def makedirs(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def write_text(self, path: Path, text: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


class SystemInterface:
    """Read-only probes, child-process launch, log tailing, atomic writes."""

    def __init__(
        self,
        *,
        workspace_host: Path = WORKSPACE_HOST,
        isaac_lab_dir: Path = ISAAC_LAB_DIR,
        gpu_lock: Path = GPU_LOCK,
        training_unit: str = TRAINING_UNIT,
    ) -> None:
        self.workspace_host = Path(workspace_host)
        self.isaac_lab_dir = Path(isaac_lab_dir)
        self.gpu_lock = Path(gpu_lock)
        self.training_unit = training_unit
        self.writer = DirectoryWriter()

    # -- clock -----------------------------------------------------------

    def now(self) -> float:
        """Monotonic seconds; deadlines must not move when the wall clock does."""
        return time.monotonic()

    def utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def utcnow_text(self) -> str:
        return self.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    # -- filesystem ------------------------------------------------------

    def exists(self, path: str | Path) -> bool:
        return Path(path).exists()

    def is_dir(self, path: str | Path) -> bool:
        return Path(path).is_dir()

    def read_chunks(self, path: str | Path) -> Iterator[bytes]:
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(_READ_CHUNK)
                if not chunk:
                    return
                yield chunk

    def read_new(self, path: str | Path, offset: int) -> tuple[str, int]:
        """Everything appended to ``path`` since ``offset``, plus the new offset."""
        target = Path(path)
        if not target.exists():
            return "", offset
        with target.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
            return data.decode("utf-8", "replace"), handle.tell()

    def tail_text(self, path: str | Path, max_bytes: int = 65536) -> str:
        target = Path(path)
        if not target.is_file():
            return ""
        size = target.stat().st_size
        with target.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", "replace")

    def inventory(self, path: str | Path) -> tuple[str, ...]:
        target = Path(path)
        if not target.is_dir():
            return ()
        return tuple(
            f"{entry.name} {entry.stat().st_size if entry.is_file() else 'dir'}"
            for entry in sorted(target.iterdir())
        )

    # -- processes -------------------------------------------------------

    def child_env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """The child environment, with unbuffered Python forced on (spec item 4)."""
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        if extra:
            environment.update(extra)
        return environment

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = _PROBE_TIMEOUT_SECONDS,
        cwd: str | Path | None = None,
    ) -> CommandResult:
        """Run one read-only probe; a missing tool is reported, not raised."""
        try:
            completed = subprocess.run(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=None if cwd is None else str(cwd),
                env=self.child_env(),
            )
        except FileNotFoundError:
            return CommandResult(tuple(argv), None, "", "command not found", available=False)
        except subprocess.TimeoutExpired:
            return CommandResult(tuple(argv), None, "", "probe timed out", available=False)
        except OSError as error:  # pragma: no cover - platform dependent
            return CommandResult(tuple(argv), None, "", str(error), available=False)
        return CommandResult(
            tuple(argv), completed.returncode, completed.stdout, completed.stderr
        )

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path,
        stdout_path: str | Path,
        extra_env: Mapping[str, str] | None = None,
    ) -> subprocess.Popen:
        """Start the launcher as an owned child, output redirected to a file."""
        target = Path(stdout_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = target.open("wb")
        return subprocess.Popen(  # noqa: S603 - argv is composed, never a shell string
            list(argv),
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=self.child_env(extra_env),
            start_new_session=False,
        )

    # -- read-only machine probes ---------------------------------------

    def process_table(self) -> CommandResult:
        return self.run(["ps", "-eo", "pid=,ppid=,args="])

    def gpu_compute_apps(self) -> CommandResult:
        return self.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )

    def docker_names(self) -> CommandResult:
        return self.run(["docker", "ps", "--format", "{{.Names}}"])

    def service_status(self) -> CommandResult:
        return self.run(["systemctl", "is-active", "--quiet", self.training_unit])

    def gpu_lock_free(self) -> bool | None:
        """Probe the shared advisory lock without holding it."""
        result = self.run(["flock", "-n", str(self.gpu_lock), "true"])
        if not result.available:
            return None
        return result.returncode == 0

    def snapshot(self) -> gates.ResourceSnapshot:
        """One resource snapshot for the gates. Read-only, no side effects."""
        processes = self.process_table()
        gpu = self.gpu_compute_apps()
        containers = self.docker_names()
        service = self.service_status()
        notes: list[str] = []
        for name, result in (
            ("ps", processes),
            ("nvidia-smi", gpu),
            ("docker", containers),
            ("systemctl", service),
        ):
            if not result.available:
                notes.append(f"{name} was unavailable: {result.stderr.strip()}")
        return gates.snapshot_from_text(
            process_table=processes.stdout if processes.available else "",
            gpu_compute_apps=gpu.stdout if gpu.available else "",
            docker_names=containers.stdout if containers.available else "",
            service_status=service.returncode if service.available else None,
            gpu_lock_free=self.gpu_lock_free(),
            observed_at_utc=self.utcnow_text(),
            notes=notes,
        )

    # -- host layout -----------------------------------------------------

    def spark_layout_problems(self) -> tuple[str, ...]:
        """Everything about this host that forbids a launch action."""
        problems: list[str] = []
        for path, label in (
            (self.workspace_host, "Spark workspace mirror"),
            (self.isaac_lab_dir, "Isaac Lab checkout"),
        ):
            if not path.is_dir():
                problems.append(f"{label} is missing: {path}")
        for tool in ("docker", "flock", "systemctl"):
            if shutil.which(tool) is None:
                problems.append(f"required command is unavailable: {tool}")
        return tuple(problems)

    def require_spark_layout(self) -> None:
        """Fail closed, with the reasons, on any host that is not the Spark."""
        problems = self.spark_layout_problems()
        if problems:
            raise SparkLayoutError(
                "this host is not the Spark training host, so no launch action is "
                "possible here:\n  " + "\n  ".join(problems) + "\n"
                "Compose the invocation here and run it from the Spark "
                "(see docs/OPERATIONS.md §9)."
            )


class DiagnosticsCapture:
    """Spec item 6: one atomic evidence bundle per no-progress trigger.

    Every command is read-only. Whatever is unavailable is recorded as
    unavailable rather than omitted, because a missing probe and a clean probe
    must never look the same afterwards.
    """

    def __init__(
        self,
        system: SystemInterface,
        attempt_dir: str | Path,
        *,
        container_name: str = "",
        child_pid: int | None = None,
        log_path: str | Path | None = None,
        run_dir: str | Path | None = None,
        parent_path: str | Path | None = None,
        parent_sha256: str = "",
    ) -> None:
        self._system = system
        self._attempt_dir = Path(attempt_dir)
        self._container_name = container_name
        self._child_pid = child_pid
        self._log_path = None if log_path is None else Path(log_path)
        self._run_dir = None if run_dir is None else Path(run_dir)
        self._parent_path = None if parent_path is None else Path(parent_path)
        self._parent_sha256 = parent_sha256
        self._sequence = 0

    def __call__(self, reason: str, elapsed_seconds: float) -> str:
        directory = self._attempt_dir / "diagnostics" / f"{self._sequence:02d}"
        self._sequence += 1
        writer = self._system.writer
        writer.makedirs(directory)

        def record(name: str, text: str) -> None:
            writer.write_text(directory / name, text if text.endswith("\n") else text + "\n")

        def record_command(name: str, argv: Sequence[str]) -> None:
            result = self._system.run(argv)
            body = (
                f"argv={' '.join(argv)}\n"
                f"available={'true' if result.available else 'false'}\n"
                f"returncode={result.returncode}\n"
                f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n"
            )
            record(name, body)

        record(
            "capture.meta",
            "\n".join(
                (
                    "schema_version=1",
                    f"reason={reason}",
                    f"elapsed_seconds={elapsed_seconds:.3f}",
                    f"captured_at_utc={self._system.utcnow_text()}",
                    f"container_name={self._container_name or 'not-observed'}",
                    f"child_pid={self._child_pid if self._child_pid is not None else 'unknown'}",
                )
            ),
        )

        if self._container_name:
            record_command(
                "container.inspect",
                [
                    "docker",
                    "container",
                    "inspect",
                    "--format",
                    "{{.Id}}|{{.Name}}|{{.State.Running}}|{{.State.Status}}|{{.State.ExitCode}}",
                    self._container_name,
                ],
            )
            record_command(
                "container.logs",
                ["docker", "container", "logs", "--tail", "200", self._container_name],
            )
            record_command("docker.top", ["docker", "top", self._container_name])
        record_command("nvidia-smi", ["nvidia-smi"])
        record_command(
            "nvidia-smi.compute-apps",
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
        )
        record_command("process.table", ["ps", "-eo", "pid=,ppid=,args="])
        record_command("oom.state", ["journalctl", "-k", "-n", "200", "--no-pager"])

        if self._child_pid is not None:
            for name in ("status", "wchan"):
                source = Path(f"/proc/{self._child_pid}/{name}")
                try:
                    body = source.read_text(encoding="utf-8", errors="replace")
                except OSError as error:
                    body = f"unavailable: {error}"
                record(f"proc.{name}", body)

        if self._log_path is not None:
            size = self._log_path.stat().st_size if self._log_path.is_file() else -1
            tail = self._system.tail_text(self._log_path, 8192)
            last_line = tail.strip().splitlines()[-1] if tail.strip() else ""
            record(
                "log.state",
                "\n".join(
                    (
                        f"path={self._log_path}",
                        f"size_bytes={size}",
                        f"last_line={last_line}",
                    )
                ),
            )
            record("log.tail", tail or "(empty)")

        if self._run_dir is not None:
            entries = self._system.inventory(self._run_dir)
            record(
                "run.inventory",
                "\n".join(
                    (
                        f"path={self._run_dir}",
                        f"exists={'true' if self._run_dir.is_dir() else 'false'}",
                        *entries,
                    )
                ),
            )

        if self._parent_path is not None:
            from .contract import verify_parent

            try:
                verification = verify_parent(
                    str(self._parent_path), self._parent_sha256, self._system.read_chunks
                )
                body = "\n".join(
                    (
                        f"path={verification.path}",
                        f"expected={verification.expected_sha256}",
                        f"actual={verification.actual_sha256}",
                        f"matches={'true' if verification.matches else 'false'}",
                    )
                )
            except OSError as error:
                body = f"path={self._parent_path}\nunavailable: {error}"
            record("parent.sha256", body)

        return str(directory)
