"""Compose the exact launcher invocation for one probe attempt.

Composition is pure: named experiment config + optional intervention deltas +
seed + label in, one argv out. It emits **only** the intervention overrides,
because the launcher pins the whole baseline itself and merges interventions
into it; re-emitting baseline keys would put the same key on the command line
twice and change nothing.

Because the launcher is the executor, this module refuses to compose at all
unless the named experiment config and the launcher's pinned bash arrays are
still one thing. The equality check is the same one
``isaaclab/tests/test_experiment_config_contract.py`` performs, run here at
composition time, so the YAML cannot silently drift away from the bash that
actually launches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from . import labels
from .configio import (
    ConfigDocument,
    ConfigError,
    load_config,
    same_scalar,
    value_category,
)


__all__ = [
    "CompositionError",
    "DEFAULT_LAUNCHER_COMMAND",
    "LauncherBaseline",
    "ProbeInvocation",
    "compose_probe",
    "launcher_path",
    "parse_launcher",
    "repo_root",
]


#: How the runbook spells the launcher on the Spark command line.
DEFAULT_LAUNCHER_COMMAND = "./isaaclab/deploy/probe-stage2c-single-current-best"
LAUNCHER_RELATIVE_PATH = "isaaclab/deploy/probe-stage2c-single-current-best"
SHARDED_SCREEN_RELATIVE_PATH = "isaaclab/deploy/screen-stage2c-probe-sharded"

MAX_SEED = 2147483647

#: ``run:`` key -> the pinned shell variable in the launcher that owns it.
RUN_KEY_TO_LAUNCHER_VARIABLE = {
    "task_id": "task_id",
    "experiment_name": "experiment",
    "parent_run": "parent_run",
    "parent_checkpoint": "parent_checkpoint",
    "parent_sha256": "parent_sha256",
    "num_envs": "num_envs",
    "max_iterations": "max_iterations",
}

# The launcher's own grammars, repeated so composition fails here rather than at
# exit 64 on the Spark.
_FINITE_NUMBER = re.compile(
    r"^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$"
)
_SHELL_INTEGER = re.compile(r"^[+-]?[0-9]+$")


class CompositionError(ValueError):
    """A run cannot be composed as specified, and must not be launched."""


def repo_root() -> Path:
    """The repository root, resolved from this file's own location."""
    return Path(__file__).resolve().parents[3]


def launcher_path(root: Path | None = None) -> Path:
    """The probe launcher inside this checkout."""
    return (root or repo_root()) / LAUNCHER_RELATIVE_PATH


def sharded_screen_path(root: Path | None = None) -> Path:
    """The formal sharded screen launcher inside this checkout."""
    return (root or repo_root()) / SHARDED_SCREEN_RELATIVE_PATH


def _shell_scalar(token: str):
    """Interpret one bash override value the way the launcher and Hydra would."""
    if token in ("true", "false"):
        return token == "true"
    if _SHELL_INTEGER.match(token):
        return int(token)
    if _FINITE_NUMBER.match(token):
        return float(token)
    return token


def _launcher_array(source: str, name: str) -> tuple[str, ...]:
    match = re.search(
        rf"(?m)^{re.escape(name)}=\(\n(?P<body>.*?)^\)$", source, flags=re.DOTALL
    )
    if match is None:
        raise CompositionError(f"launcher is missing the shell array {name}")
    return tuple(
        line.strip()
        for line in match.group("body").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _launcher_variable(source: str, name: str) -> str:
    match = re.search(rf"(?m)^{re.escape(name)}=(?P<value>\S+)$", source)
    if match is None:
        raise CompositionError(f"launcher is missing the pinned variable {name}")
    return match.group("value")


def _launcher_overrides(tokens: Sequence[str], prefix: str) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for token in tokens:
        if "=" not in token:
            raise CompositionError(f"malformed launcher override token: {token}")
        key, _, value = token.partition("=")
        if not key.startswith(f"{prefix}."):
            raise CompositionError(f"launcher override {key} is not a {prefix}.* key")
        bare = key[len(prefix) + 1 :]
        if bare in overrides:
            raise CompositionError(f"duplicate launcher override key: {key}")
        overrides[bare] = _shell_scalar(value)
    return overrides


@dataclass(frozen=True)
class LauncherBaseline:
    """What the launcher pins, read out of the launcher itself."""

    allowed_intervention_keys: tuple[str, ...]
    baseline_env: Mapping[str, object]
    fixed_agent: Mapping[str, object]
    run_identity: Mapping[str, object]

    def allows(self, dotted_key: str) -> bool:
        return dotted_key in self.allowed_intervention_keys

    @property
    def allowlist_order(self) -> dict[str, int]:
        return {key: index for index, key in enumerate(self.allowed_intervention_keys)}


def parse_launcher(source: str) -> LauncherBaseline:
    """Read the launcher's pinned baseline and intervention surface."""
    return LauncherBaseline(
        allowed_intervention_keys=_launcher_array(source, "allowed_intervention_keys"),
        baseline_env=_launcher_overrides(
            _launcher_array(source, "baseline_env_overrides"), "env"
        ),
        fixed_agent=_launcher_overrides(
            _launcher_array(source, "fixed_agent_overrides"), "agent"
        ),
        run_identity={
            key: _shell_scalar(_launcher_variable(source, variable))
            for key, variable in RUN_KEY_TO_LAUNCHER_VARIABLE.items()
        },
    )


def load_launcher(path: str | Path) -> LauncherBaseline:
    """Read the launcher text from disk and parse its pinned baseline."""
    resolved = Path(path)
    try:
        source = resolved.read_text(encoding="utf-8")
    except OSError as error:
        raise CompositionError(f"cannot read launcher {resolved}: {error}") from error
    return parse_launcher(source)


def assert_experiment_matches_launcher(
    experiment: ConfigDocument, baseline: LauncherBaseline
) -> None:
    """Refuse to compose when the named config and the executor disagree.

    Any divergence is reported, never reconciled: whichever side is wrong, a run
    launched from a config that does not describe what the launcher will do is
    unattributable evidence.
    """
    divergences: list[str] = []

    config_env = experiment.section("env")
    missing_in_launcher = sorted(set(config_env) - set(baseline.baseline_env))
    missing_in_config = sorted(set(baseline.baseline_env) - set(config_env))
    for key in missing_in_launcher:
        divergences.append(f"env.{key} is in the config but not in baseline_env_overrides")
    for key in missing_in_config:
        divergences.append(f"env.{key} is in baseline_env_overrides but not in the config")
    for key in sorted(set(config_env) & set(baseline.baseline_env)):
        if not same_scalar(config_env[key], baseline.baseline_env[key]):
            divergences.append(
                f"env.{key}: config {config_env[key]!r} "
                f"({value_category(config_env[key])}) != launcher "
                f"{baseline.baseline_env[key]!r} ({value_category(baseline.baseline_env[key])})"
            )

    config_agent = experiment.section("agent")
    for key in sorted(set(config_agent) ^ set(baseline.fixed_agent)):
        divergences.append(f"agent.{key} is pinned on only one side")
    for key in sorted(set(config_agent) & set(baseline.fixed_agent)):
        if not same_scalar(config_agent[key], baseline.fixed_agent[key]):
            divergences.append(
                f"agent.{key}: config {config_agent[key]!r} != launcher "
                f"{baseline.fixed_agent[key]!r}"
            )

    config_run = experiment.section("run")
    for key in sorted(set(config_run) ^ set(baseline.run_identity)):
        divergences.append(f"run.{key} is declared on only one side")
    for key in sorted(set(config_run) & set(baseline.run_identity)):
        if not same_scalar(config_run[key], baseline.run_identity[key]):
            divergences.append(
                f"run.{key}: config {config_run[key]!r} != launcher "
                f"{baseline.run_identity[key]!r}"
            )

    if divergences:
        raise CompositionError(
            f"{experiment.name} and the launcher baseline have diverged; refusing to "
            "compose:\n  " + "\n  ".join(divergences)
        )


@dataclass(frozen=True)
class ProbeInvocation:
    """The exact argv one probe attempt should be launched with."""

    launcher: str
    label: str
    seed: int
    overrides: tuple[str, ...]
    experiment: str = ""
    interventions: tuple[str, ...] = ()

    @property
    def argv(self) -> list[str]:
        return [self.launcher, self.label, str(self.seed), *self.overrides]

    def render(self) -> str:
        return " ".join(self.argv)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CompositionError(f"seed must be an integer: {seed!r}")
    if not 0 <= seed <= MAX_SEED:
        raise CompositionError(f"seed must be in [0, {MAX_SEED}]: {seed}")
    return seed


def _validate_intervention_token(dotted_key: str, token: str) -> None:
    if not _FINITE_NUMBER.match(token):
        raise CompositionError(
            f"intervention value must be a finite number: {dotted_key}={token}"
        )
    value = float(token)
    if value != value or value in (float("inf"), float("-inf")):
        raise CompositionError(
            f"intervention value must be finite: {dotted_key}={token}"
        )
    if "_reference_" in dotted_key and value <= 0.0:
        raise CompositionError(
            f"intervention reference must be strictly positive: {dotted_key}={token}"
        )


def _collect_interventions(
    documents: Sequence[ConfigDocument],
) -> list[tuple[str, str, str]]:
    """``(bare key, source token, origin file)`` for every intervention key."""
    collected: dict[str, tuple[str, str]] = {}
    ordered: list[tuple[str, str, str]] = []
    for document in documents:
        extra_sections = sorted(set(document.sections) - {"env"})
        if extra_sections:
            raise CompositionError(
                f"{document.name}: an intervention is an env-only delta; found section(s) "
                + ", ".join(extra_sections)
            )
        for key in document.section("env"):
            if key in collected:
                raise CompositionError(
                    f"{document.name}: duplicate intervention key {key} (already set by "
                    f"{collected[key][1]})"
                )
            token = document.token("env", key)
            collected[key] = (token, document.name)
            ordered.append((key, token, document.name))
    return ordered


def compose_probe(
    experiment: str | Path | ConfigDocument,
    interventions: Iterable[str | Path | ConfigDocument] = (),
    *,
    seed: int,
    label: str,
    launcher: str | Path | None = None,
    launcher_command: str | None = None,
) -> ProbeInvocation:
    """Compose the exact probe-launcher argv, or refuse and say why.

    ``interventions`` may be empty, which composes the zero-intervention control
    exactly as Probe20 was launched.
    """
    experiment_document = (
        experiment
        if isinstance(experiment, ConfigDocument)
        else load_config(experiment)
    )
    intervention_documents = [
        document if isinstance(document, ConfigDocument) else load_config(document)
        for document in interventions
    ]

    launcher_file = Path(launcher) if launcher is not None else launcher_path()
    baseline = load_launcher(launcher_file)

    if launcher_command is None:
        try:
            relative = launcher_file.resolve().relative_to(repo_root())
        except ValueError:
            launcher_command = DEFAULT_LAUNCHER_COMMAND
        else:
            launcher_command = f"./{relative.as_posix()}"

    try:
        labels.assert_launcher_safe(label)
    except labels.LabelError as error:
        raise CompositionError(str(error)) from error
    _validate_seed(seed)
    assert_experiment_matches_launcher(experiment_document, baseline)

    config_env = experiment_document.section("env")
    order = baseline.allowlist_order
    overrides: list[tuple[int, str]] = []
    for key, token, origin in _collect_interventions(intervention_documents):
        dotted_key = f"env.{key}"
        if key not in config_env:
            raise CompositionError(
                f"{origin}: {dotted_key} is not part of the named experiment baseline "
                f"{experiment_document.name}; an intervention must replace a control value"
            )
        if not baseline.allows(dotted_key):
            raise CompositionError(
                f"{origin}: {dotted_key} is not in the launcher's allowed_intervention_keys; "
                "the launcher would refuse this override"
            )
        _validate_intervention_token(dotted_key, token)
        overrides.append((order[dotted_key], f"{dotted_key}={token}"))

    # Stable order: the launcher's own allowlist declaration order, so the same
    # intervention set always composes to the same argv regardless of how the
    # config files happen to be written or supplied.
    ordered_overrides = tuple(token for _, token in sorted(overrides, key=lambda item: item[0]))

    return ProbeInvocation(
        launcher=launcher_command,
        label=label,
        seed=seed,
        overrides=ordered_overrides,
        experiment=experiment_document.name,
        interventions=tuple(document.name for document in intervention_documents),
    )


def parent_pin(experiment: str | Path | ConfigDocument):
    """The immutable parent named by an experiment config."""
    from .contract import ParentPin

    document = (
        experiment if isinstance(experiment, ConfigDocument) else load_config(experiment)
    )
    run = document.section("run")
    try:
        return ParentPin(
            run_dir=str(run["parent_run"]),
            checkpoint=str(run["parent_checkpoint"]),
            sha256=str(run["parent_sha256"]),
        )
    except KeyError as error:
        raise ConfigError(f"{document.name}: run section is missing {error}") from error
