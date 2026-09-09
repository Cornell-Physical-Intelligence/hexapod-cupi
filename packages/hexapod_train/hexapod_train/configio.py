"""Dependency-free reader for the restricted YAML subset used by ``configs/``.

The named configs under ``configs/experiment`` and ``configs/intervention`` are
deliberately tiny: two levels, flat scalars under a ``run:``/``agent:``/``env:``
section, and nothing else. This module reads exactly that subset and raises
:class:`ConfigError` on anything outside it rather than guessing, so a config
file can never mean one thing to PyYAML and another thing to this repository.

The scalar rules mirror ``isaaclab/tests/test_experiment_config_contract.py``
key for key: a plain scalar becomes a float only when it carries a decimal
point, which is also PyYAML's rule, so the two parsers cannot disagree about any
value in these files.

Both the parsed value and the exact source token are kept. Composition emits the
source token, so ``1.8`` is passed to the launcher as ``1.8`` and never as a
re-rendered float.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


__all__ = [
    "ALLOWED_SECTIONS",
    "ConfigDocument",
    "ConfigError",
    "load_config",
    "parse_config",
    "same_scalar",
    "scalar_from_token",
    "value_category",
]


#: The only section names this subset admits.
ALLOWED_SECTIONS = ("run", "agent", "env")

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
# PyYAML resolves a plain scalar as a float only when it carries a decimal
# point; these two patterns encode exactly that rule.
_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_FLOAT = re.compile(r"^[+-]?([0-9]+[.][0-9]*|[.][0-9]+)([eE][+-]?[0-9]+)?$")

Scalar = bool | int | float | str


class ConfigError(ValueError):
    """A config file left the restricted subset this repository allows."""


def scalar_from_token(token: str) -> Scalar:
    """Interpret one plain scalar from the restricted subset."""
    if token in ("true", "false"):
        return token == "true"
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    if _INTEGER.match(token):
        return int(token)
    if _FLOAT.match(token):
        return float(token)
    return token


def value_category(value: object) -> str:
    """``boolean``/``number``/``string`` — the comparison classes this subset has."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def same_scalar(left: object, right: object) -> bool:
    """Value-based equality: ``2.0e-5`` and ``0.00002`` are the same override."""
    if value_category(left) != value_category(right):
        return False
    if value_category(left) == "number":
        return float(left) == float(right)  # type: ignore[arg-type]
    return left == right


@dataclass(frozen=True)
class ConfigDocument:
    """One parsed config file: values, the source tokens, and its origin."""

    name: str
    sections: Mapping[str, Mapping[str, Scalar]]
    tokens: Mapping[str, Mapping[str, str]]

    def section(self, name: str) -> Mapping[str, Scalar]:
        try:
            return self.sections[name]
        except KeyError:
            raise ConfigError(f"{self.name}: missing required section {name}:") from None

    def optional_section(self, name: str) -> Mapping[str, Scalar]:
        return self.sections.get(name, {})

    def token(self, section: str, key: str) -> str:
        try:
            return self.tokens[section][key]
        except KeyError:
            raise ConfigError(f"{self.name}: no source token for {section}.{key}") from None


def parse_config(text: str, name: str = "<config>") -> ConfigDocument:
    """Parse the ``section:``/``  key: value`` subset these config files use."""
    sections: dict[str, dict[str, Scalar]] = {}
    tokens: dict[str, dict[str, str]] = {}
    current: str | None = None

    for number, raw_line in enumerate(text.splitlines(), 1):
        where = f"{name}:{number}"
        if "\t" in raw_line:
            raise ConfigError(f"{where}: tabs are not part of this subset")
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.split(" #", 1)[0].rstrip()
        if not line:
            continue
        stripped = line.strip()
        if stripped.startswith("-"):
            raise ConfigError(f"{where}: sequences are not part of this subset")
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in stripped:
            raise ConfigError(f"{where}: expected 'key: value'")
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if not _KEY.match(key):
            raise ConfigError(f"{where}: unsupported key name {key!r}")

        if indent == 0:
            if value:
                raise ConfigError(f"{where}: top-level scalars are not part of this subset")
            if key not in ALLOWED_SECTIONS:
                raise ConfigError(
                    f"{where}: unknown section {key}: "
                    f"(allowed: {', '.join(ALLOWED_SECTIONS)})"
                )
            if key in sections:
                raise ConfigError(f"{where}: duplicate section {key}:")
            current = key
            sections[key] = {}
            tokens[key] = {}
        elif indent == 2:
            if current is None:
                raise ConfigError(f"{where}: key outside any section")
            if not value:
                raise ConfigError(f"{where}: nested mappings are not part of this subset")
            if key in sections[current]:
                raise ConfigError(f"{where}: duplicate key {current}.{key}")
            scalar = scalar_from_token(value)
            if value_category(scalar) not in ("boolean", "number", "string"):
                raise ConfigError(f"{where}: unsupported value {value!r}")
            sections[current][key] = scalar
            tokens[current][key] = value
        else:
            raise ConfigError(f"{where}: unexpected indentation {indent} (expected 0 or 2)")

    if not sections:
        raise ConfigError(f"{name}: config declares no sections")
    return ConfigDocument(name=name, sections=sections, tokens=tokens)


def load_config(path: str | Path) -> ConfigDocument:
    """Read and parse one config file, naming errors after the file."""
    resolved = Path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read config {resolved}: {error}") from error
    return parse_config(text, resolved.name)
