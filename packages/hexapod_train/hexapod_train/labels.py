"""Label and attempt grammar for probe batches.

A batch label is ``<slug>_<UTCstamp>Z``; a bounded retry of that attempt is
``<slug>_retryN_<UTCstamp>Z`` with a fresh stamp. The launcher accepts a label
only as one safe directory component of at most 96 characters, so this module
enforces that grammar too — a label this module produces is always a label the
launcher will accept.

Labels are immutable. A failed attempt keeps its label and its artifact
directory forever, so allocation refuses any label whose batch directory already
exists and any label already known to this run. The existence check is injected,
which keeps the grammar itself pure string logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


__all__ = [
    "LabelCollisionError",
    "LabelError",
    "MAX_LABEL_LENGTH",
    "ParsedLabel",
    "STAMP_FORMAT",
    "allocate",
    "assert_available",
    "assert_launcher_safe",
    "batch_directory",
    "build_label",
    "format_stamp",
    "parse_label",
    "retry_label",
]


STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
MAX_LABEL_LENGTH = 96

# The launcher's own batch-label grammar, repeated here so composition fails on
# this side rather than at exit 64 on the Spark.
_LAUNCHER_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LABEL = re.compile(
    r"^(?P<slug>[A-Za-z0-9][A-Za-z0-9._-]*?)"
    r"(?:_retry(?P<attempt>[1-9][0-9]*))?"
    r"_(?P<stamp>[0-9]{8}T[0-9]{6}Z)$"
)


class LabelError(ValueError):
    """A label is outside the batch-label grammar."""


class LabelCollisionError(LabelError):
    """A label is already taken; attempt labels are never reused."""


@dataclass(frozen=True)
class ParsedLabel:
    """One decomposed batch label."""

    text: str
    slug: str
    attempt: int
    stamp: datetime

    @property
    def is_retry(self) -> bool:
        return self.attempt > 0


def assert_launcher_safe(label: str) -> str:
    """Refuse anything the launcher itself would refuse as a batch label."""
    if not isinstance(label, str) or not label:
        raise LabelError("label must be a non-empty string")
    if label in (".", ".."):
        raise LabelError("label must not be a relative directory component")
    if len(label) > MAX_LABEL_LENGTH:
        raise LabelError(
            f"label must be at most {MAX_LABEL_LENGTH} characters: {label!r} is {len(label)}"
        )
    if not _LAUNCHER_SAFE.match(label):
        raise LabelError(
            "label must be one safe directory component matching "
            f"[A-Za-z0-9][A-Za-z0-9._-]*: {label!r}"
        )
    return label


def format_stamp(moment: datetime) -> str:
    """Render one UTC stamp; a naive moment is read as UTC."""
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime(STAMP_FORMAT)


def parse_label(label: str) -> ParsedLabel:
    """Decompose ``<slug>[_retryN]_<UTCstamp>Z`` or raise :class:`LabelError`."""
    assert_launcher_safe(label)
    match = _LABEL.match(label)
    if match is None:
        raise LabelError(
            f"label {label!r} is not <slug>[_retryN]_<YYYYMMDDTHHMMSSZ>"
        )
    try:
        stamp = datetime.strptime(match.group("stamp"), STAMP_FORMAT).replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise LabelError(f"label {label!r} has an impossible UTC stamp: {error}") from error
    attempt_group = match.group("attempt")
    return ParsedLabel(
        text=label,
        slug=match.group("slug"),
        attempt=int(attempt_group) if attempt_group else 0,
        stamp=stamp,
    )


def build_label(slug: str, moment: datetime, attempt: int = 0) -> str:
    """Generate ``<slug>[_retryN]_<UTCstamp>Z`` for one attempt."""
    if not _SLUG.match(slug or ""):
        raise LabelError(
            f"slug must match [A-Za-z0-9][A-Za-z0-9._-]*: {slug!r}"
        )
    if slug.endswith("_"):
        raise LabelError(f"slug must not end with an underscore: {slug!r}")
    if attempt < 0:
        raise LabelError(f"attempt must not be negative: {attempt}")
    stamp = format_stamp(moment)
    label = f"{slug}_{stamp}" if attempt == 0 else f"{slug}_retry{attempt}_{stamp}"
    assert_launcher_safe(label)
    # Round-trip: anything this function emits must parse back to what went in.
    parsed = parse_label(label)
    if parsed.slug != slug or parsed.attempt != attempt:
        raise LabelError(f"generated label {label!r} does not round-trip")
    return label


def retry_label(label: str | ParsedLabel, moment: datetime) -> str:
    """The next attempt label for ``label``, under a fresh immutable stamp."""
    parsed = label if isinstance(label, ParsedLabel) else parse_label(label)
    return build_label(parsed.slug, moment, parsed.attempt + 1)


def batch_directory(logs_root: str | Path, label: str) -> Path:
    """Where the launcher would publish this label's artifact leaf."""
    assert_launcher_safe(label)
    return Path(logs_root) / label


def assert_available(
    label: str,
    logs_root: str | Path,
    exists: Callable[[Path], bool],
    known_labels: Iterable[str] = (),
) -> str:
    """Refuse a label that any earlier attempt already owns."""
    assert_launcher_safe(label)
    if label in set(known_labels):
        raise LabelCollisionError(
            f"attempt label {label!r} was already used; labels are immutable"
        )
    directory = batch_directory(logs_root, label)
    if exists(directory):
        raise LabelCollisionError(
            f"batch directory already exists: {directory}; labels are never reused"
        )
    return label


def allocate(
    slug: str,
    moment: datetime,
    logs_root: str | Path,
    exists: Callable[[Path], bool],
    attempt: int = 0,
    known_labels: Iterable[str] = (),
) -> str:
    """Generate one attempt label and prove it is free before returning it."""
    return assert_available(
        build_label(slug, moment, attempt), logs_root, exists, known_labels
    )
