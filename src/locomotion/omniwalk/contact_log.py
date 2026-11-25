"""Audit the Isaac contact-buffer warnings emitted during an omniwalk allocation."""

import hashlib
import re
from pathlib import Path

_INCOMPLETE = re.compile(r"incomplete\s+(?:contact|friction)\s+data", re.IGNORECASE)


def audit_contact_log(path):
    path = Path(path)
    data = path.read_bytes()  # A missing/unreadable log must never pass.
    lines = data.decode("utf-8", errors="replace").splitlines()
    matches = [line for line in lines if _INCOMPLETE.search(line)]
    capacities = sorted(
        {
            int(value)
            for line in matches
            for value in re.findall(r"maxContactDataCount\s*=\s*(\d+)", line)
        }
    )
    return dict(
        log_sha256=hashlib.sha256(data).hexdigest(),
        incomplete_data_warning_count=len(matches),
        reported_capacities=capacities,
        first_warning=matches[0] if matches else None,
        last_warning=matches[-1] if matches else None,
        passed=not matches,
        scope="Required absence of reported contact/friction data truncation; raw physical gate must also pass",
    )
