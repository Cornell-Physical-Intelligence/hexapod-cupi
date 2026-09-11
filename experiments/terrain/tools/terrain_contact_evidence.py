"""Reject known PhysX contact-data truncation before admitting standing.

This host-side check runs after the owned process exits and closes its log.
Absence of a warning is a required check, not proof of all possible SDK errors.
The complete simulator log and its hash remain the underlying evidence.
"""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pathlib import Path
import hashlib
import re

_INCOMPLETE = re.compile(r"incomplete\s+(?:contact|friction)\s+data", re.IGNORECASE)


def audit_contact_log(path):
    path = Path(path)
    data = path.read_bytes()  # A missing/unreadable log must never pass.
    lines = data.decode("utf-8", errors="replace").splitlines()
    matches = [line for line in lines if _INCOMPLETE.search(line)]
    capacities = sorted({int(value) for line in matches
                         for value in re.findall(r"maxContactDataCount\s*=\s*(\d+)", line)})
    return dict(log_sha256=hashlib.sha256(data).hexdigest(),
        incomplete_data_warning_count=len(matches), reported_capacities=capacities,
        first_warning=matches[0] if matches else None,
        last_warning=matches[-1] if matches else None,
        passed=not matches,
        scope="Required absence of reported contact/friction data truncation; raw physical gate must also pass")
