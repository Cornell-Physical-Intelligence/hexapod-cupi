#!/usr/bin/env python3
"""List owned tools and check the maintained-source boundary without scanning evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.code_map import imported_modules, module_name  # noqa: E402

INVENTORY = "configs/source_inventory.json"
SCOPES = ("tools", "src/contracts", "src/navigation", "src/mission", "src/locomotion")

# Cross-package dependencies. Same-package imports need no entry.
IMPORT_BOUNDARIES = {
    "contracts": set(),
    "navigation": {"contracts"},
    "mission": {"contracts"},
    "locomotion.robots": set(),
    "locomotion.omniwalk": {"contracts", "locomotion.robots"},
    "locomotion.evaluation": {
        "contracts",
        "navigation",
        "locomotion.omniwalk",
        "locomotion.robots",
    },
    "locomotion.spark": {"contracts", "locomotion.omniwalk", "locomotion.robots"},
    "locomotion.rl": set(),
    "locomotion.amp": {
        "locomotion.evaluation",
        "locomotion.omniwalk",
        "locomotion.rl",
        "locomotion.robots",
        "locomotion.spark",
    },
    "locomotion.scripts": {
        "locomotion.amp",
        "locomotion.evaluation",
        "locomotion.omniwalk",
        "locomotion.rl",
        "locomotion.robots",
        "locomotion.spark",
    },
    "locomotion.surrogate": {
        "contracts",
        "locomotion.amp",
        "locomotion.evaluation",
        "locomotion.omniwalk",
        "locomotion.rl",
        "locomotion.robots",
        "locomotion.spark",
        "locomotion.randomization",
    },
    "locomotion.randomization": {"locomotion.robots"},
}


def boundary(module):
    parts = module.split(".")
    return ".".join(parts[:2]) if parts[0] == "locomotion" else parts[0]


def check_imports(path, text):
    source = boundary(module_name(path))
    # The retained experiment is evidence, not part of the maintained runtime.
    if source == "locomotion.hexapod":
        return
    for name in imported_modules(ast.parse(text), path):
        target = boundary(name)
        if target == "artifacts":
            raise ValueError(f"Production package imports evidence: {path}")
        if target == "locomotion.hexapod":
            raise ValueError(f"Production package imports retained experiment: {path}")
        if target == source or (target == "locomotion" and source.startswith("locomotion.")):
            continue
        owned = target in {"contracts", "navigation", "mission", "locomotion"} or target.startswith(
            "locomotion."
        )
        if owned and target not in IMPORT_BOUNDARIES.get(source, set()):
            raise ValueError(f"Forbidden import boundary: {source} -> {target} in {path}")


def read_inventory(root=ROOT):
    return json.loads((root / INVENTORY).read_text())


def check(root=ROOT):
    inventory = read_inventory(root)
    rows = inventory["tools"]
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate source ownership")
    actual = {
        str(p.relative_to(root))
        for directory in SCOPES
        for p in (root / directory).rglob("*.py")
        if "__pycache__" not in p.parts and "tests" not in p.parts
    }
    if actual != set(paths):
        raise ValueError(
            f"Source inventory differs: unowned={sorted(actual - set(paths))}; missing={sorted(set(paths) - actual)}"
        )
    for row in rows:
        path = root / row["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source must be a regular in-repository file")
        if (
            row["classification"] not in {"current", "historical", "prototype"}
            or not row["owner"]
            or not row["purpose"]
        ):
            raise ValueError("Every source needs classification, ownership and purpose")
    retired = inventory.get("retired_sources", {})
    for name, binding in inventory.get("source_archives", {}).items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not re.fullmatch("[0-9a-f]{40}", binding["source_commit"])
        ):
            raise ValueError("Invalid archived source binding")
        original = subprocess.check_output(
            ["git", "show", binding["source_commit"] + ":" + name], cwd=root
        )
        if hashlib.sha256(original).hexdigest() != binding["sha256"]:
            raise ValueError("Archived source bytes differ: " + name)
    for name, expected in retired.get("files", {}).items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or (root / relative).exists()
            or not re.fullmatch("[0-9a-f]{40}", retired["source_commit"])
        ):
            raise ValueError("Invalid retired source binding")
        original = subprocess.check_output(
            ["git", "show", retired["source_commit"] + ":" + name], cwd=root
        )
        if hashlib.sha256(original).hexdigest() != expected:
            raise ValueError("Retired source bytes differ: " + name)
    # Production code must not import frozen evidence or historical runtimes.
    for path in [
        p
        for scope in ("src/contracts", "src/navigation", "src/mission", "src/locomotion")
        for p in (root / scope).rglob("*.py")
    ]:
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        check_imports(path.relative_to(root).as_posix(), path.read_text())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "check"])
    parser.add_argument("--classification", choices=["current", "historical", "prototype"])
    args = parser.parse_args()
    rows = check()
    if args.command == "check":
        print(
            f"Ownership and paths verified for {len(rows)} tools; production import boundaries verified"
        )
    else:
        for kind in ["current", "prototype", "historical"]:
            if args.classification and args.classification != kind:
                continue
            print(kind.upper())
            for row in rows:
                if row["classification"] == kind:
                    print(f"  {row['path']} — {row['owner']}: {row['purpose']}")


if __name__ == "__main__":
    main()
