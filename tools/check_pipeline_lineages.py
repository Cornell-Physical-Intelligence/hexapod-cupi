#!/usr/bin/env python3
"""Verify the frozen Stage2 Git lineage and a separately versioned current release.

This checks source integrity only. It does not qualify dynamics, admit training,
load checkpoints, or modify the historical manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_MANIFEST = "configs/releases/stage2_pipeline.sha256"
ARCHIVED_SOURCE_MANIFEST = "isaaclab/deploy/stage2_pipeline.sha256"
ARCHIVED_REF = "81d7c6f2a43c7de99f32cd6bb1b7efb0f54874df"
ARCHIVED_MANIFEST_SHA256 = "19fc816cf9c53a79be8e14831daa58a12eba3f7f07c5fca80d03f2dc947ccda1"
ARCHIVED_ENTRY_COUNT = 112
CURRENT_MANIFEST = "configs/releases/tripod_feedback_20260922_v1.sha256"
CURRENT_HEADER = (
    "# hexapod.locomotion_kernel.v1\n"
    f"# Archived source commit: {ARCHIVED_REF}\n"
    f"# Archived manifest SHA256: {ARCHIVED_MANIFEST_SHA256}\n"
    "# Current coverage is independent of historical model dependencies.\n"
)
# Cover maintained source and tests, without recursively pinning old releases or results.
CURRENT_TREES = ("tools", "contracts", "navigation", "mission", "tests", "robot/tests")
CURRENT_TOP_LEVEL = ("locomotion", "locomotion/tests", "locomotion/priors",
                     "locomotion/priors/tests")
CURRENT_FILES = ("robot/active_model.json", "configs/source_inventory.json", "configs/archive.json", "configs/locomotion_spark.json",
                 "robot/hexapod_mkii_updated_v1/inputs.json",
                 ".github/workflows/tests.yml", "pyproject.toml", "uv.lock", ".python-version")
SOURCE_SUFFIXES = {".py", ".toml", ".json", ".yaml", ".yml"}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def parse_manifest(content):
    records = {}
    for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._/-]+)", line)
        if not match:
            raise ValueError(f"Malformed manifest entry at line {number}")
        digest, relative = match.groups()
        path = PurePosixPath(relative)
        if (path.is_absolute() or path.as_posix() != relative or ".." in path.parts
                or ".git" in path.parts or relative in records):
            raise ValueError(f"Unsafe or duplicate manifest path at line {number}")
        records[relative] = digest
    if not records:
        raise ValueError("Manifest contains no files")
    return records


def read_archived_manifest(root, *, expected_sha256=ARCHIVED_MANIFEST_SHA256,
                           expected_count=ARCHIVED_ENTRY_COUNT):
    content = (Path(root) / ARCHIVED_MANIFEST).read_bytes()
    if digest_bytes(content) != expected_sha256:
        raise ValueError("Frozen historical manifest bytes changed")
    records = parse_manifest(content)
    if len(records) != expected_count:
        raise ValueError("Unexpected historical manifest entry count")
    return content, records


def verify_historical(root=ROOT, *, source_ref=ARCHIVED_REF,
                      expected_manifest_sha256=ARCHIVED_MANIFEST_SHA256,
                      expected_count=ARCHIVED_ENTRY_COUNT, source_manifest=ARCHIVED_SOURCE_MANIFEST):
    """Read immutable Git objects into an isolated tar; never check out old code."""
    root = Path(root).resolve()
    content, expected = read_archived_manifest(root,
        expected_sha256=expected_manifest_sha256, expected_count=expected_count)
    result = subprocess.run(["git", "show", f"{source_ref}:{source_manifest}"], cwd=root,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        raise ValueError("Pinned historical source is unavailable; fetch its Git history before checking")
    if result.stdout != content:
        raise ValueError("Historical Git manifest differs from the frozen local manifest")
    with tempfile.TemporaryFile() as archive:
        result = subprocess.run(["git", "archive", "--format=tar", source_ref, "--", *sorted(expected)],
            cwd=root, stdout=archive, stderr=subprocess.PIPE, timeout=120)
        if result.returncode:
            raise ValueError("Cannot archive all historical manifest paths from the pinned Git source")
        archive.seek(0)
        actual = {}
        with tarfile.open(fileobj=archive, mode="r|") as bundle:
            for member in bundle:
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in expected or member.name in actual:
                    raise ValueError("Historical Git archive contains an unexpected or nonregular entry")
                with bundle.extractfile(member) as stream:
                    actual[member.name] = digest_stream(stream)
    if actual != expected:
        changed = sorted(path for path in expected if actual.get(path) != expected[path])
        raise ValueError(f"Historical Git files do not match all frozen entries: {changed}")
    return {"pass": True, "lineage": "archived_stage2", "source_commit": source_ref,
            "manifest_sha256": digest_bytes(content), "files_verified": len(expected)}


def current_records(root=ROOT):
    """Hash current source and the selected robot without loading historical task code."""
    root = Path(root).resolve()
    relative_paths = set(CURRENT_FILES)
    for directory in CURRENT_TREES:
        relative_paths.update(p.relative_to(root).as_posix() for p in (root/directory).rglob("*")
            if p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts and p.is_file())
    for directory in CURRENT_TOP_LEVEL:
        relative_paths.update(p.relative_to(root).as_posix() for p in (root/directory).glob("*.py"))
    model = json.loads((root/"robot/active_model.json").read_text())
    selected = {model[key]["path"]: model[key]["sha256"] for key in ("urdf", "model")}
    selected[model["usd"]] = model["usd_sha256"]
    relative_paths.update(selected)
    records = {}
    for relative in sorted(relative_paths):
        path = root/relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink()
                or not path.is_file() or not path.resolve().is_relative_to(root)):
            raise ValueError(f"Current release path is missing or not an in-root regular file: {relative}")
        with path.open("rb") as stream:
            records[relative] = digest_stream(stream)
        if relative in selected and records[relative] != selected[relative]:
            raise ValueError("Selected robot identity differs: "+relative)
    return records, []


def compare_current_manifest(content, actual):
    if not content.startswith(CURRENT_HEADER.encode()):
        raise ValueError("Current manifest has the wrong schema or historical lineage header")
    expected = parse_manifest(content)
    if set(expected) != set(actual):
        raise ValueError(f"Current manifest coverage differs: missing={sorted(set(actual)-set(expected))}; "
                         f"extra={sorted(set(expected)-set(actual))}")
    changed = sorted(path for path in expected if expected[path] != actual[path])
    if changed:
        raise ValueError(f"Current release hashes differ from their versioned manifest: {changed}")
    return len(expected)


def verify_current(root=ROOT, manifest=None):
    root = Path(root).resolve()
    path = Path(manifest) if manifest is not None else root / CURRENT_MANIFEST
    actual, revisions = current_records(root)
    content = path.read_bytes()
    count = compare_current_manifest(content, actual)
    return {"pass": True, "lineage": "locomotion_kernel_v1", "manifest": str(path),
            "manifest_sha256": digest_bytes(content), "files_verified": count,
            "historical_verification": "separate pinned Git lineage"}


def generate_current(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output) if output is not None else root / CURRENT_MANIFEST
    if output.resolve() == (root / ARCHIVED_MANIFEST).resolve():
        raise ValueError("The historical manifest cannot be a generation destination")
    # Generation requires the pinned lineage to pass, not just matching metadata.
    verify_historical(root)
    records, revisions = current_records(root)
    content = CURRENT_HEADER + "".join(f"{records[path]}  {path}\n" for path in sorted(records))
    with output.open("x") as stream:
        stream.write(content)
    return {"generated": str(output), "files": len(records), "sha256": digest_bytes(content.encode()),
            "historical_verification": "separate pinned Git lineage"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("historical", "current", "generate"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, help="Current manifest to verify, or a new generation destination")
    args = parser.parse_args(argv)
    if args.mode == "historical" and args.manifest is not None:
        parser.error("Historical verification always uses the pinned manifest and commit")
    try:
        result = (verify_historical(args.root) if args.mode == "historical" else
                  verify_current(args.root, args.manifest) if args.mode == "current" else
                  generate_current(args.root, args.manifest))
    except (OSError, ValueError, subprocess.TimeoutExpired, tarfile.TarError) as exc:
        print(json.dumps({"pass": False, "mode": args.mode, "error": str(exc)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
