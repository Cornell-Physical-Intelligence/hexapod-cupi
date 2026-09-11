"""Exact source/asset lineage and checkpoint admission for four-bar scratch runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-Fourbar-V1-Direct-v0"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(root=ROOT):
    root = Path(root).resolve()
    files = set()
    for directory, suffixes in (("packages", {".py", ".json", ".yaml", ".yml", ".toml"}), ("configs", {".json", ".yaml"}),
                                ("tools", {".py"}), ("experiments/c_length_study/tools", {".py"}),
                                ("experiments/terrain/tools", {".py"}), ("isaaclab", {".py"}),
                                ("robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v3", None),
                                ("robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v4", None),
                                ("robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v5", None),
                                ("robot/hexapod_mkii_assy/meshes", None)):
        for path in (root / directory).rglob("*"):
            if path.is_file() and not any(p in {"__pycache__", "tests"} for p in path.parts):
                if suffixes is None or path.suffix in suffixes:
                    files.add(path)
    for relative in ("robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf",
                     "artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/cad_pin_frames.json"):
        files.add(root / relative)
    required = root / "robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v3/hexapod_mkii_fourbar_v3.usda"
    if required not in files:
        raise ValueError("Physical four-bar USD bundle is missing")
    records = {}
    for path in sorted(files):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Contract source escapes checkout")
        records[str(path.relative_to(root))] = digest(path)
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"schema": "hexapod.fourbar_training_contract.v1", "task_id": TASK_ID,
            "sha256": hashlib.sha256(encoded).hexdigest(), "files": records}


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def require_admission(path, contract):
    report = read_json(path)
    if (report.get("pass") is not True or report.get("simulation_training_admission") is not True
            or report.get("task_id") != TASK_ID or report.get("errors") != []
            or report.get("contract") != contract):
        raise ValueError("Training needs a passing physical-model admission report for these exact files")
    if (type(report.get("num_envs")) is not int or type(report.get("steps_completed")) is not int
            or report["num_envs"] < 32 or report["steps_completed"] < 1000):
        raise ValueError("Short probes cannot admit learning")
    return report


def require_checkpoint(path, contract):
    path = Path(path)
    sidecar = read_json(path.with_suffix(path.suffix + ".json"))
    if (sidecar.get("contract") != contract or sidecar.get("checkpoint_sha256") != digest(path)
            or type(sidecar.get("next_iteration")) is not int or sidecar["next_iteration"] < 1):
        raise ValueError("Checkpoint bytes or physical task lineage do not match")
    return sidecar


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
