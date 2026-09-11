#!/usr/bin/env python3
"""Import and prepare a verified, portable floating-base MKII USD bundle.

Fresh import (requires Isaac Sim and may use the GPU):

    python tools/assets/import_urdf_to_usd.py robot/.../hexapod_mkii_serial.urdf \
        robot/.../usd/new_version/hexapod_mkii_serial.usda

Prepare an existing import entirely on the CPU:

    uv run python tools/assets/import_urdf_to_usd.py robot/.../hexapod_mkii_serial.urdf \
        robot/.../usd/new_version/hexapod_mkii_serial.usda \
        --source-usd robot/.../usd/old_import/hexapod_mkii_serial.usda

The destination directory must be absent or empty. Raw import/copy and contact
metadata authoring occur only in an owned temporary directory; tensor repair,
geometric verification and dependency hashing must pass before publishing the
new bundle. Existing USDs, URDFs and installed converter code stay unchanged.
The legacy Isaac importer operation is retained: the fresh GPU import path has
not been exercised by the CPU-only September 2026 repair. Its compatibility with
the installed Isaac SDK must be checked when fresh imports are authorized.
"""

from __future__ import annotations

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET


def _preflight(urdf, output, source_usd, fix_base, stiffness, damping):
    if not urdf.is_file():
        raise ValueError(f"URDF not found: {urdf}")
    if output.suffix not in (".usd", ".usda") or output.exists():
        raise ValueError("Choose a new .usd/.usda output; existing assets are never overwritten")
    if output.parent.exists() and (not output.parent.is_dir() or any(output.parent.iterdir())):
        raise ValueError("Output bundle directory must be absent or empty")
    if fix_base:
        raise ValueError("Verified MKII training assets require a floating base; fixed-base stand models need a separate contract")
    if not all(math.isfinite(value) and value >= 0 for value in (stiffness, damping)):
        raise ValueError("Drive stiffness and damping must be finite and nonnegative")
    root = ET.parse(urdf).getroot()
    links, joints = root.findall("link"), root.findall("joint")
    if (len(links) != 19 or len(joints) != 18
            or any(not element.get("name") for element in [*links, *joints])
            or len({link.get("name") for link in links}) != 19
            or len({joint.get("name") for joint in joints}) != 18
            or any(joint.get("type") != "revolute" or joint.find("mimic") is not None for joint in joints)):
        raise ValueError("Use the serial MKII URDF: 19 named links and 18 independent revolute joints")
    if source_usd is not None:
        if stiffness != 30.0 or damping != 0.6:
            raise ValueError("Drive overrides apply only to fresh imports; --source-usd preserves existing drives")
        if not source_usd.is_file():
            raise ValueError(f"Source USD not found: {source_usd}")
        if output.parent == source_usd.parent or output.parent.is_relative_to(source_usd.parent):
            raise ValueError("Output must be outside the existing source bundle")


@contextmanager
def _isaac_import(urdf, raw_usd, stiffness, damping):
    """Keep the pre-existing SDK operation and close its app on every exit."""
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.kit.commands

        try:
            from isaacsim.asset.importer.urdf import _urdf
        except ImportError:  # pragma: no cover - old SDK fallback, not CPU-tested
            from omni.importer.urdf import _urdf

        cfg = _urdf.ImportConfig()
        cfg.merge_fixed_joints = True
        cfg.fix_base = False
        cfg.make_default_prim = True
        cfg.create_physics_scene = False
        cfg.import_inertia_tensor = True
        cfg.density = 0.0
        cfg.distance_scale = 1.0
        cfg.self_collision = False
        cfg.convex_decomp = False
        cfg.collision_from_visuals = False
        cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        cfg.default_drive_strength = stiffness
        cfg.default_position_drive_damping = damping
        status, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=str(urdf), import_config=cfg,
            dest_path=str(raw_usd), get_articulation_root=True,
        )
        if not status:
            raise RuntimeError("URDF import failed")
        print(f"IMPORTED_RAW prim={prim_path} usd={raw_usd}")
        yield
    finally:
        app.close()


def _dependencies(path):
    from tools.assets.prepare_mkii_usd import dependencies

    return dependencies(path, path.parent)


def _enable_contacts(path):
    from tools.assets.enable_nested_contact_reports import enable_contact_reports

    return enable_contact_reports(path, expected_bodies=19)


def _prepare(urdf, source, output):
    from tools.assets.prepare_mkii_usd import prepare

    return prepare(urdf, source, output)


def build_verified_asset(urdf, output, *, source_usd=None, fix_base=False,
                         drive_stiffness=30.0, drive_damping=0.6):
    """Run preflight -> raw source -> contact metadata -> full preparation.

    ``source_usd`` never imports Isaac Sim. It copies and fingerprints the source
    before any contact-report edits, preserving the original import byte-for-byte.
    A fresh import uses the same verification path after the SDK returns.
    """
    urdf, output = Path(urdf).resolve(), Path(output).resolve()
    source_usd = Path(source_usd).resolve() if source_usd is not None else None
    _preflight(urdf, output, source_usd, fix_base, drive_stiffness, drive_damping)
    output.parent.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mkii-import-", dir=output.parent.parent) as temporary:
        raw = Path(temporary) / "raw"
        if source_usd is not None:
            original_dependencies = _dependencies(source_usd)
            if any(path.is_symlink() for path in source_usd.parent.rglob("*")):
                raise ValueError("Source bundle contains a symlink")
            shutil.copytree(source_usd.parent, raw)
            raw_usd = raw / source_usd.name
            for record in original_dependencies:
                copied_hash = hashlib.sha256((raw / record["path"]).read_bytes()).hexdigest()
                if copied_hash != record["sha256"]:
                    raise ValueError("Source dependency changed while copying")
            context = nullcontext()
        else:
            raw.mkdir()
            raw_usd = raw / f"{urdf.stem}.usda"
            original_dependencies = None
            context = _isaac_import(urdf, raw_usd, drive_stiffness, drive_damping)
        with context:
            if not raw_usd.is_file():
                raise RuntimeError("Importer did not produce the requested raw USD")
            if original_dependencies is None:
                original_dependencies = _dependencies(raw_usd)
            origin_record = {
                "mode": "cpu_source_usd" if source_usd is not None else "fresh_isaac_import",
                "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                "dependencies_before_contact_authoring": original_dependencies,
            }
            if source_usd is None:
                origin_record["drive_stiffness"] = drive_stiffness
                origin_record["drive_damping"] = drive_damping
            contacts = _enable_contacts(raw_usd)
            if contacts != 19:
                raise RuntimeError(f"Contact authoring returned {contacts} bodies, expected 19")
            origin_record["contact_report_bodies"] = contacts
            (raw / "import_origin.json").write_text(json.dumps(origin_record, indent=2, allow_nan=False) + "\n")
            result = _prepare(urdf, raw_usd, output)
            if not result.get("pass"):
                raise RuntimeError(f"Prepared USD did not pass validation: {result.get('errors')}")
            return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("usd", type=Path, help="new verified bundle root .usd/.usda; parent directory must be absent or empty")
    parser.add_argument("--source-usd", type=Path, help="prepare an existing import on CPU without starting Isaac Sim")
    parser.add_argument("--fix-base", action="store_true", help="unsupported for verified floating-base MKII; rejected before starting Isaac Sim")
    parser.add_argument("--drive-stiffness", type=float, default=30.0, help="fresh imports only")
    parser.add_argument("--drive-damping", type=float, default=0.6, help="fresh imports only")
    args = parser.parse_args(argv)
    try:
        result = build_verified_asset(args.urdf, args.usd, source_usd=args.source_usd,
                                      fix_base=args.fix_base, drive_stiffness=args.drive_stiffness,
                                      drive_damping=args.drive_damping)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, ET.ParseError) as error:
        print(json.dumps({"pass": False, "errors": [str(error)]}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
