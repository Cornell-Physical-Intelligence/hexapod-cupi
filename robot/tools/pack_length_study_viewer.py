#!/usr/bin/env python3
"""Embed all exact study URDFs and shared detailed STL surfaces for inspection."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from pack_urdf_viewer import load_stl, pack_mesh

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "robot/hexapod_mkii_length_study"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    urdfs, records = {}, {}
    for record in manifest["variants"]:
        path = PACKAGE / record["urdf"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        ET.fromstring(raw)
        urdfs[record["variant"]] = raw.decode()
        records[record["variant"]] = {key: record[key] for key in ("sha256", "femur_length_m", "tibia_length_m")}
    blob, meshes = bytearray(), []
    for path in sorted((PACKAGE / "meshes").glob("*.stl")):
        packed = pack_mesh(load_stl(path), blob)
        packed["file"] = path.name
        meshes.append(packed)
    production_path = ROOT / "robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf"
    production_raw = production_path.read_bytes()
    production = {"file":production_path.name, "sha256":hashlib.sha256(production_raw).hexdigest(),
                  "urdf":production_raw.decode()}
    data = json.dumps({"urdfs": urdfs, "records": records, "production": production}, separators=(",", ":")).encode()
    fragment = (PACKAGE / "preview_template.html").read_text()
    fragment = fragment.replace("__PACK__", json.dumps(meshes, separators=(",", ":")))
    fragment = fragment.replace("__URDFS__", base64.b64encode(gzip.compress(data, mtime=0)).decode())
    fragment = fragment.replace("__MESH_BLOB__", base64.b64encode(blob).decode())
    assert len(fragment.encode()) < 1_000_000
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(fragment)
    print(f"Packed {len(urdfs)} exact URDFs, {len(meshes)} shared meshes, {len(fragment.encode())} bytes: {args.out}")


if __name__ == "__main__":
    main()
