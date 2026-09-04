"""Read-only URDF-to-USD tensor comparison using a saved dump_usd.py result.

Usage: python3 compare_inertia.py [--urdf FILE] [--dump FILE]
Requires numpy and scipy. Exits 1 on a tensor mismatch; never edits either asset.
USD quaternions in the dump are [w, x, y, z]. Tensor comparison is at each COM,
in the link frame, so no parallel-axis translation belongs in this comparison.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=root / "robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf")
    parser.add_argument("--dump", type=Path, default=Path(__file__).parent / "evidence/spark_usd.json")
    args = parser.parse_args()
    links = {e.get("name"): e for e in ET.parse(args.urdf).getroot().findall("link")}
    dump = json.loads(args.dump.read_text())
    output = []
    failures = []
    for body in dump["bodies"]:
        name = body["name"]
        inertial = links[name].find("inertial")
        a = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
        tensor = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]])
        origin = inertial.find("origin")
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        frame = Rotation.from_euler("xyz", rpy).as_matrix()
        expected = frame @ tensor @ frame.T
        attrs = body["attributes"]
        w, x, y, z = attrs["physics:principalAxes"]
        principal_to_link = Rotation.from_quat([x, y, z, w]).as_matrix()
        diagonal = np.diag(attrs["physics:diagonalInertia"])
        actual = principal_to_link @ diagonal @ principal_to_link.T
        # Diagnostic for the installed converter's row/column convention defect.
        inverse_axes = principal_to_link.T @ diagonal @ principal_to_link
        error = float(np.max(np.abs(actual - expected)))
        passed = bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-8))
        if not passed:
            failures.append(name)
        output.append({"link": name, "pass": passed, "max_abs_error_kg_m2": error,
                       "urdf_izz_kg_m2": float(expected[2, 2]), "usd_izz_kg_m2": float(actual[2, 2]),
                       "inverse_axes_max_abs_error_kg_m2": float(np.max(np.abs(inverse_axes - expected)))})
    missing = sorted(set(links) - {b["name"] for b in dump["bodies"]})
    result = {"urdf": str(args.urdf), "dump": str(args.dump), "failed_links": failures,
              "missing_links": missing, "comparison": output}
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(bool(failures or missing))


if __name__ == "__main__":
    sys.exit(main())
