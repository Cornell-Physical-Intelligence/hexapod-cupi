"""Surrogate body with the hexagonal centre piece in nylon plus an added mass (diagnostic arm).

The repeating job: any surrogate script (training or judging) run with the body link's mass, centre of
mass and inertia changed as follows, and nothing else. The nine printed parts of the body link (top and
bottom enclosure, bottom plate, six standoff plates; body_center_parts.json, from the CAD mass ledger)
keep their geometry but take nylon density instead of the CAD's aluminium (2680-2810 kg/m^3): each
part's mass, and its inertia about its own centre, scale by NYLON / its CAD density. ADDED kg is then
added as a point mass at the body's original centre of mass. The body's new mass, centre of mass and
inertia tensor (about the new centre, body axes) are computed from the old ones by the parallel-axis
theorem and written into every MuJoCo model the surrogate keeps (base model and per-replica copies) and
into its per-episode base arrays, so mass randomization scales the new body. Repository files and
robot/active_model.json are untouched; this file wraps SurrogateEnv.__init__.
Check: fitted_eval.py run through this file (judges under the same body).
Record: a "BODY MASS:" line with every number and this file's hash; <output>/body_mass.json when the
target script takes --output, else ROOT/body_mass.json for fitted_eval.py.

Usage (from the hexapod-cupi checkout):
  uv run python body_mass.py show
  uv run python body_mass.py run SCRIPT.py [that script's arguments ...]
"""

import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
PARTS = HERE / "body_center_parts.json"
NYLON = 1140.0  # kg/m^3, nylon 6 (PA6); PA12 would be about 1010
ADDED = 0.75  # kg
SEEN = {"environments": 0, "models": 0}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def offset(mass, d):
    d = np.asarray(d, float)
    return mass * (d @ d * np.eye(3) - np.outer(d, d))


def tensor(i):
    xx, xy, xz, yy, yz, zz = i
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def new_body(mass0, com0, inertia0):
    """Old body (mass, centre, tensor about centre, body axes) -> new body and the part table."""
    parts = json.loads(PARTS.read_text())["parts"]
    com0 = np.asarray(com0, float)
    about_origin = inertia0 + offset(mass0, com0)
    moment = mass0 * com0
    mass = mass0
    table = []
    for p in parts:
        factor = NYLON / p["cad_density_kg_m3"]
        delta = (factor - 1.0) * p["cad_mass_kg"]
        y = p["yaw_rad"]
        rot = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        own = rot @ tensor(p["inertia_part_axes"]) @ rot.T
        at = np.asarray(p["com_body_m"])
        about_origin = about_origin + (factor - 1.0) * (own + offset(p["cad_mass_kg"], at))
        moment = moment + delta * at
        mass += delta
        table.append({"part": p["part"], "volume_cm3": p["mesh_volume_m3"] * 1e6,
                      "cad_kg": p["cad_mass_kg"], "nylon_kg": p["cad_mass_kg"] * factor})
    about_origin = about_origin + offset(ADDED, com0)
    moment = moment + ADDED * com0
    mass += ADDED
    com = moment / mass
    inertia = about_origin - offset(mass, com)
    return mass, com, inertia, table


def body_id(m):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "body")


def old_body(m):
    b = body_id(m)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.body_iquat[b])
    rot = rot.reshape(3, 3)
    return float(m.body_mass[b]), m.body_ipos[b].copy(), rot @ np.diag(m.body_inertia[b]) @ rot.T


def write(m, mass, com, inertia):
    b = body_id(m)
    values, vectors = np.linalg.eigh(inertia)
    if np.linalg.det(vectors) < 0:
        vectors[:, 0] = -vectors[:, 0]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, vectors.ravel())
    m.body_mass[b], m.body_ipos[b], m.body_inertia[b], m.body_iquat[b] = mass, com, values, quat
    mujoco.mj_setConst(m, mujoco.MjData(m))
    SEEN["models"] += 1


RESULT = {}


def install():
    from locomotion.surrogate import env as senv

    stock = senv.SurrogateEnv.__init__

    def patched(self, *a, **k):
        stock(self, *a, **k)
        m = self.mj_model
        if not RESULT:
            mass0, com0, inertia0 = old_body(m)
            mass, com, inertia, table = new_body(mass0, com0, inertia0)
            RESULT.update(old_mass_kg=mass0, old_com_m=com0.tolist(), old_inertia=inertia0.tolist(),
                          mass_kg=mass, com_m=com.tolist(), inertia=inertia.tolist(), parts=table)
        args = (RESULT["mass_kg"], np.array(RESULT["com_m"]), np.array(RESULT["inertia"]))
        models = {id(m): m}
        models.update({id(x): x for x in getattr(self, "_models", [])})
        for x in models.values():
            write(x, *args)
        if hasattr(self, "_base_body_mass"):
            self._base_body_mass = m.body_mass.copy()
            self._base_body_inertia = m.body_inertia.copy()
        SEEN["environments"] += 1
        RESULT["robot_mass_kg"] = float(m.body_mass.sum())

    senv.SurrogateEnv.__init__ = patched


def record(path=None):
    data = {"label": "hexagonal centre piece in nylon + added point mass; surrogate diagnostic arm",
            "nylon_kg_m3": NYLON, "added_kg": ADDED, "parts_file_sha256": sha256(PARTS),
            "script_sha256": sha256(__file__), **SEEN, **RESULT}
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=1) + "\n")
    short = {k: v for k, v in data.items() if k not in ("parts", "old_inertia", "inertia")}
    print(f"BODY MASS: {json.dumps(short)}", flush=True)
    return data


def main(argv):
    if argv[:1] == ["show"]:
        from locomotion.robots.hexapod import MASS_KG

        model = json.loads((Path(__import__("locomotion").__file__).parents[2] / "robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json").read_text())
        body = [x for x in model["links"] if x["name"] == "body"][0]
        mass, com, inertia, table = new_body(body["mass"], body["com"], np.array(body["inertia"]))
        for row in table:
            print(f"{row['part']:18s} {row['volume_cm3']:7.1f} cm3  CAD {row['cad_kg']:.4f} kg  nylon {row['nylon_kg']:.4f} kg")
        vol = sum(r["volume_cm3"] for r in table)
        print(f"centre piece: {vol:.1f} cm3; CAD {sum(r['cad_kg'] for r in table):.4f} kg -> nylon {sum(r['nylon_kg'] for r in table):.4f} kg at {NYLON:.0f} kg/m3")
        print(f"body: {body['mass']:.4f} -> {mass:.4f} kg (incl. +{ADDED} kg); robot {MASS_KG:.4f} -> {MASS_KG - body['mass'] + mass:.4f} kg")
        print("body centre of mass", np.round(body["com"], 5), "->", np.round(com, 5))
        print("body inertia diagonal", np.round(np.diag(body["inertia"]), 5), "->", np.round(np.diag(inertia), 5))
        return 0
    if argv[:1] != ["run"] or len(argv) < 2:
        raise SystemExit(__doc__)
    script, rest = argv[1], argv[2:]
    sys.path.insert(0, str(Path(script).resolve().parent))
    install()
    sys.argv = [script, *rest]
    status = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as stop:
        status = stop.code or 0
    if "--output" in rest:
        out = Path(rest[rest.index("--output") + 1]) / "body_mass.json"
    elif Path(script).name == "fitted_eval.py":
        out = Path(rest[0]) / "body_mass.json"
    else:
        out = None
    record(out)
    if SEEN["environments"] == 0:
        print("BODY MASS: no surrogate environment was built", flush=True)
        return 1
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
