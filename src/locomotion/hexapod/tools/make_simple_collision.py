# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replace the hexapod's 153 SDF mesh colliders with one primitive per link (19 shapes), fitted to the CAD.

Body: a box = the bounding box of its collision parts. Each leg segment (coxa, femur, tibia): a capsule along the
link's +x axis (the joint-to-joint direction in this URDF), spanning the parts' x extent, with radius from the
cross-section of the distal half of the segment (so the tibia's radius is its shaft/toe, not its joint housing).
The tibia capsule's far end-cap therefore sits at the real toe tip. Visual meshes, masses, inertias, joints and
limits are untouched; the output is a new file in the source directory, so its relative references stay valid.

Run inside the Isaac Lab container (needs pxr), from the Isaac Lab root:
  ./isaaclab.sh -p <repo>/src/locomotion/hexapod/tools/make_simple_collision.py [--visible] [--radius_scale 1.0]
or on CPU with `uv run python <repo>/src/locomotion/hexapod/tools/make_simple_collision.py`.
It writes robot_simple.usda next to the club's robot.usda (gitignored, regenerate on demand). The approved asset is
never modified: the tool exits non-zero without writing when --out is outside the source directory, resolves to the
source or any layer or asset it depends on, or names an existing file this tool did not write, and when a radius
setting or a fitted dimension is not finite and positive. These checks raise explicit exceptions, so `python -O`
keeps them. Each primitive carries the PhysxCollisionAPI contact and rest offsets of the SDF parts it replaces (the
approved recipe authors contactOffset 0.001, restOffset 0).
The variant has no native admission, so `training_policy` in robot/active_model.json forbids new training on it. It
serves throughput and fidelity diagnostics until it is admitted; HEXAPOD_USD=<path> selects it for those.
Results (speed-up, fidelity checks): site/assets/collision_simplification_20261015_001/README.md.
"""

import argparse
import math
import os
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, Vt

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
DEFAULT_SRC = os.path.normpath(os.path.join(_REPO, "robot", "hexapod_mkii_updated_v1", "usd", "robot.usda"))
RECIPE_KEY = "collision_recipe"
RECIPE_PREFIX = "one primitive per link, fitted by make_simple_collision.py"


class ConversionError(Exception):
    """The conversion was refused; nothing was written."""


def _real(path):
    return os.path.normcase(os.path.realpath(path))


def _written_by_this_tool(path):
    try:
        layer = Sdf.Layer.OpenAsAnonymous(path)
    except Exception:
        return False
    return bool(layer) and str(dict(layer.customLayerData).get(RECIPE_KEY, "")).startswith(RECIPE_PREFIX)


def check_output_path(src, out):
    """Refuse an output that could replace an input or break the output's relative references."""
    if not os.path.isfile(src):
        raise ConversionError(f"--src {src} is not a file")
    src_dir = os.path.dirname(_real(src))
    out_real = _real(out)
    if os.path.dirname(out_real) != src_dir:
        # robot.usda references ./geometry.usdc; a layer exported elsewhere resolves that against its new directory
        # and loses every visual mesh
        raise ConversionError(f"--out {out} must be in the source directory {src_dir}, beside the layers it references")
    layers, assets, _unresolved = UsdUtils.ComputeAllDependencies(src)
    inputs = {_real(src)} | {_real(layer.realPath) for layer in layers if layer.realPath} | {_real(a) for a in assets}
    if out_real in inputs or (os.path.exists(out) and any(os.path.samefile(out, p) for p in inputs if os.path.exists(p))):
        raise ConversionError(f"--out {out} would overwrite an input of {src}")
    if os.path.lexists(out) and not (os.path.isfile(out) and _written_by_this_tool(out)):
        raise ConversionError(f"--out {out} exists and was not written by this tool")


def _positive(value, name):
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ConversionError(f"{name} must be finite and positive, got {value}")
    return value


def convert(src=DEFAULT_SRC, out=None, radius_scale=1.0, radius_quantile=0.95, visible=False):
    """Write the simplified-collision variant of ``src`` to ``out`` and return (out, report lines)."""
    radius_scale = _positive(radius_scale, "--radius_scale")
    if not (math.isfinite(radius_quantile) and 0.0 < radius_quantile <= 1.0):
        raise ConversionError(f"--radius_quantile must be in (0, 1], got {radius_quantile}")
    out = out or os.path.join(os.path.dirname(src), "robot_simple_visible.usda" if visible else "robot_simple.usda")
    check_output_path(src, out)

    stage = Usd.Stage.Open(src)
    robot = stage.GetPrimAtPath("/Robot")
    if not robot:
        raise ConversionError(f"{src} has no /Robot prim")
    links = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    xc = UsdGeom.XformCache()
    report = []

    for link in links:
        col = link.GetChild("collisions")
        parts = list(col.GetChildren()) if col else []
        if not parts:
            raise ConversionError(f"{link.GetName()}: no collision parts to replace")
        pts = []
        for part in parts:
            p = np.array(UsdGeom.Mesh(part).GetPointsAttr().Get(), dtype=np.float64)
            M = np.array(xc.ComputeRelativeTransform(part, link)[0])  # row-vector convention: p_link = [p, 1] @ M
            pts.append((np.c_[p, np.ones(len(p))] @ M)[:, :3])
        pts = np.vstack(pts)
        if not np.isfinite(pts).all():
            raise ConversionError(f"{link.GetName()}: collision points are not finite")
        lo, hi = pts.min(0), pts.max(0)
        # contact / rest offsets of the replaced parts; without them PhysX picks its automatic contact offset and the
        # tibia capsule reports ground contact ~2 cm before touchdown
        offsets = {
            tuple(part.GetAttribute(f"physxCollision:{name}").Get() for name in ("contactOffset", "restOffset"))
            for part in parts
        }
        if len(offsets) != 1 or None in next(iter(offsets)):
            raise ConversionError(f"{link.GetName()}: parts disagree on or lack offsets {offsets}")
        contact_offset, rest_offset = next(iter(offsets))
        for part in parts:
            stage.RemovePrim(part.GetPath())
        shape_path = col.GetPath().AppendChild("shape")
        if link.GetName() == "body":
            size = hi - lo
            for axis, extent in zip("xyz", size):
                _positive(extent, f"body box {axis} size")
            box = UsdGeom.Cube.Define(stage, shape_path)
            box.GetSizeAttr().Set(1.0)
            xf = UsdGeom.Xformable(box)
            xf.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            desc = f"box {np.round(size, 4)} at {np.round((lo + hi) / 2, 4)}"
            prim = box.GetPrim()
        else:
            distal = pts[pts[:, 0] > (lo[0] + hi[0]) / 2]
            if len(distal) == 0:
                raise ConversionError(f"{link.GetName()}: no collision points in the distal half")
            r = float(np.quantile(np.linalg.norm(distal[:, 1:], axis=1), radius_quantile)) * radius_scale
            r = _positive(r, f"{link.GetName()} capsule radius")
            x0, x1 = lo[0] + r, hi[0] - r  # end-cap centres; far cap surface = hi[0] (the toe tip for the tibia)
            if x1 <= x0:
                x0 = x1 = (lo[0] + hi[0]) / 2
            cap = UsdGeom.Capsule.Define(stage, shape_path)
            cap.GetAxisAttr().Set("X")
            cap.GetRadiusAttr().Set(r)
            cap.GetHeightAttr().Set(max(x1 - x0, 0.0))
            UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, float((lo[1] + hi[1]) / 2), float((lo[2] + hi[2]) / 2)))
            desc = f"capsule r={r:.4f} x=[{lo[0]:.4f},{hi[0]:.4f}] (cylinder {x1 - x0:.4f})"
            prim = cap.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float, custom=False).Set(contact_offset)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float, custom=False).Set(rest_offset)
        gp = UsdGeom.Gprim(prim)
        if visible:
            gp.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.9, 0.3, 0.1)]))
            gp.GetDisplayOpacityAttr().Set(Vt.FloatArray([0.5]))
        else:
            gp.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        report.append(f"{link.GetName():10s} {len(parts):3d} parts -> {desc}, contactOffset {contact_offset}, restOffset {rest_offset}")

    # The fitted femur capsules (their motor housings, r ~3 cm) overlap the body box at rest. PhysX already ignores
    # contacts between links joined by a joint (body-coxa, coxa-femur), but body-femur is two joints apart and would
    # generate large spurious forces (measured: motors pinned at the cap, body "contact" at rest). Filter those pairs.
    body = stage.GetPrimAtPath("/Robot/body")
    femurs = [l.GetPath() for l in links if l.GetName().endswith("_femur")]
    UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets(femurs)
    report.append(f"filtered pairs: body <-> {len(femurs)} femur links")

    layer = stage.GetRootLayer()
    cd = dict(layer.customLayerData)
    cd[RECIPE_KEY] = f"{RECIPE_PREFIX} (radius_scale {radius_scale}, quantile {radius_quantile})"
    layer.customLayerData = cd
    # export beside the output and rename, so a failed export or check leaves no partial artifact
    root, ext = os.path.splitext(out)
    tmp = f"{root}.tmp{os.getpid()}{ext}"
    try:
        if not layer.Export(tmp):
            raise ConversionError(f"exporting {tmp} failed")
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(tmp)
        if unresolved:
            raise ConversionError(f"the output does not resolve {unresolved}")
        os.replace(tmp, out)
    finally:
        if os.path.lexists(tmp):
            os.remove(tmp)
    return out, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--out", default=None, help="default: robot_simple.usda (or robot_simple_visible.usda) next to src")
    parser.add_argument("--radius_scale", type=float, default=1.0, help="multiplier on the fitted capsule radii")
    parser.add_argument("--radius_quantile", type=float, default=0.95, help="cross-section quantile used as the radius")
    parser.add_argument("--visible", action="store_true", help="render the shapes (for a visual check); otherwise purpose=guide")
    args = parser.parse_args(argv)
    try:
        out, report = convert(args.src, args.out, args.radius_scale, args.radius_quantile, args.visible)
    except ConversionError as error:
        print(f"make_simple_collision: refused, nothing written: {error}", file=sys.stderr)
        return 2
    print("\n".join(report))
    check = Usd.Stage.Open(out)
    n = sum(1 for p in Usd.PrimRange(check.GetPrimAtPath("/Robot")) if p.HasAPI(UsdPhysics.CollisionAPI))
    print(f"wrote {out}: {n} collision shapes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
