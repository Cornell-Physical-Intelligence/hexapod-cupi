#!/usr/bin/env python3
"""Import the hexapod URDF into a floating-base USD with Isaac Sim's URDF importer.

Run with Isaac Sim's Python (headless, no GUI needed), e.g. on the Spark:

    python tools/import_urdf_to_usd.py \\
        robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf \\
        robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda
    python tools/enable_nested_contact_reports.py \\
        robot/hexapod_mkii_assy/usd/hexapod_mkii_serial/hexapod_mkii_serial.usda

Import settings mirror the ones used for the mock asset: floating base, URDF
inertials kept (no density override), collision primitives taken from the
URDF (no convex decomposition, nothing derived from visuals), self-collision
off (Isaac Lab sets it per task), position drives whose gains Isaac Lab
overrides through the actuator config.  Mesh paths use ``package://`` and
resolve from the package directory that contains the ``urdf/`` folder.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf")
    parser.add_argument("usd", help="destination .usd/.usda (payload files are written next to it)")
    parser.add_argument("--fix-base", action="store_true", help="import with a fixed base (default floating)")
    parser.add_argument("--drive-stiffness", type=float, default=30.0)
    parser.add_argument("--drive-damping", type=float, default=0.6)
    args = parser.parse_args()
    urdf_path = os.path.abspath(args.urdf)
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(urdf_path):
        raise SystemExit(f"URDF not found: {urdf_path}")
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)

    from isaacsim import SimulationApp  # noqa: E402  (Isaac Sim python only)

    app = SimulationApp({"headless": True})
    try:
        import omni.kit.commands  # noqa: E402

        try:
            from isaacsim.asset.importer.urdf import _urdf  # Isaac Sim >= 4.5
        except ImportError:  # pragma: no cover - older Isaac Sim
            from omni.importer.urdf import _urdf  # type: ignore

        cfg = _urdf.ImportConfig()
        cfg.merge_fixed_joints = True
        cfg.fix_base = bool(args.fix_base)
        cfg.make_default_prim = True
        cfg.create_physics_scene = False
        cfg.import_inertia_tensor = True
        cfg.density = 0.0  # keep the URDF masses
        cfg.distance_scale = 1.0
        cfg.self_collision = False
        cfg.convex_decomp = False
        cfg.collision_from_visuals = False
        cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        cfg.default_drive_strength = float(args.drive_stiffness)
        cfg.default_position_drive_damping = float(args.drive_damping)

        status, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=cfg,
            dest_path=usd_path,
            get_articulation_root=True,
        )
        if not status:
            raise SystemExit("URDF import failed")
        print(f"IMPORTED prim={prim_path} usd={usd_path}")

        from pxr import Usd, UsdPhysics  # noqa: E402

        stage = Usd.Stage.Open(usd_path)
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
        print(f"rigid_bodies={len(bodies)} revolute_joints={len(joints)}")
        if len(bodies) != 19 or len(joints) != 18:
            raise SystemExit("expected 19 rigid bodies and 18 revolute joints for hexapod_mkii_serial.urdf")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
