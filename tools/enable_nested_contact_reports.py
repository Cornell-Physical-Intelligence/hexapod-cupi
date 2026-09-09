#!/usr/bin/env python3
"""Author PhysX contact-report APIs on every rigid body in an imported USD."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics


def enable_contact_reports(usd_path: str | Path, expected_bodies: int = 19) -> int:
    """Author reports on a raw import, before its immutable validation manifest."""
    stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")
    if stage.GetRootLayer().customLayerData.get("hexapod_inertia_repair_version"):
        raise ValueError("Prepared USD bundles are immutable; author contact reports on the raw import before preparation")

    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != expected_bodies:
        raise RuntimeError(
            f"Expected {expected_bodies} rigid bodies, found {len(bodies)}"
        )

    for prim in bodies:
        # PhysX plugins are optional on the CPU. Authored tokens remain visible
        # even when GetAppliedSchemas() omits unregistered PhysX schemas.
        authored = prim.GetMetadata("apiSchemas")
        schemas = set(authored.GetAppliedItems()) if authored else set()
        if "PhysxRigidBodyAPI" not in schemas:
            prim.AddAppliedSchema("PhysxRigidBodyAPI")
        if "PhysxContactReportAPI" not in schemas:
            prim.AddAppliedSchema("PhysxContactReportAPI")
        prim.CreateAttribute(
            "physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float, custom=False
        ).Set(0.0)
        prim.CreateAttribute(
            "physxContactReport:threshold", Sdf.ValueTypeNames.Float, custom=False
        ).Set(0.0)

    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll)
    for body in bodies:
        prim = reopened.GetPrimAtPath(body.GetPath())
        schemas = set(prim.GetMetadata("apiSchemas").GetAppliedItems())
        if not {"PhysxRigidBodyAPI", "PhysxContactReportAPI"}.issubset(schemas):
            raise RuntimeError(f"Contact-report metadata did not persist on {prim.GetPath()}")
        for attribute in ("physxRigidBody:sleepThreshold", "physxContactReport:threshold"):
            if prim.GetAttribute(attribute).Get() != 0.0:
                raise RuntimeError(f"Contact-report setting did not persist: {prim.GetPath()} {attribute}")
    return len(bodies)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_path")
    parser.add_argument("--expected-bodies", type=int, default=19)
    args = parser.parse_args()
    count = enable_contact_reports(args.usd_path, args.expected_bodies)
    print(f"CONTACT_REPORTS_ENABLED bodies={count} path={args.usd_path}")


if __name__ == "__main__":
    main()
