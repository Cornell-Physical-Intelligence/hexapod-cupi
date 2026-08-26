#!/usr/bin/env python3
"""Author PhysX contact-report APIs on every rigid body in an imported USD."""

from __future__ import annotations

import argparse

from pxr import Sdf, Usd, UsdPhysics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_path")
    parser.add_argument("--expected-bodies", type=int, default=19)
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {args.usd_path}")

    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != args.expected_bodies:
        raise RuntimeError(
            f"Expected {args.expected_bodies} rigid bodies, found {len(bodies)}"
        )

    for prim in bodies:
        schemas = set(prim.GetAppliedSchemas())
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
    print(f"CONTACT_REPORTS_ENABLED bodies={len(bodies)} path={args.usd_path}")


if __name__ == "__main__":
    main()
