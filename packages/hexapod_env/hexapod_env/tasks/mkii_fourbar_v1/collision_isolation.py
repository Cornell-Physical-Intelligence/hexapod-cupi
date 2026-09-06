"""Verify authored PhysX clone-isolation groups after explicit scene filtering.

This proves USD topology, not the backend's actual collision response. A live
overlap positive/negative control is still required for that separate claim.
"""
from __future__ import annotations


def verify_collision_isolation(stage, *, physics_scene_path, env_prim_paths, global_prim_paths):
    """Fail closed on stale, partial, merged or nonreciprocal allowlist groups.

    Isaac Lab 3's PhysX cloner authors an inverted scene-level collision filter:
    ``filteredGroups`` then lists allowed groups. Each environment must allow
    only itself and the global group, which must reciprocally allow every env.
    The runtime identity deliberately omits environment count and instance paths
    so qualification and training can use different numbers of isolated clones.
    """
    from pxr import Sdf, Usd

    environments, globals_ = list(env_prim_paths), list(global_prim_paths)
    if not environments or environments != [f"/World/envs/env_{i}" for i in range(len(environments))]:
        raise ValueError("Collision isolation requires the complete ordered physical environment paths")
    if (not globals_ or any(not isinstance(path, str) for path in globals_)
            or len(set(globals_)) != len(globals_)):
        raise ValueError("Collision isolation requires unique explicit global collider paths")
    for path in [*environments, *globals_]:
        parsed = Sdf.Path(path)
        if not parsed.IsAbsolutePath() or not parsed.IsPrimPath() or not stage.GetPrimAtPath(path):
            raise ValueError(f"Collision collection target is not an existing absolute prim: {path}")
    for global_path in globals_:
        if any(Sdf.Path(global_path).HasPrefix(Sdf.Path(env)) or Sdf.Path(env).HasPrefix(Sdf.Path(global_path))
               for env in environments):
            raise ValueError("Global collision collection overlaps an environment namespace")
    scene = stage.GetPrimAtPath(physics_scene_path)
    if (not scene or scene.GetTypeName() != "PhysicsScene"
            or scene.GetAttribute("physxScene:invertCollisionGroupFilter").Get() is not True):
        raise ValueError("Actual physics scene does not use inverted collision-group filtering")
    root = "/World/collisions"
    scope = stage.GetPrimAtPath(root)
    if not scope or scope.GetTypeName() != "Scope":
        raise ValueError("Explicit collision-group scope is missing or has the wrong type")
    groups = [f"{root}/group{i}" for i in range(len(environments))]
    global_group = f"{root}/global_group"
    expected_groups = set([*groups, global_group])
    if {str(prim.GetPath()) for prim in scope.GetChildren()} != expected_groups:
        raise ValueError("Collision scope has missing or unexpected group children")
    if {str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "PhysicsCollisionGroup"} != expected_groups:
        raise ValueError("Stage has missing or additional unreviewed collision groups")

    def check_group(path, includes, allowed):
        prim = stage.GetPrimAtPath(path)
        if (prim.GetTypeName() != "PhysicsCollisionGroup"
                or "CollectionAPI:colliders" not in prim.GetAppliedSchemas()):
            raise ValueError(f"Collision group has the wrong schema: {path}")
        collection = Usd.CollectionAPI(prim, "colliders")
        if (str(collection.GetExpansionRuleAttr().Get()) != "expandPrims"
                or collection.GetIncludeRootAttr().Get() not in (None, False)
                or collection.GetExcludesRel().GetTargets()
                or prim.GetAttribute("collection:colliders:membershipExpression").HasAuthoredValueOpinion()
                or prim.GetAttribute("collection:colliders:mode").HasAuthoredValueOpinion()
                or prim.GetAttribute("physics:invertFilteredGroups").Get() not in (None, False)
                or prim.GetAttribute("physics:mergeGroup").Get() not in (None, "")):
            raise ValueError(f"Collision group changes collection or filtering semantics: {path}")
        actual_includes = [str(value) for value in collection.GetIncludesRel().GetTargets()]
        actual_allowed = [str(value) for value in prim.GetRelationship("physics:filteredGroups").GetTargets()]
        if len(actual_includes) != len(includes) or set(actual_includes) != set(includes):
            raise ValueError(f"Collision group membership differs from its assigned namespace: {path}")
        if len(actual_allowed) != len(allowed) or set(actual_allowed) != set(allowed):
            raise ValueError(f"Collision group allowlist is not isolated and reciprocal: {path}")
        query = collection.ComputeMembershipQuery()
        if not all(query.IsPathIncluded(Sdf.Path(value)) for value in includes):
            raise ValueError(f"Actual USD collection query excludes its required members: {path}")
        return {"group_path": path, "includes": sorted(actual_includes),
                "allowed_groups": sorted(actual_allowed)}

    details = []
    for environment, group in zip(environments, groups):
        details.append(dict(check_group(group, [environment], [group, global_group]), env_prim_path=environment))
    global_details = check_group(global_group, globals_, [global_group, *groups])
    runtime_identity = {
        "schema": "hexapod.physx_collision_isolation.v1",
        "mode": "explicit_usd_inverted_collision_groups",
        "authored_topology_verified": True,
        "invert_collision_group_filter": True,
        "collection_expansion_rule": "expandPrims",
        "environment_collection_membership": "exactly_own_environment_subtree",
        "environment_allowlist": "own_group_and_global_group_only",
        "global_allowlist": "global_group_and_every_environment_group",
        "global_prim_paths": sorted(globals_),
        "cross_environment_allowlinks": False,
        "global_allowlinks_reciprocal": True,
    }
    return {"runtime_identity": runtime_identity, "physics_scene_path": str(physics_scene_path),
            "collision_scope_path": root, "environment_groups": details, "global_group": global_details,
            "live_collision_response_verified": False}


def verify_physical_environment_ownership(stage, isolation, kinematics):
    """Audit every actual rigid body/collider and native-mimic USD reference."""
    from pxr import Sdf, Usd, UsdPhysics

    groups = isolation["environment_groups"]
    queries = [(row["env_prim_path"], Usd.CollectionAPI(stage.GetPrimAtPath(row["group_path"]),
                  "colliders").ComputeMembershipQuery()) for row in groups]
    global_query = Usd.CollectionAPI(stage.GetPrimAtPath(isolation["global_group"]["group_path"]),
                                     "colliders").ComputeMembershipQuery()
    results = []
    for environment, own_query in queries:
        robot = environment + "/Robot"
        root = stage.GetPrimAtPath(robot)
        if not root:
            raise ValueError("Physical environment Robot prim is missing")
        expected = {f"{robot}/{path}" for path in kinematics["body_paths"].values()}
        bodies = {str(prim.GetPath()) for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        if len(expected) != 31 or bodies != expected:
            raise ValueError("Physical environment rigid body paths differ from the 31-body model")
        colliders, covered = [], set()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(environment)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            path = prim.GetPath()
            if (not own_query.IsPathIncluded(path) or global_query.IsPathIncluded(path)
                    or any(query.IsPathIncluded(path) for other, query in queries if other != environment)):
                raise ValueError(f"Collider belongs to an incorrect collision group: {path}")
            ancestor = prim
            while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                ancestor = ancestor.GetParent()
            if not ancestor or str(ancestor.GetPath()) not in expected:
                raise ValueError(f"Collider is outside its robot's declared rigid bodies: {path}")
            covered.add(str(ancestor.GetPath()))
            colliders.append(str(path))
        if covered != expected:
            raise ValueError("Physical ownership audit did not cover every body's colliders")
        references = []
        for name, relation in kinematics["passive_relations"].items():
            path = f"{robot}/Physics/{name}"
            prim = stage.GetPrimAtPath(path)
            schemas = set(prim.GetAppliedSchemas()) if prim else set()
            authored = prim.GetMetadata("apiSchemas") if prim else None
            if authored:
                schemas.update(authored.GetAppliedItems())
            reference = f"{robot}/Physics/{relation['source_joint']}"
            if (not prim or "PhysxMimicJointAPI:rotZ" not in schemas
                    or list(prim.GetRelationship("physxMimicJoint:rotZ:referenceJoint").GetTargets()) != [Sdf.Path(reference)]
                    or not stage.GetPrimAtPath(reference)):
                raise ValueError(f"Mimic reference escapes or differs from its physical articulation: {path}")
            references.append({"joint": path, "reference_joint": reference})
        if len(references) != 12:
            raise ValueError("Physical environment does not have all twelve native mimic references")
        results.append({"environment_prim_path": environment, "rigid_body_paths": sorted(bodies),
                        "collider_paths": sorted(colliders), "mimic_references": references})
    return {"verified": True, "environment_count": len(results), "environments": results,
            "live_collision_response_verified": False}
