"""Authored clone isolation and actual task wiring; no backend/GPU admission."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
for package in ("hexapod_core", "hexapod_env"):
    sys.path.insert(0, str(ROOT / "packages" / package))
from hexapod_core import fourbar_v1 as contract
from hexapod_env.tasks.mkii_fourbar_v1.collision_isolation import verify_collision_isolation


def fixture(count, *, authored=True):
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/World/customPhysicsScene")
    UsdGeom.Xform.Define(stage, "/World/ground")
    environments = [f"/World/envs/env_{i}" for i in range(count)]
    for path in environments:
        UsdGeom.Xform.Define(stage, path + "/Robot")
    scene = SimpleNamespace(stage=stage, physics_scene_path="/World/customPhysicsScene", env_prim_paths=environments)
    if authored:
        author_groups(scene)
    return scene


def author_groups(scene):
    """Author the exact installed cloner_utils.filter_collisions USD layout."""
    stage = scene.stage
    stage.GetPrimAtPath(scene.physics_scene_path).CreateAttribute(
        "physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool).Set(True)
    UsdGeom.Scope.Define(stage, "/World/collisions")
    groups = [f"/World/collisions/group{i}" for i in range(len(scene.env_prim_paths))]
    global_group = "/World/collisions/global_group"
    for path, includes, allowed in [*[(group, [env], [group, global_group])
            for group, env in zip(groups, scene.env_prim_paths)],
            (global_group, ["/World/ground"], [global_group, *groups])]:
        prim = UsdPhysics.CollisionGroup.Define(stage, path).GetPrim()
        collection = Usd.CollectionAPI.Apply(prim, "colliders")
        collection.CreateExpansionRuleAttr().Set("expandPrims")
        collection.CreateIncludesRel().SetTargets(includes)
        prim.CreateRelationship("physics:filteredGroups").SetTargets(allowed)


def verify(scene):
    return verify_collision_isolation(scene.stage, physics_scene_path=scene.physics_scene_path,
        env_prim_paths=scene.env_prim_paths, global_prim_paths=["/World/ground"])


class CollisionIsolationTests(unittest.TestCase):
    def test_real_usd_collections_are_isolated_reciprocal_and_count_invariant(self):
        identities = []
        for count in (1, 2, 32):
            scene = fixture(count)
            report = verify(scene)
            identities.append(report["runtime_identity"])
            self.assertEqual(len(report["environment_groups"]), count)
            self.assertEqual(report["physics_scene_path"], scene.physics_scene_path)
            self.assertIs(report["live_collision_response_verified"], False)
            for index, group in enumerate(report["environment_groups"]):
                collection = Usd.CollectionAPI(scene.stage.GetPrimAtPath(group["group_path"]), "colliders")
                query = collection.ComputeMembershipQuery()
                for other in range(count):
                    self.assertEqual(query.IsPathIncluded(Sdf.Path(f"/World/envs/env_{other}/Robot")), index == other)
            self.assertEqual(report["global_group"]["includes"], ["/World/ground"])
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[1], identities[2])

    def test_missing_stale_extra_or_mistyped_groups_fail_closed(self):
        for mutation in ("scope", "missing", "extra", "wrong_type", "wrong_scene", "not_inverted"):
            scene = fixture(2)
            stage = scene.stage
            if mutation == "scope": stage.RemovePrim("/World/collisions")
            elif mutation == "missing": stage.RemovePrim("/World/collisions/group1")
            elif mutation == "extra": UsdPhysics.CollisionGroup.Define(stage, "/World/unreviewed")
            elif mutation == "wrong_type": stage.GetPrimAtPath("/World/collisions/group0").SetTypeName("Scope")
            elif mutation == "wrong_scene": scene.physics_scene_path = "/World/ground"
            elif mutation == "not_inverted": stage.GetPrimAtPath(scene.physics_scene_path).GetAttribute("physxScene:invertCollisionGroupFilter").Set(False)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): verify(scene)

    def test_membership_cross_environment_and_nonreciprocal_allowlinks_fail(self):
        for mutation in ("wrong_env", "extra_env", "exclude_robot", "missing_ground", "cross_env", "nonreciprocal", "global_covers_env"):
            scene = fixture(2)
            group = scene.stage.GetPrimAtPath("/World/collisions/group0")
            collection = Usd.CollectionAPI(group, "colliders")
            global_group = scene.stage.GetPrimAtPath("/World/collisions/global_group")
            if mutation == "wrong_env": collection.GetIncludesRel().SetTargets(["/World/envs/env_1"])
            elif mutation == "extra_env": collection.GetIncludesRel().AddTarget("/World/envs/env_1")
            elif mutation == "exclude_robot": collection.CreateExcludesRel().AddTarget("/World/envs/env_0/Robot")
            elif mutation == "missing_ground": Usd.CollectionAPI(global_group, "colliders").GetIncludesRel().ClearTargets(True)
            elif mutation == "cross_env": group.GetRelationship("physics:filteredGroups").AddTarget("/World/collisions/group1")
            elif mutation == "nonreciprocal": global_group.GetRelationship("physics:filteredGroups").RemoveTarget("/World/collisions/group0")
            elif mutation == "global_covers_env": Usd.CollectionAPI(global_group, "colliders").GetIncludesRel().AddTarget("/World/envs")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): verify(scene)

    def test_collection_overrides_and_per_group_inversion_cannot_bypass_membership(self):
        for name, kind, value in (("physics:mergeGroup", Sdf.ValueTypeNames.Token, "merge_all"),
                ("physics:invertFilteredGroups", Sdf.ValueTypeNames.Bool, True),
                ("collection:colliders:includeRoot", Sdf.ValueTypeNames.Bool, True),
                ("collection:colliders:expansionRule", Sdf.ValueTypeNames.Token, "explicitOnly"),
                ("collection:colliders:mode", Sdf.ValueTypeNames.Token, "expression")):
            scene = fixture(2)
            scene.stage.GetPrimAtPath("/World/collisions/group0").CreateAttribute(name, kind).Set(value)
            with self.subTest(name=name), self.assertRaises(ValueError): verify(scene)
        scene = fixture(2)
        with self.assertRaises(ValueError):
            verify_collision_isolation(scene.stage, physics_scene_path=scene.physics_scene_path,
                env_prim_paths=scene.env_prim_paths, global_prim_paths=["/World"])

    def test_actual_setup_calls_filter_then_verification_on_cpu_and_gpu(self):
        path = ROOT / "packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py"
        tree = ast.parse(path.read_text())
        setup = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_setup_scene")
        for device in ("cpu", "cuda:0"):
            scene = fixture(2, authored=False)
            scene.cfg = SimpleNamespace(num_envs=2, env_spacing=2.)
            scene.articulations, scene.sensors = {}, {}
            events = []
            scene.clone_environments = lambda **kwargs: events.append("clone")
            def filter_collisions(*, global_prim_paths):
                self.assertEqual(events, ["clone"])
                self.assertEqual(global_prim_paths, ["/World/ground"])
                author_groups(scene)
                events.append("filter")
            scene.filter_collisions = filter_collisions
            def checked(stage, **kwargs):
                self.assertEqual(events, ["clone", "filter"])
                events.append("verify")
                return verify_collision_isolation(stage, **kwargs)
            terrain = SimpleNamespace(prim_path="/World/ground", class_type=lambda cfg: SimpleNamespace())
            raw = SimpleNamespace(device=device, scene=scene, cfg=SimpleNamespace(robot=None, terrain=terrain,
                body_contact_sensors={f"{leg}_tibia": None for leg in contract.LEGS}))
            scope = {"Articulation": lambda cfg: SimpleNamespace(), "ContactSensor": lambda cfg: SimpleNamespace(),
                     "contract": contract, "verify_collision_isolation": checked,
                     "sim_utils": SimpleNamespace(DomeLightCfg=lambda **kwargs: SimpleNamespace(func=lambda *args: None))}
            exec(compile(ast.Module(body=[setup], type_ignores=[]), str(path), "exec"), scope)
            scope["_setup_scene"](raw)
            self.assertEqual(events, ["clone", "filter", "verify"])
            self.assertTrue(raw.collision_isolation_report["runtime_identity"]["authored_topology_verified"])
            # Simulate Isaac's early return for an existing, malformed scope.
            scene.stage.RemovePrim("/World/collisions/group1")
            events.clear()
            scene.filter_collisions = lambda **kwargs: events.append("filter")
            with self.assertRaises(ValueError): scope["_setup_scene"](raw)

    def test_actual_runtime_manifest_receives_only_count_independent_identity(self):
        path = ROOT / "packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py"
        tree = ast.parse(path.read_text())
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "resolved_collision_isolation" for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        outputs = []
        for count in (1, 32):
            raw = SimpleNamespace(runtime_manifest={}, collision_isolation_report=verify(fixture(count)))
            exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), {"self": raw})
            outputs.append(copy.deepcopy(raw.runtime_manifest))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
