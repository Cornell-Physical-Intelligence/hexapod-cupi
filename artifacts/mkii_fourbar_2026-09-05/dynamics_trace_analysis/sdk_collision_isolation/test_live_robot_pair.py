"""CPU tests of fixture controls/report semantics and composed v5 relationships."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest

from pxr import Usd

DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_live_robot_pair_cpu_tests", DIRECTORY / "live_robot_pair.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
fixture.load_source()
sys.path.insert(0, str(fixture.ROOT / "isaaclab/tests"))
from test_mkii_fourbar_collision_isolation import fixture as usd_fixture, verify


class LivePairFixtureTests(unittest.TestCase):
    def test_actual_recipe_checker_accepts_emitted_fields_and_rejects_cfg_runtime_drift(self):
        for multiplier in (1, 2):
            recipe = fixture.contract.numerical_recipe(multiplier)
            cfg = NS(sim=NS(dt=recipe["physics_dt_s"], physics=NS(
                solver_type=recipe["solver_type"],
                enable_external_forces_every_iteration=recipe["enable_external_forces_every_iteration"])),
                decimation=recipe["decimation"], robot=NS(spawn=NS(articulation_props=NS(
                    solver_position_iteration_count=recipe["solver_position_iterations"],
                    solver_velocity_iteration_count=recipe["solver_velocity_iterations"]))))
            report = {"runtime_manifest": {"resolved_simulation": {
                key: value for key, value in recipe.items() if key != "recipe_id"}}}
            self.assertEqual(fixture.record_numerical_recipe(report, cfg, multiplier), recipe)
            self.assertEqual(fixture.contract.validate_numerical_recipe_report(report, multiplier), recipe)
            self.assertEqual(report["solver_iterations"], [recipe["solver_position_iterations"], recipe["solver_velocity_iterations"]])
            for key in ("solver_multiplier", "solver_iterations"):
                changed = copy.deepcopy(report)
                del changed[key]
                with self.assertRaisesRegex(ValueError, "Solver iteration report"):
                    fixture.contract.validate_numerical_recipe_report(changed, multiplier)
            changed = copy.deepcopy(report)
            changed["runtime_manifest"]["resolved_simulation"]["physics_dt_s"] *= 2
            with self.assertRaisesRegex(ValueError, "Actual resolved simulation"):
                fixture.record_numerical_recipe(changed, cfg, multiplier)
            cfg.robot.spawn.articulation_props.solver_velocity_iteration_count += 1
            with self.assertRaisesRegex(ValueError, "changed during environment"):
                fixture.record_numerical_recipe(report, cfg, multiplier)
            with self.assertRaisesRegex(ValueError, "selected TGS contract"):
                fixture.record_numerical_recipe({"runtime_manifest": report["runtime_manifest"]}, cfg, multiplier)

    def metrics(self, case):
        return {"samples": 256, "finite": True, "contact_buffer_capacity_reached": False,
                "max_pair_contact_count": 0 if case == "filtered" else 4,
                "max_pair_force_n": 0. if case == "filtered" else 20.,
                "ground_support_observed_per_robot": [True, True]}

    def test_control_grades_require_real_pair_counts_force_and_separate_ground_health(self):
        for case in fixture.CASES:
            self.assertEqual(fixture.grade_control(case, self.metrics(case), 256), [])
            for changes in ({"samples": 255}, {"finite": False}, {"max_pair_force_n": float("nan")},
                            {"max_pair_contact_count": True}, {"contact_buffer_capacity_reached": True}):
                self.assertTrue(fixture.grade_control(case, dict(self.metrics(case), **changes), 256))
        for changes in ({"max_pair_contact_count": 1}, {"max_pair_force_n": .001},
                        {"ground_support_observed_per_robot": [True, False]},
                        {"ground_support_observed_per_robot": [1, 1]}):
            self.assertTrue(fixture.grade_control("filtered", dict(self.metrics("filtered"), **changes), 256))
        for changes in ({"max_pair_contact_count": 0}, {"max_pair_force_n": 0.}):
            self.assertTrue(fixture.grade_control("unfiltered_negative", dict(self.metrics("unfiltered_negative"), **changes), 256))

    def test_negative_control_only_adds_reciprocal_cross_environment_allowlinks(self):
        scene = usd_fixture(2)
        original = verify(scene)
        global_targets = scene.stage.GetPrimAtPath("/World/collisions/global_group").GetRelationship("physics:filteredGroups").GetTargets()
        evidence = fixture.add_negative_allowlinks(scene.stage)
        self.assertEqual(len(evidence), 2)
        for index, record in enumerate(evidence):
            self.assertEqual(set(record["after"])-set(record["before"]), {f"/World/collisions/group{1-index}"})
        self.assertEqual(scene.stage.GetPrimAtPath("/World/collisions/global_group").GetRelationship("physics:filteredGroups").GetTargets(), global_targets)
        self.assertTrue(original["runtime_identity"]["authored_topology_verified"])
        with self.assertRaises(ValueError): verify(scene)
        with self.assertRaises(ValueError): fixture.add_negative_allowlinks(scene.stage)

    def test_actual_v5_reference_composition_resolves_all24_mimics_and_rejects_cross_clone(self):
        stage = Usd.Stage.CreateInMemory()
        usd = fixture.ROOT / fixture.contract.ASSET_BUNDLES["mkii_fourbar_v5"]["usd_path_relative"]
        environments = [f"/World/envs/env_{i}" for i in range(2)]
        for environment in environments:
            stage.DefinePrim(environment + "/Robot").GetReferences().AddReference(str(usd))
        result = fixture.mimic_reference_audit(stage, environments)
        self.assertTrue(result["resolved_relationships_verified"])
        self.assertEqual(len(result["joints"]), 24)
        bad = stage.GetPrimAtPath("/World/envs/env_1/Robot/Physics/lm_tibia_rod_pivot")
        bad.GetRelationship("physxMimicJoint:rotZ:referenceJoint").SetTargets(
            ["/World/envs/env_0/Robot/Physics/lm_tibia_lever_pivot"])
        with self.assertRaisesRegex(ValueError, "escapes or differs"):
            fixture.mimic_reference_audit(stage, environments)

    def test_pair_comparison_requires_both_controls_and_same_source_physics(self):
        reports = []
        for case in fixture.CASES:
            reports.append({"case": case, "fixture_complete": True, "control_expectation_met": True,
                "errors": [], "pass": False, "simulation_training_admission": False, "hardware_admission": False,
                "metrics": self.metrics(case), "physics_steps_requested": 256,
                "source_identity": {"sha256": "same source"}, "fixture_source_sha256": "same fixture",
                "solver_multiplier": 2, "solver_iterations": [128, 1],
                "numerical_recipe": {"dt": .00125}, "initial_reset_roots_m": [[0., 0., .14297]]*2,
                "runtime_manifest": {"motor": {"kp": 30., "kd": .3}, "resolved_collision_isolation": {"mode": case}}})
        self.assertEqual(fixture.compare_cases(*reports), [])
        for key, value in (("source_identity", {}), ("pass", True), ("fixture_complete", False),
                           ("physics_steps_requested", 512), ("runtime_manifest", {"motor": {"kd": .6}})):
            changed = copy.deepcopy(reports)
            changed[1][key] = value
            self.assertTrue(fixture.compare_cases(*changed))

    def test_external_fixture_import_and_compare_parser_need_no_production_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "live_robot_pair.py"
            copied.write_bytes((DIRECTORY / "live_robot_pair.py").read_bytes())
            code = """
import importlib.util, sys
spec=importlib.util.spec_from_file_location('external_pair',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.ROOT is None
assert not any(name.startswith(('hexapod_', 'isaaclab', 'pxr', 'torch')) for name in sys.modules)
args=m.parser().parse_args(['--compare','a/report.json','b/report.json','--report','pair/report.json'])
assert args.source_dir is None and args.compare is not None
assert sys.dont_write_bytecode is True
print('EXTERNAL_FIXTURE_IMPORT_PASS')
"""
            result = subprocess.run([sys.executable, "-I", "-c", code, str(copied)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("EXTERNAL_FIXTURE_IMPORT_PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
