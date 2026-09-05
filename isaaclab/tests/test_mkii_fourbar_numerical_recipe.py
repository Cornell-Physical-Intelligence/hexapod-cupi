"""TGS recipe settings and fail-closed agreement between declared and actual configuration."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'packages/hexapod_core'), str(ROOT/'tools')]
from hexapod_core import fourbar_v1 as contract
from qualify_mkii_fourbar import qualify

spec = importlib.util.spec_from_file_location('_fourbar_recipe_validator', ROOT/'isaaclab/validate_mkii_fourbar.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def report(multiplier=1):
    recipe = contract.numerical_recipe(multiplier)
    return {'solver_multiplier': multiplier, 'numerical_recipe': recipe,
            'solver_iterations': [recipe['solver_position_iterations'], recipe['solver_velocity_iterations']],
            'runtime_manifest': {'resolved_simulation': {key: value for key, value in recipe.items() if key != 'recipe_id'}}}


class NumericalRecipeTests(unittest.TestCase):
    def test_nominal_and_refined_change_only_position_iterations(self):
        nominal, refined = contract.numerical_recipe(), contract.numerical_recipe(2)
        self.assertEqual(nominal['solver_position_iterations'], 64)
        self.assertEqual(refined['solver_position_iterations'], 128)
        self.assertEqual(nominal['solver_velocity_iterations'], 1)
        self.assertEqual(refined['solver_velocity_iterations'], 1)
        self.assertEqual(nominal['solver_type'], 1)
        self.assertIs(nominal['enable_external_forces_every_iteration'], True)
        self.assertEqual(nominal['physics_dt_s'], .005)
        self.assertEqual(nominal['decimation'], 4)
        self.assertEqual({key for key in nominal if nominal[key] != refined[key]}, {'solver_position_iterations'})
        for invalid in (True, 0, 3, 1., '1'):
            with self.assertRaises(ValueError):
                contract.numerical_recipe(invalid)

    def test_validator_applies_exact_recipe_without_modifying_timing_or_limits(self):
        cfg = SimpleNamespace(sim=SimpleNamespace(dt=.005, physics=SimpleNamespace(solver_type=0,
            enable_external_forces_every_iteration=False)), decimation=4,
            robot=SimpleNamespace(spawn=SimpleNamespace(articulation_props=SimpleNamespace(
                solver_position_iteration_count=32, solver_velocity_iteration_count=4))))
        for multiplier in (1, 2):
            self.assertEqual(validator.apply_numerical_recipe(cfg, multiplier), contract.numerical_recipe(multiplier))
            self.assertEqual(cfg.robot.spawn.articulation_props.solver_velocity_iteration_count, 1)
        cfg.sim.dt = .01
        observed = validator.apply_numerical_recipe(cfg, 1)
        self.assertEqual(observed['physics_dt_s'], .01)
        changed = report()
        changed['numerical_recipe'] = observed
        with self.assertRaises(ValueError):
            contract.validate_numerical_recipe_report(changed, 1)

    def test_declared_recipe_cannot_hide_old_actual_tgs_settings(self):
        for multiplier in (1, 2):
            contract.validate_numerical_recipe_report(report(multiplier), multiplier)
            for key, wrong in (('solver_type', 0), ('solver_position_iterations', 32),
                               ('solver_velocity_iterations', 8), ('enable_external_forces_every_iteration', False),
                               ('physics_dt_s', .01), ('decimation', 2)):
                with self.subTest(multiplier=multiplier, key=key):
                    changed = report(multiplier)
                    changed['runtime_manifest']['resolved_simulation'][key] = wrong
                    with self.assertRaisesRegex(ValueError, 'Actual resolved simulation'):
                        contract.validate_numerical_recipe_report(changed, multiplier)
            for key in ('numerical_recipe', 'runtime_manifest', 'solver_iterations', 'solver_multiplier'):
                changed = report(multiplier)
                del changed[key]
                with self.assertRaises(ValueError):
                    contract.validate_numerical_recipe_report(changed, multiplier)

    def test_stale_recipe_and_bool_integer_substitutions_fail(self):
        for changed in ({'solver_iterations': [32, 4]}, {'solver_iterations': [64, 8]},
                        {'solver_iterations': [64, True]}, {'solver_multiplier': True}, {'runtime_manifest': None}):
            with self.assertRaises(ValueError):
                contract.validate_numerical_recipe_report(dict(report(), **changed), 1)
        for field in ('numerical_recipe', 'resolved_simulation'):
            changed = report()
            target = changed[field] if field == 'numerical_recipe' else changed['runtime_manifest'][field]
            target['enable_external_forces_every_iteration'] = 1
            with self.assertRaises(ValueError):
                contract.validate_numerical_recipe_report(changed, 1)

    def test_config_uses_explicit_backend_and_iteration_constants(self):
        # Evaluate only the constructor expressions with recording stdlib stubs;
        # importing the full Isaac configuration is reserved for the SDK probe.
        tree = ast.parse((ROOT/'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/config.py').read_text())
        calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr: node
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ((isinstance(node.func, ast.Name) and node.func.id == 'PhysxCfg')
                      or (isinstance(node.func, ast.Attribute) and node.func.attr == 'ArticulationRootPropertiesCfg'))}
        backend = eval(compile(ast.Expression(calls['PhysxCfg']), '<config-expression>', 'eval'),
                       {'contract': contract, 'PhysxCfg': lambda **values: values})
        articulation = eval(compile(ast.Expression(calls['ArticulationRootPropertiesCfg']), '<config-expression>', 'eval'),
            {'contract': contract, 'sim_utils': SimpleNamespace(ArticulationRootPropertiesCfg=lambda **values: values)})
        self.assertEqual(backend['solver_type'], 1)
        self.assertIs(backend['enable_external_forces_every_iteration'], True)
        self.assertEqual(articulation['solver_position_iteration_count'], 64)
        self.assertEqual(articulation['solver_velocity_iteration_count'], 1)
        self.assertFalse(articulation['enabled_self_collisions'])

    def test_qualifier_requires_matching_actual_external_force_flag(self):
        identity = {'task_id': contract.TASK_ID}
        def valid(multiplier):
            value = report(multiplier)
            value.update({'pass': True, 'errors': [], 'contract': identity, 'task_id': contract.TASK_ID,
                'num_envs': 32, 'steps_completed': 1000, 'steps_requested': 1000, 'driven_steps': 2400,
                'driven_coordinate_pass': True, 'windows': {window: {'mean_height_m': .138, 'max_applied_nm': 1.2}
                    for window in ('settled', 'driven')}})
            return value
        nominal, refined = valid(1), valid(2)
        self.assertTrue(qualify(nominal, refined, identity)['pass'])
        refined['runtime_manifest']['resolved_simulation']['enable_external_forces_every_iteration'] = False
        self.assertFalse(qualify(nominal, refined, identity)['pass'])


if __name__ == '__main__':
    unittest.main()
