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
        self.assertEqual(nominal['solver_velocity_iterations'], 16)
        self.assertEqual(refined['solver_velocity_iterations'], 16)
        self.assertEqual(nominal['solver_type'], 1)
        self.assertIs(nominal['enable_external_forces_every_iteration'], True)
        self.assertEqual(nominal['physics_dt_s'], .00125)
        self.assertEqual(nominal['decimation'], 16)
        self.assertEqual(contract.POLICY_DT_S, .02)
        self.assertEqual({key for key in nominal if nominal[key] != refined[key]}, {'solver_position_iterations'})
        for invalid in (True, 0, 3, 1., '1'):
            with self.assertRaises(ValueError):
                contract.numerical_recipe(invalid)

    def test_validator_applies_exact_recipe_without_modifying_timing_or_limits(self):
        cfg = SimpleNamespace(sim=SimpleNamespace(dt=.00125, physics=SimpleNamespace(solver_type=0,
            enable_external_forces_every_iteration=False)), decimation=16,
            robot=SimpleNamespace(spawn=SimpleNamespace(articulation_props=SimpleNamespace(
                solver_position_iteration_count=32, solver_velocity_iteration_count=1))))
        for multiplier in (1, 2):
            self.assertEqual(validator.apply_numerical_recipe(cfg, multiplier), contract.numerical_recipe(multiplier))
            self.assertEqual(cfg.robot.spawn.articulation_props.solver_velocity_iteration_count, 16)
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

    def test_final_velocity_recipe_rejects_coherent_old_one_pass_evidence(self):
        self.assertEqual(contract.NUMERICAL_RECIPE_ID,
                         'mkii_fourbar_tgs_external_forces_800hz_final_velocity16_v5')
        for multiplier in (1, 2):
            old = report(multiplier)
            old['numerical_recipe'].update(
                recipe_id='mkii_fourbar_tgs_external_forces_800hz_v3', solver_velocity_iterations=1)
            old['runtime_manifest']['resolved_simulation']['solver_velocity_iterations'] = 1
            old['solver_iterations'][1] = 1
            with self.assertRaisesRegex(ValueError, 'selected TGS contract'):
                contract.validate_numerical_recipe_report(old, multiplier)
            previous_four = report(multiplier)
            previous_four['numerical_recipe'].update(
                recipe_id='mkii_fourbar_tgs_external_forces_800hz_final_velocity4_v4', solver_velocity_iterations=4)
            previous_four['runtime_manifest']['resolved_simulation']['solver_velocity_iterations'] = 4
            previous_four['solver_iterations'][1] = 4
            with self.assertRaisesRegex(ValueError, 'selected TGS contract'):
                contract.validate_numerical_recipe_report(previous_four, multiplier)
            # A fresh recipe label cannot conceal the prior actual backend setting.
            stale_backend = report(multiplier)
            stale_backend['runtime_manifest']['resolved_simulation']['solver_velocity_iterations'] = 1
            with self.assertRaisesRegex(ValueError, 'Actual resolved simulation'):
                contract.validate_numerical_recipe_report(stale_backend, multiplier)

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
        changed = report()
        for target in (changed['numerical_recipe'], changed['runtime_manifest']['resolved_simulation']):
            target.update(physics_dt_s=.005, decimation=4)
        with self.assertRaises(ValueError):
            contract.validate_numerical_recipe_report(changed, 1)

    def test_config_resolves_motor_contact_physics_and_policy_timing_together(self):
        tree = ast.parse((ROOT/'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/config.py').read_text())
        calls = {node.func.id: node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id in ('SimulationCfg', 'make_rs05_v2_cfg')}
        simulation = eval(compile(ast.Expression(calls['SimulationCfg']), '<simulation-config>', 'eval'),
            {'contract': contract, 'SimulationCfg': lambda **values: values, 'PhysxCfg': lambda **values: values,
             'sim_utils': SimpleNamespace(RigidBodyMaterialCfg=lambda **values: values)})
        motor = eval(compile(ast.Expression(calls['make_rs05_v2_cfg']), '<motor-config>', 'eval'),
            {'contract': contract, 'make_rs05_v2_cfg': lambda names, **values: values})
        sensor = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_sensor')
        namespace = {'contract': contract, 'ContactSensorCfg': lambda **values: values, 'ROBOT_PRIM': '/Robot', 'GROUND': '/Ground'}
        exec(compile(ast.Module(body=[sensor], type_ignores=[]), '<sensor-config>', 'exec'), namespace)
        contact = namespace['_sensor']('lf_tibia', 'Geometry/lf_tibia')
        self.assertEqual(simulation['dt'], .00125)
        self.assertEqual(motor['physics_dt_s'], .00125)
        self.assertEqual(contact['update_period'], .00125)
        self.assertEqual(simulation['render_interval'], 16)
        self.assertEqual(simulation['dt']*simulation['render_interval'], .02)

    def test_environment_rejects_old_or_divergent_motor_and_policy_timing(self):
        tree = ast.parse((ROOT/'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py').read_text())
        guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and any(isinstance(item, ast.Constant) and item.value == 'Physics/control timing differs from the motor/runtime contract'
                             for item in ast.walk(node)))
        code = compile(ast.Module(body=[guard], type_ignores=[]), '<actual-env-timing-guard>', 'exec')
        def check(physics=.00125, motor=.00125, decimation=16, policy=.02):
            exec(code, {'contract': contract, 'cfg': SimpleNamespace(sim=SimpleNamespace(dt=physics), decimation=decimation),
                        'motor_cfg': {'physics_dt_s': motor}, 'self': SimpleNamespace(step_dt=policy)})
        check()
        for changed in ({'physics': .005, 'motor': .005}, {'physics': .0025, 'motor': .0025},
                        {'motor': .0025}, {'motor': .005}, {'decimation': 8}, {'decimation': 4}, {'policy': .04}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    check(**changed)

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
        self.assertEqual(articulation['solver_velocity_iteration_count'], 16)
        self.assertFalse(articulation['enabled_self_collisions'])

    def test_qualifier_requires_matching_actual_external_force_flag(self):
        identity = {'task_id': contract.TASK_ID}
        def valid(multiplier):
            value = report(multiplier)
            value.update({'asset_binding': {'pass': True}, 'pass': True, 'errors': [], 'contract': identity, 'task_id': contract.TASK_ID,
                'num_envs': 32, 'steps_completed': 1000, 'steps_requested': 1000, 'driven_steps': 2400,
                'reset_root_positions_m': [[0., 0., .14297]]*32,
                'driven_coordinate_pass': True, 'windows': {window: {'mean_height_m': .138, 'max_applied_nm': 1.2, 'max_demand_nm': 1.3}
                    for window in ('settled', 'driven')}})
            return value
        nominal, refined = valid(1), valid(2)
        self.assertTrue(qualify(nominal, refined, identity)['pass'])
        refined['runtime_manifest']['resolved_simulation']['enable_external_forces_every_iteration'] = False
        self.assertFalse(qualify(nominal, refined, identity)['pass'])


if __name__ == '__main__':
    unittest.main()
