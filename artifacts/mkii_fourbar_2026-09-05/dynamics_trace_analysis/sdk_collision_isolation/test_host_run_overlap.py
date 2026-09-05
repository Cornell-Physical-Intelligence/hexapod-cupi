"""CPU-only launch/report/ownership regression checks; never invokes Docker."""
import copy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('overlap_host_test', Path(__file__).with_name('host_run_overlap.py'))
host = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host)
ROOT = Path(__file__).resolve().parents[4]
supervisor = host.load_supervisor(ROOT)


class OverlapHostTests(unittest.TestCase):
    def fixture(self, directory, case='filtered'):
        trace = directory / 'trace.npz'
        trace.write_bytes(b'opaque archived trace bytes')
        paths = [f'/World/envs/env_{i}/Robot/Geometry/body' for i in range(2)]
        value = {'schema': 'hexapod.live_collision_pair_fixture.v1', 'case': case,
                 'pass': False, 'fixture_complete': True, 'control_expectation_met': True,
                 'simulation_training_admission': False, 'hardware_admission': False,
                 'physics_steps_requested': 256, 'errors': [], 'source_identity': {'identity': 'frozen'},
                 'fixture_source_sha256': 'a' * 64,
                 'native_pair_bindings': [{'source_body': paths[i], 'target_filter': paths[1-i]} for i in range(2)],
                 'metrics': {'samples': 256, 'finite': True, 'contact_buffer_capacity_reached': False,
                     'max_pair_contact_count': 0 if case == 'filtered' else 3,
                     'max_pair_force_n': 0. if case == 'filtered' else 5.,
                     'ground_support_observed_per_robot': [True, True]},
                 'trace': {'file': 'trace.npz', 'samples': 256,
                           'sha256': hashlib.sha256(trace.read_bytes()).hexdigest()}}
        return value

    def verify(self, path, value):
        path.write_text(json.dumps(value))
        return host.require_report(path, case=value['case'], steps=256, fixture_hash='a' * 64,
                                   contract={'identity': 'frozen'}, solver_multiplier=2)

    def test_compose_separates_readonly_inputs_and_owned_output_with_cpu_barrier(self):
        argv = host.compose_argv(supervisor, Path('/frozen'), Path('/fixture'), Path('/output'),
                                 'owned-name', 'unpredictable-token', 'unfiltered_negative', 256)
        for item in ('/frozen:/workspace/hexapod:ro', '/fixture:/workspace/overlap_fixture:ro',
                     '/output:/workspace/validation_artifacts:rw',
                     '/workspace/overlap_fixture/live_robot_pair.py'):
            self.assertIn(item, argv)
        self.assertIn(f'{supervisor.OWNER_LABEL}=unpredictable-token', argv)
        self.assertEqual(argv[argv.index('--source-dir')+1], '/workspace/hexapod')
        self.assertEqual(argv[argv.index('--physics-steps')+1], '256')
        self.assertEqual(argv[argv.index('--solver-multiplier')+1], '2')
        self.assertIn('while [ ! -f /workspace/validation_artifacts/admitted ]', argv[argv.index('-c')+1])
        self.assertIn('exec "$@"', argv[argv.index('-c')+1])
        self.assertNotIn('--rm', argv)

    def test_both_controls_require_real_contact_evidence_and_keep_admission_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case in host.CASES:
                result = self.verify(root / 'report.json', self.fixture(root, case))
                self.assertIs(result['pass'], False)
                self.assertIs(result['simulation_training_admission'], False)

    def test_report_failure_modes_are_rejected_despite_exit_zero_and_complete_flag(self):
        mutations = [lambda r: r.update(pass_=True),
                     lambda r: r.update(simulation_training_admission=True),
                     lambda r: r.update(fixture_complete=False),
                     lambda r: r.update(source_identity={'identity': 'changed'}),
                     lambda r: r.update(fixture_source_sha256='b' * 64),
                     lambda r: r.update(native_pair_bindings=[]),
                     lambda r: r['metrics'].update(samples=255),
                     lambda r: r['metrics'].update(contact_buffer_capacity_reached=True),
                     lambda r: r['metrics'].update(max_pair_contact_count=1),
                     lambda r: r['metrics'].update(max_pair_force_n=float('nan')),
                     lambda r: r['metrics'].update(ground_support_observed_per_robot=[True, False]),
                     lambda r: r['metrics'].update(ground_support_observed_per_robot=[1, 1]),
                     lambda r: r.update(metrics=None),
                     lambda r: r.update(trace=None),
                     lambda r: r['trace'].update(sha256='b' * 64)]
        # 'pass' is a reserved Python keyword; mutate it explicitly.
        mutations[0] = lambda r: r.__setitem__('pass', True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self.fixture(root)
            for mutate in mutations:
                result = copy.deepcopy(base); mutate(result)
                with self.subTest(value=result), self.assertRaises(ValueError):
                    self.verify(root / 'report.json', result)
            negative = self.fixture(root, 'unfiltered_negative')
            negative['metrics'].update(max_pair_contact_count=0, max_pair_force_n=0.)
            with self.assertRaises(ValueError):
                self.verify(root / 'report.json', negative)

    def test_cleanup_stops_and_removes_only_verified_immutable_id(self):
        identifier, owner, name = 'a' * 64, 'token', 'owned'
        current = {'id': identifier, 'name': '/' + name, 'running': True, 'pid': 123,
                   'exit_code': 0, 'labels': {supervisor.OWNER_LABEL: owner}}
        after = dict(current, running=False)
        fake = SimpleNamespace(Blocked=supervisor.Blocked, check_identity=supervisor.check_identity,
            inspect_container=Mock(side_effect=[current, after]),
            command=Mock(return_value=SimpleNamespace(returncode=0)))
        with tempfile.TemporaryDirectory() as tmp, patch.object(host.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as logs:
            report = {}
            host.cleanup_owned(fake, current, name, owner, Path(tmp), report)
            self.assertEqual(report['cleanup'], 'removed_exact_id')
            self.assertEqual(fake.command.call_args_list[0].args[0], ['docker', 'stop', '--time', '25', identifier])
            self.assertEqual(fake.command.call_args_list[1].args[0], ['docker', 'rm', identifier])
            self.assertEqual(logs.call_args.args[0], ['docker', 'logs', '--timestamps', identifier])

    def test_cleanup_refuses_foreign_name_or_label_and_replaced_identity(self):
        own = {'id': 'a' * 64, 'name': '/owned', 'running': True, 'pid': 123,
               'exit_code': 0, 'labels': {supervisor.OWNER_LABEL: 'token'}}
        foreign = dict(own, labels={supervisor.OWNER_LABEL: 'someone-else'})
        for initial, inspected in ((foreign, own), (own, foreign)):
            fake = SimpleNamespace(Blocked=supervisor.Blocked, check_identity=supervisor.check_identity,
                                   inspect_container=Mock(return_value=inspected), command=Mock())
            with tempfile.TemporaryDirectory() as tmp, patch.object(host.subprocess, 'run') as logs:
                with self.assertRaises(supervisor.Blocked):
                    host.cleanup_owned(fake, initial, 'owned', 'token', Path(tmp), {})
                fake.command.assert_not_called(); logs.assert_not_called()

    def test_resource_gate_failure_before_creation_never_releases_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture = root / 'fixture'; fixture.mkdir()
            (fixture / host.FIXTURE_SCRIPT).write_text('# fixture')
            fake = SimpleNamespace(**{key: getattr(supervisor, key) for key in
                    ('Blocked', 'OWNER_LABEL', 'CONTAINER_SOURCE', 'CONTAINER_OUTPUT', 'TELEMETRY_ARGUMENT',
                     'LAB', 'atomic_json', 'check_identity')})
            fake.resource_gate = Mock(side_effect=supervisor.Blocked('Foreign GPU process'))
            fake.command = Mock(); fake.inspect_container = Mock(return_value=None)
            fake.require_unchanged_source = Mock(); fake.coordination_snapshot = Mock(return_value='shared')
            args = SimpleNamespace(steps=256, solver_multiplier=2, timeout_seconds=600, source_commit='abcdef0')
            output = root / 'output'
            self.assertFalse(host.run_case(fake, args, root / 'source', fixture, output,
                             {'identity': 'frozen'}, host.tree_identity(fixture), 'shared', 'filtered'))
            fake.command.assert_not_called()
            self.assertFalse((output / 'admitted').exists())
            report = json.loads((output / 'supervisor.json').read_text())
            self.assertEqual(report['execution'], 'failed')
            self.assertIn('Foreign GPU', report['error'])

    def test_external_fixture_edits_and_symlinks_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); script = root / 'driver.py'; script.write_text('first')
            before = host.tree_identity(root)
            script.write_text('second')
            self.assertNotEqual(before, host.tree_identity(root))
            (root / 'escape').symlink_to(script)
            with self.assertRaises(ValueError):
                host.tree_identity(root)

    def test_actual_fixture_metadata_writer_satisfies_real_host_recipe_and_asset_checks(self):
        from hexapod_core import fourbar_v1 as core
        fixture_path = Path(__file__).with_name('live_robot_pair.py')
        spec = importlib.util.spec_from_file_location('overlap_fixture_actual_integration', fixture_path)
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        fixture_module.load_source(ROOT)
        source_identity = supervisor.identity(ROOT)
        bundle = core.resolve_asset_bundle(ROOT / core.ASSET_BUNDLES['mkii_fourbar_v5']['usd_path_relative'], repo_root=ROOT)
        tree = ast.parse(fixture_path.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == 'record_numerical_recipe' for node in ast.walk(main)))
        for multiplier in (1, 2):
            recipe = core.numerical_recipe(multiplier)
            cfg = SimpleNamespace(sim=SimpleNamespace(dt=recipe['physics_dt_s'],
                physics=SimpleNamespace(solver_type=recipe['solver_type'],
                    enable_external_forces_every_iteration=recipe['enable_external_forces_every_iteration'])),
                decimation=recipe['decimation'], robot=SimpleNamespace(spawn=SimpleNamespace(
                    articulation_props=SimpleNamespace(
                        solver_position_iteration_count=recipe['solver_position_iterations'],
                        solver_velocity_iteration_count=recipe['solver_velocity_iterations']))))
            with self.subTest(multiplier=multiplier), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                result = self.fixture(directory)
                result['source_identity'] = source_identity
                result['runtime_manifest'] = dict(bundle, asset_bundle=bundle,
                    resolved_simulation={key: value for key, value in recipe.items() if key != 'recipe_id'})
                fixture_module.record_numerical_recipe(result, cfg, multiplier)
                path = directory / 'report.json'
                path.write_text(json.dumps(result))
                host.require_report(path, case='filtered', steps=256, fixture_hash='a' * 64,
                                    contract=source_identity, solver_multiplier=multiplier)
                self.assertEqual(supervisor.validate_numerical_recipe_report(result, multiplier), recipe)
                self.assertEqual(supervisor.require_requested_asset(result, 'mkii_fourbar_v5', source_identity), bundle)
                incomplete = copy.deepcopy(result); incomplete.pop('solver_iterations')
                with self.assertRaises(ValueError):
                    supervisor.validate_numerical_recipe_report(incomplete, multiplier)
                changed = copy.deepcopy(result)
                changed['runtime_manifest']['resolved_simulation']['solver_position_iterations'] += 1
                with self.assertRaises(ValueError):
                    supervisor.validate_numerical_recipe_report(changed, multiplier)

    def test_live_loop_records_owned_gpu_and_shared_note_change_stops_only_owned_case(self):
        for shared_change in (False, True):
            with self.subTest(shared_change=shared_change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); fixture = root / 'fixture'; fixture.mkdir()
                (fixture / host.FIXTURE_SCRIPT).write_text('# fixture')
                state = {'id': 'a' * 64, 'name': '/hexapod-overlap-filtered-0123456789ab',
                         'running': True, 'pid': 123, 'exit_code': 0,
                         'labels': {supervisor.OWNER_LABEL: '0123456789abcdef'}}
                fake = SimpleNamespace(**{key: getattr(supervisor, key) for key in
                    ('Blocked', 'OWNER_LABEL', 'CONTAINER_SOURCE', 'CONTAINER_OUTPUT', 'TELEMETRY_ARGUMENT',
                     'LAB', 'atomic_json', 'check_identity')})
                def command(argv, **kwargs):
                    if argv[:2] == ['docker', 'stop']:
                        state['running'] = False
                    return SimpleNamespace(returncode=0)
                def gate(**kwargs):
                    if kwargs.get('allow_owned_gpu'):
                        state['running'] = False
                        return {'gpu_pids': [123]}
                    return {'gpu_pids': []}
                fake.command = Mock(side_effect=command)
                fake.resource_gate = Mock(side_effect=gate)
                fake.inspect_container = Mock(side_effect=lambda *_: dict(state))
                fake.require_unchanged_source = Mock()
                fake.coordination_snapshot = Mock(side_effect=['shared', 'shared', 'changed', 'changed']
                                                 if shared_change else lambda: 'shared')
                fake.validate_numerical_recipe_report = Mock(); fake.require_requested_asset = Mock()
                args = SimpleNamespace(steps=256, solver_multiplier=2, timeout_seconds=600, source_commit='abcdef0')
                output = root / 'output'
                with patch.object(host.uuid, 'uuid4', return_value=SimpleNamespace(hex='0123456789abcdef')), \
                        patch.object(host.time, 'sleep'), patch.object(host.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                        patch.object(host, 'require_report', return_value={'fixture_complete': True}):
                    # A placeholder final report is used only after the successful mocked native loop.
                    original_atomic = fake.atomic_json
                    def atomic(path, value):
                        original_atomic(path, value)
                        if path.name == 'supervisor.json' and value['execution'] == 'running':
                            (path.parent / 'report.json').write_text('{}')
                    fake.atomic_json = atomic
                    completed = host.run_case(fake, args, root / 'source', fixture, output,
                        {'identity': 'frozen'}, host.tree_identity(fixture), 'shared', 'filtered')
                self.assertEqual(completed, not shared_change)
                result = json.loads((output / 'supervisor.json').read_text())
                self.assertEqual(result['cleanup'], 'removed_exact_id')
                self.assertIs(result['pass'], False)
                stops = [call.args[0] for call in fake.command.call_args_list if call.args[0][:2] == ['docker', 'stop']]
                if shared_change:
                    self.assertIn('Coordination changed', result['error'])
                    self.assertEqual(stops, [['docker', 'stop', '--time', '25', 'a' * 64]])
                else:
                    self.assertEqual(result['observed_owned_gpu_pids'], [123])
                    self.assertEqual(result['execution'], 'diagnostic_complete')
                    self.assertEqual(stops, [])


if __name__ == '__main__':
    unittest.main()
