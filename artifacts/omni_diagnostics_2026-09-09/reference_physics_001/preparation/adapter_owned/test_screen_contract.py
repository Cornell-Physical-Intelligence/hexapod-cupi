import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from screen_contract import preflight, verify_source

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'tmp/omni_velocity_launch_003/source_003/robot/hexapod_mkii_length_study'
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('reference_host', HERE / 'launch_reference_physics_spark.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


def source_fixture(root):
    source = root / 'source'; package = source / 'robot/hexapod_mkii_length_study'
    package.mkdir(parents=True)
    for name in ('manifest.json', 'training_plan.json', 'candidate_c_reference.json', 'urdf/f050_t060.urdf'):
        target = package / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, target)
    freeze(source)
    args = SimpleNamespace(package=package, variant='f050_t060', stance_index=0,
        mode='standing', num_envs=32, steps=1000, output=root/'unused_output', admission=None,
        geometry_reference=package/'candidate_c_reference.json')
    return source, args


def freeze(source):
    manifest = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob('*') if p.is_file() and p.name!='campaign_source_hashes.json'}
    (source/'campaign_source_hashes.json').write_text(json.dumps(manifest))


class ScreenContractTests(unittest.TestCase):
    def test_exact_plan_and_source_admit_standing_but_wave_needs_matching_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source,args=source_fixture(root)
            identity,_=preflight(args,source)
            args.mode='wave';args.num_envs=1;args.steps=2400
            with self.assertRaisesRegex(ValueError,'admission'):
                preflight(args,source)
            args.admission=root/'admission.json'
            args.admission.write_text(json.dumps({'identity':identity,'status':'completed','gate':{'passed':True}}))
            preflight(args,source)
            changed={**identity,'source_manifest_sha256':'0'*64}
            args.admission.write_text(json.dumps({'identity':changed,'status':'completed','gate':{'passed':True}}))
            with self.assertRaisesRegex(ValueError,'Mismatched'):
                preflight(args,source)

    def test_frozen_source_rejects_changed_and_extra_files(self):
        with tempfile.TemporaryDirectory() as td:
            source,args=source_fixture(Path(td))
            (source/'unrecorded.py').write_text('pass\n')
            with self.assertRaisesRegex(ValueError,'extra'):
                verify_source(source)
            (source/'unrecorded.py').unlink()
            args.geometry_reference.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Changed'):
                verify_source(source)

    def test_fresh_source_identity_cannot_silently_change_motor_contract(self):
        with tempfile.TemporaryDirectory() as td:
            source,args=source_fixture(Path(td))
            path=args.package/'manifest.json';manifest=json.loads(path.read_text())
            manifest['actuator_config_snapshot']['effort_limit_sim']=9.
            path.write_text(json.dumps(manifest));freeze(source)
            with self.assertRaisesRegex(ValueError,'motor'):
                preflight(args,source)

    def test_host_has_only_bounded_read_only_nontraining_commands(self):
        for phase, count, steps in [('standing','32','1000'),('wave','1','2400')]:
            command=host.command(Path('/source'),Path('/output'),'owned',phase)
            self.assertIn('/source:/workspace/hexapod:ro',command)
            self.assertIn('/output/inputs/study:/study:ro',command)
            self.assertIn('PYTHONUNBUFFERED=1',command)
            self.assertEqual(command[command.index('--mode')+1],phase)
            self.assertEqual(command[command.index('--num-envs')+1],count)
            self.assertEqual(command[command.index('--steps')+1],steps)
            self.assertNotIn('--checkpoint',command)
        with self.assertRaises(ValueError):
            host.command(Path('/source'),Path('/output'),'owned','train')

    def test_cleanup_binds_exact_container_even_after_client_exit(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)
            for name in ('jobs','logs','standing'):(output/name).mkdir()
            (output/'standing/state.json').write_text(json.dumps({'status':'completed','runtime_binding':{'runtime_tree_sha256':host.RUNTIME_TREE}}))
            args=SimpleNamespace(source=Path('/source'),output=output,isaaclab=Path('/isaaclab'),coordination_sha256='same')
            process=Mock(returncode=0);process.poll.return_value=0
            with patch.object(host.os,'open',return_value=77),patch.object(host.os,'close'),patch.object(host.fcntl,'flock'), \
                 patch.object(host,'preflight',return_value={}),patch.object(host,'verified_source'), \
                 patch.object(host.subprocess,'Popen',return_value=process),patch.object(host,'digest',return_value='same'), \
                 patch.object(host,'owned_container',side_effect=[('owned-id',True),('owned-id',False)]), \
                 patch.object(host.subprocess,'run') as action:
                host.run_owned(args,'standing')
                action.assert_called_once_with(['docker','stop','--time','20','owned-id'],timeout=30,check=True,capture_output=True)

    def test_unrelated_container_identity_is_never_signaled(self):
        result=SimpleNamespace(returncode=0,stdout='unrelated /other true')
        with patch.object(host.subprocess,'run',return_value=result):
            with self.assertRaisesRegex(RuntimeError,'identity'):
                host.owned_container('owned','owned-id')

    def test_unknown_daemon_error_is_not_container_absence(self):
        result=SimpleNamespace(returncode=1,stdout='',stderr='Cannot connect to Docker daemon')
        with patch.object(host.subprocess,'run',return_value=result):
            with self.assertRaisesRegex(RuntimeError,'unknown'):
                host.owned_container('owned','owned-id')
        result.stderr='Error: No such object: owned-id'
        with patch.object(host.subprocess,'run',return_value=result):
            self.assertIsNone(host.owned_container('owned','owned-id'))


if __name__ == '__main__':unittest.main()
