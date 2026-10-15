"""Require package entry points and complete dependency identities."""
import importlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackageCutoverTests(unittest.TestCase):
    def test_root_contains_packages_and_initializer(self):
        self.assertEqual(sorted(p.name for p in ROOT.iterdir() if p.is_file()),
                         ['README.md', '__init__.py'])

    def test_learner_configuration_uses_package_classes(self):
        from locomotion.omni.config import ppo_config
        config = ppo_config(17, action_mean='tanh', learning_rate_max=.0003)
        self.assertEqual(config['algorithm']['class_name'],
                         'locomotion.omni.algorithm:CappedRatePPO')
        self.assertEqual(config['actor']['distribution_cfg']['class_name'],
                         'locomotion.omni.algorithm:BoundedMeanGaussian')

    def test_dependency_roots_use_implementations(self):
        from locomotion.spark.sources import physics_files, source_files
        paths = physics_files(source_files(ROOT))
        self.assertIn('omni/env.py', paths)
        self.assertIn('omni/config.py', paths)
        self.assertNotIn('env.py', paths)

    def test_compatibility_has_no_global_import_hook(self):
        import sys
        before = list(sys.meta_path)
        importlib.import_module('locomotion.evaluation.checkpoint')
        self.assertEqual(before, sys.meta_path)


from locomotion.tests import test_refactor_parity as baseline_tests


class PackageSnapshotParityTests(baseline_tests.RefactorParityTests):
    fixture = ROOT/'tests/fixtures/packages_02f551fd.tar.gz'
    fixture_sha = '5857bb0e60f1785e770b6c35dc01cdb1737fe24f70d029b6da61331f74bb16d2'


if __name__ == '__main__':
    unittest.main()
