import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import smoke_contract as contract

HERE=Path(__file__).resolve().parent

class ContractTests(unittest.TestCase):
    def test_changed_added_deleted_and_symlink_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'payload').write_text('source')
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'payload':contract.sha(root/'payload')}))
            bound=contract.sha(manifest)
            contract.verify_tree(root,'manifest.json',bound)
            (root/'payload').write_text('changed')
            with self.assertRaises(ValueError):contract.verify_tree(root,'manifest.json',bound)
            (root/'payload').write_text('source');(root/'extra').write_text('extra')
            with self.assertRaises(ValueError):contract.verify_tree(root,'manifest.json',bound)
            (root/'extra').unlink();(root/'payload').unlink()
            with self.assertRaises(ValueError):contract.verify_tree(root,'manifest.json',bound)
            (root/'payload').symlink_to(manifest)
            with self.assertRaises(ValueError):contract.verify_tree(root,'manifest.json',bound)
            with self.assertRaises(ValueError):contract.verify_tree(root,'manifest.json','wrong')

    def test_invalid_allocation_rejected_before_any_input_access(self):
        for replicas,controls in ((0,264),(2,264),(1024,264),(1,263),(32,1000)):
            with self.subTest(replicas=replicas,controls=controls),patch.object(contract,'verify_tree') as guard:
                with self.assertRaises(ValueError):contract.verify(SimpleNamespace(num_envs=replicas,controls=controls))
                guard.assert_not_called()

    def test_entrypoint_retains_frozen_builder_and_no_actor_or_pose_setter(self):
        parsed=ast.parse((HERE/'run_device_smoke.py').read_text())
        calls=[node for node in ast.walk(parsed) if isinstance(node,ast.Call)]
        forbidden={'learn','load','load_state_dict','write_root_pose_to_sim_index',
                   'write_root_velocity_to_sim_index','write_joint_position_to_sim_index',
                   'write_joint_velocity_to_sim_index'}
        self.assertFalse([n.func.attr for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr in forbidden])
        builder=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='build_reference_environment']
        self.assertEqual(len(builder),1)
        self.assertEqual(ast.dump(builder[0].args[1]),ast.dump(ast.Name(id='OPTIONS',ctx=ast.Load())))
        self.assertFalse([n for n in ast.walk(parsed) if isinstance(n,ast.ImportFrom) and any(w in (n.module or '').lower() for w in ('rsl_rl','runner','ppo'))])
        self.assertTrue(any(isinstance(n.func,ast.Name) and n.func.id=='PhysicsSubstepRecorder' for n in calls))

    def test_sensor_receipt_binds_exact_copied_source_no_clock_write_api(self):
        receipt=json.loads((HERE/'sensor_source_contract.json').read_text())
        self.assertEqual(len(receipt['installed_modules']),4)
        for entry in receipt['installed_modules'].values():
            self.assertEqual(contract.sha(HERE/entry['local_source']),entry['sha256'])
        tree=ast.parse((HERE/'sensor_freshness.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse([n.func.attr for n in calls if n.func.attr in ('update','reset','step','forward')])

if __name__=='__main__':unittest.main()
