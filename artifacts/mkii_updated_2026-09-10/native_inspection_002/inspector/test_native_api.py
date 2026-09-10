"""Narrow regression for the actual110.1.13 public API and optional legacy counter."""
import ast
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_inspection as entry
import inspect_core as core

HERE=Path(__file__).resolve().parent

class Compatibility(unittest.TestCase):
 def test_actual_extracted_factory_provider_without_private_import(self):
  root=HERE.parent/'canonical_native_sdk_readback_001/files'
  paths=list(root.glob('**/tensors/api.py'))
  if not paths:self.skipTest('External exact image readback not supplied; retained actual run receipt is separate')
  path=paths[0];package=path.with_name('__init__.py')
  self.assertEqual(entry.sha(path),entry.TENSOR_API_SHA)
  self.assertEqual(entry.sha(package),entry.TENSOR_PACKAGE_SHA)
  tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='create_simulation_view')
  ns={'__name__':'omni.physics.tensors.api'}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
  # Defining the exact factory does not execute it, acquire a backend or import CUDA.
  tensors=SimpleNamespace(__file__=str(package),create_simulation_view=ns['create_simulation_view'])
  r=entry.tensor_provider(tensors)
  self.assertEqual(r['tensor_api_sha256'],entry.TENSOR_API_SHA)
  self.assertNotIn('omni.physics.tensors.impl.api',sys.modules)
  with patch.object(entry,'TENSOR_API_SHA','0'*64):
   with self.assertRaises(ValueError):entry.tensor_provider(tensors)

 def test_old_private_path_fails_in_public_only_package(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'fake_tensors';p.mkdir();(p/'__init__.py').write_text('from .api import create_simulation_view\n')
   (p/'api.py').write_text('def create_simulation_view(frontend_name, stage_id=-1, backend="physx"):\n return None\n')
   sys.path.insert(0,d)
   try:
    public=importlib.import_module('fake_tensors')
    with self.assertRaises(ModuleNotFoundError):importlib.import_module('fake_tensors.impl.api')
    with patch.object(entry,'TENSOR_API_SHA',entry.sha(p/'api.py')),patch.object(entry,'TENSOR_PACKAGE_SHA',entry.sha(p/'__init__.py')):
     self.assertEqual(Path(entry.tensor_provider(public)['tensor_api_source']),(p/'api.py').resolve())
   finally:
    sys.path.remove(d)
    for key in list(sys.modules):
     if key.startswith('fake_tensors'):sys.modules.pop(key)

 def test_optional_counter_absent_is_null_never_zero(self):
  self.assertEqual(entry.legacy_cooking_counter(SimpleNamespace())['pending'],None)
  a=SimpleNamespace(get_physx_cooking_interface=lambda:SimpleNamespace())
  r=entry.legacy_cooking_counter(a);self.assertFalse(r['available']);self.assertIsNone(r['pending'])
  a.get_physx_cooking_interface=lambda:SimpleNamespace(get_num_collision_tasks=lambda:0)
  self.assertEqual(entry.legacy_cooking_counter(a),{'available':True,'pending':0,'reason':None})
  a.get_physx_cooking_interface=lambda:SimpleNamespace(get_num_collision_tasks=lambda:-1)
  with self.assertRaises(ValueError):entry.legacy_cooking_counter(a)

 def test_returned_initialization_required_for_unknown_counter(self):
  usd={'colliders':[{'path':f'/Robot/c{i}'}for i in range(153)]}
  r={'count':153,'valid':True,'paths':[x['path']for x in usd['colliders']],
     'contract':'canonical_native_sdf_initialization_v2',
     'initialization_barrier':{'sim_reset_returned':True,'physics_view_valid':True,'articulation_view_valid':True},
     'legacy_task_counter':{'available':False,'pending':None}}
  self.assertTrue(core.validate_sdf(r,usd))
  r['initialization_barrier']['physics_view_valid']=False
  with self.assertRaises(ValueError):core.validate_sdf(r,usd)
  r['initialization_barrier']['physics_view_valid']=True;r['legacy_task_counter']['pending']=0
  with self.assertRaises(ValueError):core.validate_sdf(r,usd)
  r['legacy_task_counter']={'available':True,'pending':3}
  with self.assertRaises(ValueError):core.validate_sdf(r,usd)

if __name__=='__main__':unittest.main()
