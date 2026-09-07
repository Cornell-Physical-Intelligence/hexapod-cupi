#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys
here=Path(__file__).resolve().parent
root=here.parents[2]
sys.path[:0]=[str(root/'tools'),str(root/'packages/hexapod_core')]
from mkii_training_contract import identity
from hexapod_core.fourbar_v1 import numerical_recipe
frozen=json.loads((here/'source_contract.json').read_text())
assert identity(root)==frozen,'Functional source identity differs'
review=json.loads((here/'validation.json').read_text())
assert review['source_sha256']==frozen['sha256']
assert review['selected_recipes']=={'nominal':numerical_recipe(1),'refined':numerical_recipe(2)}
for name,expected in review['changed_tracked_paths'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
for name,expected in review['protected_paths_byte_identical_to_base'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
for row in review['cpu_checks']:
 assert row['exit_code']==0 and row['pass'] is True
 assert hashlib.sha256((here/row['log']).read_bytes()).hexdigest()==row['sha256']
assert review['simulation_training_admission'] is False and review['hardware_admission'] is False
print('PASS: exact candidate source, recipes, changed/protected files and CPU evidence')
print('source_sha256='+frozen['sha256'])
