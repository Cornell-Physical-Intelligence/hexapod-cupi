"""Read-only solver diagnostics. None of these channels replaces a physical gate.

The two friction getters report different native properties. A zero new-model
triple does not establish that the deprecated legacy coefficient is zero.
"""
import math
from pathlib import Path

SCHEMA='canonical_solver_observability_v1'
LINK_FIELD='link_com_velocity_world'
FLOOR_FIELD='floor_contact_force_matrix_world_n'

def collect_legacy_friction(view,native,output,save):
 record={'schema':SCHEMA,'joint_names':list(native['joint_names']),
  'count':view.count,'getter':'get_dof_friction_coefficients',
  'scope':'Read-only deprecated legacy coefficients; no setter or zero-value requirement. New friction properties remain separately recorded in native_readback.json.'}
 path=Path(output)/'legacy_friction_readback.json';save(path,record)
 try:
  # Copy before validating so an unexpected native value survives in the receipt.
  import numpy as np
  values=np.array(view.get_dof_friction_coefficients().numpy(),copy=True)
  record['shape']=list(values.shape);record['coefficients']=values.tolist();save(path,record)
  validate_legacy(record,view.count,native['joint_names'])
 except Exception as error:
  record['failed_getter']={'name':record['getter'],'error':repr(error)};save(path,record);raise
 return record

def validate_legacy(record,n,names):
 if record.get('schema')!=SCHEMA or record.get('getter')!='get_dof_friction_coefficients' or record.get('joint_names')!=list(names) or record.get('count')!=n or record.get('shape')!=[n,18] or 'failed_getter'in record:
  raise ValueError('Incomplete legacy friction readback')
 values=record.get('coefficients')
 if not isinstance(values,list)or len(values)!=n or any(not isinstance(row,list)or len(row)!=18 for row in values):raise ValueError('Legacy friction readback shape differs')
 if any(type(value)not in(int,float)or not math.isfinite(value)for row in values for value in row):raise ValueError('Nonfinite legacy friction readback')
 # Nonzero coefficients are observations, never silently cleared or rejected.
 return True

def declaration(n,captured):
 return {'schema':SCHEMA,'captured_steps':captured,
  'channels':{LINK_FIELD:[n,19,6],FLOOR_FIELD:[19*n,1,3]},
  'link_velocity_scope':'Native global linear velocity at each link COM, then angular velocity; native articulation body order. Derived from articulation state, not independent angle-difference truth.',
  'floor_force_scope':'Native filtered contact-force matrix at dt=.0025, sensor order in contact_view.json and sole /Ground filter; copied independently of detailed patches but may share their backend. No replacement of toe/shaft classification.'}
