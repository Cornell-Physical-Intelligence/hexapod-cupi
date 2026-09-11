"""Additional source005 composition readbacks; no solver mutation or physical gates."""
from pathlib import Path
import zipfile
from . import source005_solver_diagnostics as diagnostics


def validate_solver(record,n,*,final):
 roots={'/Robot'}|{f'/Robot_{i:03d}'for i in range(1,n)}
 phases=['before','after_authoring','after_reset','after_neutral_steps']
 if final:phases.append('after_controlled_steps')
 for phase in phases:
  values=record.get(phase,{})
  if not isinstance(values,dict)or set(values)!=roots:raise ValueError('Incomplete solver recipe roots:'+phase)
  expected=(32,1,[False,False])if phase=='before'else(32,0,[True,True])
  for v in values.values():
   if not isinstance(v,dict)or type(v.get('position_iterations'))is not int or type(v.get('velocity_iterations'))is not int or (v['position_iterations'],v['velocity_iterations'],v.get('authored'))!=expected:raise ValueError('Solver recipe differs:'+phase)
 return True


def validate_diagnostic_members(directory,session):
 # Same stdlib member-presence check as exact source005 standing_contract.
 # Runtime validates shapes/finite values and preserves all actual NPZ bytes;
 # this host check does not pretend to independently recompute numerical traces.
 d=Path(directory)
 for name in ['control_trace.npz']+session.get('substep_files',[]):
  path=d/name
  if Path(name).name!=name or path.is_symlink()or not path.resolve().is_relative_to(d.resolve()):raise ValueError('Unsafe raw diagnostic path')
  with zipfile.ZipFile(path)as archive:
   if not {diagnostics.LINK_FIELD+'.npy',diagnostics.FLOOR_FIELD+'.npy'}.issubset(archive.namelist()):raise ValueError('Missing raw diagnostic channel:'+name)
 return True
