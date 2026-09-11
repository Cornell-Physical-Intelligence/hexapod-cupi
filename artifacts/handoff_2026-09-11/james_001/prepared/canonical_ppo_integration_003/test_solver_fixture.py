"""Clearly synthetic CPU receipt helpers; no native or standing-admission evidence."""
import json,zipfile
from pathlib import Path
from canonical_direct_ppo import source005_solver_diagnostics as diag


def solver_record(n=32,final=True):
 roots=['/Robot']+[f'/Robot_{i:03d}'for i in range(1,n)]
 phases=['before','after_authoring','after_reset','after_neutral_steps']+(['after_controlled_steps']if final else [])
 return {phase:{r:{'position_iterations':32,'velocity_iterations':1 if phase=='before'else 0,'authored':[False,False]if phase=='before'else[True,True]}for r in roots}for phase in phases}


def write_native_records(root,n=32,final=True):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);names=[f'CPU_q{i}'for i in range(18)]
 records={'solver_readback.json':solver_record(n,final),
 'native_readback.json':{'joint_names':names,'CPU_PROTOCOL_FIXTURE_ONLY':True},
 'legacy_friction_readback.json':{'schema':diag.SCHEMA,'getter':'get_dof_friction_coefficients','joint_names':names,'count':n,'shape':[n,18],'coefficients':[[.125]*18 for _ in range(n)]},
 'contact_view.json':{'CPU_PROTOCOL_FIXTURE_ONLY':True,'filter_paths':['/Ground']}}
 for name,data in records.items():(root/name).write_text(json.dumps(data))
 return names


def write_member_fixture(path,omit=None):
 # This exercises the stdlib host member barrier only. It deliberately does
 # not claim these tiny bytes are complete native NumPy arrays or a scorer fixture.
 with zipfile.ZipFile(path,'w')as z:
  for field in [diag.LINK_FIELD,diag.FLOOR_FIELD]:
   if field!=omit:z.writestr(field+'.npy',b'CPU MEMBER-PRESENCE FIXTURE ONLY')
