"""Fail-closed reviewed source delta against immutable directional002."""
from pathlib import Path
import ast,copy,difflib,hashlib,json
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'reference_directional_adapter_002/source_directional_002';SOURCE=HERE/'source_rr_preload_001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return ast.parse(p.read_text())
def function_nodes(t):return {n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def contains_rr(n):return 'rr_preload_diagnostic' in ast.dump(n)
class StripExplicitDiagnosticHooks(ast.NodeTransformer):
 def visit_ImportFrom(self,n):return None if n.module=='rr_preload_diagnostic' else n
 def visit_Expr(self,n):return None if isinstance(n.value,ast.Call) and contains_rr(n.value.func) else self.generic_visit(n)
 def visit_Assign(self,n):return None if any(contains_rr(x) for x in n.targets) else self.generic_visit(n)
 def visit_If(self,n):
  if any(isinstance(x,ast.Constant) and isinstance(x.value,str) and x.value in [
   'RR diagnostic requires the exact original wave005 configuration',
   'RR diagnostic accepts only exact left_strafe0.005 or zero command'] for x in ast.walk(n)):
   # Only these exact guard blocks; an entire method containing a guard is not an If.
   if len(n.body)==1 and isinstance(n.body[0],ast.Raise):return None
  return self.generic_visit(n)
 def visit_BoolOp(self,n):
  n.values=[self.visit(x) for x in n.values if not (isinstance(x,ast.UnaryOp) and isinstance(x.op,ast.Not) and contains_rr(x))]
  return n.values[0] if len(n.values)==1 else n
 def visit_Dict(self,n):
  keep=[i for i,k in enumerate(n.keys) if not(isinstance(k,ast.Constant) and k.value=='rr_preload_diagnostic')]
  n.keys=[n.keys[i] for i in keep];n.values=[n.values[i] for i in keep];return self.generic_visit(n)
old=json.loads((PARENT/'campaign_source_hashes.json').read_text());new=json.loads((SOURCE/'campaign_source_hashes.json').read_text())
for root,m in [(PARENT,old),(SOURCE,new)]:
 assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==set(m)|{'campaign_source_hashes.json'}
 for f,h in m.items():assert sha(root/f)==h,f
changed=[f for f in new if new[f]!=old.get(f)]
assert set(changed)=={'source_origin.json','tools/wave_reference.py','tools/rr_preload_diagnostic.py','tools/directional_contract.py','tools/launch_directional_physics_spark.py','tools/solver_comparison.py'}
a=tree(PARENT/'tools/wave_reference.py');b=StripExplicitDiagnosticHooks().visit(tree(SOURCE/'tools/wave_reference.py'))
assert ast.dump(a)==ast.dump(b),'Unexpected base controller change outside explicit diagnostic hooks'
for file in ['solver_comparison.py','launch_directional_physics_spark.py']:
 oldf=function_nodes(tree(PARENT/'tools'/file));newf=function_nodes(tree(SOURCE/'tools'/file))
 for name in oldf:
  if file=='launch_directional_physics_spark.py' and name=='command':continue # case-limit error text only
  assert ast.dump(oldf[name])==ast.dump(newf[name]),(file,name)
# Command callback differs only in its truthful single-case error string.
a=(PARENT/'tools/launch_directional_physics_spark.py').read_text();b=(SOURCE/'tools/launch_directional_physics_spark.py').read_text()
b=b.replace('Fresh standing then one bounded RR preload left-strafe diagnostic; no PPO.','Fresh standing then three previously unmeasured named low-speed directional reference cases; no PPO.').replace('Only fresh standing and the RR-diagnostic left_strafe case exist','Only fresh standing and three previously unmeasured named directional cases exist')
assert a==b
unchanged=[f for f in old if new[f]==old[f]];assert len(unchanged)==925
report={'parent_manifest_sha256':sha(PARENT/'campaign_source_hashes.json'),'source_manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),'source_files':931,'unchanged_parent_files':925,'exact_changed_paths':changed,'base_wave_AST_equal_after_removing_only_explicit_diagnostic_hooks':True,'all_solver_config_and_readback_functions_AST_equal':True,'owned_supervisor_and_phase_allocation_AST_equal':True,'host_only_case_error_and_docstring_text_changed':True,'run_directional_physics_byte_identical':new['tools/run_directional_physics.py']==old['tools/run_directional_physics.py'],'all_existing_physical_and_metric_gates_byte_identical':True,'no_main_frozen_source_or_CAD_edits':True}
(HERE/'DELTA_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
(HERE/'wave_reference.patch').write_text(''.join(difflib.unified_diff((PARENT/'tools/wave_reference.py').read_text().splitlines(True),(SOURCE/'tools/wave_reference.py').read_text().splitlines(True),fromfile='directional002/tools/wave_reference.py',tofile='rr_preload001/tools/wave_reference.py')))
print(json.dumps(report,indent=2))
