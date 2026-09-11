"""Run the actual current load() with a minimal late-import main; no native startup."""
from pathlib import Path
import sys,json,hashlib,tempfile
OUTPUT=Path('tmp/canonical_ppo_independent_review_001/lazy_import_counterexample_001.json')
if OUTPUT.exists():raise SystemExit('Preserve prior counterexample; choose a fresh output for a successor run.')
SOURCE=Path('tmp/canonical_ppo_integration_001').resolve();sys.path.insert(0,str(SOURCE))
from canonical_direct_ppo import native_entry_adapter as a
result={'scope':'CPU actual load-function import-lifetime counterexample','source_sha256':hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()}
with tempfile.TemporaryDirectory()as x:
 root=Path(x);(root/'standing_session.py').write_text('marker=42\n')
 original=a.adapted_source
 a.adapted_source=lambda _:('def main():\n import standing_session\n return standing_session.marker\n',{'test':'no native initialization'})
 try:
  fn,_=a.load(root,None,None,None,'test')
  try:result['returned']=fn();result['late_import_succeeded']=True
  except Exception as e:result['late_import_succeeded']=False;result['error']=repr(e)
 finally:a.adapted_source=original;sys.modules.pop('standing_session',None)
Path('tmp/canonical_ppo_independent_review_001/lazy_import_counterexample_001.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
