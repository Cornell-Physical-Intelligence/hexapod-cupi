"""Metadata-only execution wrapper; no AppLauncher monkeypatch or control-function edits."""
from pathlib import Path
import argparse,ast,hashlib,json,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
from launch_train_spark import SOURCE_MAP,LEGACY_RUNTIME_TREE,sha,read,verify_tree,verify_legacy_runtime

verify_runtime=verify_legacy_runtime

def binding(source,expected):
 package=(source/'isaaclab/hexapod_rl').resolve();origins={}
 for name,module in list(sys.modules.items()):
  if name=='hexapod_rl' or name.startswith('hexapod_rl.'):
   path=getattr(module,'__file__',None)
   if path is None:raise ValueError('Missing legacy module origin: '+name)
   p=Path(path).resolve()
   if not p.is_relative_to(package):raise ValueError('Wrong legacy module origin: '+name)
   key=p.relative_to(source).as_posix()
   if key not in expected or sha(p)!=expected[key]:raise ValueError('Unpinned legacy import: '+name)
   origins[name]={'path':str(p),'sha256':expected[key]}
 if 'hexapod_rl.env' not in origins:raise ValueError('Actual legacy environment not imported')
 return {'mode':'legacy_exact_source_metadata_wrapper','package_directory':str(package),
         'scope':'This binding verifies the external metadata wrapper and legacy package; native training updates and overlays are separately declared in source/selection/receipts.',
         'runtime_tree_sha256':LEGACY_RUNTIME_TREE,'runtime_tree_hash_method':'SHA256 of sorted compact JSON map from package-relative Python path to file SHA256; same method as pinned C runtime',
         'python_files_verified':16,'source_manifest_sha256':SOURCE_MAP,'python_files':expected,
         'imported_module_origins':origins,'metadata_only':True,'actor_or_physics_changed':False}

def instrument(text):
 tree=ast.parse(text)
 app=[i for i,node in enumerate(tree.body) if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='app' for x in node.targets)]
 saves=[i for i,node in enumerate(tree.body) if isinstance(node,ast.FunctionDef) and node.name=='save']
 if len(app)!=1 or len(saves)!=1:raise ValueError('Unexpected legacy metadata insertion sites')
 node=tree.body[app[0]]
 if ast.dump(node.value)!=ast.dump(ast.parse('AppLauncher(args).app',mode='eval').body):raise ValueError('AppLauncher construction shape changed')
 for i,code in sorted([(app[0]+1,'_cold_app_ready()'),(saves[0]+1,'save = _cold_bind_save(save)')],reverse=True):
  tree.body.insert(i,ast.copy_location(ast.parse(code).body[0],tree.body[i-1]))
 return ast.fix_missing_locations(tree)

def execute(source,argv):
 verify_tree(source,'campaign_source_hashes.json',SOURCE_MAP);expected=verify_runtime(source)
 if not argv or Path(argv[0]).resolve()!=source/'tools/train_length_study.py':raise ValueError('Only exact frozen entrypoint is allowed')
 entry=source/'tools/train_length_study.py';tree=instrument(entry.read_text())
 def ready():print('REFERENCE_SCREEN_APP_READY',flush=True)
 def bind_save(original):
  binding(source,expected) # Verify actual origins after old imports and before main.
  def save_metadata(path,data):
   if Path(path).name=='state.json':
    if 'runtime_binding' in data:raise ValueError('Legacy state unexpectedly already has a runtime binding')
    data={**data,'runtime_binding':binding(source,expected)}
   return original(path,data)
  return save_metadata
 sys.argv=list(argv)
 namespace={'__name__':'__main__','__file__':str(entry),'__package__':None,
            '_cold_app_ready':ready,'_cold_bind_save':bind_save}
 exec(compile(tree,str(entry),'exec'),namespace)

def main():
 try:i=sys.argv.index('--')
 except ValueError:raise ValueError('Separate immutable legacy CLI with --')
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--source-root',type=Path,required=True)
 args=p.parse_args(sys.argv[1:i]);execute(args.source_root.resolve(),sys.argv[i+1:])
if __name__=='__main__':main()
