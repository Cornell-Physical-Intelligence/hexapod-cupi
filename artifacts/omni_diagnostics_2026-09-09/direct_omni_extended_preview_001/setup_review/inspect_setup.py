"""Local Python3.12, standard-library-only import/compile review; no execution allocation."""
import ast,hashlib,importlib.util,json,platform,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode=True
REPO=Path(sys.argv[1]).resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
out={'python_version':platform.python_version(),'platform':platform.platform(),'remote_probe':False,'GPU_allocation':False,'checks':{}}
guard=REPO/'tmp/direct_omni_extended_preview_guard_001';host=REPO/'tmp/direct_omni_preview_host_003';adapter=REPO/'tmp/direct_omni_preview_adapter_004';supervisor=REPO/'tmp/reference_physics_adapter_009/source_009';native=REPO/'tmp/direct_omni_recovery_001/native_005'
def forbidden(*args,**kwargs):raise AssertionError('No external process or GPU probe authorized in this local setup review')
with patch.object(subprocess,'run',side_effect=forbidden),patch.object(subprocess,'Popen',side_effect=forbidden),patch.object(subprocess,'check_output',side_effect=forbidden):
 g=load('_actual_guard_review',guard/'launch_guarded_remote.py');h=load('_actual_preview_host_review',host/'launch_preview_spark.py')
 for folder,bound in [(guard,'4454f8083ad0983f352b9d1e78a7ea537c21835dd53deba4dfe7ec8b2c13c413'),(host,g.HOST_FREEZE_SHA256),(adapter,g.ADAPTER_SHA256),(native,g.CONTRACT_SHA256)]:g.verify_frozen(folder,bound)
 h.verify_tree(supervisor,'campaign_source_hashes.json',g.SUPERVISOR_SHA256)
 out['checks']['all_reviewed_dependency_payloads_unchanged']=True
 # Exact installed-source modules are imported, not fake supervisor stubs.
 sys.path.insert(0,str(supervisor/'tools'));original=load('_unmodified_parent_review',supervisor/'tools/launch_reference_physics_spark.py')
 args=SimpleNamespace(supervisor_source=supervisor,source=Path('/explicit/native/source004'),output=Path('/uncreated/review/output'),adapter=adapter,contract=native,pilot=Path('/explicit/completed/extended002'),checkpoint=Path('/explicit/completed/extended002/train/policy/final.pt'),checkpoint_sha256='a'*64)
 parent=h.load_supervisor(args)
 for name in ('run_owned','owned_container'):
  old=getattr(original,name);new=getattr(parent,name)
  assert old.__code__==new.__code__,name
  assert new.__module__=='_frozen_preview_source009_supervisor'
 out['checks']['actual_load_supervisor_import_passed_no_run_owned_call']=True
 out['checks']['run_owned_and_cleanup_code_objects_equal_without_recompile']=True
 assert not any('deadline_adapter' in name for name in sys.modules)
 out['checks']['no_extended_training_deadline_adapter_loaded']=True
 assert parent.verified_source is not original.verified_source
 # Call dispatch callback, which only builds argv and performs no external work.
 argv=parent.command(args.source,args.output,'never-started-owner','recording');assert '/recording/record_direct_preview.py' in argv
 out['checks']['inherited_command_callback_calls_actual_preview_command']=True
 try:parent.verified_source(Path('/wrong/source'))
 except ValueError:pass
 else:raise AssertionError('Unexpected native source alias accepted')
 out['checks']['native_verified_source_override_installed']=True
 # Guard-style import has no cwd/sys.path helper requirement and remains stdlib only.
 assert not any(k in sys.modules for k in ('torch','numpy','isaaclab'))
 sys.path.insert(0,str(adapter));entry=load('_actual_entry_adapter_review',adapter/'entry_adapter.py')
 tree,counts=entry.instrument((adapter/'inputs/native_entry.py').read_text())
 compiled=compile(tree,str(args.source/'tools/train_length_study.py'),'exec')
 assert counts=={'app':1,'save':1,'config':1,'render':1,'dispatch':1,'shutdown':1}
 # Full-module AST compilation has no isolated function CodeType comparison.
 out['checks']['source004_whole_instrumented_entry_compiles_on_python312']=True
 out['entry_seam_counts']=counts
 recorder=load('_actual_recording_entry_review',adapter/'record_direct_preview.py')
 argv=recorder.native_arguments(SimpleNamespace(source_root=args.source,output=args.output,admission=Path('/admission/admission.json'),checkpoint=args.checkpoint,device='cuda:0',kit_args='--/exts/omni.kit.telemetry/skipDeferredStartup=true'))
 assert argv[argv.index('--mode')+1]=='evaluate' and argv[argv.index('--direct-evaluation')+1]=='constant'
 assert '--direct-allocation' not in argv and '--direct-branch' not in argv
 out['checks']['native_evaluate_cli_has_no_training_allocation']=True
 c=load('_actual_direct_config_review',native/'direct_config.py')
 assert c.selection('evaluate',None,None,'constant',None)=={'schema':'direct315_extended_native_v4','evaluation':'constant','replicas':48}
 out['checks']['actual_native005_evaluation_selector_accepted']=True
 assert not any(k in sys.modules for k in ('torch','numpy','isaaclab'))
 out['checks']['pre_app_imports_remain_stdlib_only']=True
out.update(passed=True,guard_freeze_sha256=sha(guard/'FREEZE_SHA256.json'),host_freeze_sha256=sha(host/'FREEZE_SHA256.json'),adapter_freeze_sha256=sha(adapter/'FREEZE_SHA256.json'),limitations=['Local macOS Python3.12.12; no new Spark3.12.3 execution or installed Isaac/RGB/selected-checkpoint readback occurred.','Actual terminal checkpoint and campaign remain unbound; no recording preflight or allocation is claimed.'])
print(json.dumps(out,indent=2))
