"""Bounded paper-walk allocation using the existing exact ownership supervisor.

The selected native command is bound before launch. No historical robot task,
checkpoint, standing gate or reservation policy is repointed by this adapter.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import signal
import sys
import types

sys.dont_write_bytecode = True
import reservation as guard

SUPERVISOR = Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009')
SUPERVISOR_SHA = '9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
SUPERVISOR_MAP = '04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
REMOTE_ROOT = Path('/home/orionh/HEXAPOD_runs/restart_20260914')


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapt_deadline(parent, path, seconds):
    """Only the three previously reviewed deadline seams change; finally is equal."""
    source=path.read_text()
    guard.require(guard.sha(path)==SUPERVISOR_SHA,'Wrong ownership supervisor')
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_owned')
    original=ast.get_source_segment(source,node)
    swaps=[('deadline_seconds=600',f'deadline_seconds={seconds}'),
           ('deadline = time.monotonic() + 600',f'deadline = time.monotonic() + {seconds}'),
           ('raise TimeoutError("Standing phase exceeded ten-minute bound")',
            f'raise TimeoutError("Paper-walk phase exceeded {seconds}-second bound")')]
    adapted=original
    for before,after in swaps:
        guard.require(adapted.count(before)==1,'Missing or ambiguous ownership seam')
        adapted=adapted.replace(before,after,1)
    final=lambda s:ast.dump(ast.Module(body=next(n for n in ast.parse(s).body[0].body
                                               if isinstance(n,ast.Try)).finalbody,type_ignores=[]))
    guard.require(final(original)==final(adapted),'Ownership cleanup changed')
    reverse=adapted
    for before,after in reversed(swaps): reverse=reverse.replace(after,before,1)
    guard.require(reverse==original,'Additional ownership source changes')
    compiled=compile(adapted,str(path)+'::paper_walk_deadline','exec')
    code=next(c for c in compiled.co_consts if isinstance(c,types.CodeType))
    parent.run_owned=types.FunctionType(code,vars(parent),'run_owned')


def verify(binding, own):
    guard.require(binding['schema']=='canonical_paper_walk_launch_v1','Wrong launch schema')
    guard.require(binding['root_review_complete'] is True,'Root review incomplete')
    guard.require(binding['mode'] in ('diagnostic','train','video'),'Unsupported native mode')
    guard.require(type(binding['max_seconds']) is int and 120<=binding['max_seconds']<=7200,'Unbounded allocation')
    paths={k:guard.canonical_path(binding[k]) for k in ('source','asset','prior','geometry_source','output')}
    guard.require(paths['source']==own,'Launcher must belong to its bound source')
    for k in ('source','output','prior'):
        guard.require(REMOTE_ROOT in paths[k].parents,'Fresh restart path required')
    for k,v in paths.items():
        if k!='output':
            guard.require(v!=paths['output'] and v not in paths['output'].parents
                          and paths['output'] not in v.parents,'Output overlaps immutable input')
    guard.verify_tree(paths['source'],binding['source_freeze_sha256'])
    for name,digest in binding['input_files'].items(): guard.pinned_file(name,digest)
    guard.require(guard.sha(SUPERVISOR/'campaign_source_hashes.json')==SUPERVISOR_MAP,'Supervisor map differs')
    guard.pinned_file(SUPERVISOR/'tools/launch_reference_physics_spark.py',SUPERVISOR_SHA)
    for relative,digest in guard.read(SUPERVISOR/'campaign_source_hashes.json').items():
        guard.pinned_file(SUPERVISOR/relative,digest)
    guard.require(binding.get('stage2_complete') is False and binding.get('physical_admission') is False,
                  'Launch cannot grant qualification')
    guard.require(binding['command_args'] and all(isinstance(x,str) for x in binding['command_args']),
                  'Exact native arguments required')
    argv=binding['command_args']
    guard.require(not any(x=='--output' or x.startswith('--output=') for x in argv),'Output override rejected')
    for option,want in (('--mode',binding['mode']),('--source-freeze-sha256',binding['source_freeze_sha256']),
                        ('--asset','/asset'),('--model','/asset/source/model.json'),('--device','cuda:0')):
        guard.require(argv.count(option)==1 and not any(x.startswith(option+'=') for x in argv),
                      'Exact single native argument required: '+option)
        index=argv.index(option)
        guard.require(index+1<len(argv) and argv[index+1]==want,'Native argument differs: '+option)
    guard.require(argv.count('--headless')==1,'Explicit headless allocation required')
    return paths


def command(binding, paths, name):
    args=['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml','--profile','base',
          'run','--rm','--no-deps','--name',name,'-w','/output','-e','PYTHONDONTWRITEBYTECODE=1',
          '-e','PYTHONUNBUFFERED=1','-e','PYTHONPATH=/source:/workspace/isaaclab/source/isaaclab',
          '-v',str(paths['output'])+':/output:rw','-v',str(paths['source'])+':/source:ro',
          '-v',str(paths['asset'])+':/asset:ro','-v',str(paths['prior'])+':/prior:ro',
          '-v',str(paths['geometry_source'])+':/geometry_source:ro']
    for source,target in binding.get('extra_mounts',[]):
        guard.canonical_path(source)
        guard.require(target in ('/standing_one','/standing_batch','/admission','/checkpoint'),'Unexpected read-only input mount')
        args+=['-v',source+':'+target+':ro']
    return args+['--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base',
                '/source/train.py','--output','/output/standing',*binding['command_args']]


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--bindings',type=Path,required=True)
    parser.add_argument('--bindings-sha256',required=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--cleanup-only',action='store_true')
    cli=parser.parse_args()
    guard.pinned_file(cli.bindings,cli.bindings_sha256)
    binding=guard.read(cli.bindings)
    paths=verify(binding,Path(__file__).resolve().parent)
    if cli.cleanup_only:
        with guard.both_locks():
            receipt=guard.cleanup_owned(paths['output'])
            receipt['reservation']=guard.verify_reservation()
            receipt['resources']=guard.no_live_compute()
            if paths['output'].exists():guard.save(paths['output']/'cleanup.json',receipt)
        print(json.dumps(receipt,indent=2));return
    guard.require(not paths['output'].exists(),'Attempt output must be fresh')
    with guard.both_locks():
        reservation=guard.verify_reservation(); resources=guard.no_live_compute()
    if cli.preflight_only:
        print(json.dumps({'reservation':reservation,'resources':resources,'mode':binding['mode'],
                          'native_started':False,'source_freeze_sha256':binding['source_freeze_sha256']},indent=2))
        return
    sys.path.insert(0,str(SUPERVISOR/'tools'))
    parent=load('_paper_walk_owned_supervisor',SUPERVISOR/'tools/launch_reference_physics_spark.py')
    adapt_deadline(parent,SUPERVISOR/'tools/launch_reference_physics_spark.py',binding['max_seconds'])
    original_preflight,original_resources,original_save=parent.preflight,parent.resources,parent.save
    def preflight():
        reserved=guard.verify_reservation();guard.no_live_compute()
        return {**original_preflight(),'current_reservation':reserved}
    def live_resources():
        guard.verify_policy_bytes();return original_resources()
    def save(path,data):
        if Path(path).parent.name=='jobs' and Path(path).name=='standing.json':
            data={**data,'no_policy_loaded':binding['mode']=='diagnostic','native_mode':binding['mode'],
                  'stage2_complete':False,'physical_admission':False,'source_freeze_sha256':binding['source_freeze_sha256']}
        return original_save(path,data)
    parent.preflight,parent.resources,parent.save=preflight,live_resources,save
    parent.command=lambda source,output,name,phase:command(binding,paths,name)
    parent.verified_source=lambda source:guard.verify_tree(source,binding['source_freeze_sha256'])
    parent.RUNTIME_TREE=binding['source_freeze_sha256']
    args=types.SimpleNamespace(source=paths['source'],output=paths['output'],isaaclab=Path('/home/orionh/IsaacLab'),
                               coordination_sha256=guard.COORDINATION_SHA256)
    args.output.mkdir(parents=True,exist_ok=False)
    for sub in ('jobs','logs'):(args.output/sub).mkdir()
    signal.signal(signal.SIGTERM,lambda *_:(args.output/'stop.request').touch())
    signal.signal(signal.SIGINT,lambda *_:(args.output/'stop.request').touch())
    guard.save(args.output/'launch_binding.json',binding)
    try:
        parent.run_owned(args,'standing')
    finally:
        with guard.both_locks():
            cleanup=guard.cleanup_owned(args.output)
            cleanup['reservation']=guard.verify_reservation();cleanup['resources']=guard.no_live_compute()
            verify(binding,Path(__file__).resolve().parent)
            guard.save(args.output/'cleanup.json',cleanup)


if __name__=='__main__':main()
