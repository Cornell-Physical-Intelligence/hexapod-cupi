#!/usr/bin/env python3
"""External rendering-only dispatch of the exact native direct315 evaluator."""
import argparse, sys, traceback
from pathlib import Path
sys.dont_write_bytecode = True
from preview_contract import read, save, verify_inputs, legacy_binding
from entry_adapter import instrument, configure_rendering

def native_arguments(args):
    return [str(args.source_root/'tools/train_length_study.py'),
        '--package',str(args.source_root/'robot/hexapod_mkii_length_study'),
        '--output',str(args.output),'--variant','f050_t060','--stance-index','0',
        '--mode','evaluate','--direct-evaluation','constant',
        '--admission',str(args.admission),'--checkpoint',str(args.checkpoint),
        '--enable_cameras','--headless','--device',args.device,'--kit_args='+args.kit_args]

def preserve_failure(output, exc):
    output.mkdir(parents=True,exist_ok=True)
    failure={'error':repr(exc),'traceback':traceback.format_exc(),
             'stage2_complete':False,'recording_only':True}
    save(output/'recording_failure.json',failure)
    state=read(output/'state.json') if (output/'state.json').is_file() else {}
    save(output/'state.json',{**state,'status':'failed','recording_failure':failure})
    video=read(output/'video.json') if (output/'video.json').is_file() else {}
    save(output/'video.json',{**video,'complete':False,'external_failure':failure})

def execute(args, provenance):
    entry=args.source_root/'tools/train_length_study.py'
    tree,counts=instrument(entry.read_text())
    save(args.output/'entry_seams.json',counts)
    # No Torch, NumPy or native robot imports before actual AppLauncher construction.
    sys.path[:0]=[str(args.source_root/'tools'),str(args.source_root/'isaaclab')]
    def ready(): print('REFERENCE_SCREEN_APP_READY',flush=True)
    def bind_save(original):
        legacy_binding(args.source_root)
        def metadata(path,data):
            if Path(path).name=='state.json':
                if 'runtime_binding' in data: raise ValueError('Unexpected preexisting runtime binding')
                data={**data,'runtime_binding':legacy_binding(args.source_root),
                      'recording_only':True,'recording_provenance':provenance,
                      'native_selection_used_for_construction_only':True,
                      'stage2_complete':False,'qualification_performed':False}
            return original(path,data)
        return metadata
    def recording(env,runner,plan,output,checkpoint_sha):
        from preview_recorder import record
        return record(env,runner,plan,output,checkpoint_sha,args,provenance)
    namespace={'__name__':'__main__','__file__':str(entry),'__package__':None,
        '_preview_app_ready':ready,'_preview_bind_save':bind_save,
        '_preview_render_config':lambda cfg:configure_rendering(cfg,args.output),
        '_preview_record':recording}
    sys.argv=native_arguments(args)
    exec(compile(tree,str(entry),'exec'),namespace)

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ('source-root','native-contract','pilot','checkpoint','admission','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--seed',type=int,default=7057)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--kit_args',default='--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry')
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(argv)
    for key in ('source_root','native_contract','pilot','checkpoint','admission','output'):
        setattr(args,key,getattr(args,key).resolve())
    return args

def main(argv=None):
    args=parse_args(argv);provenance=verify_inputs(args)
    if args.preflight_only:
        import json
        print(json.dumps(provenance,indent=2));return
    if args.output.exists():raise ValueError('Recording output must be fresh')
    args.output.mkdir(parents=True)
    save(args.output/'provenance.json',provenance)
    save(args.output/'native_arguments.json',native_arguments(args))
    try:
        execute(args,provenance)
        # Includes successful close of the real native App, not only the recording loop.
        if verify_inputs(args)!=provenance:raise ValueError('Inputs changed after native App closed')
        video=read(args.output/'video.json')
        if not video.get('complete'):raise ValueError('Missing complete recording receipt')
        save(args.output/'final_integrity.json',{'passed':True,'provenance':provenance})
    except BaseException as exc:
        preserve_failure(args.output,exc)
        raise

if __name__=='__main__':main()
