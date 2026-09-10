#!/usr/bin/env python3
"""External native recording of exact source009's unchanged 48 s physical screen."""
import argparse,faulthandler,json,os,sys,threading,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
from recording_contract import verify_inputs,save_json,external_hashes,load_frozen_functions,STEPS

def parser_base():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    for flag in ['source-root','package','output','campaign','admission','study-tree-receipt','geometry-reference']:p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    return p

def preflight(early):
    for key in ['source_root','package','output','campaign','admission','study_tree_receipt','geometry_reference']:setattr(early,key,getattr(early,key).resolve())
    provenance=verify_inputs(early)
    for protected in [early.source_root,early.package,early.campaign.parent,Path(__file__).resolve().parent]:
        if early.output==protected or protected in early.output.parents:raise ValueError('Output must be outside immutable source, inputs and recording code')
    early.mode='wave';early.num_envs=1;early.steps=STEPS;early.variant='f050_t060';early.stance_index=0
    sys.path.insert(0,str(early.source_root/'tools'))
    from screen_contract import preflight as frozen_preflight
    identity,geometry=frozen_preflight(early,early.source_root)
    if identity!=provenance['identity']:raise ValueError('Exact source009 preflight identity mismatch')
    provenance.update(recording_source_hashes=external_hashes(Path(__file__).resolve().parent),fresh_physics_recording=True,original009_video=False,stage2_complete=False,policy_training_started=False,pose_forcing=False,rendering_overrides={'num_envs':1,'width':1280,'height':720,'enable_cameras':True,'render_mode':'rgb_array'})
    return identity,geometry,provenance

def main():
    parser=parser_base();early,_=parser.parse_known_args()
    # AppLauncher is imported before any Torch-dependent runtime import. The
    # simulator is constructed only after exact source/assets/admission checks.
    if not early.preflight_only:
        verify_inputs(early);sys.path.insert(0,str(early.source_root.resolve()/'tools'))
        from c_study_runtime import bootstrap_c_study_runtime
        runtime=bootstrap_c_study_runtime()
        from isaaclab.app import AppLauncher
    identity,geometry,provenance=preflight(early)
    if early.preflight_only:print(json.dumps(provenance,indent=2));return 0
    AppLauncher.add_app_launcher_args(parser);parser.add_argument('-h','--help',action='help');args=parser.parse_args()
    for key in vars(early):setattr(args,key,getattr(early,key))
    args.enable_cameras=True;args.output.mkdir(parents=True,exist_ok=False)
    provenance['runtime_binding']=runtime;provenance['started_unix']=time.time();save_json(args.output/'provenance.json',provenance)
    faulthandler.enable();faulthandler.dump_traceback_later(45,repeat=True)
    def timeout():
        save_json(args.output/'startup_timeout.json',{'failed':True,'deadline_s':90,'stage2_complete':False})
        os._exit(124)
    timer=threading.Timer(90,timeout);timer.daemon=True;timer.start()
    def startup_ready():timer.cancel();faulthandler.cancel_dump_traceback_later()
    app=None;recorder=None;exit_code=1
    try:
        print('REFERENCE_SCREEN_APP_START',flush=True);app=AppLauncher(args).app
        print('REFERENCE_SCREEN_APP_READY',flush=True)
        import numpy as np
        import torch
        from screen_contract import OPTIONS,WAVE,STARTUP,save
        from physics_telemetry import capture_measured_state as raw_capture,pre_reset_capture,require_single_pre_reset_sample,array
        from screen_metrics import standing_screen,standing_quiet_review,physical_metrics,measured_flight_touchdowns,measured_progress
        from canonical_stance_startup import CanonicalStanceStartup
        from physics_substeps import PhysicsSubstepRecorder
        from native_recorder import NativeRecorder,build_recording_environment
        recorder=NativeRecorder(args,provenance,startup_ready)
        def capture_measured_state(*a,**kw):return recorder.capture(raw_capture(*a,**kw))
        def build_reference_environment(a,options):return build_recording_environment(a,options,recorder)
        namespace=dict(args=args,identity=identity,geometry_reference=geometry,RUNTIME=runtime,np=np,torch=torch,time=time,traceback=traceback,
            OPTIONS=OPTIONS,WAVE=WAVE,STARTUP=STARTUP,save=save,capture_measured_state=capture_measured_state,pre_reset_capture=pre_reset_capture,
            require_single_pre_reset_sample=require_single_pre_reset_sample,array=array,standing_screen=standing_screen,standing_quiet_review=standing_quiet_review,
            physical_metrics=physical_metrics,measured_flight_touchdowns=measured_flight_touchdowns,measured_progress=measured_progress,
            CanonicalStanceStartup=CanonicalStanceStartup,PhysicsSubstepRecorder=PhysicsSubstepRecorder,build_reference_environment=build_reference_environment)
        provenance['unchanged_frozen_functions']=load_frozen_functions(args.source_root,namespace);save_json(args.output/'provenance.json',provenance)
        exit_code=namespace['main']()
    except Exception as error:
        save_json(args.output/'recording_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'stage2_complete':False})
        raise
    finally:
        startup_ready()
        try:
            verified=verify_inputs(args)=={k:v for k,v in provenance.items() if k in ['source_manifest_sha256','campaign_sha256','standing_admission_sha256','original_wave_state_sha256','study_tree_sha256','asset_files','identity']}
            verified=verified and external_hashes(Path(__file__).resolve().parent)==provenance['recording_source_hashes']
            if recorder is not None:
                report=recorder.finish(verified)
                if not report['complete']:exit_code=1
            elif not verified:exit_code=1
        finally:
            if app is not None:app.close()
    return exit_code

if __name__=='__main__':raise SystemExit(main())
