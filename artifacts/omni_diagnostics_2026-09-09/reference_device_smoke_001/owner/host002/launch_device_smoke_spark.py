"""Two sequential source-bound CUDA bridge checks; no policy or walking dispatch."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib.util
import json
import math
import signal
import sys
import time
from numeric_evidence import numeric,require_eight_float32_ticks

sys.dont_write_bytecode=True
SOURCE_MAP='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
ADAPTER_FREEZE='be4870a8f7ca0857ba50ffd3a0c92ac5925b9a816767d66d826af7b59957aa0e'
OBSERVATION_FREEZE='22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63'
RUNTIME_TREE='abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'
HOST_SOURCE_SHA='9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
PHASES={'replicas_1':1,'replicas_32':32}
READER_COMPARISON_KEYS={'position_world_m','quaternion_world_xyzw','velocity_world_mps',
    'joint_position_rad','joint_velocity_rad_s','joint_target_rad','computed_torque_nm','applied_torque_nm',
    'distal_contact','contact_point_world_m','reference_point_world_m','reference_point_velocity_world_mps'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def verify_tree(path,manifest,bound):
    if sha(path/manifest)!=bound:raise ValueError('Wrong pinned input manifest: '+str(path))
    if any(p.is_symlink() for p in path.rglob('*')):raise ValueError('Symbolic input substitution')
    actual={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()};actual.pop(manifest)
    if actual!=read(path/manifest):raise ValueError('Changed or unlisted input: '+str(path))

def verify_own_bundle(expected=None):
    root=Path(__file__).resolve().parent;actual=sha(root/'FREEZE_SHA256.json')
    if expected is not None and actual!=expected:raise ValueError('Host freeze changed during campaign')
    verify_tree(root,'FREEZE_SHA256.json',actual)
    return actual

def validate_inputs(args):
    """Read-only host preflight, callable before a forecasting pause or GPU lock."""
    verify_own_bundle(getattr(args,'host_freeze_sha256',None))
    verify_tree(args.source,'campaign_source_hashes.json',SOURCE_MAP)
    verify_tree(args.adapter,'FREEZE_SHA256.json',ADAPTER_FREEZE)
    verify_tree(args.observation,'FREEZE_SHA256.json',OBSERVATION_FREEZE)
    if sha(args.source/'tools/launch_reference_physics_spark.py')!=HOST_SOURCE_SHA:
        raise ValueError('Exact reviewed source009 supervisor required')
    path=args.adapter/'smoke_contract.py'
    spec=importlib.util.spec_from_file_location('_pinned_device_smoke_contract',path)
    contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
    return {phase:contract.verify(SimpleNamespace(source_root=args.source,run=args.run,
        observation_bundle=args.observation,package=args.run/'inputs/study',num_envs=n,controls=264))
        for phase,n in PHASES.items()}

def command(args,name,phase):
    if phase not in PHASES:raise ValueError('Only replicas_1 and replicas_32 bridge phases exist')
    return ['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml',
        '--profile','base','run','--rm','--no-deps','--name',name,'-w','/outputs',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
        '-e','PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/hexapod/tools:/workspace/hexapod/isaaclab',
        '-v',f'{args.source}:/workspace/hexapod:ro','-v',f'{args.run}:/qualified:ro',
        '-v',f'{args.adapter}:/adapter:ro','-v',f'{args.observation}:/observation:ro',
        '-v',f'{args.output}:/outputs:rw',
        '--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base',
        '/adapter/run_device_smoke.py','--source-root','/workspace/hexapod','--run','/qualified',
        '--observation-bundle','/observation','--package','/qualified/inputs/study',
        '--output','/outputs/'+phase,'--num-envs',str(PHASES[phase]),'--controls','264',
        '--headless','--device','cuda:0','--info',
        '--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry']

def validate_result(output,identity):
    """Independently check declared outcomes and the compact numeric evidence."""
    n=identity['replicas'];s=read(output/'state.json')
    expected={'status':'completed','identity':identity,'controls':264,'bridge_passed':True,
        'source_inputs_unchanged':True,'actor_width':846,'critic_width':849,'substep_torque_checked':True,
        'short_hold_is_not_quiet_admission':True,'policy_training_started':False,'stage2_complete':False,
        'legacy_env_step_includes_original_CPU_telemetry_copies':True}
    if any(s.get(key)!=value for key,value in expected.items()) or s.get('runtime_binding',{}).get('runtime_tree_sha256')!=RUNTIME_TREE:
        raise ValueError('Bridge state is incomplete or has a different source/runtime contract')
    compared=s.get('comparison_keys',[])
    if not isinstance(compared,list) or any(not isinstance(k,str) for k in compared) or len(set(compared))!=len(compared) or not READER_COMPARISON_KEYS<=set(compared):
        raise ValueError('Independent source009 reader comparison evidence missing')
    for key in ('postsettle_max_requested_nm','postsettle_max_applied_nm'):
        value=s.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1.6:
            raise ValueError('Invalid postsettle torque maximum: '+key)
    sub=read(output/'physics_substep_review.json')
    expected_sub={'samples_including_initial':2113,'completed_controls':264,'error':None,
        'method_restored':True,'physical_acceptance_gate_changed':False,'decimation':8,'physics_dt_s':.0025,
        'physics_handles_decimation':False,'postsettle_worst_joint_substep_saturation_fraction':0.}
    if s.get('substep_review')!=sub or any(sub.get(key)!=value for key,value in expected_sub.items()):
        raise ValueError('Substep observer did not complete its exact original contract')
    schema=read(output/'observation_schema.json');schema_copy=dict(schema);claimed=schema_copy.pop('schema_sha256',None)
    if claimed!=hashlib.sha256(json.dumps(schema_copy,sort_keys=True).encode()).hexdigest():
        raise ValueError('Observation schema receipt changed')
    if (schema.get('actor_width')!=846 or schema.get('critic_width')!=849
        or schema.get('schema')!='c_wave005_residual002_device_observation_v2'
        or schema.get('policy_training_allowed') is not False or schema.get('raw_reported_velocity_preserved') is not True
        or schema.get('reported_velocity_physical_consistency_verified') is not False
        or schema.get('old_checkpoint_compatible') is not False
        or schema.get('joint_names_runtime')!=sub.get('joint_names_runtime')):
        raise ValueError('Wrong observation, joint-order or velocity provenance contract')
    obs=output/'final_observation.npz'
    numeric(obs,'policy',(n,846));numeric(obs,'critic',(n,849))
    raw=numeric(obs,'raw_sdk_joint_velocity_rad_s',(n,18));numeric(obs,'interval_joint_rate_rad_s',(n,18))
    if not all(numeric(obs,'interval_rate_valid',(n,),'bool')):raise ValueError('Missing interval-rate history')
    sample=output/'last_device_sample.npz'
    if raw!=numeric(sample,'joint_velocity_rad_s',(n,18)):raise ValueError('Raw SDK velocity channel changed')
    for key,shape in [('measurement_valid',(n,)),('distal_contact',(n,6)),('contact_valid',(n,6))]:
        if not all(numeric(sample,key,shape,'bool')):raise ValueError('Invalid final measurement/contact: '+key)
    for key,shape in [('terminated',(n,)),('truncated',(n,)),('shaft_contact',(n,6)),('coxa_contact',(n,6)),('femur_contact',(n,6)),('base_contact',(n,))]:
        if any(numeric(sample,key,shape,'bool')):raise ValueError('Final terminal/nonfoot contact: '+key)
    clocks=output/'sensor_clocks.npz';shape=(264,n,14)
    current=numeric(clocks,'sensor_timestamp_s',shape);last=numeric(clocks,'sensor_last_update_s',shape)
    expected_clock=numeric(clocks,'expected_timestamp_s',shape)
    if current!=last or current!=expected_clock or min(current)<0 or any(numeric(clocks,'sensor_outdated',shape,'bool')):
        raise ValueError('Actual sensor freshness evidence failed')
    require_eight_float32_ticks(current,n*14)
    if not all(numeric(clocks,'all_sensors_valid',(264,n),'bool')) or not all(numeric(clocks,'contact_valid',(264,n,6),'bool')):
        raise ValueError('Stale sensor row cannot be promoted by its final sample')
    if any(numeric(clocks,'sensor_age_s',shape)) or any(numeric(clocks,'contact_age_s',(264,n,6))):
        raise ValueError('Source-bound synchronous lazy sensor cache was not current')
    subpath=output/'physics_substeps.npz'
    for field,key in [('computed_torque_nm','postsettle_max_requested_nm'),('applied_torque_nm','postsettle_max_applied_nm')]:
        values=numeric(subpath,field,(2113,n,18));maximum=max(abs(v) for v in values[1601*n*18:])
        if maximum!=s[key] or maximum>1.6:raise ValueError('Interior substep torque differs from declared maximum')
    numeric(subpath,'joint_position_rad',(2113,n,18));numeric(subpath,'joint_velocity_rad_s',(2113,n,18))
    times=numeric(subpath,'time_s',(2113,))
    if any(abs(value-index*.0025)>1e-9 for index,value in enumerate(times)):
        raise ValueError('Missing/reordered physical timestamps')
    timing=read(output/'timings.json')
    if len(timing.get('samples',[]))!=64:raise ValueError('Incomplete hold timing window')
    for index,row in enumerate(timing['samples']):
        if row.get('control')!=201+index:raise ValueError('Out-of-order hold timing')
        for key in ('reference_and_target_set_ms','env_step_including_capture_ms','nested_device_capture_ms','observation_history_ms'):
            value=row.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Nonfinite or missing timing measurement')
    required=('state.json','environment.yaml','solver_comparison.json','sensor_clocks.npz',
        'physics_substeps.npz','physics_control_integrals.npz','physics_substep_review.json','last_device_sample.npz',
        'last_reference_state.npz','final_observation.npz','observation_schema.json','timings.json')
    return {'passed':True,'replicas':n,'controls':264,'schema_sha256':claimed,
        'files_sha256':{name:sha(output/name) for name in required},'PPO_admitted':False,
        'fresh_quiet_admission':False,'native_velocity_fidelity_qualified':False}

def run_phases(args,host,identities,report):
    """No second allocation is possible before first result and input rechecks."""
    for phase in PHASES:
        if validate_inputs(args)!=identities:raise ValueError('Inputs changed between phases')
        host.check_source(args.source)
        host.run_owned(args,phase)
        result=validate_result(args.output/phase,identities[phase])
        save(args.output/(phase+'_accepted.json'),result)
        report.setdefault('accepted_phases',{})[phase]=result
        report.update(status='running',last_completed_phase=phase)
        save(args.output/'campaign.json',report)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','run','adapter','observation','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'))
    args=parser.parse_args()
    for name in ('source','run','adapter','observation','output','isaaclab'):setattr(args,name,getattr(args,name).resolve())
    if args.output.exists() or any(args.output==p or p in args.output.parents for p in (args.source,args.run,args.adapter,args.observation,Path(__file__).resolve().parent)):
        raise ValueError('Fresh output outside all immutable inputs required')
    args.host_freeze_sha256=verify_own_bundle()
    identities=validate_inputs(args)
    sys.path.insert(0,str(args.source/'tools'))
    import launch_reference_physics_spark as host
    if Path(host.__file__).resolve()!=args.source/'tools/launch_reference_physics_spark.py':raise ValueError('Wrong imported supervisor')
    runtime=host.check_source(args.source);args.coordination_sha256=host.digest(host.COORDINATION)
    args.output.mkdir(parents=True)
    for name in ('jobs','logs'):(args.output/name).mkdir()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:(args.output/'stop.request').touch())
    original_command=host.command
    host.command=lambda source,output,name,phase:command(args,name,phase)
    report=dict(status='preparing',started_unix=time.time(),source_manifest_sha256=SOURCE_MAP,
        host_freeze_sha256=args.host_freeze_sha256,
        adapter_freeze_sha256=ADAPTER_FREEZE,observation_freeze_sha256=OBSERVATION_FREEZE,
        runtime_binding=runtime,identities=identities,policy_training_started=False,stage2_complete=False,
        scope='Two short zero-command device bridge allocations; no automatic walking/PPO successor')
    try:
        save(args.output/'campaign.json',report)
        run_phases(args,host,identities,report)
        report.update(status='completed',both_bridge_phases_passed=True)
    except Exception as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        host.command=original_command
        try:
            host.check_source(args.source)
            if validate_inputs(args)!=identities:raise ValueError('Terminal input identity changed')
            report['source_and_inputs_unchanged']=True
        except Exception as integrity_error:
            report.update(status='failed',source_and_inputs_unchanged=False,integrity_error=repr(integrity_error));raise
        finally:
            report['finished_unix']=time.time();save(args.output/'campaign.json',report)

if __name__=='__main__':main()
