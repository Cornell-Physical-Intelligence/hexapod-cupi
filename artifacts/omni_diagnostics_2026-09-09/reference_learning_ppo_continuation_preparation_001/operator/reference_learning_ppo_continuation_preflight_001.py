from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,sys,time
sys.dont_write_bytecode=True
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
hroot=base/'reference_learning_ppo_host_001';groot=base/'reference_learning_ppo_continuation_guard_001'
h=load('bound_learning_host',hroot/'launch_learning_ppo_spark.py');g=load('bound_learning_guard',groot/'launch_guarded_remote.py')
h.verify_tree(hroot,'FREEZE_SHA256.json','45f623d298c00136ca18d159261d8df601bd42840ebae467fa94bf38283e6792')
h.verify_tree(groot,'FREEZE_SHA256.json','8b03cadd1224b8615d536e2fce2b37b7a6e723410dada59f6292d0dfb50de8f3')
g.require_final_bindings()
args=SimpleNamespace(source=g.SOURCE,run=g.RUN,device_run=g.DEVICE_RUN,bridge=g.BRIDGE,consumer=g.CONSUMER,observation=g.OBSERVATION,output=g.OUTPUT,phase_group='learn',decision_receipt=g.DECISION_RECEIPT,host_freeze_sha256=g.HOST_FREEZE_SHA256)
assert not g.PAUSE.exists() and not (g.OUTPUT/'learning_campaign.json').exists()
baseline=h.validate_inputs(args)
g.verify_previous_owner()
assets,decision=g.admit_learning_before_pause(h,args,baseline)
coord=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md');assert sha(coord)=='fdad0bb0126158988dc36ea38538b2f7804299cffb5ff29601ba597089672234'
report={'passed':True,'checked_unix':time.time(),'baseline':baseline,'decision':decision,'admitted_asset_files':len(assets),'prior_payloads_verified':len(args.prior_file_hashes),'source_manifest_sha256':sha(g.SOURCE/'campaign_source_hashes.json'),'host_sha256':sha(g.HOST),'host_freeze_sha256':sha(hroot/'FREEZE_SHA256.json'),'guard_sha256':sha(groot/'launch_guarded_remote.py'),'guard_freeze_sha256':sha(groot/'FREEZE_SHA256.json'),'coordination_sha256':sha(coord),'previous_receipts':g.PRIOR_PINS,'previous_exact_owned_identifiers_absent':6,'fresh_pause':str(g.PAUSE),'phases':h.LEARNING_PHASES,'GPU_allocated':False,'prior_admission_modified':False}
p=base/'reference_learning_ppo_continuation_preflight_001.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'preflight':str(p),'decision_sha256':decision['decision_sha256'],'phases':h.LEARNING_PHASES}))
