from pathlib import Path
import hashlib,importlib.util,json,sys,time
sys.dont_write_bytecode=True
b=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');root=b/'direct_omni_cold_guard_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'FREEZE_SHA256.json')=='07d6462ab27120c223e7d12d4b5dfef5d9948afea822236c88f5963f2625cde5'
s=importlib.util.spec_from_file_location('guard',root/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
g.verify_frozen(root,'07d6462ab27120c223e7d12d4b5dfef5d9948afea822236c88f5963f2625cde5');g.require_final_bindings()
s=importlib.util.spec_from_file_location('host',g.HOST);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
assert not g.PAUSE.exists() and not g.OUTPUT.exists()
identity=g.validate_cold_inputs(h);g.verify_previous_owner()
coord=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md');assert sha(coord)=='fdad0bb0126158988dc36ea38538b2f7804299cffb5ff29601ba597089672234'
r={'passed':True,'checked_unix':time.time(),'identity':identity,'guard_sha256':sha(root/'launch_guarded_remote.py'),'guard_freeze_sha256':sha(root/'FREEZE_SHA256.json'),'host_freeze_sha256':g.HOST_FREEZE_SHA256,'source_manifest_sha256':g.SOURCE_SHA256,'previous_receipts':g.PRIOR_PINS,'previous_owned_name_ID_absent':True,'coordination_sha256':sha(coord),'unit':g.UNIT,'pause':str(g.PAUSE),'output':str(g.OUTPUT),'phases':['standing','baseline'],'GPU_allocated':False}
p=b/'direct_omni_cold_final_preflight_001.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
